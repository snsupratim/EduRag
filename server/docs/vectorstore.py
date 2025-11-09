import os
import time
import asyncio
import uuid
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

# === New Imports for Multimodal ===
import cohere
import fitz  # PyMuPDF for PDF extraction
from PIL import Image
# ==================================

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# === API Keys and Setup ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAMES")
COHERE_API_KEY = os.getenv("COHERE_API_KEY") 

# Initialize Cohere Client
co = cohere.Client(api_key=COHERE_API_KEY)

# --- CRITICAL FIX: Set Model and Dimension for embed-v4.0 ---
EMBED_MODEL = "embed-v4.0" 
# Use 1024 for a balance, but 1536 is the default if not specified
EMBEDDING_DIM = 1536 
# -----------------------------------------------------------

UPLOAD_DIR = "./uploaded_docs"
IMAGE_DIR = "./uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# --- Pinecone Setup ---
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud='aws', region=PINECONE_ENV)
existing_index = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_index:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM, # Match dimension to the model's output
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


# --- Multimodal Utility Functions ---
MAX_PIXELS = 1568 * 1568 # Cohere image size limit

def resize_image(pil_image):
    """Resizes image to fit Cohere max pixel constraint if necessary."""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        # Use Image.Resampling.LANCZOS for high quality resizing
        pil_image.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
    return pil_image

def image_to_base64_data_url(img_path):
    """Converts a local image path to a base64 Data URL (required for some Cohere API calls)."""
    pil_image = Image.open(img_path).convert("RGB")
    resized_image = resize_image(pil_image)

    img_buffer = io.BytesIO()
    resized_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    encoded_string = base64.b64encode(img_buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded_string


# --- Main Vectorstore Loading Function ---
async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        # 1. Extract Text and Images using PyMuPDF (fitz)
        doc = fitz.open(save_path)
        text_chunks_raw = []
        image_data = [] # Stores: (img_path, page_num)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        print(f"Extracting content from {file.filename}...")
        for page_num in tqdm(range(len(doc)), desc="Extracting pages"):
            page = doc[page_num]
            
            # Text Extraction and Splitting
            text = page.get_text()
            if text.strip():
                temp_doc = Document(
                    page_content=text, 
                    metadata={"page": page_num, "source": file.filename}
                )
                chunks = splitter.split_documents([temp_doc])
                text_chunks_raw.extend(chunks)

            # Image Extraction
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                img_filename = f"{doc_id}_page{page_num+1}_img{img_index+1}.{img_ext}"
                img_path = Path(IMAGE_DIR) / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                image_data.append({
                    "path": str(img_path),
                    "page": page_num,
                    "filename": img_filename
                })
        
        
        # 2. Embed and Upsert Text Chunks
        texts = [chunk.page_content for chunk in text_chunks_raw]
        text_ids = [f"text-{doc_id}-{uuid.uuid4()}" for _ in range(len(texts))]
        text_metadatas = [
            {
                "source": file.filename,
                "doc_id": doc_id,
                "role": role,
                "page": chunk.metadata.get("page", 0),
                "type": "text",
                "text": chunk.page_content,
            }
            for i, chunk in enumerate(text_chunks_raw)
        ]

        print(f"Embedding {len(texts)} text chunks with Cohere...")
        
        # Cohere Embedding Call for Text
        text_embeddings_res = await asyncio.to_thread(
            co.embed,
            model=EMBED_MODEL,
            texts=texts,
            input_type="search_document",
            # output_dimension=EMBEDDING_DIM # 👈 PASS DIMENSION HERE
        )
        text_embeddings = text_embeddings_res.embeddings
        
        # Upsert Text Vectors
        print("Uploading text to Pinecone...")
        # Since text_vectors is a zip object, we convert it to a list before iterating
        text_vectors = list(zip(text_ids, text_embeddings, text_metadatas)) 
        
        # Batch upsert 
        for i in tqdm(range(0, len(text_vectors), 100), desc="Upserting Text Batches"):
            batch_vectors = text_vectors[i:i+100]
            index.upsert(vectors=batch_vectors)


        # 3. Embed and Upsert Images 
        image_vectors = []
        
        print(f"Embedding {len(image_data)} images with Cohere...")
        for i, img_data in tqdm(enumerate(image_data), desc="Embedding Images"):
            img_path = img_data["path"]
            img_filename = img_data["filename"]
            
            try:
                # CRITICAL FIX: Use the base64 data URL input format for multimodal
                base64_url = image_to_base64_data_url(img_path)
                
                image_input = {
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}}
                    ]
                }
                
                image_embed_res = await asyncio.to_thread(
                    co.embed,
                    model=EMBED_MODEL,
                    inputs=[image_input], # Pass multimodal input via 'inputs'
                    input_type="search_document",
                    # output_dimension=EMBEDDING_DIM # 👈 PASS DIMENSION HERE
                )
                image_emb = image_embed_res.embeddings[0]
                
                # No need for co.files.upload/delete if using base64 URL directly

                image_vectors.append((
                    f"image-{doc_id}-{uuid.uuid4()}",
                    image_emb,
                    {
                        "source": file.filename,
                        "doc_id": doc_id,
                        "role": role,
                        "page": img_data["page"],
                        "type": "image",
                        "img_path": img_path, 
                        "text": f"A relevant image from page {img_data['page']} of document {file.filename}." 
                    }
                ))
            except Exception as e:
                print(f"⚠️ Error processing image {img_filename}: {e}")
                
        # Upsert Image Vectors
        print(f"Uploading {len(image_vectors)} image vectors to Pinecone...")
        if image_vectors:
            # Batch upsert for images
            for i in tqdm(range(0, len(image_vectors), 100), desc="Upserting Image Batches"):
                batch_vectors = image_vectors[i:i+100]
                index.upsert(vectors=batch_vectors)


        print(f"Upload complete for {file.filename} (Text and Images)")