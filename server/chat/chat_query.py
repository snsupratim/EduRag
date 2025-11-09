import os
import asyncio
import io
from dotenv import load_dotenv
from pathlib import Path # Added for robust path handling in utility

# === New Imports for Multimodal ===
import cohere
from PIL import Image
from langchain_core.messages import HumanMessage
# ==================================

from pinecone import Pinecone
from langchain_groq import ChatGroq # Kept, though LLM is Gemini for multimodal
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()

# === API Keys and Setup ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Ensure PINECONE_INDEX_NAME is correctly set in your .env
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAMES") 
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere Client
co = cohere.Client(api_key=COHERE_API_KEY)
EMBED_MODEL = "embed-v4.0"
EMBEDDING_DIM = 1536 # Must match Pinecone index dimension (1536 is default for v4.0)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Use the Gemini model that supports multimodal input
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY 
)

# --- Multimodal Utility Function ---

def create_image_part(img_path: str):
    """
    Loads an image from the local path and returns a PIL Image object
    for use in the LangChain/Gemini multimodal message.
    """
    try:
        # Check if file exists before trying to open
        if not Path(img_path).exists():
            print(f"⚠️ Image file not found at {img_path}")
            return None
            
        pil_image = Image.open(img_path).convert("RGB")
        return pil_image 
    except Exception as e:
        print(f"⚠️ Error loading image {img_path}: {e}")
        return None

# --- RAG QUERY FUNCTION (FIXED) ---

async def answer_query(query:str, user_role:str):

    # 1. Embed Query using Cohere
    print("Embedding query with Cohere...")
    query_res = await asyncio.to_thread(
        co.embed,
        model=EMBED_MODEL,
        texts=[query],
        input_type="search_query"
    )
    embedding = query_res.embeddings[0]
    
    # 2. Query Pinecone
    results = await asyncio.to_thread(index.query, vector=embedding, top_k=5, include_metadata=True)

    # 3. Process Results and build Multimodal Parts
    sources = set()
    
    # List to collect image paths for Streamlit
    retrieved_image_paths = [] 
    
    # Textual context instruction for the LLM
    text_parts = [
        "You are a helpful assistant. Answer the user's question based ONLY on the provided context, which includes both text excerpts and images. Analyze the images if they are relevant.\n\n**RETRIEVED CONTEXT:**\n"
    ]
    
    # List to hold PIL Image objects for the LLM
    image_parts = []

    for match in results["matches"]:
        metadata = match["metadata"]
        
        # Role-based filtering
        if metadata.get("role") == user_role:
            sources.add(metadata.get("source"))
            
            if metadata.get("type") == "text":
                text_content = metadata.get("text")
                if text_content:
                    text_parts.append(f"--- TEXT CHUNK (Source: {metadata.get('source')}, Page: {metadata.get('page')}) ---\n{text_content}\n")

            elif metadata.get("type") == "image":
                img_path = metadata.get("img_path")
                pil_image = create_image_part(img_path)
                
                if pil_image:
                    # Add PIL image object to the image list (for LLM analysis)
                    image_parts.append(pil_image)
                    
                    # Collect the path for the Streamlit frontend
                    retrieved_image_paths.append(img_path) 
                    
                    # Add a descriptive text reference to the textual context
                    text_parts.append(f"--- IMAGE REFERENCE: A relevant image from Page {metadata.get('page')} of {metadata.get('source')} is attached for analysis. ---")


    if not text_parts and not image_parts:
        return {"answer": "No relevant text or images found in your accessible documents.", "sources": []}
    
    # 4. Construct the Multimodal Message
    
    full_context_text = "\n".join(text_parts)
    
    multimodal_content = [
        full_context_text,
        *image_parts, 
        f"\n\n**USER QUESTION:** {query}"
    ]
    
    multimodal_message = HumanMessage(content=multimodal_content)
    
    # 5. Invoke the LLM
    final_answer = await asyncio.to_thread(llm.invoke, [multimodal_message])

    # 6. Return the result
    return {
        "answer": final_answer.content,
        "sources": list(sources),
        "retrieved_images": retrieved_image_paths # This sends the paths to the frontend
    }