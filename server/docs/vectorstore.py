import os, io, cv2, fitz, pdfplumber, asyncio, time
from PIL import Image
import numpy as np
import easyocr
from tqdm.auto import tqdm
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "medical-rag-index"
IMAGE_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/page_images"  # static image URL base

reader = easyocr.Reader(['en'], gpu=False)


async def load_vectorstore(uploaded_files, role: str, doc_id: str, upload_dir="./uploaded_docs"):
    """Extracts text, tables, and images (with OCR) → embeds with Gemini → stores in Pinecone."""

    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud='aws', region=PINECONE_ENV)

    # --- Ensure Index Exists ---
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🧱 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,
            metric="dotproduct",
            spec=spec
        )

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("⏳ Waiting for Pinecone index to be ready...")
        time.sleep(2)

    index = pc.Index(PINECONE_INDEX_NAME)
    # Ensure query endpoint is live
    for i in range(10):
        try:
            index.describe_index_stats()
            print("✅ Pinecone index ready & accessible.")
            break
        except Exception:
            print(f"⏳ Waiting for query endpoint ({i+1}/10)...")
            time.sleep(3)

    # --- Clear old vectors for same role ---
    try:
        index.delete(filter={"role": role})
        print(f"🧹 Cleared old vectors for role '{role}'")
    except Exception as e:
        if "Namespace not found" in str(e):
            print(f"ℹ️ No previous namespace found for role '{role}', skipping delete.")
        else:
            print(f"⚠️ Cleanup warning: {e}")

    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    os.makedirs(upload_dir, exist_ok=True)

    for file in uploaded_files:
        save_path = Path(upload_dir) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        print(f"📄 Processing: {file.filename}")
        text_blocks = []

        with pdfplumber.open(save_path) as pdf, fitz.open(save_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                pdf_page = pdf.pages[page_num - 1]

                # ---- Table extraction (strong settings) ----
                tables = pdf_page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_x_tolerance": 5,
                    "intersection_y_tolerance": 5,
                })
                if tables:
                    for table in tables:
                        table_text = "\n".join([" | ".join([cell or '' for cell in row]) for row in table])
                        text_blocks.append({
                            "text": "TABLE DATA:\n" + table_text,
                            "type": "table",
                            "page": page_num,
                            "source": file.filename
                        })
                else:
                    # Fallback OCR if pdfplumber found nothing
                    pix = page.get_pixmap(dpi=200)
                    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                    ocr_text = "\n".join(reader.readtext(img_arr, detail=0, paragraph=True)).strip()
                    if " | " in ocr_text or ":" in ocr_text:
                        text_blocks.append({
                            "text": "OCR TABLE (fallback):\n" + ocr_text,
                            "type": "ocr_table",
                            "page": page_num,
                            "source": file.filename
                        })

                # ---- OCR for scanned pages ----
                if len(page_text) < 50:
                    pix = page.get_pixmap(dpi=350)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ocr_text = "\n".join(reader.readtext(gray, detail=0, paragraph=True)).strip()
                    if ocr_text:
                        text_blocks.append({
                            "text": ocr_text,
                            "type": "ocr_page",
                            "page": page_num,
                            "source": file.filename
                        })
                        print(f"📃 Page {page_num}: OCR text extracted.")
                else:
                    text_blocks.append({
                        "text": page_text,
                        "type": "text",
                        "page": page_num,
                        "source": file.filename
                    })

                # ---- Image extraction ----
                images = page.get_images(full=True)
                for i_idx, img in enumerate(images):
                    xref = img[0]
                    base_img = doc.extract_image(xref)
                    img_data = base_img["image"]
                    img_ext = base_img["ext"]

                    image_dir = Path(upload_dir) / "page_images"
                    image_dir.mkdir(parents=True, exist_ok=True)
                    file_stem = Path(file.filename).stem
                    img_path = image_dir / f"{file_stem}_page{page_num}_img{i_idx}.{img_ext}"
                    with open(img_path, "wb") as out:
                        out.write(img_data)

                    img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    img_text = "\n".join(reader.readtext(img_cv, detail=0, paragraph=True)).strip()

                    image_url = f"{IMAGE_BASE_URL}/{img_path.name}"
                    text_blocks.append({
                        "text": img_text if img_text else "[Image with no detected text]",
                        "type": "image",
                        "page": page_num,
                        "source": file.filename,
                        "image_path": image_url
                    })
                    print(f"🖼️ Page {page_num}: Image {i_idx} ({'with text' if img_text else 'no text'})")

        # ---- Embed and upsert ----
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_chunks, metas = [], []
        for blk in text_blocks:
            chunks = splitter.split_text(blk["text"])
            for c in chunks:
                all_chunks.append(c)
                metas.append({
                    "source": blk["source"],
                    "doc_id": doc_id,
                    "role": role,
                    "page": blk["page"],
                    "type": blk["type"],
                    "image_path": blk.get("image_path") or "",
                    "text": c
                })

        if not all_chunks:
            print(f"⚠️ No text extracted from {file.filename}")
            continue

        print(f"🧠 Embedding {len(all_chunks)} chunks...")
        embeddings = await asyncio.to_thread(embed_model.embed_documents, all_chunks)
        ids = [f"{doc_id}-{i}" for i in range(len(all_chunks))]

        print("🚀 Upserting to Pinecone...")
        with tqdm(total=len(embeddings), desc="Pinecone Upsert") as bar:
            index.upsert(vectors=zip(ids, embeddings, metas))
            bar.update(len(embeddings))

        print(f"✅ Upload complete for {file.filename}")

    print("🎉 All documents processed successfully!")
