# 📁 docs/vectorstore.py

import os
import time
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "enterprise-rag-index"


async def load_vectorstore(uploaded_files, role: str, doc_id: str, upload_dir: str = "./uploaded_docs"):
    """
    Ingest enterprise PDFs (reports, manuals, policies) into Pinecone.

    Pipeline:
    - Save uploaded PDF(s)
    - Extract page-wise TEXT only (no OCR, no images)
    - Chunk text into semantically meaningful pieces
    - Store chunks in Pinecone with rich metadata:
        - source (filename)
        - doc_id (UUID for this upload)
        - role (access control)
        - page (page number)
        - text (chunk content)
    """

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    # --- Ensure Index Exists ---
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🧱 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,          # Gemini embedding dimension
            metric="dotproduct",
            spec=spec,
        )

    # Wait until index is ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("⏳ Waiting for Pinecone index to be ready...")
        time.sleep(2)

    index = pc.Index(PINECONE_INDEX_NAME)

    # Optional: clear old vectors for same role (fresh workspace per role)
    try:
        index.delete(filter={"role": role})
        print(f"🧹 Cleared old vectors for role '{role}'")
    except Exception as e:
        print(f"⚠️ Cleanup warning (can ignore if first run): {e}")

    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    os.makedirs(upload_dir, exist_ok=True)

    for file in uploaded_files:
        save_path = Path(upload_dir) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        print(f"📄 Processing: {file.filename}")
        text_blocks = []

        # Extract text per page
        with pdfplumber.open(save_path) as pdf, fitz.open(save_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                page_text = page_text.strip()

                if not page_text or len(page_text) < 20:
                    # Skip pages with almost no text (scanned / empty)
                    print(f"⚠️ Page {page_num}: Ignored (no extractable text)")
                    continue

                text_blocks.append(
                    {
                        "text": page_text,
                        "page": page_num,
                        "source": file.filename,
                    }
                )

        # ---- Chunk + Embed + Upsert ----
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,  # slightly larger chunks for enterprise docs
            chunk_overlap=80,
        )

        all_chunks = []
        metas = []

        for blk in text_blocks:
            chunks = splitter.split_text(blk["text"])
            for c in chunks:
                all_chunks.append(c)
                metas.append(
                    {
                        "source": blk["source"],
                        "doc_id": doc_id,
                        "role": role,
                        "page": blk["page"],
                        "text": c,
                    }
                )

        if not all_chunks:
            print(f"⚠️ No usable text found in {file.filename}")
            continue

        print(f"🧠 Embedding {len(all_chunks)} chunks...")
        # Run embeddings in a thread to avoid blocking event loop
        import asyncio

        embeddings = await asyncio.to_thread(embed_model.embed_documents, all_chunks)
        ids = [f"{doc_id}-{i}" for i in range(len(all_chunks))]

        print("🚀 Upserting to Pinecone...")
        index.upsert(vectors=zip(ids, embeddings, metas))

        print(f"✅ Upload complete for {file.filename}")

    print("🎉 All documents processed and stored as searchable knowledge!")
