# 📁 routes/docs.py

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from auth.routes import authenticate
from docs.vectorstore import load_vectorstore
import uuid
from pinecone import Pinecone
import os




from dotenv import load_dotenv



router = APIRouter()
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "enterprise-rag-index"


@router.post("/upload_docs")
async def upload_docs(
    user=Depends(authenticate),
    file: UploadFile = File(...),
    role: str = Form(...),
):
    """
    Upload an enterprise PDF (report/manual/policy) and ingest it into Pinecone.
    Only admins can upload.

    role = which user role can later access/search this document
           e.g., "admin", "hr", "finance", "engineer", "sales", etc.
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can upload files")

    doc_id = str(uuid.uuid4())

    try:
        await load_vectorstore([file], role, doc_id)

        return {
            "status": "success",
            "message": f"{file.filename} uploaded and indexed successfully.",
            "doc_id": doc_id,
            "accessible_to_role": role,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@router.get("/uploaded_docs")
async def uploaded_docs(user = Depends(authenticate)):
    """Return uploaded docs accessible to this user's role."""
    pc = Pinecone()
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()

    docs = set()   # avoid duplicates
    for ns in stats.get("namespaces", {}).values():
        # depending on Pinecone version this may be empty — safer approach:
        pass

    # ---------- Updated Safe Retrieval ----------
    # Fetch all vectors via query scan
    result = index.query(vector=[0]*3072, top_k=5000, include_metadata=True)

    for match in result.get("matches", []):
        meta = match.get("metadata", {})
        if meta.get("role") == user["role"]:
            docs.add((meta.get("doc_id"), meta.get("source")))

    return [{"doc_id": d[0], "name": d[1]} for d in docs]