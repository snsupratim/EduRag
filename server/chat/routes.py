# 📁 routes/chat.py

from fastapi import APIRouter, Depends, Form
from auth.routes import authenticate
from chat.chat_query import answer_query

router = APIRouter()

@router.post("/chat")
async def chat(
    user=Depends(authenticate),
    message: str = Form(...)
):
    """
    Query uploaded documents.
    Returns LLM answer + tables + image references from Pinecone metadata.
    """
    try:
        response = await answer_query(message, user["role"])
        return {
            "answer": response["answer"],
            "sources": response.get("sources", []),
            "tables": response.get("tables", []),
            "images": response.get("images", [])
        }
    except Exception as e:
        return {"error": str(e)}
