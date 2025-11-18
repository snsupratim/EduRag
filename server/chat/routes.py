# 📁 routes/chat.py

from fastapi import APIRouter, Depends, Form, HTTPException
from auth.routes import authenticate
from chat.chat_query import answer_query

router = APIRouter()


@router.post("/chat")
async def chat(
    user=Depends(authenticate),
    message: str = Form(...),
):
    """
    Enterprise search endpoint.

    - Takes a natural language question.
    - Searches role-accessible documents.
    - Returns an answer + list of source documents.
    """
    try:
        response = await answer_query(message, user["role"])
        return {
            "answer": response["answer"],
            "sources": response.get("sources", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
