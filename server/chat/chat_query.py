# 📁 chat/chat_query.py

import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "enterprise-rag-index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
)

prompt = ChatPromptTemplate.from_template(
    """
You are an enterprise knowledge assistant.
You answer questions strictly based on the provided context, which comes from internal company PDFs
such as reports, manuals, and policies.

Guidelines:
- Use only the information found in the Context.
- If the answer is not clearly present in the Context, say:
  "I do not have information on this from the available documents."
- Be concise but clear. If useful, enumerate key points.

Question:
{question}

Context:
{context}
"""
)

rag_chain = prompt | llm


async def answer_query(query: str, user_role: str):
    """
    Retrieve role-filtered chunks from Pinecone and answer using LLM.

    - Only chunks with metadata.role == user_role are considered.
    - Returns: { "answer": str, "sources": [filenames...] }
    """
    # 1. Embed query
    embedding = await asyncio.to_thread(embed_model.embed_query, query)

    # 2. Query Pinecone
    results = await asyncio.to_thread(
        index.query,
        vector=embedding,
        top_k=8,
        include_metadata=True,
    )

    contexts = []
    sources = set()

    for match in results.get("matches", []):
        meta = match.get("metadata", {})

        # Role-based filtering
        if meta.get("role") != user_role:
            continue

        text = meta.get("text", "")
        if text and text.strip():
            contexts.append(text.strip())
            sources.add(meta.get("source"))

    if not contexts:
        return {
            "answer": "No relevant information found in the accessible documents.",
            "sources": [],
        }

    docs_text = "\n\n".join(contexts)

    # 3. Run RAG LLM
    answer = await asyncio.to_thread(
        rag_chain.invoke,
        {"question": query, "context": docs_text},
    )

    return {
        "answer": answer.content,
        "sources": list(sources),
    }
