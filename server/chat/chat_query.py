import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
# ---- Multimodal LLM (Gemini 2.5 Flash) ----
# llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3,google_api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that can analyze both text and images.
Use the provided text context and attached images to answer the user's question accurately.

If an image is relevant, describe it briefly before analyzing.
If not, answer only from text and tables.

Question:
{question}

Context:
{context}
""")

rag_chain = prompt | llm


async def answer_query(query: str, user_role: str):
    """Retrieve role-specific info, filter image/table context intelligently."""
    embedding = await asyncio.to_thread(embed_model.embed_query, query)
    results = await asyncio.to_thread(index.query, vector=embedding, top_k=5, include_metadata=True)

    filtered_contexts, sources, retrieved_images, retrieved_tables = [], set(), [], []

    for match in results["matches"]:
        meta = match["metadata"]
        if meta.get("role") != user_role:
            continue

        # ---- Smart filtering ----
        chunk_type = meta.get("type", "")
        # If user didn't ask for charts/images/visuals, skip image chunks
        if chunk_type == "image" and not any(
            word in query.lower() for word in ["see", "chart", "graph", "figure", "image", "diagram", "visual"]
        ):
            continue

        # Collect tables separately
        if "table" in chunk_type:
            retrieved_tables.append(meta.get("text", ""))

        filtered_contexts.append(meta.get("text", ""))
        if meta.get("image_path"):
            retrieved_images.append(meta.get("image_path"))
        sources.add(meta.get("source"))

    if not filtered_contexts:
        return {"answer": "No relevant information found.", "sources": [], "retrieved_images": [], "retrieved_tables": []}

    docs_text = "\n".join(filtered_contexts)
    final_answer = await asyncio.to_thread(rag_chain.invoke, {"question": query, "context": docs_text})

    return {
        "answer": final_answer.content,
        "sources": list(sources),
        "retrieved_images": list(set(retrieved_images)),
        "retrieved_tables": list(set(retrieved_tables)),
    }
