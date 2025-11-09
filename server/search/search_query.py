import os
import asyncio
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional
from .database import get_extract_collection # Import the MongoDB utility

load_dotenv()

# ... existing environment variables ...
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # Fixed typo
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Use a specific, fast model for structured extraction
llm = ChatGroq(
    temperature=0.1, # Low temp for factual extraction
    model_name="llama-3.1-405b-versatile", # High capability model for better extraction
    groq_api_key=GROQ_API_KEY
)

# --- 🛑 Pydantic Schema for Structured Extraction 🛑 ---
class DocumentSection(BaseModel):
    """A meaningful section extracted from the source document."""
    section_title: str = Field(description="The heading or main topic of this section (e.g., 'Remote Work Policy' or 'Chapter 3: Maintenance').")
    summary: str = Field(description="A comprehensive summary of the text content in this section.")
    page_number: int = Field(description="The starting page number of this section.")
    extracted_keywords: List[str] = Field(description="A list of 3 to 5 most important keywords from this section.")

class ExtractedDocument(BaseModel):
    """The full structured representation of a single PDF document."""
    document_title: str = Field(description="The original filename of the document.")
    document_id: str = Field(description="The unique identifier (doc_id) of the document.")
    access_role: str = Field(description="The access role tag of the document.")
    sections: List[DocumentSection] = Field(description="A list of all meaningful sections and their extracted data.")


# --- Modified RAG Chain for Structured Output ---
# We do not need the old prompt template or simple rag_chain anymore.
# We will use llm.with_structured_output directly within the function.


async def extract_knowledge(user_role: str):
    """
    Retrieves all document chunks for a given role, compiles them by source document, 
    and uses the LLM to extract structured data.
    """
    extract_collection = get_extract_collection()
    
    # 1. Retrieve ALL chunks for the given role (no query vector needed)
    # We query the Pinecone index metadata directly to get all relevant doc IDs/sources
    # NOTE: Pinecone's standard query is for vector search. For metadata-only retrieval, 
    # a dedicated metadata store (like MongoDB) would be faster.
    # For now, we'll retrieve all vectors and rely on the metadata filter.
    
    # We must first get all unique document IDs for the user_role
    # This often requires a full scan or a secondary index in a real-world scenario.
    # Since Pinecone doesn't have a direct "list all metadata for filter" API in a simple way,
    # we will query for a common vector (e.g., 'a' * 3072) with a high k, 
    # or rely on a function to list all document IDs which is not standard.
    # A simplified approach for prototyping: fetch the top 1000 and hope it covers all documents.
    
    # A cleaner solution is to use a secondary index for document metadata. 
    # Assuming you have a list of all known doc_ids for this `user_role` in a separate database (like MongoDB).
    # Since we are adding MongoDB now, we can query it for the doc_ids.

    # 🛑 For this exercise, we will assume a simple scenario: retrieve all chunks for the role 
    # to feed to the LLM for extraction, but this is inefficient for large scale.
    
    # 2. Get the unique list of (doc_id, source) for the user_role
    # Since Pinecone's API is difficult for getting ALL records, we'll rely on an external call 
    # to fetch all indexed doc IDs or use a high k. Since we don't have that helper, 
    # we'll implement a robust filter search:
    
    # Placeholder for a dedicated function to get all document chunks, grouped by doc_id
    # A simple but inefficient way to get ALL data for a role: use a query with no semantic bias.
    
    # --- Simplified Step (Requires a list of doc_ids for the user_role to be efficient) ---
    # For a proper solution: use an additional index (like MongoDB) to store document metadata
    # and list all unique document IDs (doc_id) for a given `user_role` first.
    
    # Since we don't have that, we'll implement the full document chunking and processing.
    
    # For the actual extraction logic: we need to retrieve and group chunks by their 
    # original document ID and source. Since Pinecone is the only source, we'll focus on 
    # extracting per document and storing.

    
    # 3. Use the LLM for Structured Extraction for a SAMPLE document
    # For the initial version, let's assume we can retrieve the *full text* of one document 
    # tagged with the user_role to show the structured extraction process.
    # To get all segments of a single document, a *query* is still required. 
    # We cannot perform true "extraction of all info from ALL pdfs" without iterating over doc IDs.
    
    # Let's simplify and mock the full context retrieval for a hypothetical single document.
    
    # Step 3.1: Retrieve all chunks for a single document ID (requires knowing the ID)
    # Since this is a GET request and we are not passing a doc_id, we will process ALL documents 
    # found for the role.
    
    # --- ACTUAL EXTRACTION PROCESS ---
    
    # This is highly inefficient but demonstrates the RAG extraction chain:
    # 1. Get ALL Pinecone data for the role (simulate a large batch)
    # 2. Group it by (doc_id, source)
    # 3. Process each document group using the LLM for structured extraction
    
    # --- 1. Get ALL relevant documents/chunks (Mock retrieval for all documents) ---
    # This step should use a tool to get ALL document metadata, but we only have Pinecone API
    # which is vector-based. We will use a mock query vector to retrieve a large batch.
    
    MOCK_QUERY_VECTOR = [0.0] * 3072 # Use a zero vector or random vector for non-semantic query

    # WARNING: This query is INSUFFICIENT for large-scale extraction, it only returns top_k=1000.
    # A proper solution iterates over doc_ids from an index list.
    results = await asyncio.to_thread(
        index.query, 
        vector=MOCK_QUERY_VECTOR,
        top_k=1000, # Retrieve a large batch
        include_metadata=True,
        filter={"role": user_role}
    )
    
    # 2. Group chunks by (doc_id, source)
    grouped_docs = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        doc_id = metadata.get("doc_id")
        source = metadata.get("source")
        text_content = metadata.get("text")
        
        if doc_id and source and text_content:
            key = (doc_id, source)
            if key not in grouped_docs:
                grouped_docs[key] = {
                    "doc_id": doc_id,
                    "source": source,
                    "role": user_role,
                    "full_text": []
                }
            grouped_docs[key]["full_text"].append(text_content)
    
    if not grouped_docs:
        return {"message": f"No documents found for role '{user_role}'."}
        
    # 3. Process each document group for structured extraction
    
    final_extracted_data = []
    
    # Bind the LLM to the Pydantic schema
    structured_llm = llm.with_structured_output(ExtractedDocument)
    
    for key, doc_data in grouped_docs.items():
        doc_id, source = key
        
        # Combine all chunks into one context string for the LLM
        full_context = "\n\n---\n\n".join(doc_data["full_text"])
        
        # Define the extraction prompt
        extraction_prompt = PromptTemplate.from_template("""
            Analyze the following document text and extract the information according to the required JSON schema.
            The document title is '{doc_title}'. The document ID is '{doc_id}'.
            Break the text into logical, titled sections (Chapter, Policy, etc.) and provide a summary and keywords for each.
            
            FULL DOCUMENT TEXT:
            ---
            {context}
            ---
            
            If no information can be logically segmented, provide one main segment summarizing the whole document.
            Ensure the output strictly follows the JSON schema.
        """)
        
        # Use a simple chain for extraction
        extraction_chain = extraction_prompt | structured_llm
        
        try:
            # Run the extraction (Async)
            extracted_model = await asyncio.to_thread(
                extraction_chain.invoke, 
                {
                    "context": full_context,
                    "doc_title": source,
                    "doc_id": doc_id
                }
            )
            
            # The result is a Pydantic model instance
            extracted_json = extracted_model.model_dump()
            
            # 4. Store in MongoDB
            # Add a timestamp and unique ID for the MongoDB record
            mongo_record = {
                "_id": str(doc_id), # Use doc_id as MongoDB ID for easy lookup
                "extracted_at": asyncio.to_thread(lambda: time.time()), # Add time
                "data": extracted_json
            }
            
            # This is a critical step: store the structured data
            await extract_collection.replace_one(
                {"_id": str(doc_id)}, # Find by doc_id
                mongo_record,
                upsert=True # Insert if not found, replace if found
            )
            
            final_extracted_data.append({"doc_id": doc_id, "title": source, "status": "Success"})
            
        except Exception as e:
            final_extracted_data.append({"doc_id": doc_id, "title": source, "status": f"Extraction Failed: {e}"})
            print(f"Extraction failed for {source}: {e}")

    return {
        "message": "Structured extraction and storage complete.",
        "documents_processed": final_extracted_data
    }