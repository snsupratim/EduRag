import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
# Import Pydantic for defining the expected schema structure for data
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Environment Setup ---
load_dotenv()
# Use better defaults for local testing if environment variables are missing
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "enterprise_knowledge")


# --- Database Connection ---
# The client is initialized globally but the connection is asynchronous
try:
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Define the new collection for structured extraction
    extracted_collection = db["extracted_data"]
    
    # You may keep the existing collection if needed
    # users_collection = db["users"] 
    
    print(f"MongoDB client initialized for database: {DB_NAME}")

except Exception as e:
    print(f"Error initializing MongoDB client: {e}")
    # Consider raising an error to stop the application if the DB connection is critical
    # raise

# --- Utility Function for Extraction Collection ---

def get_extract_collection():
    """Returns the asynchronous MongoDB collection instance for extracted data."""
    if 'db' not in globals() or not db:
        raise ConnectionError("MongoDB database connection is not initialized.")
    return extracted_collection

# --- Pydantic Models for Data Validation (Optional but Recommended) ---
# NOTE: These models are defined here to ensure both the retrieval logic (query.py) 
# and the database utility (database.py) use the same schema definitions.

class DocumentSection(BaseModel):
    """A meaningful section extracted from the source document."""
    section_title: str = Field(description="The heading or main topic of this section.")
    summary: str = Field(description="A comprehensive summary of the text content in this section.")
    page_number: int = Field(description="The starting page number of this section.")
    extracted_keywords: List[str] = Field(description="A list of 3 to 5 most important keywords.")

class ExtractedDocumentData(BaseModel):
    """The Pydantic model for the data extracted from a single PDF document."""
    document_title: str = Field(description="The original filename of the document.")
    document_id: str = Field(description="The unique identifier (doc_id) of the document.")
    access_role: str = Field(description="The access role tag of the document.")
    sections: List[DocumentSection] = Field(description="A list of all meaningful sections.")

class ExtractedMongoRecord(BaseModel):
    """The full record to be stored in MongoDB."""
    # We use _id to match MongoDB's convention for the primary key
    _id: str = Field(alias="doc_id", description="The unique document ID used as MongoDB's primary key.") 
    extracted_at: float = Field(description="Timestamp of when the extraction occurred.")
    data: ExtractedDocumentData = Field(description="The structured data extracted by the LLM.")