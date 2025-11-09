from fastapi import APIRouter, Depends, HTTPException

# 🛑 TEMPORARY: Remove the import of 'authenticate'
# from auth.routes import authenticate 
from .search_query import extract_knowledge 

router=APIRouter()

# --- NEW /extract GET Endpoint (FOR TESTING ONLY) ---
@router.get("/extract")
async def extract():
    """
    Retrieves ALL documents accessible to a placeholder role, extracts structured 
    knowledge, and stores the result in MongoDB.
    
    🛑 NOTE: The authentication dependency has been REMOVED for testing.
    """
    
    # 🛑 Hardcode a testing role here. 
    # Use a role that is present in your Pinecone metadata (e.g., 'admin' or 'Public').
    TESTING_USER_ROLE = "admin" 
    
    try:
        # Call the extraction function with the placeholder role
        result = await extract_knowledge(TESTING_USER_ROLE)
        return result
    
    except Exception as e:
        # Include the specific role used in the error detail for debugging
        raise HTTPException(status_code=500, detail=f"Extraction process failed for role '{TESTING_USER_ROLE}': {e}")

# NOTE: REMEMBER TO REVERT THIS CHANGE ONCE YOU IMPLEMENT REAL AUTHENTICATION!