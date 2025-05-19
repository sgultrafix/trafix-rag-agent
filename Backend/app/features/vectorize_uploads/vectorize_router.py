from fastapi import APIRouter, HTTPException
import os
import logging
from typing import List
from langchain.schema import Document
from app.core.config import settings
from app.services.hybrid_store import hybrid_store
from app.core.file_vectorizer import process_all_files_in_directory

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/vectorize-uploads")
async def vectorize_uploads():
    try:
        # Clear existing stores
        hybrid_store.clear_store("document")
        hybrid_store.clear_store("schema")
        
        # Process all files
        uploads_dir = settings.UPLOADS_DIR
        if not os.path.exists(uploads_dir):
            raise HTTPException(status_code=404, detail="Uploads directory not found")
            
        docs = process_all_files_in_directory(uploads_dir, hybrid_store)
        
        return {
            "message": f"Successfully processed {len(docs)} documents",
            "document_count": len(docs)
        }
    except Exception as e:
        logger.error(f"Error in vectorize_uploads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 