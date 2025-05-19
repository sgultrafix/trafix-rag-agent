from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.config import settings
from app.core.utils import save_upload_file, SUPPORTED_EXTENSIONS
from app.services.rag_service import RAGService
from app.services.hybrid_store import HybridStorageManager
from app.features.schema_qa.service import SchemaQAService
import os
import shutil
import time
from app.api.health import router as health_router
from fastapi import BackgroundTasks
from typing import Dict
from app.core.file_vectorizer import process_all_files_in_directory, summarize_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
hybrid_store = HybridStorageManager()
rag_service = RAGService()
schema_qa_service = SchemaQAService(hybrid_store)

app.include_router(health_router)

@app.get("/")
async def root():
    return {"message": "Welcome to LangChain RAG API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the system. Supported file types: PDF, JSON, SQL, YAML.
    Files are saved to the uploads directory but not processed until vectorize-uploads is called.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS).upper()}"
            )
        
        logger.info(f"Attempting to save file: {file.filename}")
        file_path = await save_upload_file(file)
        if not file_path:
            raise HTTPException(status_code=400, detail="Failed to save file")
        
        logger.info(f"File saved successfully at: {file_path}")
        return {
            "message": f"{file.filename} uploaded successfully",
            "file_path": file_path,
            "file_type": file_extension[1:],
            "status": "success"
        }
            
    except HTTPException as e:
        raise e
    except ValueError as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize-uploads")
async def vectorize_uploads():
    """
    Process all files in the uploads directory, create vector embeddings, and store them in the vector store.
    Clears previous vector data before processing.
    """
    try:
        uploads_dir = settings.UPLOAD_DIR
        logger.info(f"Clearing vector store and processing files in {uploads_dir}")
        
        # Clear all vector stores
        hybrid_store.clear_store("document")
        hybrid_store.clear_store("schema")
        hybrid_store.clear_store("memory")
        
        # Process all files in the uploads directory
        all_docs = process_all_files_in_directory(uploads_dir, hybrid_store)
        
        logger.info(f"[VECTORIZE-ENDPOINT] Processed files: {os.listdir(uploads_dir)}")
        logger.info(f"[VECTORIZE-ENDPOINT] Total documents to add: {len(all_docs)}")
        if all_docs:
            logger.info(f"[VECTORIZE-ENDPOINT] Sample doc: {all_docs[0].page_content[:100]} ... Metadata: {all_docs[0].metadata}")
        
        # Add documents to appropriate stores based on their type
        if all_docs:
            # Add in batches for efficiency
            batch_size = 32
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i+batch_size]
                # Route documents to appropriate store based on their type
                for doc in batch:
                    if doc.metadata.get("file_type") in ["json", "sql", "yaml"]:
                        logger.info(f"Routing doc to SCHEMA store: {doc.metadata}")
                        hybrid_store.add_document(doc, store_type="schema")
                    else:
                        logger.info(f"Routing doc to DOCUMENT store: {doc.metadata}")
                        hybrid_store.add_document(doc, store_type="document")
                logger.info(f"[VECTORIZE-ENDPOINT] Batch {i//batch_size+1}: Added {len(batch)} docs")
        
        return {
            "status": "success",
            "processed_files": os.listdir(uploads_dir),
            "num_documents": len(all_docs)
        }
    except Exception as e:
        logger.error(f"Error in vectorize_uploads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error vectorizing uploads: {str(e)}")

@app.post("/ask")
async def ask_question(
    question: str = Query(None, description="Your question"),
    payload: dict = Body(None)
):
    """Ask questions about the uploaded documents"""
    # Prefer query param, fallback to JSON body
    if not question and payload:
        question = payload.get("question")
    if not question:
        raise HTTPException(status_code=422, detail="Missing 'question'.")
    try:
        logger.info(f"Received question: {question}")
        
        # Search both document and schema stores
        doc_docs = hybrid_store.search(question, store_type="document", k=2)
        schema_docs = hybrid_store.search(question, store_type="schema", k=2)
        
        # Combine and sort results by relevance
        all_docs = doc_docs + schema_docs
        logger.info(f"[ASK-ENDPOINT] Retrieved {len(all_docs)} docs for question '{question}'")
        if all_docs:
            logger.info(f"[ASK-ENDPOINT] Top doc source: {all_docs[0].metadata.get('source', 'Unknown')}, content: {all_docs[0].page_content[:100]}")
            answer = all_docs[0].page_content
        else:
            logger.warning(f"[ASK-ENDPOINT] No relevant docs found for question '{question}'")
            answer = "Sorry, I could not find an answer to your question in the ingested content."
        
        logger.info("Question answered successfully")
        return {"answer": answer}
    except ValueError as e:
        logger.error(f"Value error in ask_question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_all")
async def reset_all():
    """
    Reset the entire RAG system, including:
    - Clear all uploaded files
    - Clear and reinitialize the vector store
    """
    try:
        # Clear uploads directory
        uploads_dir = settings.UPLOAD_DIR
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        # Clear vector stores
        hybrid_store.clear_store("document")
        hybrid_store.clear_store("schema")
        hybrid_store.clear_store("memory")
        
        return {"message": "System reset successfully"}
    except Exception as e:
        logger.error(f"Error in reset_all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
async def refresh():
    return rag_service.refresh()

@app.post("/summarize")
async def summarize_document(file_path: str = Query(..., description="Path to the uploaded document")):
    try:
        logger.info(f"[SUMMARIZE] Attempting to summarize file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"[SUMMARIZE] File not found: {file_path}")
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        summary = summarize_document(file_path)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary.")

def remove_old_schema_upload():
    pass 