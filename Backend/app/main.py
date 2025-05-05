from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.config import settings
from app.core.utils import save_upload_file
from app.features.pdf_qa.service import PDFQAService

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

# Initialize PDF QA service
pdf_qa_service = PDFQAService()

@app.get("/")
async def root():
    return {"message": "Welcome to LangChain RAG API"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        logger.info(f"Attempting to save file: {file.filename}")
        file_path = await save_upload_file(file)
        
        if not file_path:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
        
        logger.info(f"File saved successfully at: {file_path}")
        logger.info("Processing PDF with QA service...")
        
        try:
            pdf_qa_service.process_pdf(file_path)
            logger.info("PDF processed successfully")
            return {"message": "PDF processed successfully"}
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str):
    try:
        logger.info(f"Received question: {question}")
        answer = pdf_qa_service.ask_question(question)
        logger.info("Question answered successfully")
        return {"answer": answer}
    except ValueError as e:
        logger.error(f"Value error in ask_question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"} 