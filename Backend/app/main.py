from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.features.pdf_qa.service import PDFQAService
from app.core.utils import save_upload_file
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF QA API")

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

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF QA API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = await save_upload_file(file)
        
        # Process the PDF
        success = pdf_qa_service.process_pdf(file_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
        
        return {"message": "PDF processed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = pdf_qa_service.ask_question(request.question)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your question") 