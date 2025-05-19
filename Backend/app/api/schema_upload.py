from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.features.schema_qa.service import SchemaQAService
from app.services.hybrid_store import HybridStorageManager
from app.core.config import settings

router = APIRouter()

# Initialize services (reuse as in main.py)
hybrid_store = HybridStorageManager()
schema_qa_service = SchemaQAService(hybrid_store)

@router.post("/schema/upload", tags=["Schema"])
async def upload_schema(file: UploadFile = File(...)):
    """Upload and process a database schema file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in settings.SCHEMA_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported schema types: {', '.join(settings.SCHEMA_TYPES)}"
            )
        content = await file.read()
        schema_content = content.decode('utf-8')
        success = await schema_qa_service.process_schema_file(schema_content, file_ext)
        if not success:
            raise HTTPException(status_code=500, detail="Error processing schema file")
        return {
            "message": "Schema processed successfully",
            "schema_summary": schema_qa_service.get_schema_summary()
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 