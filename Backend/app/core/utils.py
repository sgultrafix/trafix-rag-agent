import os
from typing import Optional
from fastapi import UploadFile
from .config import settings

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in settings.ALLOWED_EXTENSIONS

async def save_upload_file(upload_file: UploadFile) -> Optional[str]:
    """Save uploaded file to the uploads directory."""
    if not allowed_file(upload_file.filename):
        return None
    
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Create file path
    file_path = os.path.join(settings.UPLOAD_DIR, upload_file.filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    
    return file_path 