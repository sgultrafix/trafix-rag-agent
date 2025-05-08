import os
from typing import Optional
from fastapi import UploadFile
from .config import Settings

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return filename.lower().endswith('.pdf')

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save the uploaded file to the upload directory."""
    if not allowed_file(upload_file.filename):
        raise ValueError("Invalid file type. Only PDF files are allowed.")
        
    # Create upload directory if it doesn't exist
    upload_dir = Settings().UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    
    # Get the file path using os.path.join
    file_path = os.path.join(upload_dir, upload_file.filename)
    
    # Save the file
    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path 