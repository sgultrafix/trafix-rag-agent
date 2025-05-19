import os
from typing import Optional, List
from fastapi import UploadFile
from .config import Settings

SUPPORTED_EXTENSIONS = ['.pdf', '.json', '.sql', '.yml', '.yaml']

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)

def get_file_type(filename: str) -> str:
    """Get the file type from the extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext[1:] if ext.startswith('.') else ext

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save the uploaded file to the upload directory."""
    if not allowed_file(upload_file.filename):
        raise ValueError(f"Invalid file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS).upper()}")
        
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