import pytest
from app.core.config import Settings
from app.core.utils import allowed_file, save_upload_file
from fastapi import UploadFile
import os

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def temp_upload_dir(tmp_path):
    original_dir = Settings().UPLOAD_DIR
    Settings.UPLOAD_DIR = str(tmp_path)
    yield tmp_path
    Settings.UPLOAD_DIR = original_dir

def test_allowed_file():
    assert allowed_file("test.pdf") is True
    assert allowed_file("test.txt") is False
    assert allowed_file("test.PDF") is True
    assert allowed_file("test") is False

@pytest.mark.asyncio
async def test_save_upload_file(temp_upload_dir):
    # Create a test file
    test_content = b"test content"
    test_filename = "test.pdf"
    
    # Create a mock UploadFile
    file = UploadFile(filename=test_filename, file=type('obj', (object,), {'read': lambda: test_content}))
    
    # Save the file
    file_path = await save_upload_file(file)
    
    # Verify the file was saved
    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.basename(file_path) == test_filename
    
    # Verify the content
    with open(file_path, "rb") as f:
        assert f.read() == test_content

@pytest.mark.asyncio
async def test_save_upload_file_invalid_extension(temp_upload_dir):
    # Create a test file with invalid extension
    test_content = b"test content"
    test_filename = "test.txt"
    
    # Create a mock UploadFile
    file = UploadFile(filename=test_filename, file=type('obj', (object,), {'read': lambda: test_content}))
    
    # Try to save the file
    file_path = await save_upload_file(file)
    
    # Verify the file was not saved
    assert file_path is None 