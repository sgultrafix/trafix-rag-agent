import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.main import app
import os
from pathlib import Path
from langchain.schema import Document

client = TestClient(app)

@pytest.fixture
def mock_uploads_dir(tmp_path):
    # Create a temporary uploads directory
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    return uploads_dir

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to LangChain RAG API"}

# Update paths to point to the correct location
TEST_PDF_PATH = Path("../../Testing_pdf_files/customsAct.pdf")
TEST_SCHEMA_PATH = Path("../../Testing_pdf_files/trading_schema.json")

def test_upload_pdf_success():
    with open("../Testing_pdf_files/customsAct.pdf", "rb") as f:
        test_content = f.read()
    files = {"file": ("customsAct.pdf", test_content, "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "uploaded successfully" in response.json()["message"]
    assert "file_path" in response.json()
    assert "file_type" in response.json()
    assert response.json()["status"] == "success"

def test_upload_json_success():
    with open("../Testing_pdf_files/trading_schema.json", "rb") as f:
        test_content = f.read()
    files = {"file": ("trading_schema.json", test_content, "application/json")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "uploaded successfully" in response.json()["message"]
    assert "file_path" in response.json()
    assert response.json()["file_type"] == "json"
    assert response.json()["status"] == "success"

def test_upload_invalid_file():
    # Create a test text file
    test_content = b"test content"
    files = {"file": ("test.txt", test_content, "text/plain")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

@patch('app.main.hybrid_store.search')
def test_ask_question_success(mock_search):
    mock_search.return_value = [MagicMock(page_content="test answer")]
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 200
    assert response.json() == {"answer": "test answer"}

def test_ask_question_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422
    assert "Missing 'question'" in response.json()["detail"]

@patch('app.main.hybrid_store.search')
def test_ask_question_no_results(mock_search):
    mock_search.return_value = []
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 200
    assert "could not find an answer" in response.json()["answer"].lower()

@patch('app.main.hybrid_store.search')
def test_ask_question_error(mock_search):
    mock_search.side_effect = Exception("Test error")
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 500
    assert "Test error" in response.json()["detail"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch('app.main.process_all_files_in_directory')
def test_vectorize_uploads_success(mock_process_files):
    mock_docs = [
        Document(page_content="Document 1", metadata={"source": "test1"}),
        Document(page_content="Document 2", metadata={"source": "test2"}),
        Document(page_content="Document 3", metadata={"source": "test3"})
    ]
    mock_process_files.return_value = mock_docs
    response = client.post("/vectorize-uploads")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "processed_files" in response.json()
    assert "num_documents" in response.json()

@patch('app.main.process_all_files_in_directory')
def test_vectorize_uploads_error(mock_process_files):
    mock_process_files.side_effect = Exception("Test error")
    response = client.post("/vectorize-uploads")
    assert response.status_code == 500
    assert "Error vectorizing uploads" in response.json()["detail"]

def test_reset_all_success():
    response = client.post("/reset_all")
    assert response.status_code == 200
    assert response.json()["message"] == "System reset successfully" 