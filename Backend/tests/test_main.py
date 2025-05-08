import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
import os
from pathlib import Path

client = TestClient(app)

@pytest.fixture
def mock_pdf_file(tmp_path):
    # Create a dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%EOF")
    return pdf_path

@pytest.fixture
def mock_uploads_dir(tmp_path):
    # Create a temporary uploads directory
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    return uploads_dir

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the PDF QA API"}

@patch('app.main.pdf_qa_service')
def test_upload_pdf_success(mock_pdf_qa):
    # Setup mock
    mock_pdf_qa.process_pdf.return_value = True
    
    # Create a test PDF file
    test_content = b"%PDF-1.4\n%Test PDF content"
    files = {"file": ("test.pdf", test_content, "application/pdf")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert response.json() == {"message": "PDF processed successfully"}

def test_upload_pdf_invalid_file():
    # Create a test text file
    test_content = b"test content"
    files = {"file": ("test.txt", test_content, "text/plain")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid file type. Only PDF files are allowed."}

@patch('app.main.pdf_qa_service')
def test_ask_question_success(mock_pdf_qa):
    mock_pdf_qa.ask_question.return_value = "test answer"
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 200
    assert response.json() == {"answer": "test answer"}

@patch('app.main.pdf_qa_service')
def test_ask_question_no_pdf_processed(mock_pdf_qa):
    mock_pdf_qa.ask_question.side_effect = ValueError("No PDF has been processed yet")
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 400
    assert response.json() == {"detail": "No PDF has been processed yet"}

@patch('app.main.pdf_qa_service')
def test_ask_question_error(mock_pdf_qa):
    mock_pdf_qa.ask_question.side_effect = Exception("Test error")
    response = client.post("/ask", json={"question": "test question"})
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred while processing your question"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"} 