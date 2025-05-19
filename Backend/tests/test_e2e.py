import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from pathlib import Path
import shutil
import json

client = TestClient(app)

# Test data setup
TEST_PDF_PATH = Path("tests/test_data/sample_manual.pdf")
TEST_SCHEMA_PATH = Path("tests/test_data/sample_schema.sql")

@pytest.fixture(autouse=True)
def setup_test_files():
    """Create test files if they don't exist"""
    os.makedirs("tests/test_data", exist_ok=True)
    # Create sample PDF if not exists
    if not TEST_PDF_PATH.exists():
        with open(TEST_PDF_PATH, "w") as f:
            f.write("Sample PDF content for testing")
    # Create sample schema if not exists
    if not TEST_SCHEMA_PATH.exists():
        with open(TEST_SCHEMA_PATH, "w") as f:
            f.write("""
            CREATE TABLE customers (
                id INT PRIMARY KEY,
                name VARCHAR(100),
                service_level VARCHAR(50)
            );
            """)
    yield
    # Cleanup after tests
    if TEST_PDF_PATH.exists():
        TEST_PDF_PATH.unlink()
    if TEST_SCHEMA_PATH.exists():
        TEST_SCHEMA_PATH.unlink()

# Helper: Upload a file using the /upload endpoint
def upload_file(file_path):
    with open(file_path, "rb") as f:
        response = client.post("/upload", files={"file": (os.path.basename(file_path), f)})
    return response

# Helper: Check if file exists in uploads directory
def is_file_in_uploads(file_name):
    uploads_dir = "uploads"
    return os.path.exists(os.path.join(uploads_dir, file_name))

# Helper: Call vectorize API
def vectorize_uploads():
    return client.post("/vectorize-uploads")

# Helper: Ask a question using query parameter
def ask_question(question):
    return client.post(f"/ask?question={question}")

# Test: End-to-end BlackRock Investment scenario
def test_json_schema_blackrock_e2e(tmp_path):
    """
    Scenario:
    1) Upload a json schema
    2) Verify the file is properly uploaded in the "uploads" directory
    3) Use Vectorize API and convert the data into vector data
    4) Use "Ask" API and ask questions about BlackRock Investment
    """
    # Copy trading_schema.json to uploads and upload via API
    sample_json = os.path.join(os.path.dirname(__file__), "..", "..", "Testing_pdf_files", "trading_schema.json")
    response = upload_file(sample_json)
    assert response.status_code == 200
    assert "success" in response.json().get("message", "").lower()
    # Verify file is in uploads directory
    assert is_file_in_uploads("trading_schema.json")
    # Vectorize
    vec_response = vectorize_uploads()
    assert vec_response.status_code == 200
    assert "success" in vec_response.json().get("status", "").lower()
    # Ask questions
    questions = [
        "Tell me about BlackRock Investment",
        "Tell me who is managing BlackRock Investment",
        "Tell me total number of orders by BlackRock"
    ]
    for q in questions:
        ask_resp = ask_question(q)
        assert ask_resp.status_code == 200
        data = ask_resp.json()
        # Accept either 'answer' or 'response' key
        answer = data.get("answer") or data.get("response")
        assert answer is not None and isinstance(answer, str) and len(answer.strip()) > 0
        assert "empty vector store" not in answer.lower()

# Update other tests to use /upload and correct /ask usage
def test_upload_and_process_pdf():
    pdf_path = Path("tests/test_data/sample_manual.pdf")
    response = upload_file(pdf_path)
    assert response.status_code == 200
    assert "success" in response.json()["message"].lower()

def test_upload_and_process_schema():
    schema_path = Path("tests/test_data/sample_schema.sql")
    response = upload_file(schema_path)
    assert response.status_code == 200
    assert "success" in response.json()["message"].lower()

def test_customer_service_query():
    query = "What settings are currently activated for customer 'XYZ'?"
    response = ask_question(query)
    assert response.status_code == 200
    data = response.json()
    answer = data.get("answer") or data.get("response")
    assert answer is not None and len(answer) > 0

def test_pricing_query():
    query = "If we offer these services to a new customer, what monthly cost should we charge?"
    response = ask_question(query)
    assert response.status_code == 200
    data = response.json()
    answer = data.get("answer") or data.get("response")
    assert answer is not None and len(answer) > 0

def test_combined_query():
    query = "What are the service settings and pricing for customer 'XYZ'?"
    response = ask_question(query)
    assert response.status_code == 200
    data = response.json()
    answer = data.get("answer") or data.get("response")
    assert answer is not None and len(answer) > 0

def test_context_aware_response():
    query1 = "What services does customer 'XYZ' have?"
    response1 = ask_question(query1)
    assert response1.status_code == 200
    query2 = "What are the current settings for these services?"
    response2 = ask_question(query2)
    assert response2.status_code == 200
    data = response2.json()
    answer = data.get("answer") or data.get("response")
    assert answer is not None and len(answer) > 0

def test_error_handling():
    query = ""  # Empty query
    response = ask_question(query)
    # Accept either 400 or 422 as valid error codes for empty input
    assert response.status_code in (400, 422)
    data = response.json()
    assert "error" in data or "detail" in data 