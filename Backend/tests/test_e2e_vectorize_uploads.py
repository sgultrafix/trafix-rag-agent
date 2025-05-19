import os
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_FILES = [
    {"filename": "trading_schema.json", "content_type": "application/json"},
    {"filename": "customsAct.pdf", "content_type": "application/pdf"},
    {"filename": "newDb.sql", "content_type": "application/sql"}
]

TEST_FILES_DIR = os.path.abspath("../Testing_pdf_files")


def upload_test_files():
    for file_info in TEST_FILES:
        file_path = os.path.join(TEST_FILES_DIR, file_info["filename"])
        with open(file_path, "rb") as f:
            files = {"file": (file_info["filename"], f, file_info["content_type"])}
            response = client.post("/upload", files=files)
            assert response.status_code == 200
            assert "uploaded successfully" in response.json()["message"].lower()
        # Add a small delay between uploads to ensure proper processing
        time.sleep(1)


def assert_any_keyword_in_answer(answer, keywords, question):
    found = any(keyword in answer for keyword in keywords)
    assert found, f"None of the expected keywords {keywords} found in answer for question: {question}\nAnswer: {answer}"


def test_vectorize_and_ask():
    # Upload test files
    upload_test_files()

    # Call /vectorize-uploads
    response = client.post("/vectorize-uploads")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    for file_info in TEST_FILES:
        assert file_info["filename"] in data["processed_files"]
    assert data["num_documents"] > 0

    # Add a delay to ensure vectorization is complete
    time.sleep(2)

    # Test cases for trading_schema.json
    test_cases = [
        {
            "question": "Tell me about BlackRock Investment",
            "expected_keywords": ["blackrock", "investment", "limit", "risk", "client"]
        },
        {
            "question": "What are the trading limits for BlackRock?",
            "expected_keywords": ["limit", "trading", "blackrock"]
        },
        {
            "question": "List all investment types in the schema",
            "expected_keywords": ["investment", "type", "client", "blackrock"]
        },
        {
            "question": "Summarize the schema",
            "expected_keywords": ["schema", "client", "order", "instrument"]
        }
    ]

    for test_case in test_cases:
        response = client.post("/ask", json={"question": test_case["question"]})
        assert response.status_code == 200
        answer = response.json().get("answer", "").lower()
        assert_any_keyword_in_answer(answer, test_case["expected_keywords"], test_case["question"])
        # Add a small delay between questions
        time.sleep(1)


def test_pdf_qa():
    # Test cases for customsAct.pdf
    test_cases = [
        {
            "question": "What is the main purpose of the Customs Act?",
            "expected_keywords": ["customs", "act", "purpose", "regulation", "law"]
        },
        {
            "question": "What are the key sections in the Customs Act?",
            "expected_keywords": ["section", "chapter", "provision", "customs"]
        },
        {
            "question": "Summarize the Customs Act document",
            "expected_keywords": ["customs", "act", "regulation", "import", "export", "law"]
        }
    ]

    for test_case in test_cases:
        response = client.post("/ask", params={"question": test_case["question"]})
        assert response.status_code == 200
        answer = response.json().get("answer", "").lower()
        assert_any_keyword_in_answer(answer, test_case["expected_keywords"], test_case["question"])


def test_sql_schema_qa():
    # Test cases for newDb.sql
    test_cases = [
        {
            "question": "List all tables in the database schema",
            "expected_keywords": ["table", "schema", "dbo", "column"]
        },
        {
            "question": "What are the relationships between tables?",
            "expected_keywords": ["relationship", "foreign", "key", "reference", "table"]
        },
        {
            "question": "Describe the structure of the database",
            "expected_keywords": ["structure", "schema", "table", "column"]
        },
        {
            "question": "Summarize the SQL schema",
            "expected_keywords": ["sql", "schema", "table", "column"]
        }
    ]

    for test_case in test_cases:
        response = client.post("/ask", params={"question": test_case["question"]})
        assert response.status_code == 200
        answer = response.json().get("answer", "").lower()
        assert_any_keyword_in_answer(answer, test_case["expected_keywords"], test_case["question"])


def test_error_handling():
    # Test invalid question format
    response = client.post("/ask", params={"question": ""})
    assert response.status_code == 422

    # Test unsupported file type
    with open("test.txt", "w") as f:
        f.write("test content")
    with open("test.txt", "rb") as f:
        response = client.post("/upload", files={"file": ("test.txt", f, "text/plain")})
    assert response.status_code in [400, 500]  # Either validation error or processing error 