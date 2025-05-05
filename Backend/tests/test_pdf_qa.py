import pytest
from unittest.mock import patch, MagicMock
from app.features.pdf_qa.service import PDFQAService
from langchain_core.documents import Document
import os
from pathlib import Path
import shutil
import sys

from reportlab.pdfgen import canvas

def create_valid_pdf(path):
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, "This is a test PDF.")
    c.save()

@pytest.fixture
def pdf_qa_service(tmp_path):
    persist_dir = tmp_path / "test_chroma_db"
    return PDFQAService(persist_directory=str(persist_dir))

@pytest.fixture
def mock_pdf_file(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    create_valid_pdf(pdf_path)
    return pdf_path

def test_pdf_qa_service_initialization(pdf_qa_service):
    assert pdf_qa_service is not None
    assert pdf_qa_service.llm is not None
    assert pdf_qa_service.vector_store is not None
    assert pdf_qa_service.text_splitter is not None
    assert pdf_qa_service.embeddings is not None

@patch('app.features.pdf_qa.service.PyPDFLoader')
@patch('app.features.pdf_qa.service.Chroma')
def test_process_pdf_success(mock_chroma, mock_loader, pdf_qa_service, mock_pdf_file):
    # Setup document
    test_doc = Document(
        page_content="Test content",
        metadata={"source": str(mock_pdf_file), "page": 0}
    )
    mock_pages = [test_doc]
    
    # Setup loader mock
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_pages
    mock_loader.return_value = mock_loader_instance
    
    # Setup Chroma mock
    mock_vector_store = MagicMock()
    mock_chroma.from_documents.return_value = mock_vector_store
    
    # Process PDF
    result = pdf_qa_service.process_pdf(str(mock_pdf_file))
    
    # Assertions
    assert result is True
    assert pdf_qa_service.vector_store is not None
    mock_loader.assert_called_once_with(str(mock_pdf_file))
    mock_loader_instance.load.assert_called_once()
    
    # Get the actual call arguments
    call_args = mock_chroma.from_documents.call_args
    assert call_args is not None
    
    # Check that from_documents was called with the correct arguments
    args, kwargs = call_args
    assert len(kwargs['documents']) == 1
    assert kwargs['documents'][0].page_content == "Test content"
    assert kwargs['documents'][0].metadata['source'] == str(mock_pdf_file)
    assert kwargs['documents'][0].metadata['page'] == 1  # Service adds 1 to make it 1-indexed
    assert kwargs['documents'][0].metadata['chunk_id'] == 0
    assert kwargs['embedding'] == pdf_qa_service.embeddings
    assert kwargs['persist_directory'] == pdf_qa_service.persist_directory

@patch('app.features.pdf_qa.service.PyPDFLoader')
def test_process_pdf_error(mock_loader, pdf_qa_service, mock_pdf_file):
    mock_loader.side_effect = Exception("Test error")
    result = pdf_qa_service.process_pdf(str(mock_pdf_file))
    assert result is False
    assert pdf_qa_service.vector_store is not None

def test_ask_question_no_vector_store(pdf_qa_service):
    pdf_qa_service.vector_store = None
    result = pdf_qa_service.ask_question("What is this about?")
    assert result == "Please upload and process a PDF first."

@patch('app.features.pdf_qa.service.RetrievalQA')
def test_ask_question_success(mock_qa, pdf_qa_service):
    pdf_qa_service.vector_store = MagicMock()
    pdf_qa_service.vector_store.as_retriever.return_value = MagicMock()
    mock_qa_instance = MagicMock()
    mock_qa.from_chain_type.return_value = mock_qa_instance
    mock_qa_instance.return_value = {"result": "Test answer"}
    result = pdf_qa_service.ask_question("What is this about?")
    assert result == "Test answer"
    mock_qa.from_chain_type.assert_called_once()
    mock_qa_instance.assert_called_once_with({"query": "What is this about?"})

@patch('app.features.pdf_qa.service.RetrievalQA')
def test_ask_question_error(mock_qa, pdf_qa_service):
    pdf_qa_service.vector_store = MagicMock()
    mock_qa.from_chain_type.side_effect = Exception("Test error")
    result = pdf_qa_service.ask_question("What is this about?")
    assert "Error getting answer" in result
    mock_qa.from_chain_type.assert_called_once()

def test_integration_with_real_pdf(tmp_path):
    """Integration test using real Ollama and Chroma components"""
    from app.features.pdf_qa.service import PDFQAService
    
    # Use absolute path to customsAct.pdf
    pdf_path = Path(__file__).parent.parent.parent / "Testing_pdf_files" / "customsAct.pdf"
    pdf_path = pdf_path.resolve()
    assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"
    
    # Initialize service with real components
    persist_dir = tmp_path / "chroma_db"
    service = PDFQAService(persist_directory=str(persist_dir))
    
    # Process PDF
    result = service.process_pdf(str(pdf_path))
    assert result is True
    assert service.vector_store is not None
    
    # Ask question
    answer = service.ask_question("What is the Customs Act?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    print(f"Integration test answer: {answer}")
    
    # Clean up
    try:
        shutil.rmtree(persist_dir)
    except PermissionError:
        if sys.platform.startswith("win"):
            pass
        else:
            raise 