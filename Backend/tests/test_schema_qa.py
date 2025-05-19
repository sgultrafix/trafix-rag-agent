import pytest
from unittest.mock import patch, MagicMock
from app.features.schema_qa.service import SchemaQAService
from app.services.hybrid_store import HybridStorageManager
from langchain.schema import Document
import json
import yaml

@pytest.fixture
def hybrid_store():
    mock_store = MagicMock(spec=HybridStorageManager)
    # Create mock schema_store
    mock_store.schema_store = MagicMock()
    mock_store.schema_store.add_document = MagicMock(return_value=True)
    mock_store.schema_store.as_retriever = MagicMock(return_value=MagicMock())
    return mock_store

@pytest.fixture
def schema_qa_service(hybrid_store):
    service = SchemaQAService(hybrid_store)
    # Mock the memory object
    service.memory = MagicMock()
    service.memory.clear = MagicMock()
    return service

def test_schema_qa_service_initialization(schema_qa_service):
    assert schema_qa_service is not None
    assert schema_qa_service.llm is not None
    assert schema_qa_service.memory is not None
    assert schema_qa_service.processor is not None

@pytest.mark.asyncio
async def test_process_schema_file_success(schema_qa_service, hybrid_store):
    # Test SQL schema
    sql_schema = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));"
    hybrid_store.schema_store.add_document.return_value = True
    result = await schema_qa_service.process_schema_file(sql_schema, ".sql")
    assert result is not None

    # Test JSON schema
    json_schema = json.dumps({
        "users": {
            "id": "integer",
            "name": "string"
        }
    })
    result = await schema_qa_service.process_schema_file(json_schema, ".json")
    assert result is not None

    # Test YAML schema (.yml)
    yml_schema = yaml.dump({
        "products": {
            "id": "integer",
            "name": "string"
        }
    })
    result = await schema_qa_service.process_schema_file(yml_schema, ".yml")
    assert result is not None

    # Test YAML schema (.yaml)
    yaml_schema = yaml.dump({
        "customers": {
            "id": "integer",
            "email": "string"
        }
    })
    result = await schema_qa_service.process_schema_file(yaml_schema, ".yaml")
    assert result is not None

@pytest.mark.asyncio
async def test_process_schema_file_invalid_type(schema_qa_service):
    with pytest.raises(ValueError):
        await schema_qa_service.process_schema_file("test", ".invalid")

@pytest.mark.asyncio
async def test_ask_schema_question_success(schema_qa_service, hybrid_store):
    # Mock the QA chain
    mock_qa_chain = MagicMock()
    mock_qa_chain.return_value = {"result": "Test answer"}
    
    with patch('app.features.schema_qa.service.RetrievalQA') as mock_qa:
        mock_qa.from_chain_type.return_value = mock_qa_chain
        hybrid_store.schema_store.as_retriever.return_value = MagicMock()
        
        answer, sources, similarity = await schema_qa_service.ask_schema_question("What tables exist?")
        
        assert answer == "Test answer"
        assert sources == []
        assert similarity == 0.0
        mock_qa.from_chain_type.assert_called_once()

@pytest.mark.asyncio
async def test_ask_schema_question_error(schema_qa_service, hybrid_store):
    with patch('app.features.schema_qa.service.RetrievalQA') as mock_qa:
        mock_qa.from_chain_type.side_effect = Exception("Test error")
        hybrid_store.schema_store.as_retriever.return_value = MagicMock()
        
        answer, sources, similarity = await schema_qa_service.ask_schema_question("What tables exist?")
        
        assert "Error getting answer" in answer
        assert sources == []
        assert similarity == 0.0

def test_clear_memory(schema_qa_service):
    schema_qa_service.clear_memory()
    schema_qa_service.memory.clear.assert_called_once()

def test_get_schema_summary(schema_qa_service, hybrid_store):
    # Mock documents with different schema types
    mock_docs = [
        Document(
            page_content="CREATE TABLE users (id INT);",
            metadata={
                "type": "sql",
                "tables": ["users"],
                "relationships": [{"type": "primary_key", "field": "id"}]
            }
        ),
        Document(
            page_content=json.dumps({"orders": {"id": "integer"}}),
            metadata={
                "type": "json",
                "table_name": "orders",
                "fields": ["id"],
                "relationships": []
            }
        ),
        Document(
            page_content=yaml.dump({"products": {"id": "integer"}}),
            metadata={
                "type": "yaml",
                "table_name": "products",
                "fields": ["id"],
                "relationships": []
            }
        )
    ]
    
    hybrid_store.search.return_value = mock_docs
    
    summary = schema_qa_service.get_schema_summary()
    
    assert "users" in summary["tables"]
    assert "orders" in summary["tables"]
    assert "products" in summary["tables"]
    assert "sql" in summary["schema_types"]
    assert "json" in summary["schema_types"]
    assert "yaml" in summary["schema_types"]
    assert len(summary["relationships"]) > 0

def test_get_schema_summary_error(schema_qa_service, hybrid_store):
    hybrid_store.search.side_effect = Exception("Test error")
    
    summary = schema_qa_service.get_schema_summary()
    
    assert summary["tables"] == []
    assert summary["relationships"] == []
    assert summary["schema_types"] == [] 