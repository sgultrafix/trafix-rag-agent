import pytest
from app.services.sql_schema_parser import SQLSchemaParser

@pytest.fixture
def sql_parser():
    return SQLSchemaParser()

@pytest.fixture
def sample_sql():
    return """
    CREATE TABLE users (
        id INT PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        email VARCHAR(100) NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE posts (
        id INT PRIMARY KEY,
        user_id INT NOT NULL,
        title VARCHAR(200) NOT NULL,
        content TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """

def test_parse_sql_file(sql_parser, sample_sql):
    """Test parsing SQL file and extracting schema information."""
    result = sql_parser.parse_sql_file(sample_sql)
    
    # Check tables
    assert len(result["tables"]) == 2
    assert any(table["name"] == "users" for table in result["tables"])
    assert any(table["name"] == "posts" for table in result["tables"])
    
    # Check columns
    users_table = next(table for table in result["tables"] if table["name"] == "users")
    assert len(users_table["columns"]) == 4
    assert any(col["name"] == "id" and col["primary_key"] for col in users_table["columns"])
    assert any(col["name"] == "username" and not col["nullable"] for col in users_table["columns"])
    
    # Check relationships
    assert len(result["relationships"]) == 1
    rel = result["relationships"][0]
    assert rel["from_table"] == "posts"
    assert rel["from_column"] == "user_id"
    assert rel["to_table"] == "users"
    assert rel["to_column"] == "id"

def test_invalid_sql(sql_parser):
    """Test handling of invalid SQL."""
    with pytest.raises(Exception):
        sql_parser.parse_sql_file("INVALID SQL")

def test_empty_sql(sql_parser):
    """Test handling of empty SQL."""
    result = sql_parser.parse_sql_file("")
    assert len(result["tables"]) == 0
    assert len(result["relationships"]) == 0 