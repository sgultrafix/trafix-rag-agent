from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LangChain RAG"
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: set = {"pdf", "sql", "json", "yaml", "graphql"}
    
    # Model Settings
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    LLM_MODEL: str = "mistral:latest"
    MODEL_TEMPERATURE: float = 0.7
    
    # Schema Settings
    SCHEMA_TYPES: set = {"sql", "json", "yaml", "graphql"}
    SCHEMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    SCHEMA_CHUNK_SIZE: int = 500
    SCHEMA_CHUNK_OVERLAP: int = 50
    
    # Storage Settings
    FAISS_INDEX_DIR: str = "faiss_index"
    CHROMA_MEMORY_DIR: str = "chroma_memory"
    
    class Config:
        case_sensitive = True

settings = Settings() 