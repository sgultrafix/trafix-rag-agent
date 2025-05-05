from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LangChain RAG"
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: set = {"pdf"}
    
    # Model Settings
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    LLM_MODEL: str = "mistral:latest"
    MODEL_TEMPERATURE: float = 0.7
    
    class Config:
        case_sensitive = True

settings = Settings() 