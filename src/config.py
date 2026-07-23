import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server & CORS Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Pinecone Vector Credentials
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "semantic-etl-index"

    # Ollama Local Service Configuration
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    TEXT_MODEL: str = "llama3"
    VISION_MODEL: str = "llava"

    # Groq Cloud AI Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_VISION_MODEL: str = "qwen/qwen3.6-27b"

    # Chunking Options
    MAX_CHUNK_CHARS: int = 1200
    OVERLAP_ELEMENTS: int = 1

    # Local Filesystem Asset Paths
    DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    CACHE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cache"))

    # Read configuration states dynamically from local configuration environment files
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiated configuration global state singleton
settings = Settings()