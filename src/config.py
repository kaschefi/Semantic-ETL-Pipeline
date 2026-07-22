import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pinecone Vector Credentials
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "semantic-etl-index"

    # Ollama Local Service Configuration
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    TEXT_MODEL: str = "llama3"
    VISION_MODEL: str = "llava"

    GROQ_API_KEY: str
    GROQ_VISION_MODEL: str = "qwen/qwen3.6-27b"
    # Local Filesystem Asset Paths
    CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "../data/cache")

    # Read configuration states dynamically from local configuration environment files
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instantiated configuration global state singleton
settings = Settings()