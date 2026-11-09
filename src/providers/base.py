# src/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models import ExtractedElement, ChunkMetadata, FinalVectorPayload


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers (Groq, OpenAI, Local Ollama).
    """
    @abstractmethod
    async def enrich_chunk_async(
        self,
        elements: List[ExtractedElement],
        parent_id: Optional[str] = None
    ) -> ChunkMetadata:
        """Enriches document element chunk with metadata tags."""
        pass

    @abstractmethod
    async def describe_image_async(self, image_path: str) -> str:
        """Generates visual technical summary for image elements."""
        pass

    @abstractmethod
    async def complete_text_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates general text response for user prompts."""
        pass

    @abstractmethod
    async def stream_text_async(self, prompt: str, system_prompt: Optional[str] = None):
        """Yields text tokens asynchronously as they are generated."""
        pass


class BaseVectorStore(ABC):
    """
    Abstract Base Class for Vector Database Stores (Pinecone, Qdrant, ChromaDB, Mock).
    """
    @abstractmethod
    def generate_embedding(self, text: str, input_type: str = "passage") -> List[float]:
        """Computes dense vector array for text input (passage or query)."""
        pass

    @abstractmethod
    def build_payload(self, elements: List[ExtractedElement], ai_metadata: ChunkMetadata) -> FinalVectorPayload:
        """Constructs vector payload contract."""
        pass

    @abstractmethod
    def upsert_payloads(self, payloads: List[FinalVectorPayload], namespace: str = "documents") -> int:
        """Upserts vector payloads to storage index."""
        pass

    @abstractmethod
    def query_vectors(
        self,
        query_vector: List[float],
        namespace: str = "documents",
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries vector index for matching candidate payloads."""
        pass
