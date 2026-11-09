# src/providers/factory.py
from src.config import settings
from src.providers.base import BaseLLMProvider, BaseVectorStore
from src.providers.groq_provider import GroqLLMProvider
from src.providers.azure_provider import AzureLLMProvider
from src.providers.pinecone_provider import PineconeVectorStore


def get_llm_provider() -> BaseLLMProvider:
    """
    Factory function instantiating the active LLM Provider based on settings.
    Prioritizes Azure if configured or specified.
    """
    provider_type = getattr(settings, 'LLM_PROVIDER', 'azure').lower()
    if provider_type == 'azure' or getattr(settings, 'AZURE_API_KEY', None):
        return AzureLLMProvider()
    elif provider_type == 'groq':
        return GroqLLMProvider()
    else:
        return AzureLLMProvider() if getattr(settings, 'AZURE_API_KEY', None) else GroqLLMProvider()


def get_vector_store() -> BaseVectorStore:
    """
    Factory function instantiating the active Vector Store Provider based on settings.
    """
    provider_type = getattr(settings, 'VECTOR_STORE_PROVIDER', 'pinecone').lower()
    if provider_type == 'pinecone':
        return PineconeVectorStore()
    else:
        # Default to PineconeVectorStore for unhandled types
        return PineconeVectorStore()
