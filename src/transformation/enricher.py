# src/transformation/enricher.py
import time
import json
import asyncio
from typing import List, Optional
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata
from src.providers.factory import get_llm_provider
from src.utils.logger import logger


class ChunkEnricher:
    """
    Metadata enrichment engine using the configured LLM provider (Azure AI / Groq).
    Populates strict Pydantic ChunkMetadata schemas for downstream vector and hybrid filtering.
    """
    def __init__(self):
        self.provider = get_llm_provider()

    def enrich_elements(self, elements: List[ExtractedElement], parent_id: Optional[str] = None) -> ChunkMetadata:
        """
        Synchronous enrichment of document elements.
        """
        try:
            return asyncio.run(self.enrich_elements_async(elements, parent_id=parent_id))
        except RuntimeError:
            # If an event loop is already running, create a new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.enrich_elements_async(elements, parent_id=parent_id))
                return future.result()

    async def enrich_elements_async(
        self,
        elements: List[ExtractedElement],
        parent_id: Optional[str] = None,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> ChunkMetadata:
        """
        Asynchronous enrichment of document elements using the active LLM provider.
        """
        async def _call():
            try:
                return await self.provider.enrich_chunk_async(elements, parent_id=parent_id)
            except Exception as e:
                logger.error(f"Enrichment exception: {e}")
                return ChunkMetadata(
                    summary=f"Metadata enrichment fallback: {str(e)}",
                    keywords=["enrichment-fallback"],
                    category="General",
                    parent_context_id=parent_id
                )

        if semaphore:
            async with semaphore:
                return await _call()
        return await _call()

    async def enrich_batch(
        self,
        windows: List[List[ExtractedElement]],
        parent_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Processes multiple semantic windows concurrently with rate-limiting semaphore.
        """
        limit = getattr(settings, "CONCURRENCY_LIMIT", 5)
        semaphore = asyncio.Semaphore(limit)

        tasks = [
            self.enrich_elements_async(window, parent_id=parent_id, semaphore=semaphore)
            for window in windows
        ]

        results = await asyncio.gather(*tasks)
        return list(results)
