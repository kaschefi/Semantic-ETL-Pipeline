# src/extraction/vision.py
import os
import asyncio
from typing import List
from src.config import settings
from src.providers.factory import get_llm_provider
from src.utils.logger import logger


class MultimodalVisionEngine:
    """
    Multimodal Vision Engine translating diagrams, flowcharts, and architecture images into searchable text.
    Leverages the configured LLM provider (Azure AI / Groq Vision).
    """
    def __init__(self):
        self.provider = get_llm_provider()

    def describe_image(self, image_path: str) -> str:
        """
        Synchronously generates a detailed technical description of an extracted image.
        """
        try:
            return asyncio.run(self.describe_image_async(image_path))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.describe_image_async(image_path))
                return future.result()

    async def describe_image_async(self, image_path: str, semaphore: asyncio.Semaphore = None) -> str:
        """
        Asynchronously generates a technical description of an image via the active LLM provider.
        """
        async def _call():
            try:
                return await self.provider.describe_image_async(image_path)
            except Exception as e:
                logger.error(f"Vision analysis error: {e}")
                return f"[Visual analysis fallback for {os.path.basename(image_path)}: {str(e)}]"

        if semaphore:
            async with semaphore:
                return await _call()
        return await _call()

    async def describe_images_batch(self, image_paths: List[str]) -> List[str]:
        """
        Executes parallel vision descriptions for a batch of images.
        """
        if not image_paths:
            return []
        limit = getattr(settings, "CONCURRENCY_LIMIT", 5)
        semaphore = asyncio.Semaphore(limit)
        tasks = [self.describe_image_async(path, semaphore=semaphore) for path in image_paths]
        results = await asyncio.gather(*tasks)
        return list(results)
