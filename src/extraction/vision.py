import os
import base64
import re
import asyncio
from typing import List
from groq import Groq, AsyncGroq
from src.config import settings


class MultimodalVisionEngine:
    def __init__(self):
        """
        Initializes the cloud-based Groq sync and async clients using environment settings.
        """
        api_key = getattr(settings, 'GROQ_API_KEY', None)
        self.client = Groq(api_key=api_key) if api_key else None
        self.async_client = AsyncGroq(api_key=api_key) if api_key else None
        self.model_name = settings.GROQ_VISION_MODEL

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Converts a local image file into a base64 string for API transmission.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _sanitize_response(self, text: str) -> str:
        """
        Removes internal reasoning tokens like <think>...</think> from the LLM output.
        """
        if not text:
            return ""
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def describe_image(self, image_path: str) -> str:
        """
        Sends a cached image to Groq's LPU cloud infrastructure for rapid technical analysis (Synchronous).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at cache target: {image_path}")

        if not self.client:
            return f"[Vision Skipped: Groq client not configured with API Key for image {os.path.basename(image_path)}]"

        print(f"[Vision] Streaming image to Groq LPU ({self.model_name}): {os.path.basename(image_path)}...")

        base64_image = self._encode_image_to_base64(image_path)

        prompt = (
            "You are an expert technical data analyst. Analyze this diagram, chart, or image "
            "extracted from a document. Provide a comprehensive, highly detailed description of "
            "everything it contains. If it is a flowchart, graph, or system architecture, explain every step, "
            "connection, and label precisely. Do not speak conversationally; respond only with "
            "the clean technical description of the contents."
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2
            )

            raw_description = chat_completion.choices[0].message.content
            clean_description = self._sanitize_response(raw_description)
            return clean_description if clean_description else "[Empty response after sanitization]"

        except Exception as e:
            print(f"[Vision Error] Groq API call failed: {e}")
            return f"[Failure during cloud visual analysis phase of: {os.path.basename(image_path)}]"

    async def describe_image_async(self, image_path: str, semaphore: asyncio.Semaphore = None) -> str:
        """
        Sends a cached image to Groq VLM asynchronously.
        """
        if not os.path.exists(image_path):
            return f"[Image file not found: {image_path}]"

        if not self.async_client:
            return f"[Vision Skipped: Groq client not configured for image {os.path.basename(image_path)}]"

        print(f"[Vision Async] Processing image via Groq VLM ({self.model_name}): {os.path.basename(image_path)}...")
        base64_image = self._encode_image_to_base64(image_path)

        prompt = (
            "You are an expert technical data analyst. Analyze this diagram, chart, or image "
            "extracted from a document. Provide a comprehensive, highly detailed description of "
            "everything it contains. If it is a flowchart, graph, or system architecture, explain every step, "
            "connection, and label precisely. Do not speak conversationally; respond only with "
            "the clean technical description of the contents."
        )

        async def _call_api():
            try:
                chat_completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=0.2
                )
                raw_description = chat_completion.choices[0].message.content
                clean_description = self._sanitize_response(raw_description)
                return clean_description if clean_description else "[Empty visual description]"
            except Exception as e:
                print(f"[Vision Async Error] {e}")
                return f"[Visual Analysis Error for {os.path.basename(image_path)}: {str(e)}]"

        if semaphore:
            async with semaphore:
                return await _call_api()
        else:
            return await _call_api()

    async def describe_images_batch(self, image_paths: List[str]) -> List[str]:
        """
        Executes parallel vision descriptions for a list of image file paths.
        """
        if not image_paths:
            return []
        limit = getattr(settings, "CONCURRENCY_LIMIT", 5)
        semaphore = asyncio.Semaphore(limit)
        tasks = [self.describe_image_async(path, semaphore=semaphore) for path in image_paths]
        results = await asyncio.gather(*tasks)
        return list(results)