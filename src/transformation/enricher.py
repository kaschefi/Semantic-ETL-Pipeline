# src/transformation/enricher.py
import json
import asyncio
from typing import List
from groq import Groq, AsyncGroq
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata


class ChunkEnricher:
    def __init__(self):
        # Initialize Groq sync and async clients safely if API key is provided
        api_key = getattr(settings, 'GROQ_API_KEY', None)
        self.client = Groq(api_key=api_key) if api_key else None
        self.async_client = AsyncGroq(api_key=api_key) if api_key else None
        self.model_name = "openai/gpt-oss-120b"

    def enrich_elements(self, elements: List[ExtractedElement], parent_id: str = None) -> ChunkMetadata:
        """
        Synchronous enrichment of document elements via Groq LLM.
        """
        combined_text = "\n".join([f"[{el.element_type}]: {el.content}" for el in elements])

        if not self.client:
            return ChunkMetadata(
                summary="Groq client not configured. Skipping enrichment.",
                keywords=["skipped"],
                category="General",
                parent_context_id=parent_id
            )

        schema_json = ChunkMetadata.model_json_schema()
        system_prompt = (
            "You are a pipeline intelligence agent. Analyze the provided document chunk elements.\n"
            "You MUST return a valid JSON object matching this exact schema:\n"
            f"{json.dumps(schema_json['properties'])}\n"
            "Do not include any introductory text, markdown code blocks, or explanations. Return ONLY the raw JSON."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze these document elements:\n\n{combined_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            response_content = completion.choices[0].message.content
            parsed_data = json.loads(response_content)
            parsed_data["parent_context_id"] = parent_id

            return ChunkMetadata(**parsed_data)

        except Exception as e:
            return ChunkMetadata(
                summary=f"Failed to extract semantic metadata via Groq: {str(e)}",
                keywords=["error", "groq-failure"],
                category="Error-Logs",
                parent_context_id=parent_id
            )

    async def enrich_elements_async(
        self,
        elements: List[ExtractedElement],
        parent_id: str = None,
        semaphore: asyncio.Semaphore = None
    ) -> ChunkMetadata:
        """
        Asynchronous enrichment of document elements using AsyncGroq.
        Optionally uses a Semaphore to limit concurrency.
        """
        combined_text = "\n".join([f"[{el.element_type}]: {el.content}" for el in elements])

        if not self.async_client:
            return ChunkMetadata(
                summary="Groq client not configured. Skipping enrichment.",
                keywords=["skipped"],
                category="General",
                parent_context_id=parent_id
            )

        schema_json = ChunkMetadata.model_json_schema()
        system_prompt = (
            "You are a pipeline intelligence agent. Analyze the provided document chunk elements.\n"
            "You MUST return a valid JSON object matching this exact schema:\n"
            f"{json.dumps(schema_json['properties'])}\n"
            "Do not include any introductory text, markdown code blocks, or explanations. Return ONLY the raw JSON."
        )

        async def _call_api():
            try:
                completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze these document elements:\n\n{combined_text}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                response_content = completion.choices[0].message.content
                parsed_data = json.loads(response_content)
                parsed_data["parent_context_id"] = parent_id
                return ChunkMetadata(**parsed_data)

            except Exception as e:
                return ChunkMetadata(
                    summary=f"Failed to extract semantic metadata via Groq: {str(e)}",
                    keywords=["error", "groq-failure"],
                    category="Error-Logs",
                    parent_context_id=parent_id
                )

        if semaphore:
            async with semaphore:
                return await _call_api()
        else:
            return await _call_api()

    async def enrich_batch(
        self,
        windows: List[List[ExtractedElement]],
        parent_id: str = None
    ) -> List[ChunkMetadata]:
        """
        Processes multiple semantic windows in parallel using asyncio.gather
        and concurrency rate limiting via Semaphore.
        """
        limit = getattr(settings, "CONCURRENCY_LIMIT", 5)
        semaphore = asyncio.Semaphore(limit)

        tasks = [
            self.enrich_elements_async(window, parent_id=parent_id, semaphore=semaphore)
            for window in windows
        ]

        results = await asyncio.gather(*tasks)
        return list(results)