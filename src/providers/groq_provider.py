# src/providers/groq_provider.py
import json
import base64
import os
import re
from typing import List, Optional
from groq import Groq, AsyncGroq
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata
from src.providers.base import BaseLLMProvider


class GroqLLMProvider(BaseLLMProvider):
    def __init__(self):
        api_key = getattr(settings, 'GROQ_API_KEY', None)
        self.client = Groq(api_key=api_key) if api_key else None
        self.async_client = AsyncGroq(api_key=api_key) if api_key else None
        self.text_model = "openai/gpt-oss-120b"
        self.vision_model = getattr(settings, 'GROQ_VISION_MODEL', "qwen/qwen3.6-27b")

    def _sanitize(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    async def enrich_chunk_async(
        self,
        elements: List[ExtractedElement],
        parent_id: Optional[str] = None
    ) -> ChunkMetadata:
        combined_text = "\n".join([f"[{el.element_type}]: {el.content}" for el in elements])

        if not self.async_client:
            return ChunkMetadata(
                summary="Groq client unconfigured. Skipping enrichment.",
                keywords=["skipped"],
                category="General",
                parent_context_id=parent_id
            )

        schema_json = ChunkMetadata.model_json_schema()
        system_prompt = (
            "You are a pipeline intelligence agent. Analyze the provided document chunk elements.\n"
            "You MUST return a valid JSON object matching this exact schema:\n"
            f"{json.dumps(schema_json['properties'])}\n"
            "Return ONLY raw JSON."
        )

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze these document elements:\n\n{combined_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            parsed = json.loads(completion.choices[0].message.content)
            parsed["parent_context_id"] = parent_id
            return ChunkMetadata(**parsed)
        except Exception as e:
            return ChunkMetadata(
                summary=f"Groq enrichment error: {str(e)}",
                keywords=["error"],
                category="Error-Logs",
                parent_context_id=parent_id
            )

    async def describe_image_async(self, image_path: str) -> str:
        if not os.path.exists(image_path) or not self.async_client:
            return f"[Image analysis skipped for {os.path.basename(image_path)}]"

        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Analyze this diagram or chart. Provide a comprehensive, detailed technical description of "
            "all labels, flowcharts, connections, and data values present."
        )

        try:
            res = await self.async_client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }],
                temperature=0.2
            )
            return self._sanitize(res.choices[0].message.content) or "[Empty vision summary]"
        except Exception as e:
            return f"[Vision processing error for {os.path.basename(image_path)}: {str(e)}]"

    async def complete_text_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.async_client:
            return "Groq client missing API Key."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            res = await self.async_client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=0.1
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"Groq completion error: {str(e)}"

    async def stream_text_async(self, prompt: str, system_prompt: Optional[str] = None):
        if not self.async_client:
            yield "Groq client missing API Key."
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=0.1,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else ""
                if content:
                    yield content
        except Exception as e:
            yield f"\n[Groq streaming error: {str(e)}]"
