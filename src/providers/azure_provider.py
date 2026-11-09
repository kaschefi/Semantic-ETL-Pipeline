# src/providers/azure_provider.py
import json
import base64
import os
import re
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata
from src.providers.base import BaseLLMProvider
from src.utils.logger import logger


class AzureLLMProvider(BaseLLMProvider):
    """
    Enterprise LLM Provider using Azure AI Foundry / Azure OpenAI endpoints.
    Provides high-throughput chat completions, vision extraction, and structured JSON outputs.
    """
    def __init__(self):
        api_key = getattr(settings, 'AZURE_API_KEY', None)
        endpoint = getattr(settings, 'AZURE_OPENAI_ENDPOINT', None)
        api_version = getattr(settings, 'AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self.model_name = getattr(settings, 'AZURE_MODEL_NAME', 'gpt-4.1-nano')
        self.client = None
        self.async_client = None

        if api_key and endpoint:
            try:
                # If endpoint points to /openai/v1 (Azure AI Foundry project gateway),
                # standard OpenAI client with base_url is the most reliable connector.
                if "/openai/v1" in endpoint:
                    self.client = OpenAI(base_url=endpoint, api_key=api_key)
                    self.async_client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
                else:
                    self.client = AzureOpenAI(
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version
                    )
                    self.async_client = AsyncAzureOpenAI(
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version
                    )
            except Exception as e:
                logger.error(f"Failed to initialize Azure LLM Provider: {e}")

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
                summary="Azure client not configured. Skipping enrichment.",
                keywords=["skipped"],
                category="General",
                parent_context_id=parent_id
            )

        system_prompt = (
            "You are a document intelligence agent. Analyze the provided document chunk elements.\n"
            "You MUST return a valid JSON object with EXACTLY these three keys:\n"
            "- \"summary\": A concise 1-2 sentence string summarizing the key concepts.\n"
            "- \"keywords\": A JSON list of 3-7 specific lowercase keyword strings.\n"
            "- \"category\": A concise domain category string (e.g. 'Database Architecture', 'DevOps', 'Software Design', 'Algorithms').\n\n"
            "Example valid response:\n"
            "{\n"
            '  "summary": "Overview of database transaction guarantees including ACID properties.",\n'
            '  "keywords": ["acid", "transactions", "atomicity", "durability"],\n'
            '  "category": "Database Architecture"\n'
            "}\n"
            "Return ONLY raw JSON with concrete data values, never schema descriptions."
        )

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze these document elements:\n\n{combined_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            parsed = json.loads(completion.choices[0].message.content)

            # Defensive parsing against models echoing schema dictionaries
            if isinstance(parsed.get("summary"), dict):
                parsed["summary"] = parsed["summary"].get("default", parsed["summary"].get("description", "Document chunk summary."))
            if isinstance(parsed.get("category"), dict):
                parsed["category"] = parsed["category"].get("default", parsed["category"].get("description", "General"))
            if isinstance(parsed.get("keywords"), dict):
                parsed["keywords"] = list(parsed["keywords"].keys())
            elif not isinstance(parsed.get("keywords"), list):
                parsed["keywords"] = [str(parsed.get("keywords"))] if parsed.get("keywords") else ["general"]

            # Ensure all keywords are strings
            parsed["keywords"] = [str(k) for k in parsed["keywords"]]
            parsed["parent_context_id"] = parent_id
            return ChunkMetadata(**parsed)
        except Exception as e:
            logger.error(f"Azure enrichment error: {e}")
            return ChunkMetadata(
                summary=f"Azure enrichment error: {str(e)}",
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
            "Analyze this diagram or chart extracted from a document. Provide a comprehensive, "
            "detailed technical description of all labels, flowcharts, connections, equations, and data values."
        )

        try:
            res = await self.async_client.chat.completions.create(
                model=self.model_name,
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
            logger.error(f"Azure vision error for {image_path}: {e}")
            return f"[Vision processing error for {os.path.basename(image_path)}: {str(e)}]"

    async def complete_text_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.async_client:
            return "Azure client missing credentials."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            res = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            return res.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure completion error: {e}")
            return f"Error from Azure LLM: {str(e)}"

    async def stream_text_async(self, prompt: str, system_prompt: Optional[str] = None):
        if not self.async_client:
            yield "Azure client missing credentials."
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Azure streaming error: {e}")
            yield f"\n[Streaming error: {str(e)}]"

    def complete_text_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            return "Azure client missing credentials."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            return res.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure sync completion error: {e}")
            return f"Error from Azure LLM: {str(e)}"
