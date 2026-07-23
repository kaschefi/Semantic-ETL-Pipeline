# src/transformation/enricher.py
import json
from typing import List
from groq import Groq
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata


class ChunkEnricher:
    def __init__(self):
        # Initialize the Groq client safely if API key is provided
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self.model_name = "openai/gpt-oss-120b"

    def enrich_elements(self, elements: List[ExtractedElement], parent_id: str = None) -> ChunkMetadata:
        """
        Combines grouped elements and prompts a local/cloud Groq LLM
        to return the exact ChunkMetadata payload format.
        """
        combined_text = "\n".join([f"[{el.element_type}]: {el.content}" for el in elements])

        if not self.client:
            return ChunkMetadata(
                summary="Groq client not configured. Skipping enrichment.",
                keywords=["skipped"],
                category="General",
                parent_context_id=parent_id
            )

        # Crafting a strict system prompt to guarantee JSON structure matches our Pydantic schema
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

            # Parse the strict JSON response directly back into the native Pydantic contract
            response_content = completion.choices[0].message.content
            parsed_data = json.loads(response_content)

            parsed_data["parent_context_id"] = parent_id

            return ChunkMetadata(**parsed_data)

        except Exception as e:
            # Graceful degradation so API limits or network drops don't break the pipeline run
            return ChunkMetadata(
                summary=f"Failed to extract semantic metadata via Groq: {str(e)}",
                keywords=["error", "groq-failure"],
                category="Error-Logs",
                parent_context_id=parent_id
            )