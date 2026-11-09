# src/providers/pinecone_provider.py
import hashlib
import re
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata, FinalVectorPayload
from src.providers.base import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    def __init__(self):
        api_key = getattr(settings, 'PINECONE_API_KEY', None)
        self.pc = Pinecone(api_key=api_key) if api_key else None
        self.index_name = getattr(settings, "PINECONE_INDEX_NAME", "semantic-etl-index")
        self.index = self.pc.Index(self.index_name) if self.pc else None

    def generate_embedding(self, text: str, input_type: str = "passage") -> List[float]:
        if not self.pc:
            return [0.00001] * 1024

        try:
            res = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[text],
                parameters={"input_type": input_type}
            )
            return res.data[0].values
        except Exception as e:
            return [0.00001] * 1024

    def build_payload(self, elements: List[ExtractedElement], ai_metadata: ChunkMetadata) -> FinalVectorPayload:
        text_content = "\n".join([el.content for el in elements])
        deterministic_id = hashlib.md5(text_content.encode('utf-8')).hexdigest()
        vector_array = self.generate_embedding(text_content)

        raw_tokens = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text_content.lower())
        unique_tokens = list(set(raw_tokens + [k.lower() for k in ai_metadata.keywords]))[:50]

        flat_metadata = {
            "summary": ai_metadata.summary,
            "keywords": ai_metadata.keywords,
            "search_tokens": unique_tokens,
            "category": ai_metadata.category,
            "parent_context_id": ai_metadata.parent_context_id or "",
            "source_pages": list(set([str(el.source_page) for el in elements if el.source_page is not None])),
            "element_types": list(set([el.element_type for el in elements if el.element_type]))
        }

        return FinalVectorPayload(
            id=deterministic_id,
            vector=vector_array,
            text_content=text_content,
            metadata=flat_metadata
        )

    def upsert_payloads(self, payloads: List[FinalVectorPayload], namespace: str = "documents") -> int:
        if not self.index:
            return 0

        records = []
        for p in payloads:
            meta = p.metadata.copy()
            meta["text_content"] = p.text_content
            records.append({
                "id": p.id,
                "values": p.vector,
                "metadata": meta
            })

        res = self.index.upsert(vectors=records, namespace=namespace)
        return res.get("upserted_count", len(records))

    def query_vectors(
        self,
        query_vector: List[float],
        namespace: str = "documents",
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.index:
            return []

        response = self.index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        return response.get("matches", [])
