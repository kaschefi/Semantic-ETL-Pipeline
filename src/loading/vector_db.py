# src/loading/vector_db.py
import hashlib
from typing import List, Dict, Any
from pinecone import Pinecone
from src.config import settings
from src.models import ExtractedElement, ChunkMetadata, FinalVectorPayload


class VectorDatabaseClient:
    def __init__(self):
        # Initialize Pinecone engine with untracked ecosystem secrets
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        self.index_name = getattr(settings, "PINECONE_INDEX_NAME", "semantic-etl-index")
        self.index = self.pc.Index(self.index_name) if self.pc else None

    def generate_embedding(self, text: str) -> List[float]:
        """
        Computes the target 1024-dimensional dense array representation.
        Using Pinecone's native inference endpoint or a local equivalent.
        """
        if not self.pc:
            return [0.0] * 1024

        try:
            # Utilizing a 1024-dimensional embedding model (e.g., bge-large-en)
            res = self.pc.inference.embed(
                model="bge-large-en",
                inputs=[text],
                parameters={"input_type": "passage"}
            )
            return res.data[0].values
        except Exception as e:
            # Fallback mock arrays if network infrastructure times out
            print(f"Embedding extraction warning: {e}. Generating empty vectors.")
            return [0.0] * 1024

    def build_payload(self, elements: List[ExtractedElement], ai_metadata: ChunkMetadata) -> FinalVectorPayload:
        """
        Takes raw elements and AI metadata to produce the exact FinalVectorPayload contract
        with a fully flattened metadata map.
        """
        # Stitch text content together
        text_content = "\n".join([el.content for el in elements])

        # Derive a unique deterministic UUID hash to avoid duplication conflicts on upserts
        deterministic_id = hashlib.md5(text_content.encode('utf-8')).hexdigest()

        # Generate the dense representation array
        vector_array = self.generate_embedding(text_content)

        # Collapse deep structures into the required flat JSON metadata map
        flat_metadata = {
            "summary": ai_metadata.summary,
            "keywords": ai_metadata.keywords,
            "category": ai_metadata.category,
            "parent_context_id": ai_metadata.parent_context_id or "",
            "source_pages": list(set([el.source_page for el in elements if el.source_page is not None])),
            "element_types": list(set([el.element_type for el in elements if el.element_type]))
        }

        return FinalVectorPayload(
            id=deterministic_id,
            vector=vector_array,
            text_content=text_content,
            metadata=flat_metadata
        )

    def upsert_payloads(self, payloads: List[FinalVectorPayload], namespace: str = "documents") -> int:
        """
        Ingests the FinalVectorPayload batch collections directly into Pinecone.
        """
        if not self.index:
            print("Ingestion bypassed: Pinecone index client uninitialized.")
            return 0

        # Transform the object model records into format required by the Pinecone native client
        pinecone_records = []
        for p in payloads:
            # Flatten the text content directly inside the metadata dictionary for RAG retrieval lookups
            record_metadata = p.metadata.copy()
            record_metadata["text_content"] = p.text_content

            pinecone_records.append({
                "id": p.id,
                "values": p.vector,
                "metadata": record_metadata
            })

        # Execute network batch operations
        res = self.index.upsert(vectors=pinecone_records, namespace=namespace)
        return res.get("upserted_count", len(pinecone_records))