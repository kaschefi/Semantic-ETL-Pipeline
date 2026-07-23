# src/agent/retriever.py
from typing import List, Dict, Any
from src.loading.vector_db import VectorDatabaseClient

class ContextRetriever:
    def __init__(self):
        self.db = VectorDatabaseClient()

    def get_relevant_context(self, query_text: str, namespace: str = "documents", top_k: int = 4, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Converts the user prompt into a vector search query and extracts matching text + tags.
        """
        if not self.db.index or not self.db.pc:
            return []

        #  Use our unified embedding generation path to construct the search vector
        query_vector = self.db.generate_embedding(query_text)

        try:
            #  Query Pinecone space
            response = self.db.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict  # Allows rapid structural metadata filtering (e.g. category="Compliance")
            )

            #  Format hits into cleanly organized chunks
            results = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                results.append({
                    "score": match.get("score"),
                    "text": meta.get("text_content", ""),
                    "summary": meta.get("summary", ""),
                    "category": meta.get("category", ""),
                    "source_pages": meta.get("source_pages", []),
                    "parent_context_id": meta.get("parent_context_id", "")
                })
            return results

        except Exception as e:
            print(f" Retrieval execution failure: {str(e)}")
            return []