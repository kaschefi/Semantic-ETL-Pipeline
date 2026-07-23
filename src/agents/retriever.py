# src/agent/retriever.py
import re
from typing import List, Dict, Any
from src.loading.vector_db import VectorDatabaseClient

class ContextRetriever:
    def __init__(self):
        self.db = VectorDatabaseClient()

    def get_relevant_context(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate context matches using Hybrid Search (Dense Vectors + Lexical Keyword Boost).
        """
        if not self.db.index or not self.db.pc:
            return []

        # Generate target dense search vector
        query_vector = self.db.generate_embedding(query_text)

        # Extract search query tokens for lexical keyword scoring boost
        query_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query_text.lower()))

        try:
            # Query Pinecone space for initial candidate pool (fetching 2x requested top_k)
            fetch_k = max(top_k * 2, 10)
            response = self.db.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=fetch_k,
                include_metadata=True,
                filter=filter_dict
            )

            results = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                dense_score = match.get("score", 0.0)
                text_content = meta.get("text_content", "")
                chunk_tokens = set(meta.get("search_tokens", [])) or set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text_content.lower()))

                # Lexical keyword match ratio calculation
                keyword_matches = query_tokens.intersection(chunk_tokens)
                keyword_score = len(keyword_matches) / len(query_tokens) if query_tokens else 0.0

                # Combined Hybrid Score: 70% Dense Semantic Vector + 30% Exact Keyword Match
                hybrid_score = (0.7 * dense_score) + (0.3 * keyword_score)

                results.append({
                    "score": hybrid_score,
                    "dense_score": dense_score,
                    "keyword_score": keyword_score,
                    "text": text_content,
                    "summary": meta.get("summary", ""),
                    "category": meta.get("category", ""),
                    "source_pages": meta.get("source_pages", []),
                    "parent_context_id": meta.get("parent_context_id", "")
                })

            # Sort by combined hybrid score descending and trim to top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f" Retrieval execution failure: {str(e)}")
            return []