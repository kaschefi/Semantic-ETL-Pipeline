# src/agents/retriever.py
from typing import List, Dict, Any, Optional
from src.loading.vector_db import VectorDatabaseClient
from src.agents.fusion import reciprocal_rank_fusion
from src.utils.logger import logger


class ContextRetriever:
    """
    Enterprise Dual-Path Hybrid Retriever.
    Combines Pinecone 1024-d Dense Vector Search with SQLite FTS5 BM25 Sparse Search,
    fused via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.db = VectorDatabaseClient()

    def get_relevant_context(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel dense and sparse retrieval, then merges candidates using RRF.
        """
        if not self.db.index or not self.db.pc:
            return []

        # Auto-sync local BM25 index from Pinecone if this is a cold namespace
        self.db.lexical_store.sync_from_pinecone_if_empty(self.db.index, namespace=namespace)

        fetch_k = max(top_k * 2, 16)

        # 1. Path A: Dense Semantic Vector Search
        dense_results = []
        try:
            query_vector = self.db.generate_embedding(query_text, input_type="query")
            response = self.db.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=fetch_k,
                include_metadata=True,
                filter=filter_dict
            )
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                dense_results.append({
                    "id": match.get("id", ""),
                    "score": float(match.get("score", 0.0)),
                    "dense_score": float(match.get("score", 0.0)),
                    "text": meta.get("text_content", ""),
                    "summary": meta.get("summary", ""),
                    "category": meta.get("category", ""),
                    "source_pages": meta.get("source_pages", []),
                    "parent_context_id": meta.get("parent_context_id", "")
                })
        except Exception as e:
            logger.error(f"Dense vector retrieval failure: {e}")

        # 2. Path B: Sparse Lexical BM25 Search
        sparse_results = []
        try:
            sparse_results = self.db.lexical_store.search(
                query_text=query_text,
                namespace=namespace,
                top_k=fetch_k
            )
        except Exception as e:
            logger.error(f"Sparse lexical retrieval failure: {e}")

        # 3. Path C: Reciprocal Rank Fusion (RRF)
        # If one search path is empty, fall back gracefully to the other
        if not sparse_results:
            return dense_results[:top_k]
        if not dense_results:
            return sparse_results[:top_k]

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60,
            top_n=top_k
        )

        return fused
