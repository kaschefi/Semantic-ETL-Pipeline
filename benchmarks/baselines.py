import re
import hashlib
from typing import List, Dict, Any, Optional
from src.models import ExtractedElement, ChunkMetadata, FinalVectorPayload
from src.loading.vector_db import VectorDatabaseClient
from src.agents.reranker import ContextReranker
from src.agents.fusion import reciprocal_rank_fusion


class NaiveFixedChunker:
    """
    Baseline naive chunking strategy:
    Concatenates all raw document text and slices strictly into fixed character windows
    (e.g., 1200 chars / ~500 tokens with fixed 200 char overlap), completely ignoring
    document layout, headers, tables, and natural semantic boundaries.
    """
    def __init__(self, max_chunk_chars: int = 1200, overlap_chars: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def group_elements(self, elements: List[ExtractedElement]) -> List[List[ExtractedElement]]:
        """
        Slices raw concatenated document text into fixed-size character windows,
        creating naive chunks that fracture structural elements.
        """
        if not elements:
            return []

        full_text = "\n".join([el.content for el in elements if el.content])
        step = max(self.max_chunk_chars - self.overlap_chars, 100)

        chunks: List[List[ExtractedElement]] = []
        for i in range(0, len(full_text), step):
            window_text = full_text[i : i + self.max_chunk_chars]
            if not window_text.strip():
                continue
            chunks.append([
                ExtractedElement(
                    content=window_text,
                    element_type="RawText",
                    source_page=1
                )
            ])

        return chunks


class BaselineRetrievalEvaluator:
    """
    Standardized evaluator executing different retrieval configurations against
    an indexed corpus to measure comparative information retrieval (IR) metrics.
    """
    def __init__(self, vector_client: Optional[VectorDatabaseClient] = None):
        self.db = vector_client or VectorDatabaseClient()
        self.reranker = ContextReranker()
        self._embedding_cache: Dict[str, List[float]] = {}

    def _get_embedding(self, query_text: str) -> List[float]:
        if query_text not in self._embedding_cache:
            self._embedding_cache[query_text] = self.db.generate_embedding(query_text, input_type="query")
        return self._embedding_cache[query_text]

    def search_vector_only(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Configuration A: Pure Dense Vector Search (1024-dim embedding cosine similarity only).
        """
        if not self.db.index or not self.db.pc:
            return []

        query_vector = self._get_embedding(query_text)
        try:
            response = self.db.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            results = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                dense_score = float(match.get("score", 0.0))
                results.append({
                    "id": match.get("id", ""),
                    "score": dense_score,
                    "dense_score": dense_score,
                    "keyword_score": 0.0,
                    "text": meta.get("text_content", ""),
                    "summary": meta.get("summary", ""),
                    "category": meta.get("category", ""),
                    "source_pages": meta.get("source_pages", []),
                    "search_tokens": meta.get("search_tokens", [])
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"[Vector Only Search Error]: {e}")
            return []

    def search_lexical_only(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Configuration B: Pure BM25 Lexical / Inverted Index Search via SQLite FTS5.
        """
        self.db.lexical_store.sync_from_pinecone_if_empty(self.db.index, namespace=namespace)
        try:
            return self.db.lexical_store.search(
                query_text=query_text,
                namespace=namespace,
                top_k=top_k
            )
        except Exception as e:
            print(f"[Lexical Only Search Error]: {e}")
            return []

    def search_hybrid(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Configuration C: True Dual-Path Hybrid Search (Dense Vectors + BM25 FTS5, fused via RRF).
        """
        if not self.db.index or not self.db.pc:
            return []

        self.db.lexical_store.sync_from_pinecone_if_empty(self.db.index, namespace=namespace)
        fetch_k = max(top_k * 2, 16)

        # Path 1: Dense Vector Candidates
        dense_results = self.search_vector_only(query_text, namespace=namespace, top_k=fetch_k)

        # Path 2: Sparse BM25 Candidates
        sparse_results = self.search_lexical_only(query_text, namespace=namespace, top_k=fetch_k)

        if not sparse_results:
            return dense_results[:top_k]
        if not dense_results:
            return sparse_results[:top_k]

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60,
            top_n=top_k
        )

    def search_hybrid_rerank(
        self,
        query_text: str,
        namespace: str = "documents",
        top_k: int = 8,
        rerank_top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Configuration D: True Hybrid Search (RRF) + FlashRank Cross-Encoder Reranking.
        """
        fetch_k = max(top_k * 2, 16)
        candidates = self.search_hybrid(query_text=query_text, namespace=namespace, top_k=fetch_k)
        if not candidates:
            return []
        effective_top_n = rerank_top_n if rerank_top_n is not None else top_k
        return self.reranker.rerank(query=query_text, candidates=candidates, top_n=effective_top_n)
