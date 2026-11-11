# src/agents/reranker.py
import os
from typing import List, Dict, Any, Optional
from flashrank import Ranker, RerankRequest
from src.config import settings
from src.utils.logger import logger


class ContextReranker:
    """
    Sub-30ms Cross-Encoder Reranker using FlashRank (ONNX-optimized ms-marco-MiniLM).
    Replaces brittle and slow LLM generative reranking with deterministic cross-attention scoring.
    """
    _instance: Optional['ContextReranker'] = None
    _ranker: Optional[Ranker] = None

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        if ContextReranker._ranker is None:
            try:
                cache_dir = os.path.join(settings.DATA_DIR, "cache", "flashrank")
                os.makedirs(cache_dir, exist_ok=True)
                ContextReranker._ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
                logger.info(f"Initialized FlashRank Cross-Encoder ({model_name}) in {cache_dir}.")
            except Exception as e:
                logger.error(f"FlashRank initialization error: {e}")
        self.ranker = ContextReranker._ranker

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks candidates using true cross-encoder attention over (query, document) pairs.
        Returns top_n highest scoring candidates in descending order of relevance.
        """
        if not candidates:
            return []

        if len(candidates) <= 1 or not self.ranker:
            return candidates[:top_n]

        try:
            # Build passages formatted for FlashRank
            passages = []
            for idx, cand in enumerate(candidates):
                summary = cand.get("summary", "")
                category = cand.get("category", "")
                text = cand.get("text", "")
                # Rich composite representation for optimal cross-attention
                composite_text = f"[{category}] {summary} {text}".strip()
                passages.append({
                    "id": idx,
                    "text": composite_text[:1200],
                    "meta": cand
                })

            rerank_req = RerankRequest(query=query, passages=passages)
            ranked_results = self.ranker.rerank(rerank_req)

            reranked_candidates = []
            for item in ranked_results[:top_n]:
                original_candidate = item["meta"]
                original_candidate["rerank_score"] = float(item["score"])
                original_candidate["score"] = float(item["score"])
                reranked_candidates.append(original_candidate)

            return reranked_candidates

        except Exception as e:
            logger.warning(f"FlashRank reranking error: {e}. Falling back to input order.")
            return candidates[:top_n]
