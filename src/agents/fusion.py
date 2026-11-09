# src/agents/fusion.py
from typing import List, Dict, Any, Optional


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Fuses ranked candidates from Dense vector search and BM25 sparse search using RRF.
    Score(d) = sum(1.0 / (k + rank_i))
    
    Guarantees that documents appearing highly in either or both lists are prioritized
    without requiring manual heuristic score calibration.
    """
    rrf_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}

    # 1. Score dense candidates
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.get("id") or str(hash(doc.get("text", "")[:50]))
        doc_registry[doc_id] = doc
        dense_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # 2. Score sparse (BM25) candidates
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc.get("id") or str(hash(doc.get("text", "")[:50]))
        if doc_id not in doc_registry:
            doc_registry[doc_id] = doc
        sparse_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # 3. Sort by combined RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)

    fused_candidates: List[Dict[str, Any]] = []
    for doc_id in sorted_doc_ids[:top_n]:
        item = dict(doc_registry[doc_id])
        item["rrf_score"] = round(rrf_scores[doc_id], 6)
        item["score"] = item["rrf_score"]
        item["dense_rank"] = dense_ranks.get(doc_id, -1)
        item["sparse_rank"] = sparse_ranks.get(doc_id, -1)
        fused_candidates.append(item)

    return fused_candidates

