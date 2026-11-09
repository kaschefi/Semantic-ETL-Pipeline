import math
import numpy as np
from typing import List, Dict, Any, Optional, Set


def calculate_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Computes statistical percentiles and summary distributions for latency arrays.
    """
    if not latencies_ms:
        return {
            "count": 0,
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "std_dev_ms": 0.0
        }

    arr = np.array(latencies_ms, dtype=float)
    return {
        "count": int(len(arr)),
        "min_ms": round(float(np.min(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_dev_ms": round(float(np.std(arr)), 2)
    }


def is_chunk_relevant(
    chunk_text: str,
    chunk_summary: str,
    chunk_category: str,
    ground_truth_keywords: List[str],
    ground_truth_concepts: List[str],
    expected_category: Optional[str] = None
) -> bool:
    """
    Evaluates whether a retrieved chunk is semantically and lexically relevant
    to a ground truth query based on keyword overlap, conceptual presence, and category alignment.
    """
    combined_text = f"{chunk_category} {chunk_summary} {chunk_text}".lower()

    # Check for keyword matches
    matched_keywords = [kw for kw in ground_truth_keywords if kw.lower() in combined_text]
    keyword_hit_ratio = len(matched_keywords) / max(len(ground_truth_keywords), 1)

    # Check for concept matches
    matched_concepts = [cp for cp in ground_truth_concepts if cp.lower() in combined_text]

    # Category match bonus
    category_match = False
    if expected_category and chunk_category:
        expected_tokens = set(expected_category.lower().replace("&", "").split())
        chunk_cat_tokens = set(chunk_category.lower().replace("&", "").split())
        category_match = bool(expected_tokens.intersection(chunk_cat_tokens))

    # A chunk is deemed relevant if:
    # 1. At least 1 full concept is present, OR
    # 2. >= 40% of ground truth keywords are present, OR
    # 3. >= 25% of keywords + Category matches
    if len(matched_concepts) >= 1:
        return True
    if keyword_hit_ratio >= 0.40:
        return True
    if keyword_hit_ratio >= 0.25 and category_match:
        return True

    return False


def calculate_retrieval_metrics_for_query(
    retrieved_hits: List[Dict[str, Any]],
    query_info: Dict[str, Any],
    k_values: List[int] = [1, 3, 5, 8]
) -> Dict[str, Any]:
    """
    Computes Recall@K, Precision@K, Reciprocal Rank (RR), and DCG@K for a single query.
    """
    keywords = query_info.get("ground_truth_keywords", [])
    concepts = query_info.get("ground_truth_concepts", [])
    expected_cat = query_info.get("expected_category", "")

    relevance_flags: List[bool] = []
    for hit in retrieved_hits:
        rel = is_chunk_relevant(
            chunk_text=hit.get("text", ""),
            chunk_summary=hit.get("summary", ""),
            chunk_category=hit.get("category", ""),
            ground_truth_keywords=keywords,
            ground_truth_concepts=concepts,
            expected_category=expected_cat
        )
        relevance_flags.append(rel)

    # Calculate Reciprocal Rank
    reciprocal_rank = 0.0
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            reciprocal_rank = 1.0 / rank
            break

    # Calculate Recall@K and Precision@K
    metrics: Dict[str, Any] = {
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": int(1.0 / reciprocal_rank) if reciprocal_rank > 0 else -1,
        "total_hits": len(retrieved_hits),
        "total_relevant_in_hits": sum(relevance_flags)
    }

    for k in k_values:
        sub_flags = relevance_flags[:k]
        rel_count = sum(sub_flags)
        # Recall@K: Did we find at least one relevant context in top K? (Targeting RAG context retrieval)
        recall_at_k = 1.0 if rel_count > 0 else 0.0
        precision_at_k = rel_count / k if k > 0 else 0.0

        # DCG@K calculation
        dcg = 0.0
        for i, is_rel in enumerate(sub_flags):
            if is_rel:
                dcg += 1.0 / math.log2((i + 1) + 1)

        metrics[f"recall@{k}"] = recall_at_k
        metrics[f"precision@{k}"] = round(precision_at_k, 3)
        metrics[f"dcg@{k}"] = round(dcg, 3)

    return metrics


def aggregate_ir_metrics(per_query_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates per-query metrics into dataset-wide IR benchmark figures:
    Recall@1, Recall@3, Recall@5, MRR, MAP@5, NDCG@5.
    """
    if not per_query_metrics:
        return {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "precision@1": 0.0,
            "precision@3": 0.0,
            "precision@5": 0.0,
            "ndcg@5": 0.0,
            "query_count": 0
        }

    n = len(per_query_metrics)
    mrr = sum(q.get("reciprocal_rank", 0.0) for q in per_query_metrics) / n
    recall_1 = sum(q.get("recall@1", 0.0) for q in per_query_metrics) / n
    recall_3 = sum(q.get("recall@3", 0.0) for q in per_query_metrics) / n
    recall_5 = sum(q.get("recall@5", 0.0) for q in per_query_metrics) / n

    precision_1 = sum(q.get("precision@1", 0.0) for q in per_query_metrics) / n
    precision_3 = sum(q.get("precision@3", 0.0) for q in per_query_metrics) / n
    precision_5 = sum(q.get("precision@5", 0.0) for q in per_query_metrics) / n

    # NDCG@5 ideal DCG calculation
    idcg_5 = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    avg_dcg_5 = sum(q.get("dcg@5", 0.0) for q in per_query_metrics) / n
    ndcg_5 = avg_dcg_5 / idcg_5 if idcg_5 > 0 else 0.0

    return {
        "query_count": n,
        "mrr": round(mrr, 4),
        "recall@1": round(recall_1, 4),
        "recall@3": round(recall_3, 4),
        "recall@5": round(recall_5, 4),
        "precision@1": round(precision_1, 4),
        "precision@3": round(precision_3, 4),
        "precision@5": round(precision_5, 4),
        "ndcg@5": round(ndcg_5, 4)
    }


def evaluate_chunking_structural_integrity(
    chunks: List[List[Any]]
) -> Dict[str, Any]:
    """
    Measures structural coherence, boundary violations (headers/tables fractured),
    and character length distributions for a chunking strategy.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_chars": 0.0,
            "std_chunk_chars": 0.0,
            "min_chunk_chars": 0,
            "max_chunk_chars": 0,
            "boundary_fracture_count": 0,
            "boundary_preservation_rate": 1.0
        }

    chunk_lengths = []
    boundary_fractures = 0
    preserved_structures = 0

    for chunk in chunks:
        chunk_text = "\n".join([getattr(el, "content", str(el)) for el in chunk])
        chunk_lengths.append(len(chunk_text))

        is_naive = any(getattr(el, "element_type", "") == "RawText" for el in chunk)
        if is_naive:
            # Check if chunk starts or ends with fragmented markdown table rows or cut-off sentences
            if "|" in chunk_text and not (chunk_text.strip().startswith("|") and chunk_text.strip().endswith("|")):
                boundary_fractures += 1
            if len(chunk_text) >= 1100 and not chunk_text.strip().endswith((".", "\n", "|", "!", "?", ":")):
                boundary_fractures += 1
        else:
            for el in chunk:
                el_type = getattr(el, "element_type", "")
                if el_type in ["Header", "Title", "SectionHeader", "Table", "TableItem"]:
                    preserved_structures += 1

    arr = np.array(chunk_lengths) if chunk_lengths else np.array([0])
    total_evaluated = len(chunks) + preserved_structures
    preservation_rate = (
        max(0.0, min(1.0, (total_evaluated - boundary_fractures) / max(total_evaluated, 1)))
        if total_evaluated > 0 else 1.0
    )

    return {
        "total_chunks": len(chunks),
        "avg_chunk_chars": round(float(np.mean(arr)), 1),
        "std_chunk_chars": round(float(np.std(arr)), 1),
        "min_chunk_chars": int(np.min(arr)),
        "max_chunk_chars": int(np.max(arr)),
        "boundary_fracture_count": boundary_fractures,
        "boundary_preservation_rate": round(preservation_rate, 4)
    }


def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Computes Jaccard Similarity index between two token sets.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return round(intersection / union, 4) if union > 0 else 0.0
