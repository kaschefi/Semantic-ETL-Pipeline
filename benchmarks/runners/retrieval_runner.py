import time
from typing import List, Dict, Any, Optional
from benchmarks.config import BenchmarkConfig, default_benchmark_config
from benchmarks.dataset import GROUND_TRUTH_EVAL_QUERIES
from benchmarks.baselines import BaselineRetrievalEvaluator
from benchmarks.metrics import (
    calculate_retrieval_metrics_for_query,
    aggregate_ir_metrics,
    calculate_latency_percentiles
)
from src.loading.vector_db import VectorDatabaseClient


class RetrievalBenchmarkRunner:
    """
    Benchmark Suite 3: Evaluates retrieval quality across 4 search paradigms:
    1. Vector-Only Dense Search
    2. Lexical / Keyword Token Search
    3. Hybrid Search (Dense + Lexical, No Rerank)
    4. Full Hybrid Search + Groq Cross-Encoder LLM Reranking
    """
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or default_benchmark_config
        self.db_client = VectorDatabaseClient()
        self.evaluator = BaselineRetrievalEvaluator(self.db_client)

    def run(
        self,
        namespace: Optional[str] = None,
        query_subset_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation dataset across all retrieval configurations.
        """
        target_namespace = namespace or self.config.benchmark_namespace
        all_queries = GROUND_TRUTH_EVAL_QUERIES
        num_queries = query_subset_size or (10 if self.config.fast_mode else len(all_queries))
        queries = all_queries[:num_queries]

        print(f"\n[Retrieval Quality Benchmark] Evaluating {len(queries)} ground truth queries against namespace '{target_namespace}'...")

        configurations = [
            ("vector_only", "Dense Vector-Only Search (Pinecone 1024-d)", lambda q: self.evaluator.search_vector_only(q, namespace=target_namespace, top_k=self.config.retrieval_top_k)),
            ("lexical_only", "Lexical Keyword Token Search", lambda q: self.evaluator.search_lexical_only(q, namespace=target_namespace, top_k=self.config.retrieval_top_k)),
            ("hybrid_no_rerank", "Hybrid Search (Dense + Lexical, No Reranking)", lambda q: self.evaluator.search_hybrid(q, namespace=target_namespace, top_k=self.config.retrieval_top_k)),
            ("hybrid_rerank", "Full Hybrid Search + FlashRank Cross-Encoder Reranking", lambda q: self.evaluator.search_hybrid_rerank(q, namespace=target_namespace, top_k=self.config.retrieval_top_k, rerank_top_n=self.config.retrieval_rerank_top_n))
        ]

        results_by_config: Dict[str, Any] = {}

        for config_key, config_name, search_fn in configurations:
            print(f" -> Benchmarking: {config_name}...")
            query_metrics = []
            latencies_ms = []

            for q in queries:
                query_text = q["query"]
                t0 = time.perf_counter()
                try:
                    hits = search_fn(query_text)
                except Exception as e:
                    print(f" [Warning] Query '{q['query_id']}' failed for {config_key}: {e}")
                    hits = []
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)

                m = calculate_retrieval_metrics_for_query(
                    retrieved_hits=hits,
                    query_info=q,
                    k_values=[1, 3, 5, 8]
                )
                m["query_id"] = q["query_id"]
                m["latency_ms"] = round(latency_ms, 2)
                query_metrics.append(m)

            aggregated_ir = aggregate_ir_metrics(query_metrics)
            latency_stats = calculate_latency_percentiles(latencies_ms)

            results_by_config[config_key] = {
                "name": config_name,
                "metrics": aggregated_ir,
                "latency_stats": latency_stats,
                "per_query_details": query_metrics
            }

        # Calculate comparative lift of Hybrid + Reranking over Vector Only Baseline
        v_only_ir = results_by_config["vector_only"]["metrics"]
        hybrid_ir = results_by_config["hybrid_no_rerank"]["metrics"]
        rerank_ir = results_by_config["hybrid_rerank"]["metrics"]

        mrr_lift = round(((rerank_ir["mrr"] - v_only_ir["mrr"]) / max(v_only_ir["mrr"], 0.0001)) * 100, 2)
        recall3_lift = round(((rerank_ir["recall@3"] - v_only_ir["recall@3"]) / max(v_only_ir["recall@3"], 0.0001)) * 100, 2)
        recall5_lift = round(((rerank_ir["recall@5"] - v_only_ir["recall@5"]) / max(v_only_ir["recall@5"], 0.0001)) * 100, 2)

        summary = {
            "evaluated_queries_count": len(queries),
            "namespace": target_namespace,
            "configurations": results_by_config,
            "comparative_lift": {
                "rerank_over_vector_only": {
                    "mrr_improvement_pct": mrr_lift,
                    "recall_at_3_improvement_pct": recall3_lift,
                    "recall_at_5_improvement_pct": recall5_lift
                },
                "hybrid_over_vector_only": {
                    "mrr_improvement_pct": round(((hybrid_ir["mrr"] - v_only_ir["mrr"]) / max(v_only_ir["mrr"], 0.0001)) * 100, 2),
                    "recall_at_3_improvement_pct": round(((hybrid_ir["recall@3"] - v_only_ir["recall@3"]) / max(v_only_ir["recall@3"], 0.0001)) * 100, 2)
                }
            }
        }

        print(f"[Retrieval Benchmark Completed] Full Hybrid+Rerank achieved MRR={rerank_ir['mrr']} (Recall@3={rerank_ir['recall@3']}, Recall@5={rerank_ir['recall@5']}).")
        return summary
