import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from benchmarks.config import BenchmarkConfig, default_benchmark_config
from benchmarks.dataset import get_available_documents, GROUND_TRUTH_EVAL_QUERIES
from benchmarks.baselines import NaiveFixedChunker, BaselineRetrievalEvaluator
from benchmarks.metrics import (
    evaluate_chunking_structural_integrity,
    calculate_retrieval_metrics_for_query,
    aggregate_ir_metrics
)
from src.extraction.doc_parser import PDFParserEngine
from src.transformation.chunker import SemanticChunker
from src.transformation.enricher import ChunkEnricher
from src.loading.vector_db import VectorDatabaseClient
from src.models import ExtractedElement, FinalVectorPayload


class ChunkingBenchmarkRunner:
    """
    Benchmark Suite 2: Compares Semantic Chunker vs. Naive Fixed-Window Chunker
    on structural boundary preservation and downstream Information Retrieval (IR) quality.
    """
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or default_benchmark_config
        self.parser = PDFParserEngine()
        self.semantic_chunker = SemanticChunker(
            max_chunk_chars=self.config.naive_chunk_size_chars,
            overlap_elements=1
        )
        self.naive_chunker = NaiveFixedChunker(
            max_chunk_chars=self.config.naive_chunk_size_chars,
            overlap_chars=self.config.naive_chunk_overlap_chars
        )
        self.enricher = ChunkEnricher()
        self.db_client = VectorDatabaseClient()
        self.evaluator = BaselineRetrievalEvaluator(self.db_client)

    async def _index_elements(
        self,
        chunked_windows: List[List[ExtractedElement]],
        namespace: str
    ) -> int:
        """
        Enriches and upserts chunked windows into a specific Pinecone namespace.
        """
        parent_id = str(uuid.uuid4())
        ai_metadatas = await self.enricher.enrich_batch(chunked_windows, parent_id=parent_id)
        payloads: List[FinalVectorPayload] = [
            self.db_client.build_payload(window, meta)
            for window, meta in zip(chunked_windows, ai_metadatas)
        ]
        return self.db_client.upsert_payloads(payloads, namespace=namespace)

    async def run(self, max_docs: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes structural and downstream retrieval comparisons between Semantic and Naive chunkers.
        """
        available_docs = get_available_documents(self.config.data_dir)
        if not available_docs:
            return {"error": "No documents found in data directory."}

        num_docs = max_docs or (self.config.ingestion_num_docs_fast if self.config.fast_mode else 3)
        selected_docs = sorted(available_docs, key=lambda d: d["size_bytes"])[:num_docs]

        print(f"\n[Chunking Quality Benchmark] Evaluating Semantic vs Naive chunking on {len(selected_docs)} documents...")

        all_extracted_elements: List[ExtractedElement] = []
        for doc in selected_docs:
            print(f" -> Parsing {doc['filename']}...")
            elements = self.parser.extract_document(doc["path"])
            all_extracted_elements.extend(elements)

        if not all_extracted_elements:
            return {"error": "No elements extracted from test documents."}

        # Step 1: Structural Integrity Evaluation
        print("[Chunking Quality Benchmark] Evaluating structural boundary preservation...")
        semantic_windows = self.semantic_chunker.group_elements(all_extracted_elements)
        naive_windows = self.naive_chunker.group_elements(all_extracted_elements)

        semantic_structural = evaluate_chunking_structural_integrity(semantic_windows)
        naive_structural = evaluate_chunking_structural_integrity(naive_windows)

        # Step 2: Index Both Strategies in Isolated Namespaces for Downstream Comparison
        semantic_namespace = f"{self.config.benchmark_namespace}-semantic"
        naive_namespace = f"{self.config.benchmark_namespace}-naive"

        print(f"[Chunking Quality Benchmark] Indexing {len(semantic_windows)} semantic chunks into '{semantic_namespace}'...")
        await self._index_elements(semantic_windows, namespace=semantic_namespace)

        print(f"[Chunking Quality Benchmark] Indexing {len(naive_windows)} naive chunks into '{naive_namespace}'...")
        await self._index_elements(naive_windows, namespace=naive_namespace)

        # Step 3: Run Downstream Retrieval Evaluation Queries
        queries = GROUND_TRUTH_EVAL_QUERIES[: (10 if self.config.fast_mode else len(GROUND_TRUTH_EVAL_QUERIES))]
        print(f"[Chunking Quality Benchmark] Running {len(queries)} retrieval queries across both chunking indices...")

        semantic_query_metrics = []
        naive_query_metrics = []

        for q in queries:
            query_text = q["query"]
            # Hybrid search against semantic namespace
            semantic_hits = self.evaluator.search_hybrid(query_text, namespace=semantic_namespace, top_k=5)
            s_metrics = calculate_retrieval_metrics_for_query(semantic_hits, q, k_values=[1, 3, 5])
            semantic_query_metrics.append(s_metrics)

            # Hybrid search against naive namespace
            naive_hits = self.evaluator.search_hybrid(query_text, namespace=naive_namespace, top_k=5)
            n_metrics = calculate_retrieval_metrics_for_query(naive_hits, q, k_values=[1, 3, 5])
            naive_query_metrics.append(n_metrics)

        semantic_ir = aggregate_ir_metrics(semantic_query_metrics)
        naive_ir = aggregate_ir_metrics(naive_query_metrics)

        # Calculate empirical improvement delta
        mrr_lift_pct = round(((semantic_ir["mrr"] - naive_ir["mrr"]) / max(naive_ir["mrr"], 0.001)) * 100, 2)
        recall3_lift_pct = round(((semantic_ir["recall@3"] - naive_ir["recall@3"]) / max(naive_ir["recall@3"], 0.001)) * 100, 2)
        recall5_lift_pct = round(((semantic_ir["recall@5"] - naive_ir["recall@5"]) / max(naive_ir["recall@5"], 0.001)) * 100, 2)

        results = {
            "test_documents_count": len(selected_docs),
            "total_elements_processed": len(all_extracted_elements),
            "structural_comparison": {
                "semantic_chunker": semantic_structural,
                "naive_fixed_chunker": naive_structural
            },
            "downstream_retrieval_comparison": {
                "semantic_chunking_ir": semantic_ir,
                "naive_chunking_ir": naive_ir,
                "empirical_lift": {
                    "mrr_improvement_pct": mrr_lift_pct,
                    "recall_at_3_improvement_pct": recall3_lift_pct,
                    "recall_at_5_improvement_pct": recall5_lift_pct
                }
            }
        }

        print(f"[Chunking Benchmark Completed] Semantic Chunking achieved {mrr_lift_pct}% MRR lift over Naive fixed chunking.")
        return results
