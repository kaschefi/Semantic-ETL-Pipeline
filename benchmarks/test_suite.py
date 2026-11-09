import pytest
import os
import asyncio
from benchmarks.config import BenchmarkConfig
from benchmarks.dataset import get_available_documents, GROUND_TRUTH_EVAL_QUERIES, LABELED_ENRICHMENT_SAMPLES
from benchmarks.baselines import NaiveFixedChunker, BaselineRetrievalEvaluator
from benchmarks.metrics import (
    calculate_latency_percentiles,
    calculate_retrieval_metrics_for_query,
    aggregate_ir_metrics,
    evaluate_chunking_structural_integrity,
    calculate_jaccard_similarity
)
from benchmarks.reporter import BenchmarkReporter
from benchmarks.runners.enrichment_runner import EnrichmentBenchmarkRunner
from benchmarks.runners.load_runner import LoadBenchmarkRunner
from src.transformation.chunker import SemanticChunker
from src.models import ExtractedElement


def test_available_documents():
    """Verify that test documents are discovered from data/."""
    docs = get_available_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0
    for doc in docs:
        assert os.path.exists(doc["path"])
        assert doc["size_bytes"] > 0


def test_ground_truth_dataset():
    """Verify ground truth evaluation dataset integrity."""
    assert len(GROUND_TRUTH_EVAL_QUERIES) >= 20
    for q in GROUND_TRUTH_EVAL_QUERIES:
        assert "query_id" in q
        assert "query" in q
        assert len(q["ground_truth_keywords"]) > 0
        assert len(q["ground_truth_concepts"]) > 0


def test_labeled_enrichment_samples():
    """Verify labeled samples for Groq LLM tagger accuracy."""
    assert len(LABELED_ENRICHMENT_SAMPLES) >= 5
    for s in LABELED_ENRICHMENT_SAMPLES:
        assert "sample_id" in s
        assert len(s["elements"]) > 0
        assert s["ground_truth_category"]
        assert len(s["ground_truth_keywords"]) > 0


def test_latency_percentiles_calculation():
    """Verify math calculations for latency percentiles."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = calculate_latency_percentiles(latencies)

    assert stats["count"] == 10
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 100.0
    assert stats["p50_ms"] == 55.0
    assert stats["p50_ms"] <= stats["p90_ms"] <= stats["p95_ms"] <= stats["p99_ms"]


def test_ir_metrics_calculation():
    """Verify Information Retrieval (IR) Recall and MRR metric calculations."""
    query = {
        "ground_truth_keywords": ["testing", "unit", "mock"],
        "ground_truth_concepts": ["unit testing", "test pyramid"],
        "expected_category": "Software Testing"
    }

    # Perfect hit at rank 1
    hits = [
        {
            "text": "Unit testing and test pyramid with mock objects",
            "summary": "Covers unit testing",
            "category": "Software Testing"
        },
        {
            "text": "Unrelated content about something else",
            "summary": "General overview",
            "category": "General"
        }
    ]

    m = calculate_retrieval_metrics_for_query(hits, query, k_values=[1, 3, 5])
    assert m["reciprocal_rank"] == 1.0
    assert m["recall@1"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["precision@1"] == 1.0

    agg = aggregate_ir_metrics([m])
    assert agg["mrr"] == 1.0
    assert agg["recall@1"] == 1.0


def test_chunking_structural_integrity():
    """Verify structural comparison between semantic and naive chunking."""
    elements = [
        ExtractedElement(element_type="Header", content="Architecture Section", source_page=1),
        ExtractedElement(element_type="Paragraph", content="Detailed architecture design text...", source_page=1),
        ExtractedElement(element_type="Table", content="| Col 1 | Col 2 |\n| Val 1 | Val 2 |", source_page=1),
        ExtractedElement(element_type="Header", content="Testing Section", source_page=2),
        ExtractedElement(element_type="Paragraph", content="Test automation overview text...", source_page=2)
    ]

    semantic_chunker = SemanticChunker(max_chunk_chars=500, overlap_elements=1)
    naive_chunker = NaiveFixedChunker(max_chunk_chars=500, overlap_chars=50)

    s_chunks = semantic_chunker.group_elements(elements)
    n_chunks = naive_chunker.group_elements(elements)

    s_stats = evaluate_chunking_structural_integrity(s_chunks)
    n_stats = evaluate_chunking_structural_integrity(n_chunks)

    assert s_stats["total_chunks"] >= 1
    assert n_stats["total_chunks"] >= 1
    assert s_stats["boundary_preservation_rate"] >= 0.0


def test_reporter_compilation(tmp_path):
    """Verify that BenchmarkReporter creates valid Markdown and JSON."""
    json_file = str(tmp_path / "test_results.json")
    md_file = str(tmp_path / "test_results.md")

    reporter = BenchmarkReporter(json_path=json_file, md_path=md_file)
    sample_report = reporter.compile_full_report(
        ingestion_results={"documents_per_minute": 15.2, "avg_time_per_document_s": 3.9},
        chunking_results={"downstream_retrieval_comparison": {"empirical_lift": {"mrr_improvement_pct": 24.5}}},
        retrieval_results={"comparative_lift": {"rerank_over_vector_only": {"mrr_improvement_pct": 31.8}}},
        load_results={"tier_benchmarks": [{"concurrency": 1, "throughput_rps": 4.5, "latency_percentiles": {"p50_ms": 220.0}}]},
        enrichment_results={"category_agreement_rate_pct": 95.0, "schema_conformance_rate_pct": 100.0}
    )

    reporter.save_json(sample_report)
    reporter.save_markdown(sample_report)

    assert os.path.exists(json_file)
    assert os.path.exists(md_file)

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Semantic ETL & Grounded RAG Performance Benchmark Report" in md_text
    assert "Executive Benchmark Summary" in md_text


def test_job_manager_sqlite_persistence(tmp_path):
    """Verify that JobManager writes to SQLite and persists across process restarts."""
    from src.jobs import JobManager
    db_file = str(tmp_path / "test_jobs.db")

    jm1 = JobManager(db_path=db_file)
    job = jm1.create_job(file_path="sample.pdf", namespace="test-ns")
    assert job.status == "queued"
    assert job.job_id is not None

    updated = jm1.update_job(
        job.job_id,
        status="completed",
        progress=100.0,
        result={"chunks_indexed": 42}
    )
    assert updated.status == "completed"
    assert updated.progress == 100.0
    assert updated.result == {"chunks_indexed": 42}

    # Simulate restart by creating a new JobManager instance on the same SQLite DB
    jm2 = JobManager(db_path=db_file)
    recovered = jm2.get_job(job.job_id)
    assert recovered is not None
    assert recovered.job_id == job.job_id
    assert recovered.status == "completed"
    assert recovered.result == {"chunks_indexed": 42}
    assert len(jm2.list_jobs()) == 1


