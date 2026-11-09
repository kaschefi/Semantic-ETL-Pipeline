import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
BENCHMARKS_DIR = os.path.join(BASE_DIR, "benchmarks")
RESULTS_MD_PATH = os.path.join(BENCHMARKS_DIR, "results.md")
RESULTS_JSON_PATH = os.path.join(BENCHMARKS_DIR, "results.json")


@dataclass
class BenchmarkConfig:
    # Environment & Sandbox Settings
    benchmark_namespace: str = "benchmark-eval-sandbox"
    data_dir: str = DATA_DIR
    cache_dir: str = CACHE_DIR
    results_md_path: str = RESULTS_MD_PATH
    results_json_path: str = RESULTS_JSON_PATH

    # Ingestion Benchmark Parameters
    fast_mode: bool = False
    ingestion_num_docs: int = 6           # Default doc count for full run
    ingestion_num_docs_fast: int = 2      # Fast mode doc count
    ingestion_timeout_seconds: int = 300  # Per document max timeout

    # Chunking Quality Parameters
    naive_chunk_size_chars: int = 1200
    naive_chunk_overlap_chars: int = 200

    # Retrieval Benchmark Parameters
    retrieval_top_k: int = 8
    retrieval_rerank_top_n: int = 8

    # Concurrency & Load Benchmark Parameters
    load_concurrencies: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    load_concurrencies_fast: List[int] = field(default_factory=lambda: [1, 5, 10])
    load_requests_per_concurrency: int = 5
    load_requests_per_concurrency_fast: int = 2

    # Enrichment Accuracy Parameters
    enrichment_sample_size: int = 15
    enrichment_sample_size_fast: int = 5


default_benchmark_config = BenchmarkConfig()
