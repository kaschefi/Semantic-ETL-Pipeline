import os
import sys
import asyncio
import argparse
from typing import Dict, Any

# Ensure stdout and stderr handle UTF-8 cleanly on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

from benchmarks.config import BenchmarkConfig, RESULTS_MD_PATH, RESULTS_JSON_PATH
from benchmarks.runners.ingestion_runner import IngestionBenchmarkRunner
from benchmarks.runners.chunking_runner import ChunkingBenchmarkRunner
from benchmarks.runners.retrieval_runner import RetrievalBenchmarkRunner
from benchmarks.runners.load_runner import LoadBenchmarkRunner
from benchmarks.runners.enrichment_runner import EnrichmentBenchmarkRunner
from benchmarks.reporter import BenchmarkReporter


async def main_async(args):
    config = BenchmarkConfig()
    config.fast_mode = args.fast

    if args.namespace:
        config.benchmark_namespace = args.namespace

    reporter = BenchmarkReporter(
        json_path=os.path.join(args.output_dir, "results.json") if args.output_dir else RESULTS_JSON_PATH,
        md_path=os.path.join(args.output_dir, "results.md") if args.output_dir else RESULTS_MD_PATH
    )

    suites_to_run = []
    if args.all or args.suite == "all":
        suites_to_run = ["ingestion", "chunking", "retrieval", "load", "enrichment"]
    elif args.suite:
        suites_to_run = [args.suite.lower()]
    else:
        # Default to running all suites
        suites_to_run = ["ingestion", "chunking", "retrieval", "load", "enrichment"]

    print("=" * 70)
    print("[START] SEMANTIC ETL & GROUNDED RAG BENCHMARK SUITE")
    print(f"Mode: {'FAST ITERATION' if config.fast_mode else 'FULL COMPREHENSIVE'}")
    print(f"Suites to execute: {', '.join(suites_to_run).upper()}")
    print("=" * 70)

    ingestion_results = {}
    chunking_results = {}
    retrieval_results = {}
    load_results = {}
    enrichment_results = {}

    # 1. Ingestion Throughput
    if "ingestion" in suites_to_run:
        runner = IngestionBenchmarkRunner(config)
        ingestion_results = await runner.run(max_docs=args.num_docs)

    # 2. Chunking Quality
    if "chunking" in suites_to_run:
        runner = ChunkingBenchmarkRunner(config)
        chunking_results = await runner.run(max_docs=args.num_docs)

    # 3. Retrieval Quality
    if "retrieval" in suites_to_run:
        runner = RetrievalBenchmarkRunner(config)
        eval_namespace = args.namespace or (
            f"{config.benchmark_namespace}-semantic" if "chunking" in suites_to_run else config.benchmark_namespace
        )
        retrieval_results = runner.run(
            namespace=eval_namespace,
            query_subset_size=args.queries
        )

    # 4. API Latency & Load
    if "load" in suites_to_run:
        runner = LoadBenchmarkRunner(config)
        concurrencies = [int(c) for c in args.concurrency.split(",")] if args.concurrency else None
        load_results = await runner.run(concurrencies=concurrencies)

    # 5. Metadata Enrichment Accuracy
    if "enrichment" in suites_to_run:
        runner = EnrichmentBenchmarkRunner(config)
        enrichment_results = await runner.run(sample_size=args.enrichment_samples)

    # Compile and save final report (merging with existing suites if running selectively)
    existing_benchmarks = {}
    json_target = os.path.join(args.output_dir, "results.json") if args.output_dir else RESULTS_JSON_PATH
    if os.path.exists(json_target):
        try:
            with open(json_target, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                existing_benchmarks = prev_data.get("benchmarks", {})
        except Exception:
            pass

    full_report = reporter.compile_full_report(
        ingestion_results=ingestion_results or existing_benchmarks.get("ingestion_throughput"),
        chunking_results=chunking_results or existing_benchmarks.get("chunking_quality"),
        retrieval_results=retrieval_results or existing_benchmarks.get("retrieval_quality"),
        load_results=load_results or existing_benchmarks.get("api_load_latency"),
        enrichment_results=enrichment_results or existing_benchmarks.get("metadata_enrichment")
    )

    json_path = reporter.save_json(full_report)
    md_path = reporter.save_markdown(full_report)

    print("\n" + "=" * 70)
    print("[COMPLETED] BENCHMARK SUITE EXECUTION FINISHED SUCCESSFULLY!")
    print(f"Markdown Report Saved: {md_path}")
    print(f"JSON Raw Data Saved:   {json_path}")
    print("=" * 70)

    # Print markdown summary snippet directly to console
    print("\n" + reporter.generate_markdown(full_report))


def main():
    parser = argparse.ArgumentParser(description="Semantic ETL & Grounded RAG Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all 5 benchmark suites")
    parser.add_argument("--suite", type=str, choices=["all", "ingestion", "chunking", "retrieval", "load", "enrichment"], default="all", help="Select specific benchmark suite")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode with reduced samples (<3 mins)")
    parser.add_argument("--num-docs", type=int, default=None, help="Number of documents to test in ingestion/chunking")
    parser.add_argument("--queries", type=int, default=None, help="Number of test queries to evaluate in retrieval")
    parser.add_argument("--concurrency", type=str, default=None, help="Comma-separated concurrency tiers (e.g. 1,5,10,20)")
    parser.add_argument("--enrichment-samples", type=int, default=None, help="Number of chunk samples for enrichment test")
    parser.add_argument("--namespace", type=str, default=None, help="Target Pinecone vector namespace")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store results.md and results.json")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
