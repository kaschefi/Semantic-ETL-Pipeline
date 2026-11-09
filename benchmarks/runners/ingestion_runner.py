import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from benchmarks.config import BenchmarkConfig, default_benchmark_config
from benchmarks.dataset import get_available_documents
from benchmarks.metrics import calculate_latency_percentiles
from src.extraction.doc_parser import PDFParserEngine
from src.extraction.vision import MultimodalVisionEngine
from src.transformation.chunker import SemanticChunker
from src.transformation.enricher import ChunkEnricher
from src.loading.vector_db import VectorDatabaseClient
from src.models import ExtractedElement, FinalVectorPayload
from src.jobs import job_manager


class IngestionBenchmarkRunner:
    """
    Benchmark Suite 1: Measures document ingestion throughput, stage-by-stage latency,
    and async job queue execution latency (p50/p95 from 202 Accepted to completion).
    """
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or default_benchmark_config
        self.parser = PDFParserEngine()
        self.chunker = SemanticChunker(
            max_chunk_chars=self.config.naive_chunk_size_chars,
            overlap_elements=1
        )
        self.enricher = ChunkEnricher()
        self.db_client = VectorDatabaseClient()

    async def benchmark_single_document_stages(
        self,
        file_path: str,
        namespace: str
    ) -> Dict[str, Any]:
        """
        Processes a single document end-to-end while recording high-precision stage timings.
        """
        doc_filename = os.path.basename(file_path)
        doc_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        t_total_start = time.perf_counter()

        # Stage 1: Layout & PDF Extraction
        t0 = time.perf_counter()
        extracted_elements = self.parser.extract_document(file_path)
        extraction_time_s = time.perf_counter() - t0

        # Stage 2: Multimodal Vision
        image_elements = [el for el in extracted_elements if el.element_type == "Image" and el.image_cache_path]
        vision_time_s = 0.0
        if image_elements:
            t0 = time.perf_counter()
            vision_engine = MultimodalVisionEngine()
            image_paths = [el.image_cache_path for el in image_elements]
            descriptions = await vision_engine.describe_images_batch(image_paths)
            for el, desc in zip(image_elements, descriptions):
                el.content = f"[Visual Image Summary]: {desc}"
            vision_time_s = time.perf_counter() - t0

        # Stage 3: Semantic Chunking
        t0 = time.perf_counter()
        grouped_windows = self.chunker.group_elements(extracted_elements)
        chunking_time_s = time.perf_counter() - t0

        # Stage 4: Groq AI Metadata Enrichment
        t0 = time.perf_counter()
        parent_id = str(uuid.uuid4())
        ai_metadatas = await self.enricher.enrich_batch(grouped_windows, parent_id=parent_id)
        enrichment_time_s = time.perf_counter() - t0

        # Stage 5: Pinecone Vector Embedding & Upsert
        t0 = time.perf_counter()
        final_payloads: List[FinalVectorPayload] = [
            self.db_client.build_payload(window, metadata)
            for window, metadata in zip(grouped_windows, ai_metadatas)
        ]
        upserted_count = self.db_client.upsert_payloads(final_payloads, namespace=namespace)
        upsert_time_s = time.perf_counter() - t0

        total_time_s = time.perf_counter() - t_total_start

        return {
            "filename": doc_filename,
            "size_kb": round(doc_size_bytes / 1024, 2),
            "elements_extracted": len(extracted_elements),
            "images_processed": len(image_elements),
            "chunks_created": len(grouped_windows),
            "vectors_upserted": upserted_count,
            "stage_extraction_s": round(extraction_time_s, 3),
            "stage_vision_s": round(vision_time_s, 3),
            "stage_chunking_s": round(chunking_time_s, 3),
            "stage_enrichment_s": round(enrichment_time_s, 3),
            "stage_upsert_s": round(upsert_time_s, 3),
            "total_time_s": round(total_time_s, 3),
            "throughput_kb_per_s": round((doc_size_bytes / 1024) / total_time_s, 2) if total_time_s > 0 else 0.0
        }

    async def benchmark_async_job_queue(
        self,
        test_files: List[str],
        namespace: str
    ) -> List[Dict[str, Any]]:
        """
        Measures the async background job execution lifecycle from job creation (202 Accepted)
        to final completion in job_manager.
        """
        from src.main import _run_background_pipeline

        job_results = []
        for file_path in test_files:
            job = job_manager.create_job(file_path=file_path, namespace=namespace)
            t_accepted = time.perf_counter()

            # Execute background pipeline runner task
            await _run_background_pipeline(
                job_id=job.job_id,
                file_path=file_path,
                namespace=namespace,
                remove_temp_file=False
            )

            t_finished = time.perf_counter()
            duration_ms = (t_finished - t_accepted) * 1000.0

            updated_job = job_manager.get_job(job.job_id)
            job_results.append({
                "job_id": job.job_id,
                "filename": os.path.basename(file_path),
                "status": updated_job.status if updated_job else "unknown",
                "queue_latency_ms": round(duration_ms, 2)
            })

        return job_results

    async def run(self, max_docs: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the comprehensive ingestion throughput and async job queue benchmark.
        """
        available_docs = get_available_documents(self.config.data_dir)
        if not available_docs:
            return {"error": "No PDF documents located in data directory."}

        # Filter out unusually huge files for rapid benchmarking if max_docs is set
        sorted_docs = sorted(available_docs, key=lambda d: d["size_bytes"])
        num_docs = max_docs or (self.config.ingestion_num_docs_fast if self.config.fast_mode else self.config.ingestion_num_docs)
        selected_docs = sorted_docs[:num_docs]
        selected_paths = [d["path"] for d in selected_docs]

        print(f"\n[Ingestion Benchmark] Commencing benchmark on {len(selected_paths)} test documents...")

        per_doc_metrics = []
        t_batch_start = time.perf_counter()

        for doc_path in selected_paths:
            print(f" -> Benchmarking document: {os.path.basename(doc_path)} ({round(os.path.getsize(doc_path)/1024, 1)} KB)...")
            try:
                metrics = await self.benchmark_single_document_stages(
                    file_path=doc_path,
                    namespace=self.config.benchmark_namespace
                )
                per_doc_metrics.append(metrics)
            except Exception as e:
                print(f" [Warning] Ingestion failed for {os.path.basename(doc_path)}: {e}")

        total_batch_duration_s = time.perf_counter() - t_batch_start
        total_chunks = sum(m["chunks_created"] for m in per_doc_metrics)
        total_kb = sum(m["size_kb"] for m in per_doc_metrics)

        # Measure async job queue latency
        print("[Ingestion Benchmark] Measuring async job queue execution latencies (202 Accepted -> Polling)...")
        queue_count = 1 if self.config.fast_mode else min(len(selected_paths), 2)
        queue_test_files = selected_paths[:queue_count]
        queue_jobs = await self.benchmark_async_job_queue(
            test_files=queue_test_files,
            namespace=self.config.benchmark_namespace
        )
        queue_latencies_ms = [j["queue_latency_ms"] for j in queue_jobs]
        queue_percentiles = calculate_latency_percentiles(queue_latencies_ms)

        # Aggregate summary metrics
        successful_docs = len(per_doc_metrics)
        avg_doc_time_s = round(total_batch_duration_s / successful_docs, 2) if successful_docs > 0 else 0.0
        docs_per_minute = round((successful_docs / total_batch_duration_s) * 60, 2) if total_batch_duration_s > 0 else 0.0
        chunks_per_minute = round((total_chunks / total_batch_duration_s) * 60, 2) if total_batch_duration_s > 0 else 0.0

        # Stage latency averages
        avg_stages = {
            "avg_extraction_s": round(sum(m["stage_extraction_s"] for m in per_doc_metrics) / max(successful_docs, 1), 3),
            "avg_vision_s": round(sum(m["stage_vision_s"] for m in per_doc_metrics) / max(successful_docs, 1), 3),
            "avg_chunking_s": round(sum(m["stage_chunking_s"] for m in per_doc_metrics) / max(successful_docs, 1), 3),
            "avg_enrichment_s": round(sum(m["stage_enrichment_s"] for m in per_doc_metrics) / max(successful_docs, 1), 3),
            "avg_upsert_s": round(sum(m["stage_upsert_s"] for m in per_doc_metrics) / max(successful_docs, 1), 3)
        }

        results = {
            "documents_processed": successful_docs,
            "total_batch_duration_s": round(total_batch_duration_s, 2),
            "avg_time_per_document_s": avg_doc_time_s,
            "documents_per_minute": docs_per_minute,
            "chunks_per_minute": chunks_per_minute,
            "total_chunks_indexed": total_chunks,
            "total_data_volume_kb": round(total_kb, 2),
            "stage_breakdowns_avg": avg_stages,
            "async_queue_latency_percentiles": queue_percentiles,
            "per_document_details": per_doc_metrics
        }

        print(f"[Ingestion Benchmark Completed] Processed {successful_docs} docs in {round(total_batch_duration_s, 1)}s ({docs_per_minute} docs/min).")
        return results
