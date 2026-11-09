import time
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from benchmarks.config import BenchmarkConfig, default_benchmark_config
from benchmarks.dataset import GROUND_TRUTH_EVAL_QUERIES
from benchmarks.metrics import calculate_latency_percentiles
from src.main import app


class LoadBenchmarkRunner:
    """
    Benchmark Suite 4: Measures FastAPI endpoint throughput, error rate,
    and p50/p90/p95/p99 latency under concurrency loads (1, 5, 10, 20 concurrent workers).
    """
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or default_benchmark_config

    async def _send_query_request(
        self,
        client: httpx.AsyncClient,
        question: str,
        namespace: str
    ) -> Dict[str, Any]:
        """
        Sends a single POST /agent/query request and tracks execution latency and HTTP status.
        """
        t0 = time.perf_counter()
        try:
            response = await client.post(
                "/agent/query",
                json={"question": question, "namespace": namespace},
                timeout=60.0
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "success": response.status_code == 200
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status_code": 500,
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            }

    async def benchmark_concurrency_tier(
        self,
        concurrency: int,
        total_requests: int,
        namespace: str
    ) -> Dict[str, Any]:
        """
        Executes a batch of requests with a bounded concurrency semaphore.
        """
        semaphore = asyncio.Semaphore(concurrency)
        queries = [q["query"] for q in GROUND_TRUTH_EVAL_QUERIES]

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async def _worker(idx: int):
                async with semaphore:
                    q_text = queries[idx % len(queries)]
                    return await self._send_query_request(client, q_text, namespace)

            t_start = time.perf_counter()
            tasks = [_worker(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)
            total_duration_s = time.perf_counter() - t_start

        latencies_ms = [r["latency_ms"] for r in results]
        successful = [r for r in results if r["success"]]
        success_rate = (len(successful) / len(results)) * 100 if results else 0.0
        rps = len(results) / total_duration_s if total_duration_s > 0 else 0.0

        percentiles = calculate_latency_percentiles(latencies_ms)

        return {
            "concurrency": concurrency,
            "total_requests": total_requests,
            "successful_requests": len(successful),
            "success_rate_pct": round(success_rate, 2),
            "total_duration_s": round(total_duration_s, 2),
            "throughput_rps": round(rps, 2),
            "latency_percentiles": percentiles
        }

    async def run(
        self,
        concurrencies: Optional[List[int]] = None,
        requests_multiplier: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs load testing across all configured concurrency levels and reports degradation.
        """
        concurrency_levels = concurrencies or (
            self.config.load_concurrencies_fast if self.config.fast_mode else self.config.load_concurrencies
        )
        multiplier = requests_multiplier or (
            self.config.load_requests_per_concurrency_fast if self.config.fast_mode else self.config.load_requests_per_concurrency
        )

        print(f"\n[Load & Concurrency Benchmark] Commencing load testing across concurrency levels: {concurrency_levels}...")

        tier_results = []
        for c in concurrency_levels:
            total_reqs = max(c * multiplier, 2)
            print(f" -> Testing concurrency={c} ({total_reqs} total requests)...")
            tier_data = await self.benchmark_concurrency_tier(
                concurrency=c,
                total_requests=total_reqs,
                namespace=self.config.benchmark_namespace
            )
            tier_results.append(tier_data)

        # Calculate latency degradation factor relative to concurrency=1 baseline
        baseline = next((t for t in tier_results if t["concurrency"] == 1), tier_results[0])
        baseline_p50 = baseline["latency_percentiles"]["p50_ms"]
        baseline_p95 = baseline["latency_percentiles"]["p95_ms"]

        degradation_curve = []
        for t in tier_results:
            c = t["concurrency"]
            p50 = t["latency_percentiles"]["p50_ms"]
            p95 = t["latency_percentiles"]["p95_ms"]
            p50_deg = round(p50 / max(baseline_p50, 0.001), 2)
            p95_deg = round(p95 / max(baseline_p95, 0.001), 2)

            degradation_curve.append({
                "concurrency": c,
                "p50_ms": p50,
                "p95_ms": p95,
                "throughput_rps": t["throughput_rps"],
                "p50_degradation_factor": p50_deg,
                "p95_degradation_factor": p95_deg
            })

        results = {
            "tested_concurrencies": concurrency_levels,
            "baseline_concurrency": baseline["concurrency"],
            "tier_benchmarks": tier_results,
            "degradation_curve": degradation_curve
        }

        print(f"[Load Benchmark Completed] Concurrency testing finished across {len(concurrency_levels)} tiers.")
        return results
