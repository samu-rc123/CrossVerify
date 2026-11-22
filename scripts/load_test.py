#!/usr/bin/env python3
"""
CrossVerify High-Throughput Load Testing & Backpressure Benchmark CLI.
Simulates production concurrency against the CrossVerify API Gateway to evaluate:
- Request throughput (RPS)
- Synchronous Champion latency distribution (p50, p95, p99)
- Async traffic shadow queue backpressure
- Canary circuit breaker status headers (active vs aborted)
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import List, Dict, Any

try:
    import httpx
except ImportError:
    print("httpx is required to run load_test.py. Install it with: pip install httpx")
    sys.exit(1)


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    request_id: int
) -> Dict[str, Any]:
    """Send a single chat completion request and measure latency."""
    start = time.perf_counter()
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    try:
        response = await client.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        canary_status = response.headers.get("X-Canary-Status", "unknown")
        req_id = response.headers.get("X-Request-ID", f"req-{request_id}")
        model_latency = response.headers.get("X-CrossVerify-Latency-Ms", "0")

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "model_latency_ms": float(model_latency),
            "canary_status": canary_status,
            "request_id": req_id,
            "error": None
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "success": False,
            "status_code": 0,
            "elapsed_ms": elapsed_ms,
            "model_latency_ms": 0.0,
            "canary_status": "error",
            "request_id": f"req-{request_id}",
            "error": str(e)
        }


async def worker(
    name: int,
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    results: List[Dict[str, Any]]
):
    """Worker task draining the queue of request indices."""
    while True:
        try:
            req_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        res = await send_request(client, url, payload, req_id)
        results.append(res)
        queue.task_done()


async def run_benchmark(
    url: str,
    total_requests: int,
    concurrency: int,
    model: str
):
    """Execute concurrent load test run and print statistical rollup."""
    print("=" * 70)
    print("  PROJECT CROSSVERIFY: LOAD TEST & QUEUE BACKPRESSURE BENCHMARK")
    print("=" * 70)
    print(f"Target URL:        {url}")
    print(f"Total Requests:    {total_requests}")
    print(f"Concurrency:       {concurrency}")
    print(f"Target Model:      {model}")
    print("-" * 70)

    queue = asyncio.Queue()
    for i in range(total_requests):
        queue.put_nowait(i)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a production assistant."},
            {"role": "user", "content": "Explain the concept of asynchronous traffic shadowing."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    results: List[Dict[str, Any]] = []

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(15.0, connect=5.0)

    start_bench = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        workers = [
            asyncio.create_task(worker(w_id, queue, client, url, payload, results))
            for w_id in range(concurrency)
        ]
        await asyncio.gather(*workers)

    total_time = time.perf_counter() - start_bench
    rps = len(results) / total_time if total_time > 0 else 0.0

    # Metrics calculation
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["elapsed_ms"] for r in results]
    model_latencies = [r["model_latency_ms"] for r in successful if r["model_latency_ms"] > 0]

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
    avg_lat = statistics.mean(latencies) if latencies else 0.0

    canary_active = sum(1 for r in results if r.get("canary_status") == "active")
    canary_aborted = sum(1 for r in results if r.get("canary_status") == "aborted")

    # Display results
    print("\nBENCHMARK RESULTS:")
    print("-" * 70)
    print(f"Total Requests Completed:   {len(results)} / {total_requests}")
    print(f"Successful (HTTP 200):      {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed / Errored:           {len(failed)}")
    print(f"Wall-Clock Duration:        {total_time:.2f} seconds")
    print(f"Throughput (RPS):           {rps:.2f} requests/sec")
    print("-" * 70)
    print("GATEWAY HTTP ROUND-TRIP LATENCIES (ms):")
    print(f"  Min:                      {min(latencies):.2f} ms")
    print(f"  Average:                  {avg_lat:.2f} ms")
    print(f"  Median (p50):             {p50:.2f} ms")
    print(f"  90th Percentile (p90):    {p90:.2f} ms")
    print(f"  95th Percentile (p95):    {p95:.2f} ms")
    print(f"  99th Percentile (p99):    {p99:.2f} ms")
    print(f"  Max:                      {max(latencies):.2f} ms")
    print("-" * 70)
    print("DYNAMIC CANARY ROUTING TELEMETRY:")
    print(f"  Shadow Traffic Active:    {canary_active} ({canary_active/len(results)*100:.1f}%)")
    print(f"  Shadow Traffic Aborted:   {canary_aborted} ({canary_aborted/len(results)*100:.1f}%)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CrossVerify Load Testing & Backpressure Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway endpoint URL")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to dispatch")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--model", default="champion-gpt-4o", help="Model name")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        model=args.model
    ))


if __name__ == "__main__":
    main()

