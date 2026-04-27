#!/usr/bin/env python3
"""vLLM throughput benchmark for DGX Spark.

Measures single-request and concurrent throughput against an OpenAI-compatible API.
Adapted from Entry 009 methodology for reuse across model evaluations.

Usage:
    python throughput_bench.py                          # defaults: localhost:8000, c1/c4/c8/c16
    python throughput_bench.py --url http://host:8000   # custom endpoint
    python throughput_bench.py --concurrency 1 4 8      # specific concurrency levels
    python throughput_bench.py --model gemma4-26b       # different model name
    python throughput_bench.py --tokens 600 --runs 3    # custom token count and runs
"""

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

def get_metric(base_url, name):
    """Fetch a single Prometheus metric from vLLM /metrics endpoint."""
    try:
        req = urllib.request.Request(f"{base_url}/metrics")
        resp = urllib.request.urlopen(req, timeout=5).read().decode()
        for line in resp.split("\n"):
            if line.startswith(f"{name}{{"):
                return float(line.split()[-1])
    except Exception:
        pass
    return 0

def single_request(url, model, max_tokens):
    """Send a single completion request and measure throughput."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 600 one per line. Output only numbers."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    start = time.monotonic()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    elapsed = time.monotonic() - start

    usage = resp["usage"]
    return {
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "elapsed": elapsed,
        "tok_s": usage["completion_tokens"] / elapsed,
    }

def run_benchmark(url, model, max_tokens, concurrency, runs):
    """Run benchmark at a given concurrency level."""
    base_url = url.rstrip("/")
    api_url = base_url if "/v1" in base_url else base_url

    results = []
    for run_num in range(runs):
        gen_before = get_metric(base_url.replace("/v1", ""), "vllm:generation_tokens_total")
        batch_start = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(single_request, base_url.replace("/v1", ""), model, max_tokens)
                       for _ in range(concurrency)]
            batch_results = [f.result() for f in concurrent.futures.as_completed(futures)]

        batch_time = time.monotonic() - batch_start
        gen_after = get_metric(base_url.replace("/v1", ""), "vllm:generation_tokens_total")

        total_tokens = sum(r["completion_tokens"] for r in batch_results)
        avg_per_req = sum(r["tok_s"] for r in batch_results) / len(batch_results)
        aggregate = total_tokens / batch_time
        server_delta = gen_after - gen_before

        results.append({
            "run": run_num + 1,
            "concurrency": concurrency,
            "per_req_tok_s": avg_per_req,
            "aggregate_tok_s": aggregate,
            "batch_time": batch_time,
            "total_tokens": total_tokens,
            "server_gen_delta": server_delta,
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--model", default="spark-llm", help="Model name")
    parser.add_argument("--tokens", type=int, default=600, help="Max completion tokens")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16], help="Concurrency levels")
    parser.add_argument("--runs", type=int, default=3, help="Runs per concurrency level")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Health check
    try:
        req = urllib.request.Request(f"{args.url}/health")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"ERROR: Cannot reach {args.url}/health: {e}", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for conc in args.concurrency:
        results = run_benchmark(args.url, args.model, args.tokens, conc, args.runs)
        all_results.extend(results)

        # Print summary for this concurrency level
        avg_per = sum(r["per_req_tok_s"] for r in results) / len(results)
        avg_agg = sum(r["aggregate_tok_s"] for r in results) / len(results)
        avg_time = sum(r["batch_time"] for r in results) / len(results)
        print(f"c{conc:2d}: per-req={avg_per:6.1f} tok/s  aggregate={avg_agg:6.1f} tok/s  batch_time={avg_time:.1f}s  ({args.runs} runs)")

    if args.json:
        print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
