"""
Benchmark for GPU/User configurations
Tests: 1 GPU/128 users, 2 GPU/128 users, 2 GPU/64 users, 2 GPU/32 users, 4 GPU/32 users, 4 GPU/16 users

Supports configurable ISL (Input Sequence Length) and OSL (Output Sequence Length)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp


def generate_prompt_with_tokens(target_tokens: int) -> str:
    """
    Generate a prompt with approximately target_tokens input tokens.
    Uses a repeating pattern to reach the target length.
    ~1.3 tokens per word is a rough estimate for English text.
    """
    base_prompt = "Explain the following concept in detail: "

    if target_tokens <= 50:
        return "Explain the concept of machine learning in simple terms."

    # Filler text that's coherent and can be repeated
    filler = (
        "Consider the implications of artificial intelligence on modern society, "
        "including its effects on employment, healthcare, education, and transportation. "
        "Analyze how machine learning algorithms process vast amounts of data to identify patterns "
        "and make predictions that were previously impossible for traditional computing methods. "
        "Examine the ethical considerations surrounding autonomous systems and the need for "
        "transparent, explainable AI that humans can trust and verify. "
    )

    # Estimate: ~1.3 tokens per word, so we need target_tokens / 1.3 words
    target_words = int(target_tokens / 1.3)
    filler_words = filler.split()

    # Build prompt by repeating filler
    words = base_prompt.split()
    while len(words) < target_words:
        words.extend(filler_words)

    # Trim to target
    words = words[:target_words]
    return " ".join(words)


@dataclass
class BenchmarkResult:
    total_time: float
    ttft: float
    tokens_generated: int
    tokens_per_second: float
    input_tokens: int


@dataclass
class BenchmarkStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    total_time: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
        self.successful_requests += 1
        self.total_tokens_generated += result.tokens_generated

    def add_failure(self):
        self.failed_requests += 1

    def summary(self) -> dict:
        if not self.results:
            return {"error": "No successful requests"}

        latencies = [r.total_time for r in self.results]
        ttfts = [r.ttft for r in self.results]
        tps = [r.tokens_per_second for r in self.results]

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_benchmark_time_seconds": self.total_time,
            "throughput_requests_per_second": self.successful_requests / self.total_time if self.total_time > 0 else 0,
            "throughput_tokens_per_second": self.total_tokens_generated / self.total_time if self.total_time > 0 else 0,
            "latency": {
                "min_s": min(latencies),
                "max_s": max(latencies),
                "avg_s": sum(latencies) / len(latencies),
                "p50_s": sorted(latencies)[len(latencies) // 2],
            },
            "time_to_first_token": {
                "min_s": min(ttfts),
                "max_s": max(ttfts),
                "avg_s": sum(ttfts) / len(ttfts),
                "p50_s": sorted(ttfts)[len(ttfts) // 2],
            },
            "tokens_per_second_per_request": {
                "min": min(tps),
                "max": max(tps),
                "avg": sum(tps) / len(tps),
            },
        }


async def send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list,
    max_tokens: int = 256,
) -> BenchmarkResult:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    start_time = time.perf_counter()
    ttft = None
    tokens_generated = 0
    input_tokens = sum(len(m.get("content", "").split()) for m in messages) * 1.3

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=180)
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]

            try:
                chunk = json.loads(line)
                if chunk.get("object") == "chat.completion.chunk":
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        tokens_generated += 1
            except json.JSONDecodeError:
                continue

    total_time = time.perf_counter() - start_time
    if ttft is None:
        ttft = total_time

    generation_time = total_time - ttft
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

    return BenchmarkResult(
        total_time=total_time,
        ttft=ttft,
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,
        input_tokens=int(input_tokens),
    )


async def run_benchmark(
    base_url: str,
    num_requests: int,
    concurrent_requests: int,
    max_tokens: int = 256,
    input_tokens: int = 50,
) -> BenchmarkStats:
    stats = BenchmarkStats(total_requests=num_requests)

    # Generate prompt with target input token count
    prompt = generate_prompt_with_tokens(input_tokens)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    start_time = time.perf_counter()

    async with aiohttp.ClientSession(base_url=base_url) as session:
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def bounded_request(request_id: int):
            async with semaphore:
                try:
                    result = await send_request(session, "llm", messages, max_tokens)
                    stats.add_result(result)
                except Exception as e:
                    stats.add_failure()
                    print(f"Request {request_id} failed: {e}")

        tasks = [bounded_request(i) for i in range(num_requests)]
        await asyncio.gather(*tasks)

    stats.total_time = time.perf_counter() - start_time
    return stats


async def health_check(base_url: str, timeout: int = 300) -> bool:
    """Check if endpoint is healthy, wait for cold start if needed"""
    async with aiohttp.ClientSession(base_url=base_url) as session:
        try:
            async with session.get("/health", timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


async def run_gpu_user_benchmark(
    endpoints: dict[int, str],  # {n_gpu: url}
    configurations: list[tuple[int, int]],  # [(n_gpu, n_users), ...]
    requests_per_config: int = 50,
    input_tokens: int = 1024,
    output_tokens: int = 1024,
):
    """
    Run benchmarks for specific GPU/user configurations

    Args:
        endpoints: Dict mapping GPU count to endpoint URL
        configurations: List of (n_gpu, n_users) tuples to test
        requests_per_config: Number of requests to send per configuration
        input_tokens: Input sequence length (ISL) - target input tokens
        output_tokens: Output sequence length (OSL) - max output tokens
    """
    results = []

    for n_gpu, n_users in configurations:
        if n_gpu not in endpoints:
            print(f"Skipping {n_gpu} GPU / {n_users} users - no endpoint available")
            continue

        base_url = endpoints[n_gpu]
        print(f"\n{'='*60}")
        print(f"Configuration: {n_gpu} GPU / {n_users} concurrent users")
        print(f"Endpoint: {base_url}")
        print(f"{'='*60}")

        # Health check / wake up
        print("Checking endpoint health (may take time for cold start)...")
        if not await health_check(base_url):
            print(f"Endpoint not healthy, skipping")
            continue
        print("Endpoint healthy!")

        # Run benchmark
        print(f"Running benchmark: {requests_per_config} requests, {n_users} concurrent...")
        print(f"ISL: {input_tokens} tokens, OSL: {output_tokens} tokens")
        stats = await run_benchmark(
            base_url=base_url,
            num_requests=requests_per_config,
            concurrent_requests=n_users,
            max_tokens=output_tokens,
            input_tokens=input_tokens,
        )

        summary = stats.summary()
        if "error" not in summary:
            throughput_total = summary["throughput_tokens_per_second"]
            throughput_per_gpu = throughput_total / n_gpu
            avg_latency = summary["latency"]["avg_s"]
            interactivity = summary["tokens_per_second_per_request"]["avg"]

            result = {
                "n_gpu": n_gpu,
                "n_users": n_users,
                "config_label": f"{n_gpu}GPU/{n_users}U",
                "isl": input_tokens,
                "osl": output_tokens,
                "throughput_total": throughput_total,
                "throughput_per_gpu": throughput_per_gpu,
                "avg_latency_s": avg_latency,
                "avg_ttft_s": summary["time_to_first_token"]["avg_s"],
                "interactivity_tokens_per_sec_per_user": interactivity,
                "total_requests": summary["total_requests"],
                "successful_requests": summary["successful_requests"],
                "failed_requests": summary["failed_requests"],
            }
            results.append(result)

            print(f"\nResults:")
            print(f"  Total Throughput: {throughput_total:.2f} tok/s")
            print(f"  Throughput/GPU: {throughput_per_gpu:.2f} tok/s/gpu")
            print(f"  Avg Latency: {avg_latency:.2f}s")
            print(f"  Avg TTFT: {summary['time_to_first_token']['avg_s']:.2f}s")
            print(f"  Interactivity: {interactivity:.2f} tok/s/user")
            print(f"  Success Rate: {summary['successful_requests']}/{summary['total_requests']}")

    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GPU/User configuration benchmarks")
    parser.add_argument("--url-1gpu", type=str, help="URL for 1-GPU endpoint")
    parser.add_argument("--url-2gpu", type=str, help="URL for 2-GPU endpoint")
    parser.add_argument("--url-4gpu", type=str, help="URL for 4-GPU endpoint")
    parser.add_argument("--requests", type=int, default=50, help="Requests per configuration")
    parser.add_argument("--isl", type=int, default=1024, help="Input Sequence Length in tokens (default: 1024)")
    parser.add_argument("--osl", type=int, default=1024, help="Output Sequence Length in tokens (default: 1024)")

    args = parser.parse_args()

    # Build endpoints dict
    endpoints = {}
    if args.url_1gpu:
        endpoints[1] = args.url_1gpu
    if args.url_2gpu:
        endpoints[2] = args.url_2gpu
    if args.url_4gpu:
        endpoints[4] = args.url_4gpu

    if not endpoints:
        print("Error: At least one endpoint URL required")
        print("Usage: python benchmark_gpu_users.py --url-1gpu URL [--url-2gpu URL] [--url-4gpu URL]")
        return

    # Define test configurations: (n_gpu, n_users)
    configurations = [
        (1, 128),   # 1 GPU / 128 users
        (2, 128),   # 2 GPU / 128 users
        (2, 64),    # 2 GPU / 64 users
        (2, 32),    # 2 GPU / 32 users
        (4, 32),    # 4 GPU / 32 users
        (4, 16),    # 4 GPU / 16 users
    ]

    # Filter to only configurations we have endpoints for
    configurations = [(g, u) for g, u in configurations if g in endpoints]

    print("="*60)
    print("GPU/USER CONFIGURATION BENCHMARK")
    print("="*60)
    print(f"Available endpoints: {list(endpoints.keys())} GPUs")
    print(f"Configurations to test: {configurations}")
    print(f"Requests per config: {args.requests}")
    print(f"ISL (Input Sequence Length): {args.isl} tokens")
    print(f"OSL (Output Sequence Length): {args.osl} tokens")
    print("="*60)

    results = await run_gpu_user_benchmark(
        endpoints=endpoints,
        configurations=configurations,
        requests_per_config=args.requests,
        input_tokens=args.isl,
        output_tokens=args.osl,
    )

    if results:
        # Save results
        with open("benchmark_gpu_users_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n\nResults saved to benchmark_gpu_users_results.json")

        # Print summary table
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"{'Config':<12} {'GPUs':<6} {'Users':<8} {'Throughput':<14} {'Thru/GPU':<14} {'Latency':<12} {'Interactivity':<14}")
        print(f"{'':12} {'':6} {'':8} {'(tok/s)':<14} {'(tok/s/gpu)':<14} {'(s)':<12} {'(tok/s/user)':<14}")
        print("-"*100)
        for r in results:
            print(f"{r['config_label']:<12} {r['n_gpu']:<6} {r['n_users']:<8} {r['throughput_total']:<14.2f} {r['throughput_per_gpu']:<14.2f} {r['avg_latency_s']:<12.2f} {r['interactivity_tokens_per_sec_per_user']:<14.2f}")
        print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
