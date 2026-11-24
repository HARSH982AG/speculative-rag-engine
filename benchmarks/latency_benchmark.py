import asyncio
import time
import statistics
from src.router import SpeculativeRAGRouter

async def benchmark(iterations: int = 100):
    router = SpeculativeRAGRouter()
    print(f"Running Speculative RAG Engine Benchmark ({iterations} iterations)...")
    
    latencies = []
    for i in range(iterations):
        q = f"Telemetry query {i} testing prefix trie reuse and speculative verification latency."
        res = await router.execute_speculative_turn(q, f"sess_{i % 10}")
        latencies.append(res["telemetry"]["total_latency_ms"])

    print("\n-- BENCHMARK RESULTS --")
    print(f"Min Latency:    {min(latencies):.2f} ms")
    print(f"Mean Latency:   {statistics.mean(latencies):.2f} ms")
    print(f"Median (P50):   {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency:    {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
    print(f"P99 Latency:    {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
    print("Cache Stats:   ", router.kv_cache.get_stats())

if __name__ == "__main__":
    asyncio.run(benchmark(200))
