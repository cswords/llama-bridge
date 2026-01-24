
import asyncio
import time
import json
import httpx
from statistics import mean, median

# Target URL
URL = "http://127.0.0.1:8000/v1/messages"

async def single_request(client, worker_id):
    payload = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": f"Request from worker {worker_id}"}],
        "max_tokens": 50,
        "stream": True
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    
    try:
        async with client.stream("POST", URL, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                return None
            
            async for line in response.aiter_lines():
                if first_token_time is None and "content_block_delta" in line:
                    first_token_time = time.perf_counter()
            
            end_time = time.perf_counter()
            
            return {
                "ttft": (first_token_time - start_time) * 1000 if first_token_time else None,
                "total": (end_time - start_time) * 1000
            }
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return None

async def run_benchmark(concurrency=5, total_requests=20):
    print(f"--- Starting Benchmark (Concurrency={concurrency}, Total={total_requests}) ---")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        # Simple semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def sem_worker(wid):
            async with sem:
                return await single_request(client, wid)

        results = await asyncio.gather(*(sem_worker(i) for i in range(total_requests)))
        
        valid_results = [r for r in results if r is not None]
        ttfts = [r["ttft"] for r in valid_results if r["ttft"] is not None]
        totals = [r["total"] for r in valid_results]
        
        print("\n--- Results ---")
        print(f"Success Rate: {len(valid_results)}/{total_requests}")
        if ttfts:
            print(f"TTFT (ms): Avg={mean(ttfts):.2f}, Median={median(ttfts):.2f}, Min={min(ttfts):.2f}, Max={max(ttfts):.2f}")
        if totals:
            print(f"Total Latency (ms): Avg={mean(totals):.2f}, Median={median(totals):.2f}")

if __name__ == "__main__":
    # This assumes the server is already running with --mock
    asyncio.run(run_benchmark())
