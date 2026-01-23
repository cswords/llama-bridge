
# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

import subprocess
import time
import requests
import json
import os
from pathlib import Path

CONFIGS = [
    "configs/qwen2.5-3b.toml",
    "configs/qwen30b-coder.toml",
    "configs/mimo-v2.toml",
    "configs/minimax-m2.1.toml",
    "configs/glm4-reap.toml",
    "configs/glm4-flash.toml",
    "configs/gpt-oss-120b.toml"
]

def wait_for_server(url, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{url}/health", timeout=1)
            if res.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def run_benchmark():
    results = []
    
    for config_path in CONFIGS:
        if not Path(config_path).exists():
            print(f"Skipping {config_path} (not found)")
            continue
            
        print(f"\n>>> Benchmarking: {config_path}")
        
        # Start server
        start_time = time.time()
        process = subprocess.Popen(
            ["uv", "run", "serve", "--config", config_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        url = "http://localhost:8000"
        ready = wait_for_server(url)
        startup_duration = time.time() - start_time
        
        if not ready:
            print(f"FAILED to start server for {config_path}")
            process.terminate()
            continue
            
        print(f"Server ready in {startup_duration:.2f}s")
        
        prompt = "Hi, who are you?"
        import httpx
        url_cmp = f"{url}/v1/chat/completions"
        payload = {
            "model": "benchmark",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        raw_latency = 0.0
        
        # 0. Sanity Check (Raw API)
        print("Performing Raw API Sanity Check...")
        try:
            with httpx.Client(timeout=60.0) as client:
                t_s = time.time()
                res_s = client.post(url_cmp, json=payload)
                t_e = time.time()
                raw_latency = t_e - t_s
                print(f"Raw API Response in {raw_latency:.2f}s")
        except Exception as e:
            print(f"Sanity check failed: {e}")

        # 0.1 Sanity Check (Streaming)
        print("Performing Streaming Raw API Sanity Check...")
        try:
            payload_stream = payload.copy()
            payload_stream["stream"] = True
            t_s = time.time()
            with httpx.stream("POST", url_cmp, json=payload_stream, timeout=60.0) as r:
                for line in r.iter_lines():
                    pass # just consume
            t_e = time.time()
            print(f"Raw Streaming API completed in {t_e - t_s:.2f}s")
        except Exception as e:
            print(f"Streaming sanity check failed: {e}")

        # 1. Cold Request via Claude Code
        print("Sending cold request via Claude Code...")
        t_cold_start = time.time()
        result_cold = subprocess.run(
            ["uv", "run", "cc", "-p", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        t_cold_end = time.time()
        cold_latency = t_cold_end - t_cold_start
        
        if result_cold.returncode != 0:
            print(f"WARNING: Cold request failed: {result_cold.stderr}")
        
        # 2. Hot Request via Claude Code
        print("Sending hot request via Claude Code...")
        t_hot_start = time.time()
        result_hot = subprocess.run(
            ["uv", "run", "cc", "-p", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        t_hot_end = time.time()
        hot_latency = t_hot_end - t_hot_start
        
        if result_hot.returncode != 0:
            print(f"WARNING: Hot request failed: {result_hot.stderr}")
        
        results.append({
            "config": config_path,
            "startup": startup_duration,
            "raw": raw_latency,
            "cold": cold_latency,
            "hot": hot_latency
        })
        
        print(f"Cold: {cold_latency:.2f}s | Hot: {hot_latency:.2f}s | Raw: {raw_latency:.2f}s")
        
        # Stop server
        process.terminate()
        process.wait()
        
    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Config':<30} | {'Startup':<8} | {'Raw API':<8} | {'CC Cold':<8} | {'CC Hot':<8}")
    print("-" * 80)
    for res in results:
        print(f"{res['config']:<30} | {res['startup']:>7.2f}s | {res['raw']:>7.2f}s | {res['cold']:>7.2f}s | {res['hot']:>7.2f}s")
    print("="*80)

if __name__ == "__main__":
    # Ensure we don't have a port conflict
    os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null")
    run_benchmark()
