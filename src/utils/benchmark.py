
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
    "configs/glm4-reap.toml"
]

def wait_for_server(url, timeout=120):
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
        
        # Test using Claude Code CLI (more realistic)
        prompt = "Hi, who are you?"
        
        # Cold Request
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
        
        # Hot Request
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
            "cold": cold_latency,
            "hot": hot_latency
        })
        
        print(f"Cold: {cold_latency:.2f}s | Hot: {hot_latency:.2f}s")
        
        # Stop server
        process.terminate()
        process.wait()
        
    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Config':<30} | {'Startup':<8} | {'Cold':<8} | {'Hot':<8}")
    print("-" * 60)
    for res in results:
        print(f"{res['config']:<30} | {res['startup']:>7.2f}s | {res['cold']:>7.2f}s | {res['hot']:>7.2f}s")
    print("="*60)

if __name__ == "__main__":
    # Ensure we don't have a port conflict
    os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null")
    run_benchmark()
