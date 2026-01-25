
# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.

import os
import sys
import subprocess
import json
import argparse
import tomllib
from pathlib import Path

def find_llama_bench():
    """Find llama-bench executable."""
    paths = [
        "/opt/homebrew/bin/llama-bench",
        "./vendor/llama.cpp/build/bin/llama-bench",
        "llama-bench"
    ]
    for p in paths:
        if subprocess.run(["which", p], capture_output=True).returncode == 0:
            return p
    return None

def run_bench(args):
    """Run llama-bench with specified arguments and return parsed JSON."""
    cmd = [args.bench_path, "-o", "json"] + args.extra
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running llama-bench: {result.stderr}")
        return None
    try:
        return json.loads(result.stdout)
    except:
        print("Failed to parse llama-bench output as JSON")
        return None

def main():
    parser = argparse.ArgumentParser(description="Llama-Bridge Parameter Optimizer")
    parser.add_argument("--config", required=True, help="Path to config TOML")
    parser.add_argument("--apply", action="store_true", help="Apply optimized settings to the config file")
    parser.add_argument("--bench-path", help="Path to llama-bench executable")
    args = parser.parse_args()

    if not args.bench_path:
        args.bench_path = find_llama_bench()
        if not args.bench_path:
            print("Error: llama-bench not found. Please install llama.cpp or specify --bench-path")
            sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Get model path from config
    # We look for the first model path
    model_path = None
    if "models" in config:
        for m in config["models"].values():
            model_path = m.get("path")
            if model_path:
                break
    
    if not model_path:
        print("Error: No model path found in config")
        sys.exit(1)

    # Resolve absolute path for model
    if not os.path.isabs(model_path):
        ws_path = Path("/Volumes/970+/Llama-Bridge")
        candidates = [
            Path(model_path),
            ws_path / model_path,
            ws_path / "models" / model_path,
        ]
        for cand in candidates:
            if cand.exists():
                model_path = str(cand.absolute())
                break
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
        
    if os.path.isdir(model_path):
        # Look for the first GGUF file in the directory, avoiding hidden files (._ files)
        gguf_files = sorted([f for f in Path(model_path).glob("*.gguf") if not f.name.startswith(".")])
        if gguf_files:
            model_path = str(gguf_files[0].absolute())
        else:
            print(f"Error: No GGUF files found in directory: {model_path}")
            sys.exit(1)

    print(f"Optimizing for model: {model_path}")

    # Phase 1: Basic compute optimization (Threads & Flash Attention)
    print("\n>>> Phase 1: Optimizing Threads & Flash Attention...")
    threads = "4,8,12,16,24,32"
    fa = "0,1"
    res_p1 = run_bench(argparse.Namespace(bench_path=args.bench_path, extra=[
        "-m", model_path, "-p", "512", "-n", "0", 
        "-t", threads, "-fa", fa, "-b", "512", "-ub", "512"
    ]))
    
    if not res_p1: sys.exit(1)
    
    def get_tps(entry):
        # Prefer avg_ts as seen in local llama-bench
        return entry.get("avg_ts") or entry.get("tps_pp") or 0

    best_p1 = max(res_p1, key=get_tps)
    best_threads = best_p1.get("n_threads")
    best_fa = bool(best_p1.get("flash_attn"))
    print(f"Best: Threads={best_threads}, FlashAttention={best_fa} ({get_tps(best_p1):.2f} t/s PP)")

    # Phase 2: KV Cache Precision (Memory vs Speed)
    print("\n>>> Phase 2: Optimizing KV Cache Precision...")
    cache_types = "f16,q8_0,q4_0"
    res_p2 = run_bench(argparse.Namespace(bench_path=args.bench_path, extra=[
        "-m", model_path, "-p", "512", "-n", "0",
        "-t", str(best_threads), "-fa", "1" if best_fa else "0",
        "-ctk", cache_types, "-ctv", cache_types,
        "-b", "512", "-ub", "512"
    ]))
    
    if not res_p2: sys.exit(1)
    best_p2 = max(res_p2, key=get_tps)
    best_ctk = best_p2.get("type_k")
    best_ctv = best_p2.get("type_v")
    print(f"Best KV Cache: K={best_ctk}, V={best_ctv} ({get_tps(best_p2):.2f} t/s PP)")

    # Phase 3: Batch & UBatch Optimization
    print("\n>>> Phase 3: Optimizing Batch/UBatch...")
    batches = "512,1024,2048,4096"
    ubatches = "128,256,512"
    res_p3 = run_bench(argparse.Namespace(bench_path=args.bench_path, extra=[
        "-m", model_path, "-p", "512", "-n", "0",
        "-t", str(best_threads), "-fa", "1" if best_fa else "0",
        "-ctk", best_ctk, "-ctv", best_ctv,
        "-b", batches, "-ub", ubatches
    ]))
    
    if not res_p3: sys.exit(1)
    # Filter valid pairs
    valid_p3 = [r for r in res_p3 if r.get("n_ubatch") <= r.get("n_batch")]
    best_p3 = max(valid_p3, key=get_tps)
    best_batch = best_p3.get("n_batch")
    best_ubatch = best_p3.get("n_ubatch")
    print(f"Best Batch: {best_batch}, UBatch: {best_ubatch} ({get_tps(best_p3):.2f} t/s PP)")

    # Summary
    print("\n" + "="*40)
    print("RECOMMENDED OPTIMIZED SETTINGS")
    print("="*40)
    print(f"n_threads     = {best_threads}")
    print(f"flash_attn    = {str(best_fa).lower()}")
    print(f"cache_type_k  = '{best_ctk}'")
    print(f"cache_type_v  = '{best_ctv}'")
    print(f"n_batch       = {best_batch}")
    print(f"n_ubatch      = {best_ubatch}")
    print("="*40)

    if args.apply:
        apply_settings(config_path, {
            "n_threads": best_threads,
            "flash_attn": str(best_fa).lower(),
            "cache_type_k": f"'{best_ctk}'",
            "cache_type_v": f"'{best_ctv}'",
            "n_batch": best_batch,
            "n_ubatch": best_ubatch
        })
    else:
        print("\nTo apply these settings, run with --apply")

def apply_settings(config_path, settings):
    import re
    with open(config_path, "r") as f:
        content = f.read()
    
    for key, val in settings.items():
        # Look for the key at the start of a line, potentially indented
        pattern = fr"^\s*{key}\s*=\s*.*$"
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, f"{key} = {val}", content, flags=re.MULTILINE)
        else:
            # Smart insertion: find [models.NAME] or [caches.NAME]
            # Since these are model/inference params, putting them in the first model block is a safe bet
            model_match = re.search(r"\[models\..*\]", content)
            if model_match:
                pos = content.find("\n", model_match.end())
                if pos == -1: pos = len(content)
                content = content[:pos] + f"\n{key} = {val}" + content[pos:]
            else:
                content += f"\n{key} = {val}"

    with open(config_path, "w") as f:
        f.write(content)
    print(f"\nSuccessfully updated {config_path}")

if __name__ == "__main__":
    main()
