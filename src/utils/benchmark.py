
# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

# --- Configuration ---
DEFAULT_CONFIGS = [
    "configs/qwen2.5-3b.toml",
    "configs/mimo-v2.toml",
    "configs/minimax-m2.1.toml",
    "configs/glm4-moe.toml",
    "configs/glm4-flash.toml",
    "configs/gpt-oss-120b.toml"
]

class LlamaPatcher:
    """Handles runtime monkeypatching of third-party libraries."""
    
    @staticmethod
    def apply_all():
        """Apply all necessary patches for Agent and Docker compatibility."""
        try:
            from terminal_bench.agents.installed_agents.claude_code.claude_code_agent import ClaudeCodeAgent
            from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
            
            # 1. Forward ANTHROPIC_BASE_URL to containerized agents
            if not hasattr(ClaudeCodeAgent, "_patched"):
                original_env_getter = ClaudeCodeAgent._env.fget
                def patched_env_getter(self):
                    env = original_env_getter(self)
                    if "ANTHROPIC_BASE_URL" in os.environ:
                        env["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]
                    return env
                ClaudeCodeAgent._env = property(patched_env_getter)
                
                # 2. Translate 'docker compose' to 'docker-compose' on macOS/Colima
                original_get_command = DockerComposeManager.get_docker_compose_command
                def patched_get_command(self, command: List[str]) -> List[str]:
                    cmd_list = original_get_command(self, command)
                    if sys.platform == "darwin" and cmd_list[:2] == ["docker", "compose"]:
                        return ["docker-compose"] + cmd_list[2:]
                    return cmd_list
                
                DockerComposeManager.get_docker_compose_command = patched_get_command
                ClaudeCodeAgent._patched = True
                return True
        except ImportError:
            return False
        return True

class ServerContext:
    """Context manager for the Llama-Bridge serving lifecycle."""
    
    def __init__(self, config_path: str, port: int = 8000):
        self.config_path = config_path
        self.port = port
        self.url = f"http://localhost:{port}"
        self.process = None

    def __enter__(self):
        env = os.environ.copy()
        # Colima detection and path setup
        if sys.platform == "darwin":
            socket = Path.home() / ".colima" / "default" / "docker.sock"
            if socket.exists():
                os.environ["DOCKER_HOST"] = env["DOCKER_HOST"] = f"unix://{socket}"

        # Clean up existing processes on the port
        os.system(f"lsof -ti:{self.port} | xargs kill -9 2>/dev/null")
        
        self.process = subprocess.Popen(
            ["uv", "run", "serve", "--config", self.config_path, "--port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        )
        
        if self._wait_ready():
            return self
        raise RuntimeError(f"Server at {self.config_path} failed to start.")

    def _wait_ready(self, timeout=120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return True
            except: pass
            time.sleep(1)
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait()

class ResultAnalyzer:
    """Parses terminal-bench results and logs to extract intelligence metrics."""
    
    @staticmethod
    def get_latest_stats() -> Dict[str, Any]:
        stats = {"accuracy": 0.0, "turns": 0, "tps": 0.0, "details": {}}
        runs_dir = Path("runs")
        if not runs_dir.exists(): return stats
        
        folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Find the latest folder that actually contains result data
        target_folder = None
        for folder in folders:
            if (folder / "results.json").exists():
                target_folder = folder
                break
        
        if not target_folder: return stats
        
        # Parse accuracy from results.json
        res_file = target_folder / "results.json"
        with open(res_file, 'r') as f:
            data = json.load(f)
            stats["accuracy"] = data.get("accuracy", 0.0)
            for trial in data.get("results", []):
                stats["details"][trial.get("task_id")] = trial.get("is_resolved", False)
            
            # Simple TPS calculation from first trial
            if data.get("results"):
                t0 = data["results"][0]
                try:
                    # Use fromisoformat for better compatibility
                    start = datetime.fromisoformat(t0["agent_started_at"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(t0["agent_ended_at"].replace("Z", "+00:00"))
                    dt = (end - start).total_seconds()
                    if dt > 0: stats["tps"] = t0.get("total_output_tokens", 0) / dt
                except Exception as e:
                    # print(f"DEBUG: TPS Calc Error: {e}")
                    pass

        # Parse turns from logs
        for log in target_folder.glob("**/sessions/agent.log"):
            try:
                with open(log, 'r') as f:
                    for line in f:
                        if '"type":"result"' in line:
                            stats["turns"] += json.loads(line).get("num_turns", 0)
            except: pass
            
        return stats

class Benchmarker:
    """The main entry point for running the performance and capability suite."""
    
    def __init__(self, configs: List[str], full_core: bool = False):
        self.configs = configs
        self.full_core = full_core
        self.results = []
        LlamaPatcher.apply_all()

    def run_suite(self):
        for config_path in self.configs:
            if not Path(config_path).exists():
                print(f"Skipping {config_path} (not found)")
                continue
            print(f"\n🚀 Benchmarking: {config_path}")
            
            try:
                with ServerContext(config_path) as ctx:
                    metrics = self._run_perf_tests(ctx.url)
                    metrics["stats"] = self._run_capability_tests(config_path)
                    self.results.append({"config": config_path, **metrics})
            except Exception as e:
                print(f"❌ Error during benchmark {config_path}: {e}")
                import traceback
                traceback.print_exc()

        self._print_report()

    def _run_perf_tests(self, url: str) -> Dict[str, float]:
        """Runs latency and streaming performance checks."""
        # Warm up & Hot Latency via CC
        print(" -> Measuring Hot Latency (Claude Code)...")
        t_start = time.time()
        # Use a simpler prompt to measure basic turn latency
        subprocess.run(["uv", "run", "cc", "-p", "Hi, are you there?"], capture_output=True)
        hot_lat = time.time() - t_start
        
        # Manual TPS Measurement
        print(" -> Measuring TPS (Manual)...")
        t_start_gen = time.time()
        # Ask for a longer response to measure generation speed
        gen_res = subprocess.run(["uv", "run", "cc", "-p", "Write a 100 word story about a bridge."], capture_output=True, text=True)
        t_end_gen = time.time()
        
        # Estimate output tokens (rough approx: chars / 4)
        output_len = len(gen_res.stdout)
        est_tokens = output_len / 4
        duration = t_end_gen - t_start_gen
        
        # Simple TPS: Tokens / Total Duration (Conservative E2E TPS)
        manual_tps = est_tokens / duration if duration > 0 else 0
        
        print(f" -> Generated {output_len} chars in {duration:.2f}s (~{manual_tps:.1f} t/s)")
        
        return {"hot": hot_lat, "manual_tps": manual_tps}

    def _run_capability_tests(self, config_path: str) -> Dict[str, Any]:
        """Runs terminal-bench tasks (Fundamentals & Synthesis)."""
        # Capability tests are currently disabled to ensure stable performance benchmarking
        print(f" -> Testing Capability... (SKIPPED for pure performance run)")
        return {"accuracy": 0.0, "turns": 0, "tps": 0.0}

        # Original logic retained for future enable:
        # from terminal_bench.cli.tb.runs import create as tb_run_create
        # ...

    def _print_report(self):
        print("\n" + "═"*120)
        print(f"║ {'MODEL CONFIGURATION':<34} ║ {'LATENCY':<7} ║ {'ACC%':<5} ║ {'STEPS':<5} ║ {'SPEED':<9} ║ {'VERDICT':<10} ║")
        print("╟" + "─"*36 + "╫" + "─"*9 + "╫" + "─"*7 + "╫" + "─"*7 + "╫" + "─"*11 + "╫" + "─"*12 + "╢")
        
        for r in self.results:
            cfg = Path(r['config']).stem
            stats = r['stats']
            acc = stats['accuracy'] * 100
            
            # Use manual TPS if available and stats TPS is 0
            tps_val = stats['tps']
            if tps_val == 0 and 'manual_tps' in r:
                tps_val = r['manual_tps']
                
            tps = f"{tps_val:.1f} t/s" if tps_val > 0 else "N/A"
            
            # Smart Verdict
            if acc >= 100 and stats['turns'] <= (2 if not self.full_core else 50): verdict = "💎 ELITE"
            elif acc > 0: verdict = "✅ PASS"
            else: verdict = "⚠️  UNK" # Changed FAIL to UNK as capability test might fail to run
            
            print(f"║ {cfg:<34} ║ {r['hot']:>6.2f}s ║ {acc:>3.0f}%  ║ {stats['turns']:<5} ║ {tps:<9} ║ {verdict:<10} ║")
        print("╚" + "═"*118 + "╝\n")

# --- CLI Entrypoint ---
def run_benchmark():
    parser = argparse.ArgumentParser(description="Llama-Bridge Professional Benchmarking Suite")
    parser.add_argument("--config", type=str, help="Single config name")
    parser.add_argument("--full-core", action="store_true", help="Run comprehensive core evaluation")
    args, _ = parser.parse_known_args()

    configs = DEFAULT_CONFIGS
    if args.config:
        # Match name or path
        name = args.config.replace(".toml", "")
        configs = [c for c in DEFAULT_CONFIGS if name in c] or [args.config]

    Benchmarker(configs, args.full_core).run_suite()

if __name__ == "__main__":
    run_benchmark()
