import argparse
import subprocess
import os
import sys
import time
import requests
from pathlib import Path

def run_evaluation(config_file, dataset_name, task_id=None):
    """Run evaluation for a single config file."""
    config_name = Path(config_file).stem
    print(f"\n" + "="*60)
    print(f">>> Evaluating model with config: {config_file}")
    print("="*60)
    
    # 1. Start the server in the background
    env = os.environ.copy()
    
    # Support Colima on macOS
    if sys.platform == "darwin":
        colima_socket = Path.home() / ".colima" / "default" / "docker.sock"
        if colima_socket.exists():
            os.environ["DOCKER_HOST"] = f"unix://{colima_socket}"
            env["DOCKER_HOST"] = f"unix://{colima_socket}"
            print(f"Set DOCKER_HOST={os.environ['DOCKER_HOST']} (Colima detected)")

    server_cmd = [
        "uv", "run", "serve",
        "--config", config_file,
        "--port", "8000",
        "--host", "0.0.0.0"
    ]
    
    print(f"Starting server: {' '.join(server_cmd)}")
    server_log = open(f"bridge_server_{config_name}.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    # Wait for server to start
    retries = 120
    started = False
    for i in range(retries):
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if resp.status_code == 200:
                started = True
                break
        except:
            time.sleep(1)
    
    if not started:
        print("Server failed to start.")
        server_proc.terminate()
        return

    # 2. Run terminal-bench via Python API with Patching
    print("Initializing terminal-bench API with compatibility patches...")
    
    # Global environment setup for the containerized agent
    os.environ["ANTHROPIC_BASE_URL"] = "http://host.docker.internal:8000"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-sid01-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-abcdefghijklmnopqrstuvwxyz0123456789"
    
    try:
        from terminal_bench.agents.installed_agents.claude_code.claude_code_agent import ClaudeCodeAgent
        from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
        from terminal_bench.cli.tb.runs import create as tb_run_create
        from terminal_bench.agents import AgentName
        
        # Patch 1: Forward ANTHROPIC_BASE_URL to ClaudeCodeAgent
        original_env_getter = ClaudeCodeAgent._env.fget
        def patched_env_getter(self):
            env = original_env_getter(self)
            if "ANTHROPIC_BASE_URL" in os.environ:
                env["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]
            return env
        ClaudeCodeAgent._env = property(patched_env_getter)

        # Patch 2: Handle docker compose vs docker-compose (Command translation)
        # We replace the command builder to use docker-compose on systems that need it
        original_get_command = DockerComposeManager.get_docker_compose_command
        def patched_get_command(self, command: list[str]) -> list[str]:
            cmd_list = original_get_command(self, command)
            # Check if we should translate ["docker", "compose", ...] -> ["docker-compose", ...]
            # This is a common requirement for Colima/older Docker setups
            if sys.platform == "darwin":
                # Check if 'docker-compose' exists and is executable
                try:
                    subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
                    # Translation logic
                    if len(cmd_list) >= 2 and cmd_list[0] == "docker" and cmd_list[1] == "compose":
                        new_cmd = ["docker-compose"] + cmd_list[2:]
                        # print(f"DEBUG: Translating command to: {' '.join(new_cmd)}")
                        return new_cmd
                except:
                    pass
            return cmd_list
        
        DockerComposeManager.get_docker_compose_command = patched_get_command
        print("Success: Applied memory patches for Agent Environment and Docker Compatibility.")

        # Run the evaluation
        ds_str = dataset_name
        if ds_str == "terminal-bench-core":
            ds_str = "terminal-bench-core==0.1.1"

        tb_run_create(
            agent=AgentName("claude-code"),
            model_name=f"bridge/{config_name}",
            dataset=ds_str,
            n_concurrent_trials=1,
            no_rebuild=True,
            task_ids=[task_id] if task_id else None
        )
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 3. Cleanup
        print("Stopping server...")
        server_proc.terminate()
        server_proc.wait()
        server_log.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama-Bridge models using terminal-bench")
    parser.add_argument("--config", type=str, help="Specific config file to evaluate (optional)")
    parser.add_argument("--dataset", type=str, default="terminal-bench-core", help="Dataset name")
    parser.add_argument("--task-id", type=str, help="Specific task ID to run")
    args = parser.parse_args()
    
    if args.config:
        run_evaluation(args.config, args.dataset, args.task_id)
    else:
        config_dir = Path("configs")
        if not config_dir.exists(): return
        configs = sorted([cfg for cfg in config_dir.glob("*.toml") if not cfg.name.startswith(".")])
        if not configs: return
        print(f"Found {len(configs)} configuration files. Starting batch evaluation...")
        for cfg in configs:
            run_evaluation(str(cfg), args.dataset, args.task_id)

if __name__ == "__main__":
    main()
