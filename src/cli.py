"""
Llama-Bridge: CLI utilities for system operations and model management.
"""
import os
import subprocess
import sys


def run_claude_code():
    """
    Run Claude Code CLI with Llama-Bridge environment variables.
    """
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8000"
    env["ANTHROPIC_API_KEY"] = "sk-ant-none"
    
    # Get any additional arguments
    args = sys.argv[1:]
    
    # Run claude with the environment
    cmd = ["claude"] + args
    subprocess.run(cmd, env=env)


def download_model():
    """
    Download a model using hfd.sh to the models/ directory.
    
    Usage: uv run hfd <repo_id> [--include "*Q5_K_M.gguf"]
    Example: uv run hfd bartowski/Llama-3.1-8B-Instruct-GGUF --include "*Q5_K_M.gguf"
    """
    if len(sys.argv) < 2:
        print("Usage: uv run hfd <repo_id> [--include \"pattern\"]")
        print("Example: uv run hfd bartowski/Llama-3.1-8B-Instruct-GGUF --include \"*Q5_K_M.gguf\"")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    args = sys.argv[2:]
    local_dir = f"models/{repo_id}"
    
    # Ensure local_dir exists
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    
    # Run hfd.sh (download if missing)
    script_path = os.path.join("scripts", "hfd.sh")
    if not os.path.exists(script_path):
        print(f"hfd.sh not found. Downloading from web...")
        os.makedirs("scripts", exist_ok=True)
        url = "https://gist.githubusercontent.com/Gwada26/f67d4b4a15993883a4c4/raw/hfd.sh" # Valid HFD script source
        subprocess.run(["curl", "-s", "-o", script_path, url], check=True)
        subprocess.run(["chmod", "+x", script_path], check=True)
        
    cmd = [f"./{script_path}", repo_id, "--local-dir", local_dir] + args
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hfd":
        sys.argv.pop(1)
        download_model()
    else:
        run_claude_code()
