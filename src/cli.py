# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

"""
Llama-Bridge: CLI utilities for system operations and model management.
"""
import os
import subprocess
import sys
import shutil
from pathlib import Path


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
    # Put it in a dedicated scripts/ folder in the root, outside of Python source
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(root_dir, "scripts", "hfd.sh")
    
    if not os.path.exists(script_path):
        print(f"hfd.sh not found. Downloading to {script_path}...")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        url = "https://gist.githubusercontent.com/Gwada26/f67d4b4a15993883a4c4/raw/hfd.sh"
        subprocess.run(["curl", "-s", "-o", script_path, url], check=True)
        subprocess.run(["chmod", "+x", script_path], check=True)
        
    cmd = [script_path, repo_id, "--local-dir", local_dir] + args
    subprocess.run(cmd)


def build():
    """
    Synchronize llama.cpp and build C++ bindings.
    Usage: uv run build [--branch <branch_name>]
    """
    root_dir = Path(__file__).parent.parent.absolute()
    llama_dir = root_dir / "vendor" / "llama.cpp"
    bindings_dir = root_dir / "bindings"
    
    # Get branch if specified
    branch = "master"
    if "--branch" in sys.argv:
        try:
            branch = sys.argv[sys.argv.index("--branch") + 1]
        except IndexError:
            pass

    print(f"--- Synchronizing llama.cpp ({branch}) ---")
    if not (llama_dir / ".git").exists():
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=root_dir, check=True)
    
    # Sync to branch/tag
    print(f"Fetching and checking out branch: {branch}")
    subprocess.run(["git", "fetch", "origin"], cwd=llama_dir, check=True)
    subprocess.run(["git", "checkout", branch], cwd=llama_dir, check=True)
    # Only pull if it's a branch, if it's a tag this might fail or do nothing
    try:
        subprocess.run(["git", "pull", "origin", branch], cwd=llama_dir, check=True)
    except subprocess.CalledProcessError:
        print(f"Note: Could not pull '{branch}', continuing (might be a tag or detached HEAD).")

    print("--- Building llama.cpp ---")
    cmake_cmd = ["cmake", "-B", "build", "-DBUILD_SHARED_LIBS=ON", "-DGGML_METAL=ON"]
    subprocess.run(cmake_cmd, cwd=llama_dir, check=True)
    subprocess.run(["cmake", "--build", "build", "-j"], cwd=llama_dir, check=True)

    print("--- Building pybind11 bindings ---")
    # Get pybind11 cmake dir
    import pybind11
    pybind_dir = pybind11.get_cmake_dir()
    print(f"Using pybind11 from: {pybind_dir}")
    
    bindings_cmake = [
        "cmake", "-B", "build",
        f"-DLLAMA_DIR=../vendor/llama.cpp",
        "-DPYBIND11_FINDPYTHON=ON",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind_dir}"
    ]
    
    print(f"Running config: {' '.join(bindings_cmake)}")
    sys.stdout.flush()
    subprocess.run(bindings_cmake, cwd=bindings_dir, check=True)
    
    print("Compiling bindings (this may take a minute)...")
    sys.stdout.flush()
    subprocess.run(["cmake", "--build", "build", "-j"], cwd=bindings_dir, check=True)

    print("--- Copying libraries ---")
    lib_dest = root_dir / "src" / "lib"
    lib_dest.mkdir(parents=True, exist_ok=True)
    
    # Copy libllama/libggml
    llama_build_bin = llama_dir / "build" / "bin"
    for pattern in ["libllama*.dylib", "libllama*.so", "libggml*.dylib", "libggml*.so"]:
        for f in llama_build_bin.glob(pattern):
            print(f"Copying {f.name} to src/lib/")
            shutil.copy(f, lib_dest)

    # Copy llama_chat binding
    bindings_build = bindings_dir / "build"
    for pattern in ["llama_chat*.so", "llama_chat*.dylib"]:
        for f in bindings_build.glob(pattern):
            print(f"Copying {f.name} to src/")
            shutil.copy(f, root_dir / "src")

    print("\n✅ Build complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "hfd":
            sys.argv.pop(1)
            download_model()
        elif cmd == "build":
            build()
        else:
            run_claude_code()
    else:
        run_claude_code()
