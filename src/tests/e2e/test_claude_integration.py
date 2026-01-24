
import subprocess
import time
import os
import signal
import pytest

def test_claude_code_e2e_mock():
    # 1. Start the server in mock mode (port 8000)
    # We use subprocess.Popen to run it in background
    server_process = subprocess.Popen(
        ["uv", "run", "serve", "--mock", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to be ready
    time.sleep(3)
    
    try:
        # 2. Run Claude Code in non-interactive mode
        # We use our 'uv run cc' shortcut which sets the correct env vars
        # --dangerously-skip-permissions is useful to avoid interactive prompts
        cmd = ["uv", "run", "cc", "-p", "Say 'Llama-Bridge is working' precisely.", "--dangerously-skip-permissions"]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        print("--- Claude Code Output ---")
        print(result.stdout)
        print("--- Claude Code Error  ---")
        print(result.stderr)
        
        # 3. Verification
        # In mock mode, our bridge returns a static or simple mock response.
        # If the bridge is working, Claude Code will print the response it received.
        assert result.returncode == 0
        # Check if the output contains our mock content or at least didn't fail with connection error
        assert "Llama-Bridge is working" in result.stdout or "Mock" in result.stdout
        
    finally:
        # 4. Cleanup: kill the server
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait()

if __name__ == "__main__":
    # For manual execution
    test_claude_code_e2e_mock()
