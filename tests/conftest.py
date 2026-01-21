"""
Global pytest configuration and test utilities.
Provides a singleton BridgeController with lazy initialization.

Usage:
    def test_something(test_bridge_controller):
        # Direct HTTP calls
        url = test_bridge_controller.get_endpoint(existing_service_preferred=True)
        response = requests.get(f"{url}/health")
        
        # Claude Code CLI calls
        result = test_bridge_controller.run_claude_code("Say hello", existing_service_preferred=False)
"""
import os
import time
import subprocess
import requests
import pytest
import pexpect
from pathlib import Path


# ============================================================================
# Configuration Constants
# ============================================================================

MOCK_SERVICE_PORT = 8001
REAL_SERVICE_PORT = 8000
REAL_SERVICE_HOST = "127.0.0.1"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LOGS_DIR = Path("logs")


# ============================================================================
# BridgeController - Singleton with Lazy Initialization
# ============================================================================

class BridgeController:
    """
    Singleton BridgeController with lazy initialization.
    
    Manages mock/real service lifecycle and provides methods to run Claude Code CLI.
    
    Behavior:
    - Only ONE instance exists across the entire test session
    - Lazy: mock service only starts when needed
    - Use get_endpoint() or run_claude_code()/run_claude_interactive() with
      existing_service_preferred parameter to control which service to use
    
    Configuration:
    - Model ID can be set via environment variable LLAMA_BRIDGE_TEST_MODEL
    - Default: Qwen/Qwen2.5-3B-Instruct-GGUF
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset_state()
        return cls._instance
    
    def _reset_state(self):
        """Reset internal state."""
        self._process = None
        self._mock_started = False
        self._original_base_url = os.environ.get("ANTHROPIC_BASE_URL")
        self._project_root = Path(__file__).parent.parent
    
    @property
    def model_id(self) -> str:
        """
        Get the model ID for mock service.
        
        Can be overridden via environment variable LLAMA_BRIDGE_TEST_MODEL.
        """
        return os.environ.get("LLAMA_BRIDGE_TEST_MODEL", DEFAULT_MODEL_ID)
    
    @property
    def using_real_service(self) -> bool:
        """Returns True if using real service (not mock)."""
        return self._is_real_service_available()
    
    def _is_real_service_available(self) -> bool:
        """Check if real service is available at port 8000."""
        try:
            res = requests.get(f"http://{REAL_SERVICE_HOST}:{REAL_SERVICE_PORT}/health", timeout=10)
            return res.status_code == 200
        except Exception:
            return False
    
    def _ensure_mock_started(self):
        """Start mock service if not already started."""
        if self._mock_started:
            return
        
        model = self.model_id
        print(f"→ Starting mock service at port {MOCK_SERVICE_PORT} with model {model}...")
        
        cmd = [
            "uv", "run", "serve",
            "--port", str(MOCK_SERVICE_PORT),
            "--model", model,
            "--debug", "--mock"
        ]

        self._process = subprocess.Popen(
            cmd,
            cwd=self._project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self._wait_for_ready(f"http://localhost:{MOCK_SERVICE_PORT}")
        self._mock_started = True
        print(f"✓ Mock service ready at port {MOCK_SERVICE_PORT}")

    def _wait_for_ready(self, url: str, timeout=30):
        """Wait for service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                res = requests.get(f"{url}/health", timeout=1)
                if res.status_code == 200:
                    return
            except Exception:
                time.sleep(1)
        self.stop()
        raise RuntimeError(f"Llama-Bridge startup timeout ({url})")

    def get_endpoint(self, existing_service_preferred: bool = True) -> str:
        """
        Get the service endpoint URL.
        
        Args:
            existing_service_preferred: If True, try to use real service at port 8000 first.
                                       If False, always use mock service.
        
        Returns:
            The service URL (e.g., "http://localhost:8000" or "http://localhost:8001")
        """
        if existing_service_preferred and self._is_real_service_available():
            url = f"http://{REAL_SERVICE_HOST}:{REAL_SERVICE_PORT}"
            print(f"✓ Using existing service at {url}")
            return url
        
        # Need mock service
        self._ensure_mock_started()
        return f"http://127.0.0.1:{MOCK_SERVICE_PORT}"

    def run_claude_code(self, prompt: str, existing_service_preferred: bool = True, extra_args: list = None):
        """
        Run Claude Code CLI in non-interactive mode.
        
        Args:
            prompt: The prompt to send to Claude
            existing_service_preferred: If True, prefer real service at port 8000.
                                       If False, always use mock service.
            extra_args: Additional CLI arguments
        
        Returns:
            subprocess.CompletedProcess with stdout, stderr, returncode
        """
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = self.get_endpoint(existing_service_preferred)
        env["ANTHROPIC_API_KEY"] = "sk-ant-none"
        
        # Use venv claude
        claude_path = self._project_root / ".venv" / "bin" / "claude"
        cmd = [str(claude_path), "--print", prompt]
        if extra_args:
            cmd.extend(extra_args)
            
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            timeout=300
        )

    def run_claude_interactive(self, prompts: list, existing_service_preferred: bool = True):
        """
        Run Claude Code in interactive mode using pexpect.
        
        Args:
            prompts: List of prompts to send
            existing_service_preferred: If True, prefer real service at port 8000.
                                       If False, always use mock service.
        
        Returns:
            Captured output log string
        
        NOTE: Claude Code's Rich-based TUI is difficult to automate
        reliably via PTY. Use for best-effort testing only.
        """
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = self.get_endpoint(existing_service_preferred)
        env["ANTHROPIC_API_KEY"] = "sk-ant-none"
        
        claude_path = self._project_root / ".venv" / "bin" / "claude"
        child = pexpect.spawn(str(claude_path), env=env, encoding='utf-8', timeout=300,
                              dimensions=(50, 200))
        output_log = ""
        
        try:
            time.sleep(5)
            
            for p in prompts:
                child.sendline(p)
                time.sleep(15)
                try:
                    output_log += child.read_nonblocking(size=100000, timeout=1)
                except:
                    pass
                
            child.sendline("/exit")
            time.sleep(2)
            try:
                output_log += child.read()
            except:
                pass
            child.close()
            
        except Exception as e:
            output_log += f"\n[INTERACTIVE ERROR: {e}]"
            
        return output_log
    
    def stop(self):
        """Stop mock service and restore environment."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
        
        # Restore original env
        if self._original_base_url:
            os.environ["ANTHROPIC_BASE_URL"] = self._original_base_url
        elif "ANTHROPIC_BASE_URL" in os.environ:
            del os.environ["ANTHROPIC_BASE_URL"]
        
        self._mock_started = False


# Global singleton instance
_bridge_controller = BridgeController()


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_bridge_controller():
    """
    Session-scoped singleton BridgeController.
    All tests share this single instance.
    
    Usage:
        def test_something(test_bridge_controller):
            # Get endpoint URL
            url = test_bridge_controller.get_endpoint(existing_service_preferred=True)
            
            # Run Claude Code CLI
            result = test_bridge_controller.run_claude_code("Say hello", existing_service_preferred=False)
    """
    yield _bridge_controller
    _bridge_controller.stop()
