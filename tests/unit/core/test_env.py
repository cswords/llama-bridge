import os
import sys
from importlib import reload
from unittest.mock import patch
import pytest

def test_metal_shader_cache_env_var():
    """Verify that GGML_METAL_SHADER_CACHE_DIR is set on Darwin."""
    
    if sys.platform != "darwin":
        pytest.skip("Test only relevant on macOS")

    # Clean env
    with patch.dict(os.environ, {}, clear=True):
        # Envvar should not exist initially
        assert "GGML_METAL_SHADER_CACHE_DIR" not in os.environ
        
        # Import server should trigger the logic
        import src.server
        reload(src.server) # Reload to re-run module level code
        
        assert "GGML_METAL_SHADER_CACHE_DIR" in os.environ
        expected_suffix = "llama_bridge/metal"
        assert os.environ["GGML_METAL_SHADER_CACHE_DIR"].endswith(expected_suffix)

def test_metal_shader_cache_env_var_respected():
    """Verify that existing GGML_METAL_SHADER_CACHE_DIR is respected."""
    
    if sys.platform != "darwin":
        pytest.skip("Test only relevant on macOS")

    custom_path = "/tmp/my_custom_cache"
    with patch.dict(os.environ, {"GGML_METAL_SHADER_CACHE_DIR": custom_path}):
        import src.server
        reload(src.server)
        
        assert os.environ["GGML_METAL_SHADER_CACHE_DIR"] == custom_path
