import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from src.bridge import Bridge
from src.config import BridgeConfig, ModelConfig, CacheConfig

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def bridge(mock_logger):
    with patch("src.bridge.logger", mock_logger):
        bridge = Bridge(
            model_path="dummy.gguf", 
            mock=True,
            cache_configs=[{"name": "test", "n_ctx": 1024}]
        )
        return bridge

@pytest.mark.asyncio
async def test_system_prompt_change_detection(bridge, caplog):
    caplog.set_level(logging.WARNING)
    
    # Request 1: System A
    req1 = {
        "model": "claude-sonnet",
        "messages": [
            {"role": "system", "content": "System A"},
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 10
    }
    
    await bridge.complete_anthropic(req1, cache_name="test")
    
    # Request 2: System A (No change)
    caplog.clear()
    await bridge.complete_anthropic(req1, cache_name="test")
    assert len(caplog.records) == 0
    
    # Request 3: System B (Change)
    req2 = {
        "model": "claude-sonnet",
        "messages": [
            {"role": "system", "content": "System B"},
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 10
    }
    
    await bridge.complete_anthropic(req2, cache_name="test")
    
    # Should warn
    assert len(caplog.records) > 0
    assert "System prompt changed" in caplog.text
    assert "test" in caplog.text
