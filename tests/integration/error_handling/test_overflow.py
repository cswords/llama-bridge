import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.server import app
from src.exceptions import ContextLimitExceededError
from src.config import BridgeConfig, ModelConfig, CacheConfig, RouteConfig

@pytest.fixture
def mock_bridge():
    with patch("src.server.Bridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        # Mock ready status
        bridge_instance.check_ready.return_value = True
        
        # Configure _generate to raise ContextLimitExceededError
        async def raise_overflow(*args, **kwargs):
            raise ContextLimitExceededError(limit=100, requested=150, cache_name="test")
            
        bridge_instance._generate = AsyncMock(side_effect=raise_overflow)
        bridge_instance.complete_openai = AsyncMock(side_effect=raise_overflow)
        bridge_instance.complete_anthropic = AsyncMock(side_effect=raise_overflow)
        
        yield bridge_instance

@pytest.fixture
def client(mock_bridge):
    # Setup dummy objects
    config = BridgeConfig()
    config.models["test"] = ModelConfig(name="test", path="test.gguf")
    config.caches["default"] = CacheConfig(name="default", model="test", n_ctx=100)
    config.routes.append(RouteConfig(cache="default", match="*"))
    
    import src.server
    src.server.bridge = mock_bridge
    src.server.config = config
    src.server.router = None # Simplify routing for this test
    
    return TestClient(app)

def test_openai_overflow_error(client):
    """Test that OpenAI endpoint returns correct error format on overflow."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Too long"}]
        }
    )
    
    # Should get 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    
    # Verify OpenAI error format
    assert "error" in data
    assert data["error"]["code"] == "context_length_exceeded"
    assert "150 tokens" in data["error"]["message"]
    assert "100 tokens" in data["error"]["message"]

def test_anthropic_overflow_error(client):
    """Test that Anthropic endpoint returns correct error format on overflow."""
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Too long"}],
            "max_tokens": 10
        }
    )
    
    # Should get 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    
    # Verify Anthropic error format
    assert data["type"] == "error"
    assert data["error"]["type"] == "invalid_request_error"
    assert "exceeds context limit" in data["error"]["message"]
