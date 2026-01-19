"""
Basic E2E tests for Llama-Bridge.
Step 0 of TDD evolution - validates core API functionality.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    async def test_health_check(self, test_client: AsyncClient):
        """Health endpoint should return status ok."""
        response = await test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAnthropicEndpoint:
    """Test the Anthropic Messages API endpoint."""
    
    async def test_simple_message(self, test_client: AsyncClient):
        """Simple text message should return valid response."""
        request = {
            "model": "test-model",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
        
        response = await test_client.post("/v1/messages", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"
    
    async def test_message_with_system(self, test_client: AsyncClient):
        """Message with system prompt should work."""
        request = {
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        }
        
        response = await test_client.post("/v1/messages", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["type"] == "message"


class TestOpenAIEndpoint:
    """Test the OpenAI Chat Completions API endpoint."""
    
    async def test_simple_chat(self, test_client: AsyncClient):
        """Simple chat message should return valid response."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100
        }
        
        response = await test_client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.fixture
async def test_client():
    """Create test client with mock bridge."""
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    from server import app
    from bridge import Bridge
    
    # Initialize with mock mode
    import server
    server.bridge = Bridge(model_path="mock", debug=False, mock=True)
    
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
