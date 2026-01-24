
import pytest
import json
from fastapi.testclient import TestClient
from src import server
from src.bridge import Bridge

@pytest.fixture(autouse=True)
def setup_mock_bridge():
    # Initialize a mock bridge and inject it into the server module
    server.bridge = Bridge(model_path="mock", mock=True)
    server.config = None # Simple config
    server.router = None # No special routing

def test_anthropic_messages_non_streaming():
    client = TestClient(server.app)
    payload = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024
    }
    
    response = client.post("/v1/messages", json=payload)
    if response.status_code != 200:
        print(f"Error: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert "content" in data

def test_anthropic_messages_streaming():
    client = TestClient(server.app)
    payload = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
        "stream": True
    }
    
    with client.stream("POST", "/v1/messages", json=payload) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        
        # Verify event sequence
        event_types = [e["type"] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "message_stop" in event_types

def test_health_check():
    client = TestClient(server.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
