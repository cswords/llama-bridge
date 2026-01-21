"""
Unit tests for Anthropic protocol adapter.
"""

import pytest
from src.adapters.anthropic import AnthropicAdapter

class TestAnthropicAdapter:
    """Test AnthropicAdapter conversion logic."""
    
    def setup_method(self):
        self.adapter = AnthropicAdapter()
    
    def test_to_internal(self):
        """Should convert request to internal format."""
        request = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "System prompt"
        }
        
        result = self.adapter.to_internal(request)
        
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["max_tokens"] == 1024
    
    def test_from_internal(self):
        """Should convert internal response to Anthropic format."""
        response = {
            "content": "Hello!",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        original_request = {"model": "claude-3-opus"}
        
        result = self.adapter.from_internal(response, original_request)
        
        assert result["role"] == "assistant"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"

    def test_reasoning_response(self):
        """Response with reasoning content should have thinking block."""
        response = {
            "content": "The answer is 4.",
            "reasoning_content": "Calculating 2+2",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
        original_request = {"model": "claude-3-opus"}
        
        result = self.adapter.from_internal(response, original_request)
        
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Calculating 2+2"
        assert result["content"][1]["type"] == "text"
        assert result["content"][1]["text"] == "The answer is 4."


class TestAnthropicSSEConversion:
    """Test SSE chunk conversion for Anthropic adapter."""
    
    def test_text_delta_sse(self):
        """Should format text delta as SSE event."""
        adapter = AnthropicAdapter()
        
        chunk = {"content": "Hello", "stop_reason": None}
        sse = adapter.chunk_to_sse(chunk, {})
        
        assert "event: content_block_delta" in sse
        assert "text_delta" in sse
        assert "Hello" in sse
    
    def test_thinking_delta_sse(self):
        """Should format thinking delta as SSE event."""
        adapter = AnthropicAdapter()
        
        chunk = {"thought": "Let me think...", "stop_reason": None}
        sse = adapter.chunk_to_sse(chunk, {})
        
        assert "event: content_block_delta" in sse
        assert "thinking_delta" in sse
        assert "Let me think..." in sse
    
    def test_final_message_delta_sse(self):
        """Should format final message delta with stop_reason."""
        adapter = AnthropicAdapter()
        
        chunk = {"content": "", "stop_reason": "end_turn", "usage": {"output_tokens": 10}}
        sse = adapter.chunk_to_sse(chunk, {})
        
        assert "message_delta" in sse
        assert "end_turn" in sse
        assert "message_stop" in sse
    
    def test_tool_use_sse(self):
        """Should format tool use as SSE events."""
        adapter = AnthropicAdapter()
        
        chunk = {
            "content": "",
            "stop_reason": "tool_use",
            "tool_calls": [{
                "id": "toolu_123",
                "name": "get_weather",
                "arguments": {"location": "Tokyo"}
            }],
            "usage": {"output_tokens": 15}
        }
        sse = adapter.chunk_to_sse(chunk, {})
        
        assert "content_block_start" in sse
        assert "tool_use" in sse
        assert "get_weather" in sse
        assert "input_json_delta" in sse
        assert "Tokyo" in sse 
