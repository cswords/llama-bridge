"""
Unit tests for protocol adapters (Anthropic, OpenAI).
"""

import pytest

from src.adapters.anthropic import AnthropicAdapter
from src.adapters.openai import OpenAIAdapter

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

class TestOpenAIAdapter:
    """Test OpenAIAdapter conversion logic."""
    
    def setup_method(self):
        self.adapter = OpenAIAdapter()
    
    def test_to_internal(self):
        """OpenAI request is already internal."""
        request = {"messages": [{"role": "user", "content": "Hi"}]}
        assert self.adapter.to_internal(request) == request
    
    def test_from_internal(self):
        """Should convert internal response to OpenAI format."""
        response = {
            "content": "Hi!",
            "stop_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        }
        original_request = {"model": "gpt-4"}
        
        result = self.adapter.from_internal(response, original_request)
        
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["object"] == "chat.completion"
