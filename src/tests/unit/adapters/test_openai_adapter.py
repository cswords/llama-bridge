"""
Unit tests for OpenAI protocol adapter.
"""

import pytest
from src.adapters.openai import OpenAIAdapter

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
