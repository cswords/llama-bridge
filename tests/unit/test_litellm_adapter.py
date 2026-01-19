"""
Unit tests for LiteLLM adapter format conversion.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from litellm_adapter import LiteLLMAdapter


class TestAnthropicToInternal:
    """Test Anthropic to internal format conversion."""
    
    def setup_method(self):
        self.adapter = LiteLLMAdapter()
    
    def test_simple_message(self):
        """Simple user message should convert correctly."""
        request = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        result = self.adapter.anthropic_to_internal(request)
        
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["max_tokens"] == 1024
    
    def test_with_system_prompt(self):
        """System prompt should become first message."""
        request = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }
        
        result = self.adapter.anthropic_to_internal(request)
        
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."
        assert result["messages"][1]["role"] == "user"
    
    def test_with_tools(self):
        """Tools should be converted to OpenAI format."""
        request = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            ]
        }
        
        result = self.adapter.anthropic_to_internal(request)
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"


class TestInternalToAnthropic:
    """Test internal to Anthropic format conversion."""
    
    def setup_method(self):
        self.adapter = LiteLLMAdapter()
    
    def test_simple_response(self):
        """Simple text response should convert correctly."""
        response = {
            "content": "Hello! I'm doing well.",
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8}
        }
        original_request = {"model": "claude-3-opus"}
        
        result = self.adapter.internal_to_anthropic(response, original_request)
        
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello! I'm doing well."
        assert result["stop_reason"] == "end_turn"
    
    def test_tool_use_response(self):
        """Response with tool calls should have tool_use blocks."""
        response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        original_request = {"model": "claude-3-opus"}
        
        result = self.adapter.internal_to_anthropic(response, original_request)
        
        assert result["stop_reason"] == "tool_use"
        # Should have tool_use block
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"location": "Tokyo"}


class TestOpenAIConversion:
    """Test OpenAI format conversion."""
    
    def setup_method(self):
        self.adapter = LiteLLMAdapter()
    
    def test_openai_to_internal(self):
        """OpenAI request should pass through with minimal changes."""
        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100
        }
        
        result = self.adapter.openai_to_internal(request)
        
        assert result["messages"] == request["messages"]
        assert result["max_tokens"] == 100
    
    def test_internal_to_openai(self):
        """Internal response should convert to OpenAI format."""
        response = {
            "content": "Hello!",
            "tool_calls": [],
            "stop_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        }
        original_request = {"model": "gpt-4"}
        
        result = self.adapter.internal_to_openai(response, original_request)
        
        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
