"""
LiteLLM adapter for API format conversion.
Handles conversion between Anthropic, OpenAI, and internal formats.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class LiteLLMAdapter:
    """
    Adapter for converting between various LLM API formats.
    Uses LiteLLM utilities where possible, with custom handling as needed.
    """
    
    def __init__(self):
        # TODO: Initialize LiteLLM utilities
        pass
    
    # ============ Anthropic Format Conversion ============
    
    def anthropic_to_internal(self, request: dict) -> dict:
        """
        Convert Anthropic Messages API request to internal format.
        
        Anthropic format:
        {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "system": "You are helpful",
            "tools": [...],
            "stream": true
        }
        
        Internal format (OpenAI-like):
        {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024,
            "tools": [...],
            "stream": true
        }
        """
        messages = []
        
        # Add system message if present
        if system := request.get("system"):
            messages.append({"role": "system", "content": system})
        
        # Convert Anthropic messages to OpenAI format
        for msg in request.get("messages", []):
            messages.append(self._convert_anthropic_message(msg))
        
        internal = {
            "messages": messages,
            "max_tokens": request.get("max_tokens", 4096),
            "stream": request.get("stream", False),
        }
        
        # Convert tools if present
        if tools := request.get("tools"):
            internal["tools"] = self._convert_anthropic_tools(tools)
        
        return internal
    
    def internal_to_anthropic(self, response: dict, original_request: dict) -> dict:
        """
        Convert internal response to Anthropic Messages API format.
        
        Internal format:
        {
            "content": "Hello!",
            "tool_calls": [...],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        Anthropic format:
        {
            "id": "msg_xxx",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "...",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        """
        content_blocks = []
        
        # Add text content if present
        if text := response.get("content"):
            content_blocks.append({"type": "text", "text": text})
        
        # Add tool use blocks if present
        for tool_call in response.get("tool_calls", []):
            content_blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": tool_call["name"],
                "input": tool_call["arguments"],
            })
        
        # Determine stop reason
        stop_reason = response.get("stop_reason", "end_turn")
        if response.get("tool_calls"):
            stop_reason = "tool_use"
        
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": original_request.get("model", "llama-bridge"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": response.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }
    
    def chunk_to_anthropic_sse(self, chunk: dict) -> str:
        """Convert internal chunk to Anthropic SSE format."""
        # TODO: Implement proper SSE formatting for streaming
        import json
        event_type = chunk.get("type", "content_block_delta")
        data = json.dumps(chunk, ensure_ascii=False)
        return f"event: {event_type}\ndata: {data}\n\n"
    
    # ============ OpenAI Format Conversion ============
    
    def openai_to_internal(self, request: dict) -> dict:
        """
        Convert OpenAI Chat Completions API request to internal format.
        OpenAI format is already close to our internal format.
        """
        return {
            "messages": request.get("messages", []),
            "max_tokens": request.get("max_tokens", 4096),
            "stream": request.get("stream", False),
            "tools": request.get("tools"),
            "temperature": request.get("temperature", 0.7),
        }
    
    def internal_to_openai(self, response: dict, original_request: dict) -> dict:
        """
        Convert internal response to OpenAI Chat Completions API format.
        """
        message = {"role": "assistant"}
        
        if content := response.get("content"):
            message["content"] = content
        
        if tool_calls := response.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] if isinstance(tc["arguments"], str) 
                                     else __import__("json").dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ]
        
        # Determine finish reason
        finish_reason = "stop"
        if response.get("tool_calls"):
            finish_reason = "tool_calls"
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": original_request.get("model", "llama-bridge"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    response.get("usage", {}).get("input_tokens", 0) +
                    response.get("usage", {}).get("output_tokens", 0)
                ),
            },
        }
    
    def chunk_to_openai_sse(self, chunk: dict) -> str:
        """Convert internal chunk to OpenAI SSE format."""
        import json
        data = json.dumps(chunk, ensure_ascii=False)
        return f"data: {data}\n\n"
    
    # ============ Helper Methods ============
    
    def _convert_anthropic_message(self, msg: dict) -> dict:
        """Convert a single Anthropic message to OpenAI format."""
        role = msg.get("role")
        content = msg.get("content")
        
        # Handle string content
        if isinstance(content, str):
            return {"role": role, "content": content}
        
        # Handle content blocks (text, tool_use, tool_result, etc.)
        if isinstance(content, list):
            # For now, concatenate text blocks
            text_parts = []
            tool_results = []
            
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tool_results.append(block)
            
            if tool_results:
                # Handle tool results specially
                return {
                    "role": "tool",
                    "content": tool_results[0].get("content", ""),
                    "tool_call_id": tool_results[0].get("tool_use_id"),
                }
            
            return {"role": role, "content": "\n".join(text_parts)}
        
        return {"role": role, "content": str(content)}
    
    def _convert_anthropic_tools(self, tools: list) -> list:
        """Convert Anthropic tools format to OpenAI format."""
        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return converted
