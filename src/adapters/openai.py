"""
Adapter for OpenAI Chat Completions API format.
"""

import json
import uuid
import time
from typing import Any, Dict
from litellm.utils import ModelResponse, Usage, Message
from .base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def to_internal(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI is already the internal base format."""
        return request

    def from_internal(self, response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal response to OpenAI format."""
        usage_obj = Usage(
            prompt_tokens=response.get("usage", {}).get("input_tokens", 0),
            completion_tokens=response.get("usage", {}).get("output_tokens", 0)
        )
        
        tool_calls = None
        if tcs := response.get("tool_calls"):
            tool_calls = []
            for tc in tcs:
                tool_calls.append({
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
                    }
                })

        model_response = ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=original_request.get("model", "llama-bridge"),
            usage=usage_obj,
            choices=[{
                "index": 0,
                "message": Message(
                    role="assistant",
                    content=response.get("content"),
                    tool_calls=tool_calls
                ),
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }]
        )
        return model_response.model_dump()

    def chunk_to_sse(self, chunk: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Convert internal chunk to OpenAI SSE."""
        return f"data: {json.dumps(chunk)}\n\n"
