# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

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
                    reasoning_content=response.get("reasoning_content"),
                    tool_calls=tool_calls
                ),
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }]
        )
        return model_response.model_dump()

    def chunk_to_sse(self, chunk: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Convert internal chunk to OpenAI SSE."""
        delta = {}
        if "thought" in chunk:
            delta["reasoning_content"] = chunk["thought"]
        if "content" in chunk:
            delta["content"] = chunk["content"]
        if "tool_calls" in chunk:
            delta["tool_calls"] = chunk["tool_calls"]

        openai_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "llama-bridge",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": chunk.get("stop_reason")
            }]
        }
        return f"data: {json.dumps(openai_chunk)}\n\n"
