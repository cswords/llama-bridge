"""
Adapter for Anthropic Messages API format.
"""

import json
import uuid
import logging
from typing import Any, Dict
from litellm.utils import Usage
from .base import BaseAdapter

logger = logging.getLogger(__name__)

class AnthropicAdapter(BaseAdapter):
    def to_internal(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic Messages API request to internal format."""
        messages = []
        
        # Add system message if present
        if system := request.get("system"):
            if isinstance(system, str):
                messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                sys_text = "\n".join([s.get("text", "") for s in system if s.get("type") == "text"])
                messages.append({"role": "system", "content": sys_text})
        
        # Convert Anthropic messages to OpenAI-like format
        for msg in request.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Collect blocks for this message
                text_parts = []
                tool_calls = []
                
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "name": block["name"],
                            "arguments": json.dumps(block["input"])
                        })
                    elif block_type == "tool_result":
                        # tool_result blocks must be their own 'tool' role messages in internal format
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": str(block.get("content", ""))
                        })
                
                # If we have text or tool_calls, add the message
                if text_parts or tool_calls:
                    msg_obj = {"role": role}
                    if text_parts:
                        msg_obj["content"] = "\n".join(text_parts)
                    if tool_calls:
                        msg_obj["tool_calls"] = tool_calls
                    messages.append(msg_obj)
        
        internal = {
            "messages": messages,
            "max_tokens": request.get("max_tokens", 4096),
            "stream": request.get("stream", False),
            "temperature": request.get("temperature", 0.7),
            "top_p": request.get("top_p"),
            "stop": request.get("stop_sequences"),
        }
        
        if tools := request.get("tools"):
            internal["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    }
                } for t in tools
            ]
        
        return internal

    def from_internal(self, response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal response to Anthropic format."""
        content_blocks = []
        
        if reasoning := response.get("reasoning_content"):
            content_blocks.append({"type": "thinking", "thinking": reasoning})

        if text := response.get("content"):
            content_blocks.append({"type": "text", "text": text})
        
        for tc in response.get("tool_calls", []):
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tc["name"],
                "input": tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"]),
            })
        
        u = response.get("usage", {})
        usage_obj = Usage(
            prompt_tokens=u.get("input_tokens", 0),
            completion_tokens=u.get("output_tokens", 0)
        )
        
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
            "usage": {
                "input_tokens": usage_obj.prompt_tokens,
                "output_tokens": usage_obj.completion_tokens
            },
        }

    def chunk_to_sse(self, chunk: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Convert internal chunk to Anthropic SSE with state tracking."""
        events = []
        
        # 1. Message Start
        if not state.get("message_started"):
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            data = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "llama-bridge",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }
            events.append(f"event: message_start\ndata: {json.dumps(data)}\n\n")
            state["message_started"] = True
            state["current_block_index"] = 0

        # 2. Handle Thought (Thinking)
        if thought := chunk.get("thought"):
            if state.get("active_block_type") != "thinking":
                # Close previous if anyway
                if state.get("active_block_type"):
                    events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['current_block_index']})}\n\n")
                    state["current_block_index"] += 1
                
                # Start thinking block
                data = {
                    "type": "content_block_start",
                    "index": state["current_block_index"],
                    "content_block": {"type": "thinking", "thinking": ""}
                }
                events.append(f"event: content_block_start\ndata: {json.dumps(data)}\n\n")
                state["active_block_type"] = "thinking"

            delta = {
                "type": "content_block_delta",
                "index": state["current_block_index"],
                "delta": {"type": "thinking_delta", "thinking": thought}
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n")

        # 3. Handle Text Content
        elif content := chunk.get("content"):
            if state.get("active_block_type") != "text":
                if state.get("active_block_type"):
                    events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['current_block_index']})}\n\n")
                    state["current_block_index"] += 1
                
                data = {
                    "type": "content_block_start",
                    "index": state["current_block_index"],
                    "content_block": {"type": "text", "text": ""}
                }
                events.append(f"event: content_block_start\ndata: {json.dumps(data)}\n\n")
                state["active_block_type"] = "text"

            delta = {
                "type": "content_block_delta",
                "index": state["current_block_index"],
                "delta": {"type": "text_delta", "text": content}
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n")

        # 4. Handle Stop / Tool Use
        if chunk.get("stop_reason") is not None:
            # Close active block if any
            if state.get("active_block_type"):
                events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['current_block_index']})}\n\n")
                state["current_block_index"] += 1
                state["active_block_type"] = None

            # Handle Tool Calls
            if tool_calls := chunk.get("tool_calls"):
                for tc in tool_calls:
                    idx = state["current_block_index"]
                    # Important: ensure we have a valid non-empty ID
                    tc_id = (tc.get("id") or tc.get("tool_call_id") or f"toolu_{uuid.uuid4().hex[:24]}")
                    
                    # Tool Use Start
                    start_data = {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "tool_use", "id": tc_id, "name": tc["name"], "input": {}}
                    }
                    events.append(f"event: content_block_start\ndata: {json.dumps(start_data)}\n\n")
                    
                    # Tool Use Delta
                    args = tc["arguments"]
                    if not isinstance(args, str): args = json.dumps(args)
                    delta_data = {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    }
                    events.append(f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n")
                    
                    # Tool Use Stop
                    events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n")
                    state["current_block_index"] += 1

            # Message Delta and Stop
            data = {
                "type": "message_delta",
                "delta": {"stop_reason": chunk["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": chunk.get("usage", {}).get("output_tokens", 0)}
            }
            events.append(f"event: message_delta\ndata: {json.dumps(data)}\n\n")
            events.append(f"event: message_stop\ndata: {{\"type\": \"message_stop\"}}\n\n")

        return "".join(events)
