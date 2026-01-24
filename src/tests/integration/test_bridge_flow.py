
import pytest
import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch
from src.bridge import Bridge

@pytest.mark.asyncio
async def test_bridge_streaming_integration():
    # 1. Mock the C++ extension class before Bridge is initialized
    with patch('llama_chat.LlamaChatWrapper') as MockWrapper:
        instance = MockWrapper.return_value
        
        # Setup the mock instance behavior
        instance.apply_template.return_value = {"prompt": "fake-prompt", "additional_stops": []}
        instance.init_inference.return_value = None
        
        token_sequence = [
            b"Hello! ",
            b"<thought>I should check use a tool.</thought>",
            b"Here is the data:",
            b"<tool_call>",
            b"<function=get_data>",
            b"<parameter=id>\"123\"</parameter>", # Use quotes to force string or just expect int in test
            b"</function>",
            b"</tool_call>",
            b"" # EOS
        ]
        # In our implementation, empty bytes means EOS
        instance.get_next_token.side_effect = token_sequence
        instance.get_usage.return_value = {"prompt_tokens": 5, "completion_tokens": 10}

        # Initialize Bridge
        dummy_model = "README.md" 
        bridge = Bridge(dummy_model, mock=False)
        
        # Force Qwen Flavor
        from src.bridge.flavors import QwenFlavor
        bridge.flavor = QwenFlavor()
        
        req = {
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "tools": [{"name": "get_data", "input_schema": {"type": "object"}}]
        }
        
        # Collect SSE events
        sse_events = []
        async for chunk in bridge.stream_anthropic(req, cache_name="main"):
            for line in chunk.split('\n'):
                if line.startswith("data: "):
                    sse_events.append(json.loads(line[6:]))
                
        # 1. Thinking block check
        thought_deltas = [e["delta"]["thinking"] for e in sse_events if e.get("type") == "content_block_delta" and e["delta"].get("type") == "thinking_delta"]
        # In current scanner, thought might come via block_complete because it's short
        assert "I should check use a tool." in "".join(thought_deltas)
        
        # 2. Text content check
        text_deltas = [e["delta"]["text"] for e in sse_events if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"]
        full_text = "".join(text_deltas)
        assert "Hello! " in full_text
        assert "Here is the data:" in full_text
        
        # 3. Tool use check
        tool_use_starts = [e for e in sse_events if e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use"]
        assert len(tool_use_starts) == 1
        assert tool_use_starts[0]["content_block"]["name"] == "get_data"
        
        # 4. Input JSON check
        json_deltas = [e["delta"]["partial_json"] for e in sse_events if e.get("type") == "content_block_delta" and e["delta"].get("type") == "input_json_delta"]
        full_json = "".join(json_deltas)
        assert json.loads(full_json) == {"id": "123"} # matches "123" in quotes now

        print("Integration test passed!")
