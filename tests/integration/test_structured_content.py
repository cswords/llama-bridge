"""
Integration tests for Bridge and Protocol Adapters interaction.
Focuses on structured content (thinking, tool calls) in streaming and non-streaming modes.
"""

import pytest
import json
import re
from unittest.mock import MagicMock, AsyncMock
from src.bridge import Bridge

@pytest.fixture
def mock_wrapper():
    wrapper = MagicMock()
    # Default usage
    wrapper.get_usage.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
    return wrapper

@pytest.fixture
def bridge(mock_wrapper):
    # Initialize bridge with mock mode to bypass file checks
    b = Bridge(model_path="mock_path", debug=False, mock=True)
    # Inject mock wrapper
    b.wrapper = mock_wrapper
    return b

@pytest.mark.asyncio
class TestBridgeStructuredContent:
    """Tests the Bridge's ability to handle structured content using mock llama.cpp wrapper."""

    async def test_non_streaming_reasoning_fallback(self, bridge, mock_wrapper):
        """Test thinking block extraction in non-streaming mode via regex fallback."""
        # Setup mock to return raw text with tags
        mock_wrapper.apply_template.return_value = {"prompt": "test prompt"}
        # Simulate wrapper NOT parsing reasoning automatically
        mock_wrapper.generate.return_value = "<thought>Calculating 2+2</thought>The answer is 4."
        mock_wrapper.parse_response.return_value = {
            "content": "<thought>Calculating 2+2</thought>The answer is 4.", # simulate unparsed
            "reasoning_content": "",
            "tool_calls": []
        }

        request = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100
        }

        # Use complete_anthropic which calls _generate
        response_json = await bridge.complete_anthropic(request)
        response = json.loads(response_json)

        # Verify content is split
        content_types = [c["type"] for c in response["content"]]
        assert "thinking" in content_types
        assert "text" in content_types
        
        thinking_block = next(c for c in response["content"] if c["type"] == "thinking")
        text_block = next(c for c in response["content"] if c["type"] == "text")
        
        assert thinking_block["thinking"] == "Calculating 2+2"
        assert text_block["text"] == "The answer is 4."

    async def test_streaming_reasoning_and_content(self, bridge, mock_wrapper):
        """Test thinking and text deltas in streaming mode."""
        mock_wrapper.apply_template.return_value = {"prompt": "test prompt"}
        
        # Generator for tokens
        tokens = ["<thought>", "I", " am", " thinking", "</thought>", "I", " am", " done."]
        token_iter = iter(tokens)
        
        def get_next_token():
            try:
                return next(token_iter)
            except StopIteration:
                return None
        
        mock_wrapper.get_next_token.side_effect = get_next_token
        
        # Partial parse results for each step
        # Note: In reality, full_raw accumulates, we mock parse_response to return partials
        def parse_partial(raw, partial):
            if "<thought>" in raw and "</thought>" not in raw:
                return {"content": "", "reasoning_content": raw.replace("<thought>", ""), "tool_calls": []}
            elif "</thought>" in raw:
                parts = raw.split("</thought>")
                return {"content": parts[1].strip(), "reasoning_content": parts[0].replace("<thought>", ""), "tool_calls": []}
            return {"content": raw, "reasoning_content": "", "tool_calls": []}
            
        mock_wrapper.parse_response.side_effect = parse_partial

        request = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }

        events = []
        async for event in bridge.stream_anthropic(request):
            events.append(event)

        # Verify event sequence
        # We expect a series of content_block_delta events
        thinking_deltas = [e for e in events if "thinking_delta" in e]
        text_deltas = [e for e in events if "text_delta" in e]
        
        assert len(thinking_deltas) > 0
        assert len(text_deltas) > 0
        
        # Check cumulative thinking content
        full_thinking = ""
        for d in thinking_deltas:
            data = json.loads(d.split("data: ")[1])
            full_thinking += data["delta"]["thinking"]
        assert "I am thinking" in full_thinking
        
        # Check cumulative text content
        full_text = ""
        for d in text_deltas:
            data = json.loads(d.split("data: ")[1])
            full_text += data["delta"]["text"]
        assert "I am done." in full_text

    async def test_streaming_tool_calls(self, bridge, mock_wrapper):
        """Test tool call emission at the end of a stream."""
        mock_wrapper.apply_template.return_value = {"prompt": "test prompt"}
        
        # Stream content then a tool call
        tokens = ["I", " will", " call", " a", " tool."]
        token_iter = iter(tokens)
        mock_wrapper.get_next_token.side_effect = lambda: next(token_iter) if hasattr(token_iter, '__next__') else None
        
        # When get_next_token returns None, the loop ends and final parse happens
        def get_none():
            try:
                return next(token_iter)
            except StopIteration:
                return None
        mock_wrapper.get_next_token.side_effect = get_none
        
        # Final parse returns a tool call
        mock_wrapper.parse_response.side_effect = [
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": []}, # during stream
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": []}, # ...
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": []}, # ...
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": []}, # ...
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": []}, # final before None
            {"content": "I will call a tool.", "reasoning_content": "", "tool_calls": [
                {"id": "toolu_1", "name": "get_weather", "arguments": {"location": "San Francisco"}}
            ]} # final after None
        ]

        request = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "Weather in SF?"}],
            "stream": True,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
        }

        events = []
        async for event in bridge.stream_anthropic(request):
            events.append(event)

        # We expect: text_deltas ..., content_block_start (tool_use), content_block_delta (json), content_block_stop, message_delta (tool_use)
        has_start = any("content_block_start" in e for e in events)
        has_json = any("input_json_delta" in e for e in events)
        has_stop = any("content_block_stop" in e for e in events)
        has_msg_delta = any("message_delta" in e for e in events)
        
        assert has_start, "Missing content_block_start for tool"
        assert has_json, "Missing input_json_delta for tool"
        assert has_stop, "Missing content_block_stop for tool"
        
        # Helper to parse SSE data from multiple potential events in a chunk
        def get_sse_data(event_str):
            datas = []
            for line in event_str.splitlines():
                if line.startswith("data: "):
                    datas.append(json.loads(line.split("data: ")[1]))
            return datas

        # Last message_delta should have stop_reason: tool_use
        msg_delta_data = None
        for e in reversed(events):
            datas = get_sse_data(e)
            for d in datas:
                if d.get("type") == "message_delta":
                    msg_delta_data = d
                    break
            if msg_delta_data:
                break
                
        assert msg_delta_data is not None
        assert msg_delta_data["delta"]["stop_reason"] == "tool_use"
