import pytest
from unittest.mock import MagicMock, patch
from src.chat_bridge.bridge import ChatBridge

@pytest.fixture
def mock_wrapper():
    """Mock LlamaChatWrapper with correct interface methods."""
    with patch('src.chat_bridge.bridge.LlamaChatWrapper') as MockWrapper:
        instance = MockWrapper.return_value
        # Mock apply_template to return dict with prompt key
        instance.apply_template.return_value = {"prompt": "Formatted Prompt"}
        yield instance

@pytest.fixture
def bridge(mock_wrapper):
    return ChatBridge("model.gguf")

def test_complete_anthropic_text_only(bridge):
    """Verify non-streaming response for plain text."""
    # Mock generate to return full text (not iterator)
    bridge.wrapper.generate.return_value = "Hello World"
    
    response = bridge.complete_anthropic({"messages": [{"role": "user", "content": "Hi"}]})
    
    assert response["type"] == "message"
    assert len(response["content"]) == 1
    assert response["content"][0]["type"] == "text"
    assert response["content"][0]["text"] == "Hello World"

def test_complete_anthropic_tool_call(bridge):
    """Verify non-streaming response parsing a tool call."""
    tool_text = 'Thinking...\n<tool_call>\n```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```\n</tool_call>\nDone.'
    bridge.wrapper.generate.return_value = tool_text
    
    response = bridge.complete_anthropic({"messages": []})
    
    content = response["content"]
    assert len(content) == 3  # Text, Tool, Text
    
    # 1. Pre-text
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Thinking..."
    
    # 2. Tool
    assert content[1]["type"] == "tool_use"
    assert content[1]["name"] == "get_weather"
    assert content[1]["input"]["city"] == "Paris"
    
    # 3. Post-text
    assert content[2]["type"] == "text"
    assert content[2]["text"] == "Done."

def test_complete_anthropic_malformed_tool(bridge):
    """Verify fallback for malformed tool JSON."""
    tool_text = '<tool_call>{bad_json}</tool_call>'
    bridge.wrapper.generate.return_value = tool_text
    
    response = bridge.complete_anthropic({"messages": []})
    
    content = response["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    # Should contain the full tag as text
    assert "<tool_call>" in content[0]["text"]

def test_complete_anthropic_with_tools(bridge):
    """Verify tool extraction is passed to apply_template."""
    bridge.wrapper.generate.return_value = "I will call the tool"
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }
    ]
    
    response = bridge.complete_anthropic({
        "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
        "tools": tools
    })
    
    # Verify apply_template was called with tools
    call_args = bridge.wrapper.apply_template.call_args
    assert call_args is not None
    # Second positional arg should be extracted tools list
    passed_tools = call_args[0][1]
    assert len(passed_tools) == 1
    assert passed_tools[0]["name"] == "get_weather"


def test_complete_anthropic_minimax_format(mock_wrapper):
    """Verify MiniMax invoke/parameter tool format parsing."""
    # Need to patch FlavorFactory to return MiniMaxFlavor
    with patch('src.chat_bridge.bridge.FlavorFactory') as MockFactory:
        from src.chat_bridge.flavors.minimax_flavor import MiniMaxFlavor
        MockFactory.get_flavor_for_model.return_value = MiniMaxFlavor()
        
        bridge = ChatBridge("minimax-model.gguf")
        bridge.wrapper.generate.return_value = '''Pre-text
<minimax:tool_call>
<invoke name="Read">
<parameter name="file_path">/path/to/file</parameter>
</invoke>
</minimax:tool_call>'''
        
        response = bridge.complete_anthropic({"messages": []})
        
        content = response["content"]
        # Should have text + tool
        assert any(b["type"] == "tool_use" for b in content)
        tool_block = next(b for b in content if b["type"] == "tool_use")
        assert tool_block["name"] == "Read"
        assert tool_block["input"]["file_path"] == "/path/to/file"


def test_complete_anthropic_mimo_format(mock_wrapper):
    """Verify Mimo function=Name tool format parsing."""
    with patch('src.chat_bridge.bridge.FlavorFactory') as MockFactory:
        from src.chat_bridge.flavors.mimo_flavor import MimoFlavor
        MockFactory.get_flavor_for_model.return_value = MimoFlavor()
        
        bridge = ChatBridge("mimo-model.gguf")
        bridge.wrapper.generate.return_value = '''<tool_call>
<function=Read>
<parameter=file_path>/path/to/file</parameter>
</function>
</tool_call>'''
        
        response = bridge.complete_anthropic({"messages": []})
        
        content = response["content"]
        assert any(b["type"] == "tool_use" for b in content)
        tool_block = next(b for b in content if b["type"] == "tool_use")
        assert tool_block["name"] == "Read"
        assert tool_block["input"]["file_path"] == "/path/to/file"

