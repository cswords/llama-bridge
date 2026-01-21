"""
Unit tests for Qwen3-Coder tool call parsing.
"""

import pytest


class TestQwenToolCallParser:
    """Test the _parse_qwen_tool_calls helper function."""
    
    def test_single_tool_call(self):
        """Should parse a single tool call."""
        from src.bridge import _parse_qwen_tool_calls
        
        raw = """<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call>"""
        
        result = _parse_qwen_tool_calls(raw)
        
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"] == {"city": "Tokyo"}
        assert "id" in result[0]
    
    def test_multiple_parameters(self):
        """Should parse multiple parameters."""
        from src.bridge import _parse_qwen_tool_calls
        
        raw = """<function=search>
<parameter=query>python tutorial</parameter>
<parameter=limit>10</parameter>
</function>"""
        
        result = _parse_qwen_tool_calls(raw)
        
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["arguments"]["query"] == "python tutorial"
        assert result[0]["arguments"]["limit"] == 10  # Should be parsed as int
    
    def test_multiple_tool_calls(self):
        """Should parse multiple tool calls."""
        from src.bridge import _parse_qwen_tool_calls
        
        raw = """<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
<function=get_time>
<parameter=timezone>JST</parameter>
</function>"""
        
        result = _parse_qwen_tool_calls(raw)
        
        assert len(result) == 2
        assert result[0]["name"] == "get_weather"
        assert result[1]["name"] == "get_time"
    
    def test_no_tool_calls(self):
        """Should return empty list for plain text."""
        from src.bridge import _parse_qwen_tool_calls
        
        result = _parse_qwen_tool_calls("Just a regular response.")
        
        assert result == []
    
    def test_json_parameter_value(self):
        """Should parse JSON values in parameters."""
        from src.bridge import _parse_qwen_tool_calls
        
        raw = """<function=create_file>
<parameter=path>/tmp/test.json</parameter>
<parameter=content>{"key": "value"}</parameter>
</function>"""
        
        result = _parse_qwen_tool_calls(raw)
        
        assert len(result) == 1
        # JSON should be parsed
        assert result[0]["arguments"]["content"] == {"key": "value"}
