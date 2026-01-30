
import pytest
import json
from src.bridge.scanner import StreamScanner, ScanEvent
from src.bridge.flavors import QwenFlavor, MiniMaxFlavor

def test_scanner_qwen_tool_call():
    flavor = QwenFlavor()
    scanner = StreamScanner(block_tokens=flavor.block_tokens)
    
    raw = "I will call a tool: <function=get_weather><parameter=city>Beijing</parameter></function> That's it."
    
    events = []
    # Feed character by character to test buffering
    for char in raw:
        events.extend(scanner.push(char))
    events.extend(scanner.flush())
    
    # Check that no protected fragments leaked into content
    content = "".join([e.data for e in events if e.type == "content"])
    assert "<function=" not in content
    assert "</function>" not in content
    assert "get_weather" not in content # Should be in block_content
    
    # Check block structure
    block_starts = [e for e in events if e.type == "block_start"]
    assert len(block_starts) >= 1
    
    # Final reconstructed content
    assert content.strip() == "I will call a tool:  That's it."

def test_scanner_minimax_tool_call():
    flavor = MiniMaxFlavor()
    scanner = StreamScanner(block_tokens=flavor.block_tokens)
    
    raw = 'Sure. <invoke name="get_weather"><parameter name="city">Shanghai</parameter></invoke> Done.'
    
    events = []
    for char in raw:
        events.extend(scanner.push(char))
    events.extend(scanner.flush())
    
    content = "".join([e.data for e in events if e.type == "content"])
    assert "invoke" not in content
    assert "weather" not in content
    assert content.strip() == "Sure.  Done."

def test_scanner_fragmented_tag():
    # Test that it buffers partial tags correctly
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    
    # Push "<tho" -> should be buffered
    ev1 = scanner.push("<tho")
    assert len(ev1) == 0
    
    # Push "ught>" -> should trigger block_start
    ev2 = scanner.push("ught>")
    assert len(ev2) == 1
    assert ev2[0].type == "block_start"
    
    # Push "Thinking..."
    ev3 = scanner.push("Thinking...")
    assert len(ev3) == 1
    assert ev3[0].type == "block_content"
    
    # Push "</tho"
    ev4 = scanner.push("</tho")
    assert len(ev4) == 0
    
    # Push "ught>"
    ev5 = scanner.push("ught>")
    assert len(ev5) == 1
    assert ev5[0].type == "block_end"

def test_scanner_false_alarm():
    scanner = StreamScanner(block_tokens=["<thought>"])
    
    # "<tho" is prefix
    assert len(scanner.push("<tho")) == 0
    
    # "at" makes it "<thoat", which is NOT a prefix
    ev = scanner.push("at")
    assert len(ev) == 1
    assert ev[0].type == "content"
    assert ev[0].data == "<thoat"

@pytest.mark.parametrize("flavor_class, raw_output, expected_tools", [
    (QwenFlavor, 
     "I'll help you. <function=get_weather><parameter=city>\"London\"</parameter></function>", 
     [{"name": "get_weather", "arguments": '{"city": "London"}'}]),
    (MiniMaxFlavor,
     'Okay. <invoke name="calculator"><parameter name="exp">"1+1"</parameter></invoke>',
     [{"name": "calculator", "arguments": '{"exp": "1+1"}'}])
])
def test_flavor_parsing(flavor_class, raw_output, expected_tools):
    flavor = flavor_class()
    # Mock the full cycle
    scanner = StreamScanner(block_tokens=flavor.block_tokens)
    events = scanner.push(raw_output) + scanner.flush()
    
    # Capture block content with stack to support nesting
    stack = [] # List of [tag, content, start_tag_data]
    tool_calls = []
    
    for ev in events:
        if ev.type == "block_start":
            for item in stack:
                item[1] += ev.data
            stack.append([ev.tag, "", ev.data])
        elif ev.type == "block_content":
            for item in stack:
                item[1] += ev.data
        elif ev.type == "block_end":
            if stack:
                tag, content, start_tag_data = stack.pop()
                for item in stack:
                    item[1] += ev.data
                res = flavor.interpret_block_complete(tag, content, start_tag=start_tag_data)
                if res and res[0] == "tool_use":
                    tool_calls.extend(res[1])
            
    assert len(tool_calls) == len(expected_tools)
    for at, et in zip(tool_calls, expected_tools):
        assert at["name"] == et["name"]
        # Arguments might differ in whitespace
        assert json.loads(at["arguments"]) == json.loads(et["arguments"])

def test_bridge_replication():
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    tokens = ["Hello ", "<", "thou", "ght>", " I am thinking.", "</", "thought>", " Done."]
    events = []
    for t in tokens:
        events.extend(scanner.push(t))
    events.extend(scanner.flush())
    
    content = "".join([e.data for e in events if e.type == "content"])
    assert content == "Hello  Done."
    
    thought_content = "".join([e.data for e in events if e.type == "block_content"])
    assert thought_content == " I am thinking."


def test_scanner_harmony_header_with_constrain():
    # Test specific regression: <|constrain|> in header should NOT act as a block starter
    flavor = MiniMaxFlavor()
    scanner = StreamScanner(block_tokens=flavor.block_tokens, skip_tokens=flavor.skip_tokens, end_tokens=flavor.end_tokens)
    
    # Raw output from GPT-OSS-120B (simulated)
    # Note: <|start|>assistant is skipped.
    raw = "<|start|>assistant Llama<|channel|>commentary to=functions.Read<|constrain|>json<|message|>{\"file\": \"README.md\"}<|end|>"
    
    events = []
    for char in raw:
        events.extend(scanner.push(char))
    events.extend(scanner.flush())
    
    # Analysis:
    # 1. <|start|>assistant -> Skipped (if defined in skip_tokens)
    # 2. Llama -> Content (oops, leaky prefix? But Scanner treats it as content)
    # 3. <|channel|> -> Block Start
    # 4. commentary to=... <|constrain|> json ... -> Block Content (should NOT start new block)
    # 5. <|message|> -> In Harmony, is this a separator? 
    # Wrapper/Protocol logic handles splitting. Scanner sees it as content inside <|channel|> block?
    # Wait, in protocols.py: block_tokens include <|channel|>, <|end|>.
    # <|message|> is NOT a block token. It is content inside the block.
    # 6. <|end|> -> Block End.
    
    # Check block events
    blocks = [e for e in events if e.type == "block_start"]
    assert len(blocks) == 1
    assert blocks[0].data == "<|channel|>"
    
    # Check that <|constrain|> did NOT trigger block_start
    assert not any(e.data == "<|constrain|>" and e.type == "block_start" for e in events)
    
    # The content of the block should contain the full header+body
    block_content = "".join([e.data for e in events if e.type == "block_content"])
    assert "commentary" in block_content
    assert "<|constrain|>" in block_content # It stays as content!
    assert "json" in block_content
    assert "README.md" in block_content

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
