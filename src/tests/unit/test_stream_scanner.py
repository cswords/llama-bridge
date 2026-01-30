
import pytest
from src.bridge.scanner import StreamScanner, ScanEvent

def test_scanner_basic_content():
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    events = scanner.push("Hello world")
    # Buffer might hold "H" if matches "<", but "H" doesn't. 
    # Actually it should flush "Hello world" immediately since no prefix matches.
    assert len(events) == 1
    assert events[0].type == "content"
    assert events[0].data == "Hello world"

def test_scanner_with_partial_tag():
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    # Push "<tho" -> should be buffered
    events = scanner.push("<tho")
    assert len(events) == 0
    assert scanner.buffer == "<tho"
    
    # Push "ught>" -> should trigger block_start
    events = scanner.push("ught>")
    assert len(events) == 1
    assert events[0].type == "block_start"
    assert events[0].tag == "thought"
    assert scanner.buffer == ""

def test_scanner_false_positive():
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    # Push "<th" (prefix)
    scanner.push("<th")
    assert len(scanner.buffer) == 3
    
    # Push "is" -> "<this" is not a prefix of "<thought>"
    # It should flush "<this"
    events = scanner.push("is")
    assert len(events) == 1
    assert events[0].type == "content"
    assert events[0].data == "<this"
    assert scanner.buffer == ""

def test_scanner_embedded_tags():
    scanner = StreamScanner(block_tokens=["<thought>", "</thought>"])
    raw_text = "Analysis: <thought>Working on it...</thought> Done."
    
    accumulated_content = ""
    accumulated_thought = ""
    in_thought = False
    
    for char in raw_text:
        events = scanner.push(char)
        for ev in events:
            if ev.type == "content":
                accumulated_content += ev.data
            elif ev.type == "block_start":
                in_thought = True
            elif ev.type == "block_content":
                if in_thought: accumulated_thought += ev.data
            elif ev.type == "block_end":
                in_thought = False
                
    # Final flush
    events = scanner.flush()
    for ev in events:
        if ev.type == "content": accumulated_content += ev.data
        
    assert accumulated_content == "Analysis:  Done."
    assert accumulated_thought == "Working on it..."

def test_scanner_qwen_tools():
    # Simulate Qwen tool call pattern
    tags = ["<function=", "</function>", "<parameter=", "</parameter>"]
    scanner = StreamScanner(block_tokens=tags)
    
    raw = "I will call a tool: <function=get_weather><parameter=city>Beijing</parameter></function>"
    
    events_log = []
    for char in raw:
        events_log.extend(scanner.push(char))
    events_log.extend(scanner.flush())
    
    # Verify we don't see any part of the tags in "content" events
    for ev in events_log:
        if ev.type == "content":
            assert "<function" not in ev.data
            assert "</function" not in ev.data
    
    # Check sequences
    event_types = [e.type for e in events_log]
    assert "block_start" in event_types
    assert "block_end" in event_types

def test_preventative_buffering_parity():
    """
    The ultimate test: compare cumulative scanner output with a mock full parse.
    """
    raw_output = "Sure! <thought>I should check the time.</thought> The time is 10:30."
    tags = ["<thought>", "</thought>"]
    scanner = StreamScanner(block_tokens=tags)
    
    # Expected results (what a full parser would give)
    # Note: common_chat_parse would return structured content
    # For this test, we verify that we can RECONSTRUCT the original or split it correctly.
    
    streamed_content = ""
    streamed_thought = ""
    in_block = False
    
    for char in raw_output:
        events = scanner.push(char)
        for ev in events:
            if ev.type == "content":
                streamed_content += ev.data
            elif ev.type == "block_start":
                in_block = True
            elif ev.type == "block_content":
                streamed_thought += ev.data
            elif ev.type == "block_end":
                in_block = False
    
    # Final flush
    for ev in scanner.flush():
        if ev.type == "content": streamed_content += ev.data

    assert streamed_content == "Sure!  The time is 10:30."
    assert streamed_thought == "I should check the time."

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
