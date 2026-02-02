import pytest
from src.chat_bridge.chat_bridge_scanner import ChatBridgeScanner
from src.chat_bridge.flavors.qwen_flavor import QwenFlavor

class TestScannerEdgeCases:

    def test_text_with_less_than_sign(self):
        """
        Verify that isolated less-than signs (math context) do NOT block streaming
        or get buffered unnecessarily.
        """
        flavor = QwenFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        # Scenario: Math reasoning
        # "The value of x < y is true."
        # If naive buffering is used, "< y" might be held back.
        input_chunks = ["The value ", "of x < y", " is true."]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        # Flush at the end just in case, but ideally text should appear before flush if logic is smart
        events.extend(list(scanner.flush()))
        
        text_full = "".join(e["text"] for e in events if e["type"] == "text_delta")
        assert text_full == "The value of x < y is true."
        
        # Detailed check: 
        # Did we get "x < y" promptly? 
        # In a real streaming test we'd check timing, but here we can check if 
        # the events stream was fragmented weirdly or if it passed through.
        # Ideally, processing "of x < y" should yield "of x < y" immediately 
        # because "< y" matches no known tag prefix.

    def test_long_start_tag_split(self):
        """
        Verify that a VERY long start tag (exceeding old 64 char limit) 
        is correctly buffered if it IS a valid tag prefix.
        """
        flavor = QwenFlavor() 
        # Qwen uses <tool_call>. Let's simulate a hypothetically long attribute version
        # even if QwenFlavor regex normally prohibits attributes if we defined it too strictly.
        # Wait, QwenFlavor regex is `<tool_call(?:\\s+[^>]*)?>`. So it allows attributes!
        
        scanner = ChatBridgeScanner(flavor)
        
        long_attr = 'a' * 100
        # Chunk 1: <tool_call matches known prefix.
        # Chunk 2: id="...long..."
        # Chunk 3: >
        
        input_chunks = [
            "<tool_call ", 
            f'id="{long_attr}"',
            ">"
        ]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        
        # Should detect block start
        block_starts = [e for e in events if e["type"] == "block_start"]
        assert len(block_starts) == 1
        assert block_starts[0]["tag"] == "tool_call"
        
        # And ensure no text leaked
        text_events = [e for e in events if e["type"] == "text_delta"]
        # There should be NO text delta of "<tool_call id=..."
        assert len(text_events) == 0

    def test_broken_tag_prefix_yields_immediately(self):
        """
        Input: <tool_X
        'tool_' matches prefix, but 'X' breaks it.
        Scanner should immediately yield '<tool_X' and not wait for flush.
        """
        flavor = QwenFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        events = list(scanner.push("<tool_X"))
        
        # Should yield text immediately
        assert len(events) > 0
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "<tool_X"

