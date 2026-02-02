import pytest
from src.chat_bridge.chat_bridge_scanner import ChatBridgeScanner
from src.chat_bridge.flavors.qwen_flavor import QwenFlavor
from src.chat_bridge.flavors.glm_flavor import GLMFlavor
from src.chat_bridge.flavors.gpt_oss_flavor import GPTOSSFlavor

class TestChatBridgeIntegration:

    def test_qwen_tool_call(self):
        """
        Verify Qwen acts as Opaque buffer for <tool_call>.
        """
        flavor = QwenFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        # Simulate stream: Text -> Start Tool -> Content -> End Tool -> More Text
        input_chunks = [
            "I will ", "read ", "the file.\n\n",
            "<tool_", "call>\n",
            '{"name": "Read", ', '"args": {}}\n',
            "</tool_call>",
            "\nDONE"
        ]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        events.extend(list(scanner.flush()))
            
        # Analyze Events
        # 1. Text Deltas
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert "".join(e["text"] for e in text_events) == "I will read the file.\n\n\nDONE"
        
        # 2. Block Logic
        # Qwen tool_call is Opaque.
        # Should see: block_start, block_end (with content). NO block_delta.
        block_starts = [e for e in events if e["type"] == "block_start"]
        block_ends = [e for e in events if e["type"] == "block_end"]
        block_deltas = [e for e in events if e["type"] == "block_delta"]
        
        assert len(block_starts) == 1
        assert block_starts[0]["tag"] == "tool_call"
        
        assert len(block_ends) == 1
        assert block_ends[0]["tag"] == "tool_call"
        assert '{"name": "Read"' in block_ends[0]["content"]
        
        assert len(block_deltas) == 0 # Opaque should NOT stream deltas

    def test_glm_reasoning_and_tool(self):
        """
        Verify GLM streams <think> (Transparent) but buffers <tool_call> (Opaque).
        """
        flavor = GLMFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        # GLM Log-like simulation
        input_chunks = [
            "<thin", "k>",
            "I need ", "to think.\n",
            "</think>\n",
            "<tool_call>",
            "Read<arg_key>file_path</arg_key><arg_value>...</arg_value>",
            "</tool_call>"
        ]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        events.extend(list(scanner.flush()))
            
        # 1. Reasoning (Transparent)
        full_think_content = ""
        think_starts = [e for e in events if e["type"] == "block_start" and e["tag"] == "reasoning_content"]
        think_deltas = [e for e in events if e["type"] == "block_delta" and e["tag"] == "reasoning_content"]
        think_ends = [e for e in events if e["type"] == "block_end" and e["tag"] == "reasoning_content"]
        
        assert len(think_starts) == 1
        assert len(think_ends) == 1
        
        full_think_content = "".join(e["text"] for e in think_deltas)
        assert "I need to think." in full_think_content
        
        # 2. Tool (Opaque)
        tool_starts = [e for e in events if e["type"] == "block_start" and e["tag"] == "tool_call"]
        tool_deltas = [e for e in events if e["type"] == "block_delta" and e["tag"] == "tool_call"]
        tool_ends = [e for e in events if e["type"] == "block_end" and e["tag"] == "tool_call"]
        
        assert len(tool_starts) == 1
        assert len(tool_deltas) == 0 # Opaque!
        assert len(tool_ends) == 1
        assert "Read<arg_key>" in tool_ends[0]["content"] # Full XML content check

    def test_gpt_oss_reasoning(self):
        """
        Verify Harmony protocol handles <|channel|>analysis... as reasoning.
        """
        flavor = GPTOSSFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        # <|channel|>analysis<|message|> is the start pattern logic in HarmonyProtocol
        input_chunks = [
            "<|channel|>analysis<|message|>",
            "Thinking ", "about life.",
            "<|end|>"
        ]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        events.extend(list(scanner.flush())) # Trigger flush to release buffered tail
        starts = [e for e in events if e["type"] == "block_start"]
        ends = [e for e in events if e["type"] == "block_end"]
        deltas = [e for e in events if e["type"] == "block_delta"]
        
        assert len(starts) == 1
        assert starts[0]["tag"] == "reasoning_content"
        
        content = "".join(e["text"] for e in deltas)
        assert content == "Thinking about life."
        
        assert len(ends) == 1

    def test_gpt_oss_tool_call(self):
        """
        Verify Harmony protocol handles complex <|channel|>commentary... as tool_call.
        Format: <|channel|>commentary to=... <|constrain|>...<|message|> {json} <|end|>
        """
        flavor = GPTOSSFlavor()
        scanner = ChatBridgeScanner(flavor)
        
        # Complex start header split across chunks for extra robustness check
        headers_chunk = "<|channel|>commentary to=functions.Read <|constrain|>json"
        
        input_chunks = [
            headers_chunk,
            "<|message|>", # End of Header
            '{"file": "READ', 'ME.md"}',
            "<|end|>"
        ]
        
        events = []
        for chunk in input_chunks:
            events.extend(list(scanner.push(chunk)))
        events.extend(list(scanner.flush()))
        
        starts = [e for e in events if e["type"] == "block_start"]
        ends = [e for e in events if e["type"] == "block_end"]
        
        assert len(starts) == 1
        assert starts[0]["tag"] == "tool_call"
        
        assert len(ends) == 1
        assert '{"file": "README.md"}' in ends[0]["content"]
