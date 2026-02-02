"""
ChatBridge Scanner + Flavor Integration Tests

Based on TESTS_CHATBRIDGE.md specification.
Uses real output samples from `samples/` directory as fixtures.
"""
import pytest
from typing import List, Dict, Any, Generator

from src.chat_bridge.chat_bridge_scanner import ChatBridgeScanner
from src.chat_bridge.flavors.qwen_flavor import QwenFlavor
from src.chat_bridge.flavors.glm_flavor import GLMFlavor
from src.chat_bridge.flavors.gpt_oss_flavor import GPTOSSFlavor


# ============================================================================
# Helper Functions
# ============================================================================

def collect_events(scanner: ChatBridgeScanner, chunks: List[str]) -> List[Dict[str, Any]]:
    """Push all chunks to scanner and collect events, including flush."""
    events = []
    for chunk in chunks:
        for event in scanner.push(chunk):
            events.append(event)
    # Flush remaining buffer
    for event in scanner.flush():
        events.append(event)
    return events


def get_text_content(events: List[Dict[str, Any]]) -> str:
    """Extract all text_delta content from events."""
    return "".join(e.get("text", "") for e in events if e.get("type") == "text_delta")


def get_block_deltas(events: List[Dict[str, Any]], tag: str) -> str:
    """Extract all block_delta content for a specific tag."""
    return "".join(
        e.get("text", "") 
        for e in events 
        if e.get("type") == "block_delta" and e.get("tag") == tag
    )


def has_block(events: List[Dict[str, Any]], tag: str) -> bool:
    """Check if a block_start event for tag exists."""
    return any(e.get("type") == "block_start" and e.get("tag") == tag for e in events)


def get_block_end_content(events: List[Dict[str, Any]], tag: str) -> str:
    """Get content from block_end event (for opaque blocks)."""
    for e in events:
        if e.get("type") == "block_end" and e.get("tag") == tag:
            return e.get("content", "")
    return ""


# ============================================================================
# 1. Scanner & Flavor Joint Tests
# ============================================================================

class TestScannerQwenFlavor:
    """Tests for QwenFlavor (Qwen 2.5/3 models)."""
    
    @pytest.fixture
    def flavor(self):
        return QwenFlavor()
    
    @pytest.fixture
    def scanner(self, flavor):
        return ChatBridgeScanner(flavor)

    # --- 1.1 Plain Content Streaming ---
    def test_plain_content_streaming(self, scanner):
        """Verify plain text flows through without blocking."""
        chunks = ["你好", "，", "世界"]
        events = collect_events(scanner, chunks)
        
        text = get_text_content(events)
        assert text == "你好，世界"
        assert not has_block(events, "tool_call")
    
    # --- 1.3 Opaque Block Buffering (Tool Call) ---
    def test_opaque_tool_call_buffering_qwen(self, scanner):
        """Verify tool_call block is buffered (opaque), not streamed.
        
        Real sample: samples/qwen/20260131_233850_477363_main/8_raw_output.txt
        """
        # Real Qwen output format
        chunks = [
            "I'll read the README.md file.\n\n",
            "<tool_call>",
            '{"name": "Read", "arguments": {"file_path": "README.md"}}',
            "</tool_call>"
        ]
        events = collect_events(scanner, chunks)
        
        # Pre-tool text should be yielded
        text = get_text_content(events)
        assert "I'll read the README.md file." in text
        
        # Tool block should exist and be opaque (content in block_end)
        assert has_block(events, "tool_call")
        tool_content = get_block_end_content(events, "tool_call")
        assert '"name": "Read"' in tool_content
    
    # --- 1.4 Fragmentation Robustness ---
    def test_fragmented_tag_detection(self, scanner):
        """Verify scanner handles extremely fragmented tag names."""
        # Extreme fragmentation: "<tool_call>" split across many chunks
        chunks = ["Some text ", "<", "tool", "_", "call", ">", 
                  "content", "<", "/tool_call", ">"]
        events = collect_events(scanner, chunks)
        
        assert has_block(events, "tool_call")
        text = get_text_content(events)
        assert "Some text " in text


class TestScannerGLMFlavor:
    """Tests for GLMFlavor (GLM-4 models)."""
    
    @pytest.fixture
    def flavor(self):
        return GLMFlavor()
    
    @pytest.fixture
    def scanner(self, flavor):
        return ChatBridgeScanner(flavor)

    # --- 1.2 Reasoning Content (Thinking) ---
    def test_reasoning_streaming_glm(self, scanner):
        """Verify <think> block streams (transparent) as reasoning.
        
        Real sample: samples/glm/20260131_234854_670060_main/8_raw_output.txt
        The GLM output starts with implicit think content.
        """
        # Simulated GLM format with explicit think tags
        chunks = [
            "<think>",
            "The user wants me to read README",
            "</think>",
            "OK, I'll read it."
        ]
        events = collect_events(scanner, chunks)
        
        # Think block should exist
        assert has_block(events, "reasoning_content")
        
        # Reasoning content should be in block_deltas (transparent)
        reasoning = get_block_deltas(events, "reasoning_content")
        assert "read README" in reasoning
        
        # Post-think text should appear
        text = get_text_content(events)
        assert "OK, I'll read it." in text
    
    # --- 1.4 Mixed Blocks ---
    def test_mixed_reasoning_and_tool_glm(self, scanner):
        """Verify think + tool_call mixed output.
        
        Real sample (modified): GLM often outputs think then tool_call inline.
        """
        # Real GLM pattern: think block, then tool_call inline
        raw = "<think>The user wants README.</think><tool_call>Read</tool_call>"
        chunks = [raw]
        events = collect_events(scanner, chunks)
        
        # Should have both blocks
        assert has_block(events, "reasoning_content")
        assert has_block(events, "tool_call")


class TestScannerGPTOSSFlavor:
    """Tests for GPTOSSFlavor (Harmony Protocol)."""
    
    @pytest.fixture
    def flavor(self):
        return GPTOSSFlavor()
    
    @pytest.fixture
    def scanner(self, flavor):
        return ChatBridgeScanner(flavor)

    def test_plain_content_with_harmony(self, scanner):
        """Harmony flavor should pass through plain text."""
        chunks = ["Hello, ", "world!"]
        events = collect_events(scanner, chunks)
        
        text = get_text_content(events)
        assert text == "Hello, world!"

    def test_gpt_oss_analysis_channel(self, scanner):
        """Test analysis channel for reasoning content.
        
        Source: samples/gpt-oss/20260131_234701_875698_main/8_raw_output.txt
        Format: <|channel|>analysis<|message|>...<|end|>
        """
        # Real GPT-OSS analysis output pattern
        chunks = [
            "<|channel|>analysis<|message|>",
            "We need to read README.md. Use Read tool.",
            "<|end|>"
        ]
        events = collect_events(scanner, chunks)
        
        # Analysis should be captured as reasoning_content
        assert has_block(events, "reasoning_content")
        reasoning = get_block_deltas(events, "reasoning_content")
        assert "read README.md" in reasoning

    def test_gpt_oss_commentary_tool_call(self, scanner):
        """Test commentary channel for tool calls.
        
        Source: samples/gpt-oss/20260131_234701_875698_main/8_raw_output.txt
        Format: <|channel|>commentary to=functions.ToolName <|constrain|>json<|message|>{...}
        """
        # Real GPT-OSS tool call pattern
        tool_json = '{\n  "file_path": "/path/to/file"\n}'
        chunks = [
            "<|channel|>commentary to=functions.Read <|constrain|>json<|message|>",
            tool_json
        ]
        events = collect_events(scanner, chunks)
        
        # Tool call should be captured
        assert has_block(events, "tool_call")
        tool_content = get_block_end_content(events, "tool_call")
        assert "file_path" in tool_content

    def test_gpt_oss_combined_analysis_and_tool(self, scanner):
        """Test combined analysis + tool call flow (real output pattern)."""
        # Full real output pattern from samples
        full_output = (
            '<|channel|>analysis<|message|>We need to read README.md.<|end|>'
            '<|start|>assistant<|channel|>commentary to=functions.Read <|constrain|>json<|message|>{\n'
            '  "file_path": "/path/to/README.md"\n'
            '}'
        )
        chunks = [full_output[i:i+40] for i in range(0, len(full_output), 40)]
        events = collect_events(scanner, chunks)
        
        # Both blocks should be detected
        assert has_block(events, "reasoning_content")
        assert has_block(events, "tool_call")


class TestScannerMiniMaxFlavor:
    """Tests for MiniMaxFlavor (namespaced tool_call tags)."""
    
    @pytest.fixture
    def flavor(self):
        from src.chat_bridge.flavors.minimax_flavor import MiniMaxFlavor
        return MiniMaxFlavor()
    
    @pytest.fixture
    def scanner(self, flavor):
        return ChatBridgeScanner(flavor)

    def test_minimax_tool_call(self, scanner):
        """Test MiniMax namespaced tool call tag.
        
        Source: samples/minimax/20260131_222906_307641_main/8_raw_output.txt
        Format: <minimax:tool_call><invoke name="...">...</invoke></minimax:tool_call>
        """
        # Real MiniMax output pattern
        chunks = [
            "Let me read the file.",
            "</think>\n\n",
            "<minimax:tool_call>",
            '<invoke name="Read">',
            '<parameter name="file_path">/path/to/file</parameter>',
            '</invoke>',
            '</minimax:tool_call>'
        ]
        events = collect_events(scanner, chunks)
        
        # Tool call should be captured
        assert has_block(events, "tool_call")
        tool_content = get_block_end_content(events, "tool_call")
        assert "invoke" in tool_content
        assert "Read" in tool_content


class TestScannerMimoFlavor:
    """Tests for MimoFlavor (function=Name syntax)."""
    
    @pytest.fixture
    def flavor(self):
        from src.chat_bridge.flavors.mimo_flavor import MimoFlavor
        return MimoFlavor()
    
    @pytest.fixture
    def scanner(self, flavor):
        return ChatBridgeScanner(flavor)

    def test_mimo_tool_call(self, scanner):
        """Test Mimo special function=Name tool call format.
        
        Source: samples/mimo/20260131_234416_989932_main/8_raw_output.txt
        Format: <tool_call><function=Name><parameter=key>value</parameter></function></tool_call>
        """
        # Real Mimo output pattern
        chunks = [
            "<tool_call>",
            "<function=Read>",
            "<parameter=file_path>/path/to/file</parameter>",
            "</function>",
            "</tool_call>"
        ]
        events = collect_events(scanner, chunks)
        
        # Tool call should be captured (standard tag wrapper)
        assert has_block(events, "tool_call")
        tool_content = get_block_end_content(events, "tool_call")
        assert "function=Read" in tool_content
        assert "file_path" in tool_content


# ============================================================================
# 2. Real Sample Data Tests
# ============================================================================

class TestRealSampleQwen:
    """Tests using actual Qwen model output samples."""
    
    @pytest.fixture
    def scanner(self):
        return ChatBridgeScanner(QwenFlavor())
    
    def test_qwen_real_output(self, scanner):
        """Test with real Qwen output from samples directory.
        
        Source: samples/qwen/20260131_233850_477363_main/8_raw_output.txt
        """
        real_output = """I'll read the README.md file to understand what this project is about.

<tool_call>
{"name": "Read", "arguments": {"file_path": "/Volumes/970+/Llama-Bridge/README.md"}}
</tool_call>"""
        
        # Simulate token-by-token streaming (realistic chunk sizes)
        chunks = [real_output[i:i+20] for i in range(0, len(real_output), 20)]
        events = collect_events(scanner, chunks)
        
        # Verify structure
        assert has_block(events, "tool_call")
        
        # Verify text before tool call is captured
        text = get_text_content(events)
        assert "I'll read the README.md" in text
        
        # Verify tool content is buffered (opaque)
        tool_content = get_block_end_content(events, "tool_call")
        assert "Read" in tool_content
        assert "file_path" in tool_content


class TestRealSampleGLM:
    """Tests using actual GLM model output samples."""
    
    @pytest.fixture
    def scanner(self):
        return ChatBridgeScanner(GLMFlavor())
    
    def test_glm_real_output(self, scanner):
        """Test with real GLM output from samples directory.
        
        Source: samples/glm/20260131_234854_670060_main/8_raw_output.txt
        Note: GLM output has </think> mid-sentence, then inline tool_call.
        """
        real_output = (
            "The user wants me to read the README.md file and provide a brief summary of the project."
            "</think>"
            "<tool_call>Read<arg_key>file_path</arg_key><arg_value>/Volumes/970+/Llama-Bridge/README.md</arg_value></tool_call>"
        )
        
        chunks = [real_output[i:i+30] for i in range(0, len(real_output), 30)]
        events = collect_events(scanner, chunks)
        
        # GLM uses non-standard tool_call format: <tool_call>Name<arg_key>...</tool_call>
        # Our XMLTagProtocol will capture this as tool_call block
        assert has_block(events, "tool_call")


# ============================================================================
# 3. Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case and stress tests."""
    
    def test_empty_input(self):
        """Scanner should handle empty input gracefully."""
        scanner = ChatBridgeScanner(QwenFlavor())
        events = collect_events(scanner, [])
        assert events == []
    
    def test_only_whitespace(self):
        """Whitespace should be passed through."""
        scanner = ChatBridgeScanner(QwenFlavor())
        events = collect_events(scanner, ["   ", "\n\n", "  "])
        text = get_text_content(events)
        assert text == "   \n\n  "
    
    def test_incomplete_tag_at_end(self):
        """Incomplete tag at stream end should be flushed as text."""
        scanner = ChatBridgeScanner(QwenFlavor())
        chunks = ["Some text <tool_ca"]  # Incomplete tag
        events = collect_events(scanner, chunks)
        
        text = get_text_content(events)
        assert "Some text" in text
        assert "<tool_ca" in text  # Flushed as text
    
    def test_nested_angle_brackets(self):
        """Angle brackets in content should not confuse scanner."""
        scanner = ChatBridgeScanner(QwenFlavor())
        chunks = ["Code: if (x < y && y > z) { return true; }"]
        events = collect_events(scanner, chunks)
        
        text = get_text_content(events)
        assert "x < y" in text


# ============================================================================
# 4. Run Info
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
