import pytest
from src.chat_bridge.flavors.qwen_flavor import QwenFlavor
from src.chat_bridge.flavors.glm_flavor import GLMFlavor
from src.chat_bridge.flavors.gpt_oss_flavor import GPTOSSFlavor

def test_qwen_flavor_tokens():
    flavor = QwenFlavor()
    tokens = flavor.block_tokens
    
    # Expectation: 
    # 1. Start Token for <tool_call> (regex)
    # 2. End Token </tool_call>
    # 3. Tag = "tool_call"
    # 4. Opaque = True
    
    found_tool_call = False
    for t in tokens:
        if t["tag"] == "tool_call":
            found_tool_call = True
            assert "<tool_call" in t["start"]
            assert t["end"] == "</tool_call>"
            assert t["is_opaque"] == True
            assert t["start_is_regex"] == True
            
    assert found_tool_call, "QwenFlavor missing 'tool_call' definition"

def test_glm_flavor_tokens():
    flavor = GLMFlavor()
    tokens = flavor.block_tokens
    
    # Expectation:
    # 1. <tool_call> (Opaque)
    # 2. <think> (Transparent/Reasoning)
    
    tool_found = False
    think_found = False
    
    for t in tokens:
        if t["tag"] == "tool_call":
            tool_found = True
            assert t["is_opaque"] == True
        elif t["tag"] == "reasoning_content":
            think_found = True
            assert t["is_opaque"] == False # Reasoning should stream
            assert "<think" in t["start"]
            assert "</think>" in t["end"]
            
    assert tool_found, "GLMFlavor missing 'tool_call'"
    assert think_found, "GLMFlavor missing 'reasoning_content'"

def test_gpt_oss_flavor_tokens():
    flavor = GPTOSSFlavor()
    tokens = flavor.block_tokens
    
    # Expectation:
    # 1. <|channel|>analysis (Reasoning)
    
    analysis_found = False
    for t in tokens:
        if t["tag"] == "reasoning_content":
            analysis_found = True
            assert "channel" in t["start"]
            assert "analysis" in t["start"]
            assert t["is_opaque"] == False
            
    assert analysis_found, "GPTOSSFlavor missing 'reasoning_content' channel definition"
