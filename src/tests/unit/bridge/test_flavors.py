
import pytest
import json
from src.bridge.flavors import QwenFlavor, MiniMaxFlavor, BaseFlavor

def test_qwen_block_interpretation():
    flavor = QwenFlavor()
    assert "tool_call" in flavor.block_tags
    
    # Test complete block interpretation
    res = flavor.interpret_block_complete("tool_call", "<function=get_now><parameter=city>Shanghai</parameter></function>")
    assert res is not None
    itype, idata = res
    assert itype == "tool_use"
    assert idata[0]["name"] == "get_now"
    assert json.loads(idata[0]["arguments"]) == {"city": "Shanghai"}

def test_minimax_block_interpretation():
    flavor = MiniMaxFlavor()
    assert "minimax:tool_call" in flavor.block_tags
    
    res = flavor.interpret_block_complete("minimax:tool_call", '<invoke name="calc"><parameter name="exp">2+2</parameter></invoke>')
    assert res is not None
    itype, idata = res
    assert itype == "tool_use"
    assert idata[0]["name"] == "calc"

def test_base_flavor_templates():
    flavor = BaseFlavor()
    # Should use default llama.cpp templates
    assert flavor.apply_template([]) is None
    
    # Test thinking interpretation
    itype, idata = flavor.interpret_block_chunk("think", "Contemplating...")
    assert itype == "thought"
    assert idata == "Contemplating..."
