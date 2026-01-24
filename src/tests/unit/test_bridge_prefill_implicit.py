
import pytest
import asyncio
from unittest.mock import MagicMock
from src.bridge.base import BridgeBase
from src.bridge.flavors import MiniMaxFlavor

class MockWrapper:
    def __init__(self, *args, **kwargs):
        self.tokens = [b"The", b" user", b" wants"]
        self.token_idx = 0
    
    def apply_template(self, messages, tools, add_gen_prompt):
        # SIMULATE THE ISSUE: Prompt ends with <think> tag
        return {"prompt": "System: ...\nUser: ...\nAssistant:\n<think>\n"}
        
    def init_inference(self, ctx, prompt, max_tokens):
        pass
        
    def get_next_token(self, ctx):
        if self.token_idx < len(self.tokens):
            t = self.tokens[self.token_idx]
            self.token_idx += 1
            return t
        return None # EOS
        
    def get_usage(self, ctx):
        return {"prompt_tokens": 10, "completion_tokens": 3}
    
    def parse_response(self, text, is_final):
        return {"tool_calls": []}

@pytest.fixture
def mock_bridge():
    # Use mock=True to bypass C++ loading
    bridge = BridgeBase(model_path="test-model", debug=True, mock=True)
    bridge.flavor = MiniMaxFlavor()
    bridge.wrapper = MockWrapper()
    
    # Manually setup locks as they are skipped in mock init
    bridge._locks = {'default': asyncio.Lock(), 'test-cache': asyncio.Lock()}
    
    return bridge

@pytest.mark.asyncio
async def test_implicit_prompt_prefill(mock_bridge):
    request = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": True
    }
    
    chunks = []
    # Note: cache_name must match one in _locks
    try:
        async for chunk in mock_bridge._stream_generate(request, "test-cache", "test-id"):
            chunks.append(chunk)
    except Exception as e:
        # If lock fails or other errors
        pytest.fail(f"Stream generation failed: {e}")
        
    thought_chunks = [c for c in chunks if "thought" in c]
    content_chunks = [c for c in chunks if "content" in c and c["content"]]
    
    print(f"Thought chunks: {thought_chunks}")
    print(f"Content chunks: {content_chunks}")
    
    # ISSUE REPRODUCTION:
    # If the bug exists, we will get Content chunks instead of Thought chunks
    if len(content_chunks) > 0 and len(thought_chunks) == 0:
        pytest.fail("Bug Reproduced: Model output treated as content, missed prompt prefill <think>")
        
    assert len(thought_chunks) > 0
    assert len(content_chunks) == 0
