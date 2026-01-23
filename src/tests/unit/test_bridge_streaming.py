
import asyncio
import pytest
from src.bridge.base import BridgeBase
from src.bridge.flavors import QwenFlavor

class MockWrapper:
    def __init__(self):
        self.tokens = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    def apply_template(self, messages, tools, flag):
        return {"prompt": "mock_prompt", "additional_stops": []}
    
    def init_inference(self, ctx, prompt, max_tokens):
        pass
    
    def get_next_token(self, ctx):
        if not self.tokens:
            return None
        return self.tokens.pop(0).encode("utf-8")
    
    def get_usage(self, ctx):
        return self.usage
    
    def parse_response(self, text, flag):
        return {"content": text, "tool_calls": []}

@pytest.mark.asyncio
async def test_streaming_preventative_buffering():
    # Setup bridge with mock wrapper
    flavor = QwenFlavor()
    bridge = BridgeBase(model_path="qwen", mock=True)
    bridge.flavor = flavor
    bridge.wrapper = MockWrapper()
    
    # We want to see if "<thought>" is buffered correctly when it comes in chunks
    # Sequence of tokens that build up a tag
    bridge.wrapper.tokens = ["Hello ", "<", "thou", "ght>", " I am thinking.", "</", "thought>", " Done."]
    
    req = {"messages": [{"role": "user", "content": "hi"}]}
    
    yielded_chunks = []
    yielded_chunks = []
    async for chunk in bridge._stream_generate(req):
        yielded_chunks.append(chunk)
    
    # Check if any chunk contains partial "<" or "<thou" etc.
    for chunk in yielded_chunks:
        content = chunk.get("content", "")
        if content:
            assert "<" not in content, f"Partial tag leaked: {content}"
            assert "thought" not in content, f"Tag name leaked: {content}"
            assert ">" not in content, f"Tag end leaked: {content}"

    # Reconstruct final content
    full_content = "".join([c.get("content", "") for c in yielded_chunks if "content" in c])
    assert full_content == "Hello  Done."

    # Check if thought was captured
    thoughts = [c.get("thought", "") for c in yielded_chunks if "thought" in c]
    assert "".join(thoughts) == " I am thinking."

if __name__ == "__main__":
    asyncio.run(test_streaming_preventative_buffering())
