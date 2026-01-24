
import pytest
import asyncio
from unittest.mock import MagicMock
from src.bridge.base import BridgeBase
from src.bridge.flavors import BaseFlavor

# 1. Custom Flavor to define what is "protected"
class MockThinkingFlavor(BaseFlavor):
    @property
    def protected_tags(self):
        return ["<think>", "</think>"]

    def interpret_block_chunk(self, tag, content):
        if tag == "think":
            return ("thought", content)  # Stream thinking
        return None

    def interpret_block_complete(self, tag, content):
        if tag == "think":
            return None # Already streamed
        return None

# 2. Mock Wrapper
class MockWrapper:
    def __init__(self, generated_tokens):
        self.generated_tokens = generated_tokens
        self.token_idx = 0
    
    def apply_template(self, messages, tools, flag):
        # Simply join content for prompt, simulating what happens in reality
        prompt = ""
        for m in messages:
            prompt += m['content']
        return {"prompt": prompt}
    
    def init_inference(self, ctx, prompt, max_tokens):
        pass
    
    def get_next_token(self, ctx):
        if self.token_idx < len(self.generated_tokens):
            token = self.generated_tokens[self.token_idx]
            self.token_idx += 1
            return token.encode("utf-8")
        return None
    
    def get_usage(self, ctx):
        return {"prompt_tokens": 10, "completion_tokens": 10}
        
    def parse_response(self, text, flag):
        return {}

@pytest.mark.asyncio
async def test_prefill_tag_handling():
    """
    Test that if the prompt ENDS with an open tag (Prefill),
    the scanner correctly inherits this state and DOES NOT leak
    the subsequent content as raw text.
    """
    
    # Setup
    bridge = BridgeBase(model_path="test", mock=True)
    bridge.flavor = MockThinkingFlavor()
    
    # 1. Simulate a request with Prefill
    # The user forces the assistant to start with <think>
    request = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>"} # PREFILL!
        ]
    }
    
    # 2. Simulate Model Output
    # IMPORTANT: The model does NOT generate <think> again.
    # It starts directly with the thought content.
    generated_text = ["I ", "am ", "thinking.", "</", "think>", " Done."]
    bridge.wrapper = MockWrapper(generated_text)
    
    # Run stream generate
    chunks = []
    async for chunk in bridge._stream_generate(request):
        chunks.append(chunk)
        
    # Analyze results
    
    # 3. Check for LEAKS
    # If the scanner missed the prefill, it will think "I am thinking." is normal content.
    # If it works, "I am thinking." should be yielded as 'thought' event, NOT 'content'.
    
    leaked_content = ""
    captured_thought = ""
    
    for c in chunks:
        if "content" in c:
            leaked_content += c["content"]
        if "thought" in c:
            captured_thought += c["thought"]
            
    print(f"\nDEBUG: Leaked Content: '{leaked_content}'")
    print(f"DEBUG: Captured Thought: '{captured_thought}'")
    
    # Assertion 1: The thought content should NOT appear in final content
    assert "I am thinking." not in leaked_content, "The thought content leaked into standard output!"
    
    # Assertion 2: The thought content SHOULD be captured as thought (if flavor supports it)
    assert "I am thinking." in captured_thought, "The thought content was not captured correctly."
    
    # Assertion 3: The final text " Done." must appear
    assert " Done." in leaked_content

if __name__ == "__main__":
    asyncio.run(test_prefill_tag_handling())
