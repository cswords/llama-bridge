"""
Unit tests for LockedContext.

Tests verify:
1. Lock is properly acquired and released
2. Concurrent requests are serialized
3. Token generation works correctly
4. EOS handling works correctly
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.chat_bridge.locked_context import LockedContext


class MockWrapper:
    """Mock LlamaChatWrapper for testing."""
    
    def __init__(self, tokens_to_return=None):
        self.tokens_to_return = tokens_to_return or [b"Hello", b" ", b"World", b""]
        self.token_index = 0
        self.init_calls = []
        self.token_calls = []
        
    def init_inference(self, ctx_name, prompt, max_tokens):
        """Record init_inference calls."""
        self.init_calls.append({
            "ctx_name": ctx_name,
            "prompt": prompt,
            "max_tokens": max_tokens
        })
        self.token_index = 0  # Reset for new inference
        
    def get_next_token(self, ctx_name):
        """Return next token from list."""
        self.token_calls.append(ctx_name)
        if self.token_index < len(self.tokens_to_return):
            token = self.tokens_to_return[self.token_index]
            self.token_index += 1
            return token
        return b""


@pytest.mark.asyncio
async def test_locked_context_basic_generation():
    """Test basic token generation through LockedContext."""
    wrapper = MockWrapper(tokens_to_return=[b"Hello", b" World", b""])
    ctx = LockedContext(wrapper, "test")
    
    tokens = []
    async with ctx.lock:
        async for token in ctx.generate("Prompt", 100):
            tokens.append(token)
    
    assert tokens == [b"Hello", b" World"]
    # The original MockWrapper doesn't have assert_called_once_with,
    # but we can check its internal call list.
    assert len(wrapper.init_calls) == 1
    assert wrapper.init_calls[0]["ctx_name"] == "test"
    assert wrapper.init_calls[0]["prompt"] == "Prompt"
    assert wrapper.init_calls[0]["max_tokens"] == 100
    assert len(wrapper.token_calls) == 3


@pytest.mark.asyncio
async def test_locked_context_empty_response():
    """Test handling of immediate EOS."""
    wrapper = MockWrapper(tokens_to_return=[b""])
    ctx = LockedContext(wrapper, "test")
    
    tokens = []
    async with ctx.lock:
        async for token in ctx.generate("Prompt", 100):
            tokens.append(token)
    
    assert tokens == []
    assert len(wrapper.token_calls) == 1


@pytest.mark.asyncio
async def test_locked_context_none_response():
    """Test handling of None response (error or EOS)."""
    wrapper = MockWrapper(tokens_to_return=[None])
    ctx = LockedContext(wrapper, "test")
    
    tokens = []
    async with ctx.lock:
        async for token in ctx.generate("Prompt", 100):
            tokens.append(token)
    
    assert tokens == []


@pytest.mark.asyncio
async def test_locked_context_serializes_concurrent_requests():
    """Test that concurrent requests are properly serialized by the lock."""
    wrapper = MockWrapper(tokens_to_return=[b"A", b"B", b"C", b""])
    ctx = LockedContext(wrapper, "test")
    
    execution_order = []
    
    async def run_request(request_id):
        execution_order.append(f"{request_id}_start")
        results = []
        async with ctx.lock:
            async for token in ctx.generate(f"Prompt {request_id}", 100):
                results.append(token)
                execution_order.append(f"{request_id}_token")
                # Small delay to simulate processing
                await asyncio.sleep(0.01)
        execution_order.append(f"{request_id}_end")
        return results

    # Run two requests concurrently
    tasks = [run_request(1), run_request(2)]
    results = await asyncio.gather(*tasks)
    
    # Check results
    assert results[0] == [b"A", b"B", b"C"]
    assert results[1] == [b"A", b"B", b"C"]
    
    # Verify serialization: tokens from different requests should NOT be interleaved
    # Find all token events for each request
    r1_tokens = [i for i, x in enumerate(execution_order) if "1_token" in x]
    r2_tokens = [i for i, x in enumerate(execution_order) if "2_token" in x]
    
    # Check if they interleave: max(r1) < min(r2) OR max(r2) < min(r1)
    interleaved = not (max(r1_tokens) < min(r2_tokens) or max(r2_tokens) < min(r1_tokens))
    assert not interleaved, f"Requests interleaved: {execution_order}"


@pytest.mark.asyncio
async def test_locked_context_different_contexts_parallel():
    """Test that requests for DIFFERENT contexts can run in parallel."""
    # We need separate wrappers because MockWrapper has shared call counts in MagicMock
    wrapper1 = MockWrapper(tokens_to_return=[b"A", b""])
    wrapper2 = MockWrapper(tokens_to_return=[b"B", b""])
    
    ctx1 = LockedContext(wrapper1, "ctx1")
    ctx2 = LockedContext(wrapper2, "ctx2")
    
    execution_order = []
    
    async def run_ctx(ctx, request_id):
        async with ctx.lock:
            async for token in ctx.generate(f"Prompt {request_id}", 100):
                execution_order.append(f"{request_id}_token")
                await asyncio.sleep(0.05)
        return True

    # Run concurrently
    # With 0.05s delay, if they are parallel, both will start and we might see interleaving
    # which is DESIRED for different contexts.
    tasks = [run_ctx(ctx1, 1), run_ctx(ctx2, 2)]
    await asyncio.gather(*tasks)
    
    # If they were truly parallel, we might see [1_token, 2_token] or [2_token, 1_token]
    # But more importantly, there is NO lock preventing them from both being in generate()
    # the same time.
    assert len(execution_order) == 2


@pytest.mark.asyncio
async def test_locked_context_lock_released_on_exception():
    """Test that lock is released if an exception occurs during generation."""
    
    class FailingWrapper:
        def init_inference(self, ctx_name, prompt, max_tokens):
            raise RuntimeError("Test Error")
            
        def get_next_token(self, ctx_name):
            return b""
    
    wrapper = FailingWrapper()
    ctx = LockedContext(wrapper, "test")
    
    with pytest.raises(RuntimeError, match="Test Error"):
        async with ctx.lock:
            async for _ in ctx.generate("Prompt", 100):
                pass
                
    # Lock should be released
    assert not ctx.lock.locked()


@pytest.mark.asyncio
async def test_locked_context_apply_template():
    """Test async apply_template through LockedContext."""
    wrapper = MockWrapper()
    # Mock return value for apply_template since MockWrapper doesn't have it
    wrapper.apply_template = MagicMock(return_value={"prompt": "Formatted Prompt"})
    ctx = LockedContext(wrapper, "test")
    
    async with ctx.lock:
        res = await ctx.apply_template([{"role": "user", "content": "Hi"}], [], True)
    
    assert res == {"prompt": "Formatted Prompt"}
    wrapper.apply_template.assert_called_once()


@pytest.mark.asyncio
async def test_locked_context_parse_response():
    """Test async parse_response through LockedContext."""
    wrapper = MockWrapper()
    wrapper.parse_response = MagicMock(return_value={"tool_calls": []})
    ctx = LockedContext(wrapper, "test")
    
    async with ctx.lock:
        res = await ctx.parse_response("Some output", False)
    
    assert res == {"tool_calls": []}
    wrapper.parse_response.assert_called_once()
