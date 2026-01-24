
import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock
from src.bridge import Bridge

def test_wrapper_performance_parameters_parity():
    """
    Verify that Bridge passes performance parameters correctly to LlamaChatWrapper,
    mimicking llama-server CLI arguments behavior.
    """
    # Use a file that exists for check_ready
    model_path = "README.md" 
    n_ctx = 2048
    n_batch = 1024
    n_threads = 12
    flash_attn = True
    
    with patch('llama_chat.LlamaChatWrapper') as MockWrapper:
        # Initialize bridge
        b = Bridge(
            model_path, 
            n_ctx=n_ctx, 
            n_batch=n_batch, 
            n_threads=n_threads, 
            flash_attn=flash_attn,
            mock=False
        )
        
        MockWrapper.assert_called_once()
        args, kwargs = MockWrapper.call_args
        
        # Verify params
        if kwargs:
            assert kwargs.get("n_ctx") == n_ctx
            assert kwargs.get("n_batch") == n_batch
            assert kwargs.get("n_threads") == n_threads
            assert kwargs.get("flash_attn") == flash_attn
        else:
            assert args[1] == n_ctx
            assert args[2] == n_batch
            assert args[4] == n_threads
            assert args[5] == flash_attn

def test_bridge_template_priority():
    """
    Verify that Bridge gives priority to Wrapper's Jinja template (llama-server behavior)
    when Flavor.apply_template returns None.
    """
    with patch('llama_chat.LlamaChatWrapper') as MockWrapper:
        instance = MockWrapper.return_value
        instance.apply_template.return_value = {"prompt": "JINJA_OUTPUT", "additional_stops": []}
        instance.generate.return_value = "response"
        instance.get_usage.return_value = {"prompt_tokens": 0, "completion_tokens": 0}
        instance.parse_response.return_value = {"content": "response", "tool_calls": []}
        
        # Use existing file
        b = Bridge("README.md", mock=False)
        req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}

        import asyncio
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(b._generate(req))
        
        # Verify that wrapper.apply_template was called
        assert instance.apply_template.called
        # The prompt used for generate should be JINJA_OUTPUT
        instance.generate.assert_called()
        call_args = instance.generate.call_args[0]
        assert call_args[1] == 'JINJA_OUTPUT'

if __name__ == "__main__":
    pytest.main([__file__])
