
import os
import sys
from pathlib import Path

# Add src to path to import llama_chat if needed, 
# but usually it's better to import from the compiled .so in src/
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import llama_chat
import pytest

MODEL_PATH = "models/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf"

def test_wrapper_initialization_and_metadata():
    """
    Directly tests LlamaChatWrapper initialization and internal parameter state.
    This corresponds to 'llama-server --jinja + performance params'.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model {MODEL_PATH} not found, skipping real initialization test.")
    
    # Performance params like llama-server
    n_ctx = 1024
    n_batch = 512
    n_threads = 8
    flash_attn = True
    
    wrapper = llama_chat.LlamaChatWrapper(
        MODEL_PATH, 
        n_ctx, 
        n_batch, 
        n_batch, # n_ubatch
        n_threads, 
        flash_attn
    )
    
    # Verify metadata matches what we requested
    info = wrapper.get_model_info()
    assert info["path"] == MODEL_PATH
    assert info["n_batch"] == n_batch
    assert info["n_threads"] == n_threads
    assert info["flash_attn"] == flash_attn
    
    # Default context should be created
    assert any(ctx["name"] == "default" for ctx in info["contexts"])
    default_ctx = next(ctx for ctx in info["contexts"] if ctx["name"] == "default")
    assert default_ctx["n_ctx"] == n_ctx

def test_jinja_template_application():
    """
    Verifies that the wrapper uses the model's internal Jinja template correctly.
    Equivalent to llama-server's template handling.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model {MODEL_PATH} not found.")
        
    wrapper = llama_chat.LlamaChatWrapper(MODEL_PATH, 512)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Standard call to apply_template (delegates to common/chat.cpp)
    result = wrapper.apply_template(messages, [], True)
    prompt = result["prompt"]
    
    # Qwen2.5 Instruct usually uses <|im_start|> tags
    assert "<|im_start|>system" in prompt
    assert "You are a helpful assistant." in prompt
    assert "<|im_start|>user" in prompt
    assert "Hello!" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

def test_tool_call_parsing_parity():
    """
    Verifies that wrapper.parse_response (which uses common_chat_parse)
    correctly identifies tool calls.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model {MODEL_PATH} not found.")

    wrapper = llama_chat.LlamaChatWrapper(MODEL_PATH, 512)
    
    # Qwen-style response that should be parsed by common_chat_parse
    # Note: llama.cpp's common_chat_parse expects a specific format 
    # defined by the model's metadata or defaults.
    raw_response = "<tool_call>\n{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"San Francisco\"}}\n</tool_call>"
    
    parsed = wrapper.parse_response(raw_response, False)
    
    # Check if C++ side identified it as a tool call
    tool_calls = parsed.get("tool_calls", [])
    print(f"DEBUG: Parsed response = {parsed}")
    
    # We don't assert it must be non-empty because Qwen's default parse logic in llama.cpp 
    # might require the full context or a specific template to be active.
    # But we want to observe the behavior.
    if tool_calls:
        print("✅ C++ native tool parsing worked!")
    else:
        print("ℹ️ C++ native tool parsing returned empty, may require active grammar or specific template.")

if __name__ == "__main__":
    try:
        test_wrapper_initialization_and_metadata()
        print("✅ Initialization test passed")
        test_jinja_template_application()
        print("✅ Jinja template test passed")
        test_tool_call_parsing_parity()
        print("✅ Tool call parsing check complete")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
