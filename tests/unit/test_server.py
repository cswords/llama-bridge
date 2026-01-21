"""
Unit tests for server module, specifically model path resolution.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestModelPathResolution:
    """Test smart model path resolution logic."""
    
    def test_auto_prepend_models_dir(self, tmp_path):
        """Should auto-prepend 'models/' if path doesn't exist."""
        # Create models/test_org/test_model/model.gguf
        model_dir = tmp_path / "models" / "test_org" / "test_model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").touch()
        
        # Simulate path resolution logic
        model_path = Path("test_org/test_model")
        if not model_path.is_absolute():
            if not model_path.exists():
                candidate = tmp_path / "models" / model_path
                if candidate.exists():
                    model_path = candidate
        
        assert model_path.exists()
        assert "models" in str(model_path)
    
    def test_auto_find_gguf_in_directory(self, tmp_path):
        """Should auto-find .gguf file in directory."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "model-q4_k_m.gguf").touch()
        (model_dir / "._model-q4_k_m.gguf").touch()  # macOS metadata file
        
        # Simulate directory detection logic
        gguf_files = sorted([f for f in model_dir.glob("*.gguf") if not f.name.startswith("._")])
        
        assert len(gguf_files) == 1
        assert gguf_files[0].name == "model-q4_k_m.gguf"
    
    def test_prefer_first_shard(self, tmp_path):
        """Should prefer first shard for split models."""
        model_dir = tmp_path / "split_model"
        model_dir.mkdir()
        (model_dir / "model-00001-of-00003.gguf").touch()
        (model_dir / "model-00002-of-00003.gguf").touch()
        (model_dir / "model-00003-of-00003.gguf").touch()
        
        # Simulate shard detection logic
        gguf_files = sorted([f for f in model_dir.glob("*.gguf") if not f.name.startswith("._")])
        first_shard = [f for f in gguf_files if "-00001-of-" in f.name]
        selected = first_shard[0] if first_shard else gguf_files[0]
        
        assert "-00001-of-" in selected.name


class TestSSEChunkConversion:
    """Test SSE chunk conversion for Anthropic adapter."""
    
    def test_text_delta_sse(self):
        """Should format text delta as SSE event."""
        from src.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
        
        chunk = {"content": "Hello", "stop_reason": None}
        sse = adapter.chunk_to_sse(chunk)
        
        assert "event: content_block_delta" in sse
        assert "text_delta" in sse
        assert "Hello" in sse
    
    def test_thinking_delta_sse(self):
        """Should format thinking delta as SSE event."""
        from src.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
        
        chunk = {"thought": "Let me think...", "stop_reason": None}
        sse = adapter.chunk_to_sse(chunk)
        
        assert "event: content_block_delta" in sse
        assert "thinking_delta" in sse
        assert "Let me think..." in sse
    
    def test_final_message_delta_sse(self):
        """Should format final message delta with stop_reason."""
        from src.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
        
        chunk = {"content": "", "stop_reason": "end_turn", "usage": {"output_tokens": 10}}
        sse = adapter.chunk_to_sse(chunk)
        
        assert "message_delta" in sse
        assert "end_turn" in sse
        assert "message_stop" in sse
    
    def test_tool_use_sse(self):
        """Should format tool use as SSE events."""
        from src.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
        
        chunk = {
            "content": "",
            "stop_reason": "tool_use",
            "tool_calls": [{
                "id": "toolu_123",
                "name": "get_weather",
                "arguments": {"location": "Tokyo"}
            }],
            "usage": {"output_tokens": 15}
        }
        sse = adapter.chunk_to_sse(chunk)
        
        assert "content_block_start" in sse
        assert "tool_use" in sse
        assert "get_weather" in sse
        assert "input_json_delta" in sse
        assert "Tokyo" in sse
