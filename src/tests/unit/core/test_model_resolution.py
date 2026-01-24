"""
Unit tests for model path resolution logic.
"""

import pytest
from pathlib import Path
from src.server import resolve_model_path

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
        
        # We need to test resolve_model_path function, but it relies on CWD.
        # So we mock Path.cwd or change directory. changing directory is safer for tmp_path.
        
        import os
        pwd = os.getcwd()
        os.chdir(tmp_path)
        try:
             resolved = resolve_model_path(str(model_path))
             assert resolved.exists()
             assert "models" in str(resolved)
        finally:
            os.chdir(pwd)

    def test_auto_find_gguf_in_directory(self, tmp_path):
        """Should auto-find .gguf file in directory."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "model-q4_k_m.gguf").touch()
        (model_dir / "._model-q4_k_m.gguf").touch()  # macOS metadata file
        
        resolved = resolve_model_path(str(model_dir))
        
        assert resolved.name == "model-q4_k_m.gguf"
    
    def test_prefer_first_shard(self, tmp_path):
        """Should prefer first shard for split models."""
        model_dir = tmp_path / "split_model"
        model_dir.mkdir()
        (model_dir / "model-00001-of-00003.gguf").touch()
        (model_dir / "model-00002-of-00003.gguf").touch()
        (model_dir / "model-00003-of-00003.gguf").touch()
        
        resolved = resolve_model_path(str(model_dir))
        
        assert "-00001-of-" in resolved.name
