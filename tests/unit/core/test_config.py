"""
Unit tests for the configuration system.
"""

import pytest
import tempfile
from pathlib import Path

from src.config import (
    BridgeConfig,
    CacheConfig,
    ConfigurationError,
    ModelConfig,
    RouteConfig,
    Router,
    create_default_config,
    load_config,
    parse_config,
)


class TestParseConfig:
    """Tests for configuration parsing."""
    
    def test_minimal_valid_config(self):
        """Test parsing a minimal valid configuration."""
        raw = {
            "models": {
                "main": {"path": "models/test.gguf"}
            },
            "caches": {
                "default": {"model": "main"}
            },
            "routes": [
                {"match": "*", "cache": "default"}
            ]
        }
        
        config = parse_config(raw)
        
        assert len(config.models) == 1
        assert config.models["main"].path == "models/test.gguf"
        assert len(config.caches) == 1
        assert config.caches["default"].model == "main"
        assert len(config.routes) == 1
    
    def test_full_config(self):
        """Test parsing a complete configuration with multiple caches."""
        raw = {
            "models": {
                "mimo": {"path": "unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/"}
            },
            "caches": {
                "main": {
                    "model": "mimo",
                    "n_ctx": 262144,
                    "description": "Main conversation"
                },
                "fast": {
                    "model": "mimo",
                    "n_ctx": 16384,
                    "description": "Background tasks"
                }
            },
            "routes": [
                {"endpoint": "/v1/messages/count_tokens", "cache": "fast"},
                {"model": "claude-3-5-haiku-20241022", "cache": "fast"},
                {"match": "*haiku*", "cache": "fast"},
                {"match": "*", "cache": "main"}
            ]
        }
        
        config = parse_config(raw)
        
        assert len(config.models) == 1
        assert len(config.caches) == 2
        assert config.caches["main"].n_ctx == 262144
        assert config.caches["fast"].n_ctx == 16384
        assert len(config.routes) == 4
    
    def test_missing_models_error(self):
        """Test error when no models are defined."""
        raw = {
            "caches": {"default": {"model": "main"}},
            "routes": [{"match": "*", "cache": "default"}]
        }
        
        with pytest.raises(ConfigurationError, match="No models defined"):
            parse_config(raw)
    
    def test_missing_caches_error(self):
        """Test error when no caches are defined."""
        raw = {
            "models": {"main": {"path": "test.gguf"}},
            "routes": [{"match": "*", "cache": "default"}]
        }
        
        with pytest.raises(ConfigurationError, match="No caches defined"):
            parse_config(raw)
    
    def test_missing_routes_error(self):
        """Test error when no routes are defined."""
        raw = {
            "models": {"main": {"path": "test.gguf"}},
            "caches": {"default": {"model": "main"}}
        }
        
        with pytest.raises(ConfigurationError, match="No routes defined"):
            parse_config(raw)
    
    def test_unknown_model_reference_error(self):
        """Test error when cache references unknown model."""
        raw = {
            "models": {"main": {"path": "test.gguf"}},
            "caches": {"default": {"model": "nonexistent"}},
            "routes": [{"match": "*", "cache": "default"}]
        }
        
        with pytest.raises(ConfigurationError, match="unknown model 'nonexistent'"):
            parse_config(raw)
    
    def test_unknown_cache_reference_error(self):
        """Test error when route references unknown cache."""
        raw = {
            "models": {"main": {"path": "test.gguf"}},
            "caches": {"default": {"model": "main"}},
            "routes": [{"match": "*", "cache": "nonexistent"}]
        }
        
        with pytest.raises(ConfigurationError, match="unknown cache 'nonexistent'"):
            parse_config(raw)
    
    def test_route_missing_match_criterion_error(self):
        """Test error when route has no matching criterion."""
        raw = {
            "models": {"main": {"path": "test.gguf"}},
            "caches": {"default": {"model": "main"}},
            "routes": [{"cache": "default"}]  # No endpoint, model, or match
        }
        
        with pytest.raises(ConfigurationError, match="missing matching criterion"):
            parse_config(raw)


class TestLoadConfig:
    """Tests for loading configuration from files."""
    
    def test_load_valid_toml(self, tmp_path):
        """Test loading a valid TOML file."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
[models.main]
path = "models/test.gguf"

[caches.default]
model = "main"
n_ctx = 8192

[[routes]]
match = "*"
cache = "default"
""")
        
        config = load_config(config_file)
        
        assert config.models["main"].path == "models/test.gguf"
        assert config.caches["default"].n_ctx == 8192
    
    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.toml")


class TestCreateDefaultConfig:
    """Tests for default configuration creation."""
    
    def test_create_default(self):
        """Test creating default configuration from model path."""
        config = create_default_config("models/test.gguf", n_ctx=4096)
        
        assert len(config.models) == 1
        assert config.models["default"].path == "models/test.gguf"
        assert len(config.caches) == 1
        assert config.caches["default"].n_ctx == 4096
        assert len(config.routes) == 1
        assert config.routes[0].match == "*"


class TestRouter:
    """Tests for the routing logic."""
    
    @pytest.fixture
    def multi_cache_config(self):
        """Create a configuration with multiple caches and routes."""
        config = BridgeConfig()
        config.models["main"] = ModelConfig(name="main", path="test.gguf")
        config.caches["primary"] = CacheConfig(name="primary", model="main", n_ctx=262144)
        config.caches["background"] = CacheConfig(name="background", model="main", n_ctx=16384)
        
        # Routes in priority order
        config.routes = [
            RouteConfig(cache="background", endpoint="/v1/messages/count_tokens"),
            RouteConfig(cache="background", model="claude-3-5-haiku-20241022"),
            RouteConfig(cache="background", match="*haiku*"),
            RouteConfig(cache="background", match="*small*"),
            RouteConfig(cache="primary", match="*"),
        ]
        
        return config
    
    def test_endpoint_match_highest_priority(self, multi_cache_config):
        """Test that endpoint matching has highest priority."""
        router = Router(multi_cache_config)
        
        # Even if model contains "sonnet", endpoint match wins
        cache = router.route("/v1/messages/count_tokens", "claude-sonnet-4")
        assert cache.name == "background"
    
    def test_model_exact_match(self, multi_cache_config):
        """Test exact model name matching."""
        router = Router(multi_cache_config)
        
        cache = router.route("/v1/messages", "claude-3-5-haiku-20241022")
        assert cache.name == "background"
    
    def test_model_wildcard_match(self, multi_cache_config):
        """Test wildcard model name matching."""
        router = Router(multi_cache_config)
        
        # Matches "*haiku*"
        cache = router.route("/v1/messages", "claude-3-haiku-latest")
        assert cache.name == "background"
        
        # Matches "*small*"
        cache = router.route("/v1/messages", "some-small-model")
        assert cache.name == "background"
    
    def test_catchall_route(self, multi_cache_config):
        """Test catch-all route for unmatched requests."""
        router = Router(multi_cache_config)
        
        cache = router.route("/v1/messages", "claude-sonnet-4")
        assert cache.name == "primary"
    
    def test_no_match_error(self):
        """Test error when no route matches."""
        config = BridgeConfig()
        config.models["main"] = ModelConfig(name="main", path="test.gguf")
        config.caches["only"] = CacheConfig(name="only", model="main")
        config.routes = [
            RouteConfig(cache="only", model="specific-model-only")
        ]
        
        router = Router(config)
        
        with pytest.raises(ConfigurationError, match="No route matches"):
            router.route("/v1/messages", "different-model")
    
    def test_none_model_matches_catchall(self, multi_cache_config):
        """Test that None model name matches catch-all."""
        router = Router(multi_cache_config)
        
        cache = router.route("/v1/messages", None)
        assert cache.name == "primary"
    
    def test_route_order_matters(self):
        """Test that route order is respected (first match wins)."""
        config = BridgeConfig()
        config.models["main"] = ModelConfig(name="main", path="test.gguf")
        config.caches["first"] = CacheConfig(name="first", model="main")
        config.caches["second"] = CacheConfig(name="second", model="main")
        
        # Both routes match "*haiku*", but first one wins
        config.routes = [
            RouteConfig(cache="first", match="*haiku*"),
            RouteConfig(cache="second", match="*haiku*"),
        ]
        
        router = Router(config)
        cache = router.route("/v1/messages", "claude-haiku")
        assert cache.name == "first"
