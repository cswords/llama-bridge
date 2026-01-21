"""
Integration tests for multi-context (multi-cache) support.

These tests verify that the configuration system correctly routes
requests to different contexts based on model name patterns.
"""

import pytest
from src.config import (
    BridgeConfig,
    CacheConfig,
    ConfigurationError,
    ModelConfig,
    RouteConfig,
    Router,
    load_config,
    parse_config,
)


class TestMultiContextRouting:
    """Tests for multi-context routing scenarios."""
    
    @pytest.fixture
    def claude_code_config(self):
        """Create a config mimicking Claude Code usage patterns."""
        return parse_config({
            "models": {
                "mimo": {"path": "test/model.gguf"}
            },
            "caches": {
                "main": {"model": "mimo", "n_ctx": 262144, "description": "Main conversation"},
                "fast": {"model": "mimo", "n_ctx": 16384, "description": "Background tasks"}
            },
            "routes": [
                {"endpoint": "/v1/messages/count_tokens", "cache": "fast"},
                {"match": "*haiku*", "cache": "fast"},
                {"match": "*small*", "cache": "fast"},
                {"match": "*", "cache": "main"}
            ]
        })
    
    def test_main_conversation_routes_to_main(self, claude_code_config):
        """Main Claude conversation should use the main cache."""
        router = Router(claude_code_config)
        
        # Claude Code's primary Sonnet model
        cache = router.route("/v1/messages", "claude-sonnet-4-5-20250927")
        assert cache.name == "main"
        assert cache.n_ctx == 262144
    
    def test_haiku_routes_to_fast(self, claude_code_config):
        """Haiku model requests should use the fast cache."""
        router = Router(claude_code_config)
        
        # Various Haiku model names
        for model in [
            "claude-3-5-haiku-20241022",
            "claude-haiku-latest",
            "anthropic.claude-haiku",
        ]:
            cache = router.route("/v1/messages", model)
            assert cache.name == "fast", f"Model {model} should route to 'fast'"
    
    def test_small_model_routes_to_fast(self, claude_code_config):
        """Small/fast model requests should use the fast cache."""
        router = Router(claude_code_config)
        
        # Claude Code's ANTHROPIC_SMALL_FAST_MODEL
        cache = router.route("/v1/messages", "claude-small-fast")
        assert cache.name == "fast"
    
    def test_token_count_endpoint_routes_to_fast(self, claude_code_config):
        """Token counting endpoint should use fast cache regardless of model."""
        router = Router(claude_code_config)
        
        # Even if model is Sonnet, endpoint takes priority
        cache = router.route("/v1/messages/count_tokens", "claude-sonnet-4")
        assert cache.name == "fast"
    
    def test_cache_n_ctx_values(self, claude_code_config):
        """Verify cache n_ctx values are correctly parsed."""
        assert claude_code_config.caches["main"].n_ctx == 262144
        assert claude_code_config.caches["fast"].n_ctx == 16384


class TestConfigFileLoading:
    """Tests for loading configuration from actual TOML files."""
    
    def test_load_claude_code_config(self):
        """Test loading the Claude Code example config."""
        try:
            config = load_config("configs/claude-code.toml")
            
            # Verify structure
            assert "mimo" in config.models
            assert "main" in config.caches
            assert "fast" in config.caches
            assert len(config.routes) >= 3
            
            # Verify routing
            router = Router(config)
            
            # Haiku -> fast
            cache = router.route("/v1/messages", "claude-haiku")
            assert cache.name == "fast"
            
            # Default -> main
            cache = router.route("/v1/messages", "claude-sonnet")
            assert cache.name == "main"
            
        except FileNotFoundError:
            pytest.skip("configs/claude-code.toml not found")


class TestMultiCacheIsolation:
    """Tests verifying that caches provide proper isolation."""
    
    @pytest.fixture
    def isolation_config(self):
        """Create a config for testing cache isolation."""
        return parse_config({
            "models": {
                "model": {"path": "test.gguf"}
            },
            "caches": {
                "cache_a": {"model": "model", "n_ctx": 8192},
                "cache_b": {"model": "model", "n_ctx": 4096}
            },
            "routes": [
                {"match": "model-a*", "cache": "cache_a"},
                {"match": "model-b*", "cache": "cache_b"},
                {"match": "*", "cache": "cache_a"}
            ]
        })
    
    def test_different_models_route_to_different_caches(self, isolation_config):
        """Different model prefixes should route to different caches."""
        router = Router(isolation_config)
        
        cache_a = router.route("/v1/messages", "model-a-v1")
        cache_b = router.route("/v1/messages", "model-b-v2")
        
        assert cache_a.name == "cache_a"
        assert cache_b.name == "cache_b"
        assert cache_a.name != cache_b.name
    
    def test_caches_have_independent_n_ctx(self, isolation_config):
        """Each cache should have its own n_ctx setting."""
        assert isolation_config.caches["cache_a"].n_ctx == 8192
        assert isolation_config.caches["cache_b"].n_ctx == 4096
