# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

"""
Llama-Bridge: Configuration system for multi-model and multi-cache routing.

Configuration structure:
    Route → Cache → Model

Supports TOML configuration files with:
- Multiple models (future: currently only one)
- Multiple caches per model (independent KV Cache contexts)
- Flexible routing rules (endpoint, model exact match, wildcard)
"""

import fnmatch
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a loaded model."""
    name: str
    path: str
    # n_ctx_train is read from model metadata at load time


@dataclass
class CacheConfig:
    """Configuration for a KV Cache context."""
    name: str
    model: str  # Reference to model name
    n_ctx: int = 0  # 0 = use model's n_ctx_train
    description: str = ""


@dataclass
class RouteConfig:
    """Configuration for a routing rule."""
    cache: str  # Reference to cache name
    endpoint: str | None = None  # Exact endpoint match (highest priority)
    model: str | None = None  # Exact model name match
    match: str | None = None  # Wildcard pattern for model name


@dataclass
class BridgeConfig:
    """Complete bridge configuration."""
    models: dict[str, ModelConfig] = field(default_factory=dict)
    caches: dict[str, CacheConfig] = field(default_factory=dict)
    routes: list[RouteConfig] = field(default_factory=list)
    
    # Server settings (can be overridden by CLI if no config file)
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    
    def get_model_for_cache(self, cache_name: str) -> ModelConfig:
        """Get the model associated with a cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache not found: {cache_name}")
        model = self.models.get(cache.model)
        if not model:
            raise ValueError(f"Model not found: {cache.model} (referenced by cache {cache_name})")
        return model


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def load_config(config_path: str | Path) -> BridgeConfig:
    """
    Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        BridgeConfig object
        
    Raises:
        ConfigurationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    
    return parse_config(raw, source=str(config_path))


def parse_config(raw: dict[str, Any], source: str = "<dict>") -> BridgeConfig:
    """
    Parse raw configuration dictionary into BridgeConfig.
    
    Args:
        raw: Raw configuration dictionary (e.g., from TOML)
        source: Source identifier for error messages
        
    Returns:
        BridgeConfig object
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = BridgeConfig()
    
    # Parse models
    models_raw = raw.get("models", {})
    if not models_raw:
        raise ConfigurationError(f"No models defined in {source}")
    
    for name, model_data in models_raw.items():
        if not isinstance(model_data, dict):
            raise ConfigurationError(f"Invalid model definition for '{name}' in {source}")
        
        path = model_data.get("path")
        if not path:
            raise ConfigurationError(f"Model '{name}' missing 'path' in {source}")
        
        config.models[name] = ModelConfig(name=name, path=path)
    
    # Parse caches
    caches_raw = raw.get("caches", {})
    if not caches_raw:
        raise ConfigurationError(f"No caches defined in {source}")
    
    for name, cache_data in caches_raw.items():
        if not isinstance(cache_data, dict):
            raise ConfigurationError(f"Invalid cache definition for '{name}' in {source}")
        
        model_ref = cache_data.get("model")
        if not model_ref:
            raise ConfigurationError(f"Cache '{name}' missing 'model' reference in {source}")
        
        if model_ref not in config.models:
            raise ConfigurationError(
                f"Cache '{name}' references unknown model '{model_ref}' in {source}"
            )
        
        config.caches[name] = CacheConfig(
            name=name,
            model=model_ref,
            n_ctx=cache_data.get("n_ctx", 0),
            description=cache_data.get("description", ""),
        )
    
    # Parse routes
    routes_raw = raw.get("routes", [])
    if not routes_raw:
        raise ConfigurationError(f"No routes defined in {source}")
    
    for i, route_data in enumerate(routes_raw):
        if not isinstance(route_data, dict):
            raise ConfigurationError(f"Invalid route definition at index {i} in {source}")
        
        cache_ref = route_data.get("cache")
        if not cache_ref:
            raise ConfigurationError(f"Route at index {i} missing 'cache' reference in {source}")
        
        if cache_ref not in config.caches:
            raise ConfigurationError(
                f"Route at index {i} references unknown cache '{cache_ref}' in {source}"
            )
        
        # At least one matching criterion is required
        endpoint = route_data.get("endpoint")
        model = route_data.get("model")
        match = route_data.get("match")
        
        if not any([endpoint, model, match]):
            raise ConfigurationError(
                f"Route at index {i} missing matching criterion "
                f"(need 'endpoint', 'model', or 'match') in {source}"
            )
        
        config.routes.append(RouteConfig(
            cache=cache_ref,
            endpoint=endpoint,
            model=model,
            match=match,
        ))
    
    # Parse server settings
    server_raw = raw.get("server", {})
    if server_raw:
        config.host = server_raw.get("host", config.host)
        config.port = server_raw.get("port", config.port)
        config.debug = server_raw.get("debug", config.debug)
    
    logger.info(f"Loaded configuration from {source}: "
                f"{len(config.models)} model(s), {len(config.caches)} cache(s), "
                f"{len(config.routes)} route(s)")
    
    return config


def create_default_config(model_path: str, n_ctx: int = 0) -> BridgeConfig:
    """
    Create a default configuration when no config file is provided.
    
    This creates a single model, single cache, catch-all route configuration
    equivalent to the legacy CLI behavior.
    
    Args:
        model_path: Path to the model
        n_ctx: Context size (0 = use model default)
        
    Returns:
        BridgeConfig object
    """
    config = BridgeConfig()
    
    config.models["default"] = ModelConfig(name="default", path=model_path)
    config.caches["default"] = CacheConfig(name="default", model="default", n_ctx=n_ctx)
    config.routes.append(RouteConfig(cache="default", match="*"))
    
    logger.info(f"Created default configuration for model: {model_path}")
    
    return config


class Router:
    """
    Routes incoming requests to the appropriate cache based on configuration.
    """
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self._validate_routes()
    
    def _validate_routes(self):
        """Ensure routes are valid and log warnings for common issues."""
        has_catchall = False
        for route in self.config.routes:
            if route.match == "*":
                has_catchall = True
        
        if not has_catchall:
            logger.warning(
                "No catch-all route (match = '*') defined. "
                "Requests not matching any route will fail."
            )
    
    def route(self, endpoint: str, model_name: str | None) -> CacheConfig:
        """
        Find the appropriate cache for a request.
        
        Args:
            endpoint: The API endpoint (e.g., "/v1/messages")
            model_name: The model name from the request payload (may be None)
            
        Returns:
            CacheConfig for the matched route
            
        Raises:
            ConfigurationError: If no route matches
        """
        for route in self.config.routes:
            if self._matches(route, endpoint, model_name):
                cache = self.config.caches[route.cache]
                logger.debug(
                    f"Routed request (endpoint={endpoint}, model={model_name}) "
                    f"to cache '{cache.name}'"
                )
                return cache
        
        raise ConfigurationError(
            f"No route matches request (endpoint={endpoint}, model={model_name})"
        )
    
    def _matches(self, route: RouteConfig, endpoint: str, model_name: str | None) -> bool:
        """Check if a route matches the request."""
        # Priority 1: Endpoint exact match
        if route.endpoint is not None:
            return endpoint == route.endpoint
        
        # Priority 2: Model exact match
        if route.model is not None:
            return model_name == route.model
        
        # Priority 3: Model wildcard match
        if route.match is not None:
            if model_name is None:
                # Wildcard "*" matches even when model_name is None
                return route.match == "*"
            return fnmatch.fnmatch(model_name, route.match)
        
        return False
