"""
FastAPI server with multi-format API endpoints.
Supports Anthropic, OpenAI, and other LiteLLM-compatible formats.
"""

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi.responses import StreamingResponse, JSONResponse
from .exceptions import ContextLimitExceededError

# --- Environment Configuration ---
if sys.platform == "darwin":
    # Ensure Metal shader cache is persistent for faster startup
    if "GGML_METAL_SHADER_CACHE_DIR" not in os.environ:
        import os
        cache_dir = Path.home() / ".cache" / "llama_bridge" / "metal"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["GGML_METAL_SHADER_CACHE_DIR"] = str(cache_dir)
        except Exception:
            pass # Fallback to default (tmp) if permissions fail

from .bridge import Bridge
from .config import (
    BridgeConfig,
    ConfigurationError,
    Router,
    create_default_config,
    load_config,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Llama-Bridge starting up...")
    yield
    logger.info("Llama-Bridge shutting down...")


app = FastAPI(
    title="Llama-Bridge",
    version="0.1.0",
    lifespan=lifespan,
)

@app.exception_handler(ContextLimitExceededError)
async def context_limit_exception_handler(request: Request, exc: ContextLimitExceededError):
    """Handle context limit exceptions with protocol-specific errors."""
    logger.warning(f"Context limit exceeded: {exc}")
    
    # Determine protocol based on URL
    if "/v1/messages" in request.url.path:
        # Anthropic Error Format
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(exc)
                }
            }
        )
    else:
        # OpenAI Error Format (default)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded"
                }
            }
        )

# Global state (initialized in main)
bridge: Bridge | None = None
router: Router | None = None
config: BridgeConfig | None = None


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    """Anthropic Messages API endpoint."""
    if bridge is None:
        return Response(content='{"error": "Bridge not initialized"}', status_code=503)
    
    body = await request.json()
    
    # Route the request to appropriate cache
    cache_name = None
    if router:
        try:
            model_name = body.get("model")
            cache_config = router.route("/v1/messages", model_name)
            cache_name = cache_config.name
        except ConfigurationError as e:
            logger.warning(f"Routing failed: {e}")
    
    # Check if streaming is requested
    stream = body.get("stream", False)
    
    if stream:
        return StreamingResponse(
            bridge.stream_anthropic(body, cache_name=cache_name),
            media_type="text/event-stream",
        )
    else:
        result = await bridge.complete_anthropic(body, cache_name=cache_name)
        return Response(
            content=result,
            media_type="application/json",
        )


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    """OpenAI Chat Completions API endpoint."""
    if bridge is None:
        return Response(content='{"error": "Bridge not initialized"}', status_code=503)
    
    body = await request.json()
    
    # Route the request to appropriate cache
    cache_name = None
    if router:
        try:
            model_name = body.get("model")
            cache_config = router.route("/v1/chat/completions", model_name)
            cache_name = cache_config.name
        except ConfigurationError as e:
            logger.warning(f"Routing failed: {e}")
    
    # Check if streaming is requested
    stream = body.get("stream", False)
    
    if stream:
        return StreamingResponse(
            bridge.stream_openai(body, cache_name=cache_name),
            media_type="text/event-stream",
        )
    else:
        result = await bridge.complete_openai(body, cache_name=cache_name)
        return Response(
            content=result,
            media_type="application/json",
        )


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request) -> Response:
    """Estimated token count for Anthropic protocol."""
    # Route to background cache if configured
    cache_name = None
    if router:
        try:
            body_peek = await request.json()
            model_name = body_peek.get("model")
            cache_config = router.route("/v1/messages/count_tokens", model_name)
            cache_name = cache_config.name
            logger.debug(f"Token count request routed to cache: {cache_name}")
        except Exception:
            pass
    
    try:
        body = await request.json()
        text = str(body.get("messages", "")) + str(body.get("system", ""))
        # Rough estimate: 4 chars per token
        tokens = len(text) // 4
        return Response(
            content=f'{{"input_tokens": {tokens}}}',
            media_type="application/json"
        )
    except:
        return Response(content='{"input_tokens": 0}', media_type="application/json")


@app.post("/api/event_logging/batch")
async def event_logging_batch(request: Request) -> Response:
    """Dummy endpoint for Claude Code telemetry logs to avoid 404 noise."""
    try:
        await request.json()
    except:
        pass
    return Response(status_code=200)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    model_info = {}
    usage = {}
    model_loaded = False
    
    if bridge:
        if bridge.mock:
            model_loaded = True
        elif bridge.wrapper:
            model_loaded = True
            model_info = bridge.wrapper.get_model_info()
            usage = bridge.wrapper.get_usage()
    
    # Include routing info
    routing_info = {}
    if config:
        routing_info = {
            "models": list(config.models.keys()),
            "caches": list(config.caches.keys()),
            "routes": len(config.routes),
        }
            
    return {
        "status": "ok", 
        "model_loaded": model_loaded,
        "model_info": model_info,
        "usage": usage,
        "routing": routing_info,
    }


def resolve_model_path(path_str: str) -> Path:
    """
    Resolve model path with smart lookup.
    
    Tries:
    1. As-is (absolute or relative to CWD)
    2. Relative to models/ directory
    3. Auto-select .gguf file if path is a directory
    """
    model_path = Path(path_str)
    
    # 1. If not absolute and doesn't exist, try relative to models/
    if not model_path.is_absolute() and not model_path.exists():
        if not path_str.startswith("models/"):
            candidate = Path.cwd() / "models" / path_str
            if candidate.exists():
                model_path = candidate
    
    # 2. If it's a directory, find the .gguf file(s)
    if model_path.is_dir():
        # Filter out macOS metadata files (._*)
        gguf_files = sorted([f for f in model_path.glob("*.gguf") if not f.name.startswith("._")])
        if gguf_files:
            # Prefer the first split shard (-00001-of-) or simply the first file
            first_shard = [f for f in gguf_files if "-00001-of-" in f.name]
            model_path = first_shard[0] if first_shard else gguf_files[0]
            logger.info(f"Auto-selected model file: {model_path.name}")
        else:
            raise FileNotFoundError(f"No .gguf files found in directory: {model_path}")
    
    return model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Llama-Bridge Server")
    
    # Configuration file (takes precedence over other model-related args)
    parser.add_argument("--config", type=str, help="Path to TOML configuration file")
    
    # Legacy arguments (ignored when --config is provided)
    parser.add_argument("--model", type=str, help="Path to GGUF model (ignored if --config is set)")
    parser.add_argument("--n-ctx", type=int, default=0, help="Context size (ignored if --config is set)")
    parser.add_argument("--n-batch", type=int, default=0, help="Batch size (default: 0 = auto)")
    parser.add_argument("--n-ubatch", type=int, default=0, help="Physical batch size (default: 0 = auto)")
    parser.add_argument("--n-threads", type=int, default=0, help="Number of threads (default: 0 = auto)")
    parser.add_argument("--flash-attn", action="store_true", help="Enable Flash Attention")
    
    # Server arguments (always respected)
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and 4-file logs")
    parser.add_argument("--mock", action="store_true", help="Use mock inference (for testing)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load or create configuration
    global config, router
    
    if args.config:
        # Load from configuration file
        try:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
            if args.model:
                logger.warning("--model argument ignored (using configuration file)")
            if args.n_ctx:
                logger.warning("--n-ctx argument ignored (using configuration file)")
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            sys.exit(1)
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
    else:
        # Legacy mode: create default config from --model
        if not args.model and not args.mock:
            logger.error("Either --config or --model must be specified")
            sys.exit(1)
        
        if args.model:
            config = create_default_config(args.model, n_ctx=args.n_ctx)
        else:
            # Mock mode without model
            config = create_default_config("mock", n_ctx=args.n_ctx)
    
    # Create router
    router = Router(config)
    
    # Resolve model paths and initialize bridge
    # For now, we only support a single model (first one in config)
    global bridge
    
    if args.mock:
        # Mock mode
        bridge = Bridge(
            model_path="mock",
            debug=args.debug,
            mock=True,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
            n_threads=args.n_threads,
            flash_attn=args.flash_attn,
        )
    else:
        # Get the first (and currently only) model
        model_config = list(config.models.values())[0]
        
        try:
            model_path = resolve_model_path(model_config.path)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            sys.exit(1)
        
        # Get cache configurations for this model
        cache_configs = [c for c in config.caches.values() if c.model == model_config.name]
        
        # For now, use the first cache's n_ctx (TODO: support multiple contexts)
        # In future, Bridge will create multiple llama_context instances
        primary_cache = cache_configs[0] if cache_configs else None
        n_ctx = primary_cache.n_ctx if primary_cache and primary_cache.n_ctx > 0 else args.n_ctx
        
        try:
            bridge = Bridge(
                model_path=str(model_path),
                debug=args.debug,
                mock=False,
                n_ctx=n_ctx,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                n_threads=args.n_threads,
                flash_attn=args.flash_attn,
                # Pass cache configs for multi-context support
                cache_configs=[
                    {"name": c.name, "n_ctx": c.n_ctx}
                    for c in cache_configs
                ],
            )
            
            logger.info(f"Model: {model_path}")
            logger.info(f"Caches configured: {[c.name for c in cache_configs]}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize Bridge: {e}")
            logger.error("Suggestion: If this is an OOM issue, try a smaller n_ctx in your config.")
            logger.error("Also, ensure you have run 'make build' to update the C++ bindings.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
