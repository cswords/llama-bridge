"""
FastAPI server with multi-format API endpoints.
Supports Anthropic, OpenAI, and other LiteLLM-compatible formats.
"""

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from .bridge import Bridge

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

# Global bridge instance (initialized in main)
bridge: Bridge | None = None


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    """Anthropic Messages API endpoint."""
    if bridge is None:
        return Response(content='{"error": "Bridge not initialized"}', status_code=503)
    
    body = await request.json()
    
    # Check if streaming is requested
    stream = body.get("stream", False)
    
    if stream:
        return StreamingResponse(
            bridge.stream_anthropic(body),
            media_type="text/event-stream",
        )
    else:
        result = await bridge.complete_anthropic(body)
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
    
    # Check if streaming is requested
    stream = body.get("stream", False)
    
    if stream:
        return StreamingResponse(
            bridge.stream_openai(body),
            media_type="text/event-stream",
        )
    else:
        result = await bridge.complete_openai(body)
        return Response(
            content=result,
            media_type="application/json",
        )


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request) -> Response:
    """Estimated token count for Anthropic protocol."""
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
    # We just consume the JSON and return 200 OK
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
    if bridge:
        if bridge.mock:
            model_loaded = True
        elif bridge.wrapper:
            model_loaded = True
            model_info = bridge.wrapper.get_model_info()
            usage = bridge.wrapper.get_usage()
            
    return {
        "status": "ok", 
        "model_loaded": model_loaded,
        "model_info": model_info,
        "usage": usage
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Llama-Bridge Server")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model (file or directory)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and 4-file logs")
    parser.add_argument("--mock", action="store_true", help="Use mock inference (for testing)")
    parser.add_argument("--n-ctx", type=int, default=0, help="Context size (default: 0 = auto/model max)")
    parser.add_argument("--n-batch", type=int, default=0, help="Batch size (default: 0 = auto)")
    parser.add_argument("--n-ubatch", type=int, default=0, help="Physical batch size (default: 0 = auto)")
    parser.add_argument("--n-threads", type=int, default=0, help="Number of threads (default: 0 = auto)")
    parser.add_argument("--flash-attn", action="store_true", help="Enable Flash Attention")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Smart model path resolution
    from pathlib import Path
    model_path = Path(args.model)
    
    # 1. If not absolute and doesn't exist, try relative to models/
    if not model_path.is_absolute() and not model_path.exists():
        # Try prepending 'models/' if it's not already there
        if not args.model.startswith("models/"):
            candidate = Path.cwd() / "models" / args.model
            if candidate.exists():
                model_path = candidate
        else:
            # Maybe it's relative to CWD but starts with models/
            pass
    
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
            logger.error(f"No .gguf files found in directory: {model_path}")
            sys.exit(1)
    
    # 3. Final existence check
    if not args.mock and not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Initialize bridge
    global bridge
    try:
        bridge = Bridge(
            model_path=str(model_path),
            debug=args.debug,
            mock=args.mock,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
            n_threads=args.n_threads,
            flash_attn=args.flash_attn,
        )
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Bridge: {e}")
        if not args.mock:
            logger.error("Suggestion: If this is an OOM (Out of Memory) issue, try specifying a smaller context size with --n-ctx (e.g., --n-ctx 4096).")
            logger.error("Also, ensure you have run 'make build' to update the C++ bindings.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Model: {model_path}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
