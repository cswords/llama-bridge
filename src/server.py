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

from bridge import Bridge

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


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": bridge is not None and bridge.model_loaded}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Llama-Bridge Server")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
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
    
    # Initialize bridge
    global bridge
    bridge = Bridge(
        model_path=args.model,
        debug=args.debug,
        mock=args.mock,
    )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
