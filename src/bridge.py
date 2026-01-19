"""
Core bridge logic for request/response handling.
Coordinates between LiteLLM format conversion and llama.cpp inference.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from litellm_adapter import LiteLLMAdapter

logger = logging.getLogger(__name__)


class Bridge:
    """
    Core bridge between API formats and llama.cpp inference.
    
    Responsibilities:
    - Accept requests in various API formats (via LiteLLM)
    - Convert to internal format
    - Call llama.cpp for inference
    - Convert response back to original format
    - Handle streaming
    """
    
    def __init__(self, model_path: str, debug: bool = False, mock: bool = False):
        self.model_path = model_path
        self.debug = debug
        self.mock = mock
        self.model_loaded = False
        
        self.adapter = LiteLLMAdapter()
        
        # TODO: Initialize llama.cpp bindings
        # self.llama = LlamaChatBindings(model_path)
        
        if mock:
            logger.info("Running in mock mode - no model loaded")
            self.model_loaded = True
        else:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the GGUF model via llama.cpp bindings."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # TODO: Load model via pybind11 bindings
        # self.llama.load(self.model_path)
        logger.info(f"Model loaded: {self.model_path}")
        self.model_loaded = True
    
    def _log_request(self, stage: int, filename: str, data: dict) -> None:
        """Log request/response data in debug mode (4-File Rule)."""
        if not self.debug:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_dir = Path("logs") / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = log_dir / f"{stage}_{filename}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Logged: {filepath}")
    
    async def complete_anthropic(self, request: dict) -> str:
        """Handle non-streaming Anthropic request."""
        self._log_request(1, "req_client_raw", request)
        
        # Convert to internal format
        internal_req = self.adapter.anthropic_to_internal(request)
        self._log_request(2, "req_model", internal_req)
        
        # Generate response
        if self.mock:
            internal_res = self._mock_generate(internal_req)
        else:
            internal_res = await self._generate(internal_req)
        
        self._log_request(3, "res_model_raw", internal_res)
        
        # Convert back to Anthropic format
        response = self.adapter.internal_to_anthropic(internal_res, request)
        self._log_request(4, "res_client_transformed", response)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def complete_openai(self, request: dict) -> str:
        """Handle non-streaming OpenAI request."""
        self._log_request(1, "req_client_raw", request)
        
        # Convert to internal format
        internal_req = self.adapter.openai_to_internal(request)
        self._log_request(2, "req_model", internal_req)
        
        # Generate response
        if self.mock:
            internal_res = self._mock_generate(internal_req)
        else:
            internal_res = await self._generate(internal_req)
        
        self._log_request(3, "res_model_raw", internal_res)
        
        # Convert back to OpenAI format
        response = self.adapter.internal_to_openai(internal_res, request)
        self._log_request(4, "res_client_transformed", response)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def stream_anthropic(self, request: dict) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic request."""
        self._log_request(1, "req_client_raw", request)
        
        internal_req = self.adapter.anthropic_to_internal(request)
        self._log_request(2, "req_model", internal_req)
        
        # Stream response
        async for chunk in self._stream_generate(internal_req):
            event = self.adapter.chunk_to_anthropic_sse(chunk)
            yield event
    
    async def stream_openai(self, request: dict) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI request."""
        self._log_request(1, "req_client_raw", request)
        
        internal_req = self.adapter.openai_to_internal(request)
        self._log_request(2, "req_model", internal_req)
        
        # Stream response
        async for chunk in self._stream_generate(internal_req):
            event = self.adapter.chunk_to_openai_sse(chunk)
            yield event
    
    async def _generate(self, internal_req: dict) -> dict:
        """Generate response using llama.cpp."""
        # TODO: Implement actual llama.cpp call
        raise NotImplementedError("llama.cpp bindings not yet implemented")
    
    async def _stream_generate(self, internal_req: dict) -> AsyncGenerator[dict, None]:
        """Stream generate response using llama.cpp."""
        # TODO: Implement actual llama.cpp streaming
        raise NotImplementedError("llama.cpp bindings not yet implemented")
        yield  # Make this a generator
    
    def _mock_generate(self, internal_req: dict) -> dict:
        """Generate mock response for testing."""
        return {
            "content": "This is a mock response for testing.",
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
            },
        }
