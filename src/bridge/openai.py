# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

import json
from datetime import datetime
from typing import AsyncGenerator

from .base import BridgeBase
from ..adapters.openai import OpenAIAdapter

class OpenAIMixin:
    """Extension for OpenAI API support."""
    
    def _init_openai(self):
        self.openai_adapter = OpenAIAdapter()

    async def complete_openai(self, request: dict, cache_name: str | None = None) -> str:
        """Handle non-streaming OpenAI request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        internal_res = await self._generate(internal_req, cache_name=cache_name)
        raw_text = internal_res.get("full_raw", "")
        self._log_request(3, "res_model_raw", raw_text, log_id)
        
        response = self.openai_adapter.from_internal(internal_res, request)
        self._log_request(4, "res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)

    async def stream_openai(self, request: dict, cache_name: str | None = None) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        full_content = ""
        final_usage = {}
        final_tool_calls = []
        final_stop_reason = None
        
        state = {} # OpenAI is stateless for now
        async for chunk in self._stream_generate(internal_req, cache_name=cache_name):
            if "content" in chunk:
                full_content += chunk["content"]
            if "usage" in chunk:
                final_usage = chunk["usage"]
            if "tool_calls" in chunk:
                final_tool_calls = chunk["tool_calls"]
            if "stop_reason" in chunk:
                final_stop_reason = chunk["stop_reason"]
            
            # The final raw text is emitted in the last chunk which has full_raw
            if "full_raw" in chunk:
                self._log_request(3, "res_model_raw", chunk["full_raw"], log_id)
                
            event = self.openai_adapter.chunk_to_sse(chunk, state)
            if event:
                yield event
        
        self._log_request(4, "res_client_sent_summary", {"status": "completed"}, log_id)
