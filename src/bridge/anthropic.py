# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

import json
from datetime import datetime
from typing import AsyncGenerator

from .base import BridgeBase
from ..adapters.anthropic import AnthropicAdapter

class AnthropicMixin:
    """Extension for Anthropic API support."""
    
    def _init_anthropic(self):
        self.anthropic_adapter = AnthropicAdapter()

    async def complete_anthropic(self, request: dict, cache_name: str | None = None) -> str:
        """Handle non-streaming Anthropic request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request("req_client_raw", request, log_id)
        
        internal_req = self.anthropic_adapter.to_internal(request)
        self._log_request("req_model", internal_req, log_id)
        
        # BridgeBase has _generate
        internal_res = await self._generate(internal_req, cache_name=cache_name, log_id=log_id)
        raw_text = internal_res.get("full_raw", "")
        self._log_request("res_model_raw", raw_text, log_id)
        
        response = self.anthropic_adapter.from_internal(internal_res, request)
        self._log_request("res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)

    async def stream_anthropic(self, request: dict, cache_name: str | None = None) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request("req_client_raw", request, log_id)
        
        internal_req = self.anthropic_adapter.to_internal(request)
        self._log_request("req_model", internal_req, log_id)
        
        full_content = ""
        final_usage = {}
        final_tool_calls = []
        final_stop_reason = None
        
        state = {
            "message_started": False,
            "active_block_type": None,
            "current_block_index": 0
        }
        
        # BridgeBase has _stream_generate
        async for chunk in self._stream_generate(internal_req, cache_name=cache_name, log_id=log_id):
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
                self._log_request("res_model_raw", chunk["full_raw"], log_id)
                
            event = self.anthropic_adapter.chunk_to_sse(chunk, state)
            if event:
                self._log_client_stream_event(event, log_id)
                yield event
        
        # Stage 4 for streaming (summary of what was sent)
        self._log_request("res_client_sent_summary", {"status": "completed", "events_sent": state["current_block_index"]}, log_id)
