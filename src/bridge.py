"""
Core bridge logic for request/response handling.
Coordinates between format adapters and llama.cpp inference.
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

from .adapters.anthropic import AnthropicAdapter
from .adapters.openai import OpenAIAdapter

# Add src/lib (where libllama.dylib is) to DYLD_LIBRARY_PATH dynamically for this process
# Also add src/ to sys.path so llama_chat.so can be imported
_src_dir = os.path.abspath(os.path.dirname(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

if sys.platform == "darwin":
    lib_path = os.path.join(_src_dir, "lib")
    if "DYLD_LIBRARY_PATH" not in os.environ:
        os.environ["DYLD_LIBRARY_PATH"] = lib_path
    elif lib_path not in os.environ["DYLD_LIBRARY_PATH"]:
        os.environ["DYLD_LIBRARY_PATH"] = lib_path + ":" + os.environ["DYLD_LIBRARY_PATH"]

logger = logging.getLogger(__name__)


def _parse_qwen_tool_calls(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse Qwen3-Coder style tool calls from raw model output.
    
    Format: <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
    Also handles: <function=name><parameter=key>value</parameter></function> without outer tags
    """
    tool_calls = []
    
    # Pattern for <function=name>...<parameter=key>value</parameter>...</function>
    func_pattern = r'<function=(\w+)>(.*?)</function>'
    param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
    
    for match in re.finditer(func_pattern, raw_text, re.DOTALL):
        func_name = match.group(1)
        func_body = match.group(2)
        
        # Extract parameters
        params = {}
        for param_match in re.finditer(param_pattern, func_body, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            # Try to parse as JSON for complex types
            try:
                params[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                params[param_name] = param_value
        
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "name": func_name,
            "arguments": json.dumps(params)
        })
    
    return tool_calls


class Bridge:
    """
    Core bridge between API formats and llama.cpp inference.
    Uses protocol-specific adapters for format conversion.
    """
    
    def __init__(self, model_path: str, debug: bool = False, mock: bool = False, 
                 n_ctx: int = 0, n_batch: int = 0, n_ubatch: int = 0, 
                 n_threads: int = 0, flash_attn: bool = False):
        self.model_path = model_path
        self.debug = debug
        self.mock = mock
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        
        # Initialize adapters directly
        self.anthropic_adapter = AnthropicAdapter()
        self.openai_adapter = OpenAIAdapter()
        
        self.wrapper = None
        
        self.check_ready()
        
        if not self.mock:
            import llama_chat
            self.wrapper = llama_chat.LlamaChatWrapper(model_path, n_ctx, n_batch)
            logger.info(f"Model loaded from {model_path} (n_ctx={n_ctx}, n_batch={n_batch})")
        else:
            logger.info("Running in MOCK mode")

    def check_ready(self) -> None:
        """Check if all dependencies and models are ready."""
        if self.mock:
            return

        # Check for llama_chat dependency
        try:
            import llama_chat
        except ImportError:
            raise ImportError(
                "llama_chat module not found. "
                "Please run 'make build' to compile the C++ bindings."
            )

        # Check for model existence
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please download a GGUF model and place it in the models/ directory."
            )
    
    def _log_request(self, stage: int, filename: str, data: Any, log_id: str) -> None:
        """Log request/response data in debug mode (4-File Rule)."""
        if not self.debug:
            return
        
        log_dir = Path("logs") / log_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = log_dir / f"{stage}_{filename}.json"
        with open(filepath, "w") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(data))
        
        logger.debug(f"Logged Step {stage}: {filename}")
    
    async def complete_anthropic(self, request: dict) -> str:
        """Handle non-streaming Anthropic request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.anthropic_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        internal_res = await self._generate(internal_req)
        self._log_request(3, "res_model_raw", internal_res, log_id)
        
        response = self.anthropic_adapter.from_internal(internal_res, request)
        self._log_request(4, "res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def complete_openai(self, request: dict) -> str:
        """Handle non-streaming OpenAI request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        internal_res = await self._generate(internal_req)
        self._log_request(3, "res_model_raw", internal_res, log_id)
        
        response = self.openai_adapter.from_internal(internal_res, request)
        self._log_request(4, "res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def stream_anthropic(self, request: dict) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.anthropic_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        full_content = ""
        state = {
            "message_started": False,
            "active_block_type": None,
            "current_block_index": 0
        }
        async for chunk in self._stream_generate(internal_req):
            if "content" in chunk:
                full_content += chunk["content"]
            event = self.anthropic_adapter.chunk_to_sse(chunk, state)
            if event:
                yield event

        # Final log for stream
        self._log_request(3, "res_model_full", {"full_content": full_content}, log_id)
    
    async def stream_openai(self, request: dict) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI request."""
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        full_content = ""
        state = {} # OpenAI is stateless for now
        async for chunk in self._stream_generate(internal_req):
            if "content" in chunk:
                full_content += chunk["content"]
            event = self.openai_adapter.chunk_to_sse(chunk, state)
            if event:
                yield event
        
        self._log_request(3, "res_model_full", {"full_content": full_content}, log_id)


    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize messages for llama.cpp, primarily flattening tool_calls."""
        normalized = []
        for msg in messages:
            n_msg = msg.copy()
            if "tool_calls" in n_msg:
                n_tcs = []
                for tc in n_msg["tool_calls"]:
                    ntc = tc.copy()
                    # If it's the OpenAI nested format, flatten it
                    if "function" in ntc and isinstance(ntc["function"], dict):
                        f = ntc["function"]
                        if "name" in f: ntc["name"] = f["name"]
                        if "arguments" in f: ntc["arguments"] = f["arguments"]
                    
                    # Ensure arguments is a string
                    if "arguments" in ntc and not isinstance(ntc["arguments"], (str, type(None))):
                        ntc["arguments"] = json.dumps(ntc["arguments"])
                    
                    n_tcs.append(ntc)
                n_msg["tool_calls"] = n_tcs
            normalized.append(n_msg)
        return normalized

    async def _generate(self, internal_req: dict) -> dict:
        """Execute non-streaming generation."""
        if self.mock:
            return self._mock_generate(internal_req)

        # 1. Apply template
        tools = []
        for t in internal_req.get("tools", []):
            if t.get("type") == "function":
                f = t["function"]
                tools.append({
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": json.dumps(f["parameters"]) if isinstance(f.get("parameters"), dict) else f.get("parameters", "{}")
                })
            else:
                tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": json.dumps(t.get("parameters", {})) if isinstance(t.get("parameters"), dict) else t.get("parameters", "{}")
                })

        messages = self._normalize_messages(internal_req["messages"])
        
        t0 = time.time()
        template_res = self.wrapper.apply_template(
            messages,
            tools,
            True
        )
        t1 = time.time()
        logger.info(f"[Non-Streaming] Template applied in {t1 - t0:.3f}s. Prompt length: {len(template_res['prompt'])}")
        
        # 2. Generate text
        raw_response = self.wrapper.generate(
            template_res["prompt"],
            internal_req.get("max_tokens", 4096)
        )
        t2 = time.time()
        logger.info(f"[Non-Streaming] Generation complete in {t2 - t1:.3f}s.")
        
        # 3. Parse result (for tool calls, reasoning, etc)
        parsed = self.wrapper.parse_response(raw_response, False)
        usage = self.wrapper.get_usage()
        
        content = parsed.get("content", "")
        reasoning = parsed.get("reasoning_content", "")
        tool_calls = parsed.get("tool_calls", [])
        
        # FALLBACK: If C++ parser didn't find tool calls, try Python parsing
        # for Qwen3-Coder style: <function=name><parameter=key>value</parameter></function>
        if not tool_calls and "<function=" in raw_response:
            tool_calls = _parse_qwen_tool_calls(raw_response)
            if tool_calls:
                logger.debug(f"Python fallback parsed {len(tool_calls)} tool call(s)")
        
        # Fallback for reasoning tags
        if not reasoning:
            # Match <think>...</think> or <thought>...</thought>
            active_match = re.search(r"<(thought|think)>(.*?)</\1>(.*)", raw_response, re.DOTALL)
            if active_match:
                reasoning = active_match.group(2).strip()
                content = active_match.group(3).strip()
                if self.debug: logger.debug(f"DEBUG: Found reasoning: {reasoning[:20]}...")
            else:
                # Still check if it just started but didn't close
                open_match = re.search(r"<(thought|think)>(.*)", raw_response, re.DOTALL)
                if open_match:
                    reasoning = open_match.group(2).strip()
                    content = ""
                else:
                    content = raw_response.strip()

        # If tool_calls were found by fallback, remove raw XML from content
        if tool_calls and "<function=" in content:
            # Remove <tool_call>...</tool_call> blocks
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
            # Also remove standalone <function=...>...</function>  
            content = re.sub(r'<function=\w+>.*?</function>', '', content, flags=re.DOTALL)
            content = content.strip()

        # Final cleanup: ensure tool_calls means stop_reason is tool_use
        final_stop_reason = "tool_use" if tool_calls else "end_turn"

        return {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "stop_reason": final_stop_reason,
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"]
            }
        }

    async def _stream_generate(self, internal_req: dict) -> AsyncGenerator[dict, None]:
        """Call llama.cpp for streaming generation."""
        if self.mock and not self.wrapper:
            yield {"content": "Mock streaming content.", "stop_reason": None}
            yield {"content": "", "stop_reason": "end_turn"}
            return

        # 1. Apply template
        tools = []
        for t in internal_req.get("tools", []):
            if t.get("type") == "function":
                f = t["function"]
                tools.append({
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": json.dumps(f["parameters"]) if isinstance(f.get("parameters"), dict) else f.get("parameters", "{}")
                })
            else:
                tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": json.dumps(t.get("parameters", {})) if isinstance(t.get("parameters"), dict) else t.get("parameters", "{}")
                })

        messages = self._normalize_messages(internal_req["messages"])
        
        logger.info("Step 1: Applying chat template...")
        t0 = time.time()
        template_res = self.wrapper.apply_template(
            messages,
            tools,
            True
        )
        t1 = time.time()
        logger.info(f"Template applied in {t1 - t0:.3f}s. Prompt length: {len(template_res['prompt'])}")
        
        # 2. Init inference
        logger.info("Step 2: Initializing inference (Prefill)...")
        self.wrapper.init_inference(template_res["prompt"])
        t2 = time.time()
        logger.info(f"Inference initialized (Prefill complete) in {t2 - t1:.3f}s. Starting token generation...")
        
        # 3. Stream loop
        full_raw = ""
        last_content_pos = 0
        last_reasoning_pos = 0
        
        while True:
            token = self.wrapper.get_next_token()
            if not token: # EOS or Error
                # Final parse (not partial)
                parsed = self.wrapper.parse_response(full_raw, False)
                usage = self.wrapper.get_usage()
                
                # Check for any final tool calls from C++ parser
                tool_calls = parsed.get("tool_calls", [])
                
                # FALLBACK: If C++ parser didn't find tool calls, try Python parsing
                # for Qwen3-Coder style: <function=name><parameter=key>value</parameter></function>
                if not tool_calls and "<function=" in full_raw:
                    tool_calls = _parse_qwen_tool_calls(full_raw)
                    if tool_calls:
                        logger.debug(f"Python fallback parsed {len(tool_calls)} tool call(s)")
                
                yield {
                    "content": "", 
                    "tool_calls": tool_calls,
                    "stop_reason": "tool_use" if tool_calls else "end_turn",
                    "usage": {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"]
                    }
                }
                break
                
            full_raw += token
            
            # Incremental parse (partial=True)
            # This handles structured content like <think>...</think> or tool calls
            parsed = self.wrapper.parse_response(full_raw, True)
            
            # 1. Improved State-aware parsing and suppression
            # We look for structured blocks like <thought>, <think>, <tool_call>, or <function
            curr_content = full_raw
            curr_reasoning = parsed.get("reasoning_content", "")
            
            # Suppression logic: find the first block start
            block_match = re.search(r"<(thought|think|tool_call|function=)", full_raw, re.IGNORECASE)
            if block_match:
                tag_start_pos = block_match.start()
                # Text before the first block is definitely content
                curr_content = full_raw[:tag_start_pos].strip()
                
                # Check for reasoning specifically
                if "thought" in block_match.group(1).lower() or "think" in block_match.group(1).lower():
                    # We found a reasoning start. 
                    # If llama.cpp didn't extract it, we extract the part between tags
                    if not curr_reasoning:
                        active_tag = block_match.group(1)
                        r_match = re.search(rf"<{active_tag}>(.*?)(?:</{active_tag}>|$)", full_raw, re.DOTALL | re.IGNORECASE)
                        if r_match:
                            curr_reasoning = r_match.group(1)
                            # If it's closed, there might be MORE content after it
                            post_match = re.search(rf"</{active_tag}>(.*)", full_raw, re.DOTALL | re.IGNORECASE)
                            if post_match:
                                # We need to re-evaluate for more blocks in the tail
                                tail = post_match.group(1)
                                second_block = re.search(r"<(thought|think|tool_call|function=)", tail, re.IGNORECASE)
                                if second_block:
                                    curr_content += "\n" + tail[:second_block.start()].strip()
                                else:
                                    curr_content += "\n" + tail.strip()
                else:
                    # It's a tool call start. If it's closed, check for text after it.
                    # Qwen uses <tool_call>...</tool_call> or just <function=...>...</function>
                    tc_tag = block_match.group(1)
                    if tc_tag == "tool_call":
                        end_tag = "</tool_call>"
                    else:
                        end_tag = "</function>"
                        
                    tc_end_match = re.search(rf"{end_tag}(.*)", full_raw, re.DOTALL | re.IGNORECASE)
                    if tc_end_match:
                         tail = tc_end_match.group(1)
                         # Recursive-like: check for more blocks in tail
                         next_block = re.search(r"<(thought|think|tool_call|function=)", tail, re.IGNORECASE)
                         if next_block:
                             curr_content += "\n" + tail[:next_block.start()].strip()
                         else:
                             curr_content += "\n" + tail.strip()
            
            # Clean up whitespace
            # CRITICAL FIX: Do NOT strip() here! It eats spaces and prevents progress.
            # curr_content = curr_content.strip()
            # curr_reasoning = curr_reasoning.strip()
            
            # 2. Handle Thoughts (Reasoning)
            if len(curr_reasoning) > last_reasoning_pos:
                delta = curr_reasoning[last_reasoning_pos:]
                yield {"thought": delta, "stop_reason": None}
                last_reasoning_pos = len(curr_reasoning)
            
            # 3. Handle Text Content
            if len(curr_content) > last_content_pos:
                delta = curr_content[last_content_pos:]
                yield {"content": delta, "stop_reason": None}
                last_content_pos = len(curr_content)
            
            # Note: For tool calls in streaming, we usually wait until they are complete 
            # unless we implement a complex JSON-patch streaming logic.
            # Here we follow the LiteLLM/Anthropic pattern of text first, then tools at EOS.

    def _mock_generate(self, internal_req: dict) -> dict:
        """Generate mock response for testing."""
        return {
            "content": "This is a mock response for testing.",
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
