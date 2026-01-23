# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

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
import hashlib
import asyncio
import contextlib
import codecs
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

from .adapters.anthropic import AnthropicAdapter
from .adapters.openai import OpenAIAdapter
from .exceptions import ContextLimitExceededError

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


def _parse_minimax_tool_calls(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse MiniMax-M2 style tool calls from raw model output.
    
    Format: <minimax:tool_call><invoke name="get_weather"><parameter name="location">Shanghai</parameter></invoke></minimax:tool_call>
    """
    tool_calls = []
    
    # Pattern for <invoke name="...">(.*?)</invoke>
    invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
    param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
    
    # Check for <minimax:tool_call> blocks
    tc_blocks = re.findall(r'<minimax:tool_call>(.*?)</minimax:tool_call>', raw_text, re.DOTALL)
    if not tc_blocks and '<invoke name="' in raw_text:
        # Fallback if the outer tag is missing but invokes are present
        tc_blocks = [raw_text]
        
    for block in tc_blocks:
        for match in re.finditer(invoke_pattern, block, re.DOTALL):
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
    Supports multiple contexts (caches) for KV cache isolation.
    """
    
    def __init__(self, model_path: str, debug: bool = False, mock: bool = False, 
                 n_ctx: int = 0, n_batch: int = 0, n_ubatch: int = 0, 
                 n_threads: int = 0, flash_attn: bool = False,
                 cache_configs: list | None = None):
        """
        Initialize the Bridge.
        
        Args:
            model_path: Path to GGUF model file
            debug: Enable debug logging
            mock: Use mock inference (no model loaded)
            n_ctx: Default context size (0 = auto)
            n_batch: Batch size (0 = auto)
            n_ubatch: Physical batch size (0 = auto)
            n_threads: Thread count (0 = auto)
            flash_attn: Enable Flash Attention
            cache_configs: List of cache config dicts with 'name' and 'n_ctx' keys.
                          If None, creates a single 'default' cache.
        """
        self.model_path = model_path
        self.debug = debug
        self.mock = mock
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        self.cache_configs = cache_configs or []
        
        # Initialize adapters directly
        self.anthropic_adapter = AnthropicAdapter()
        self.openai_adapter = OpenAIAdapter()
        
        self.wrapper = None
        self.cache_metadata = {}
        self._locks = {} # Per-cache locks for thread safety
        
        self.check_ready()
        
        if not self.mock:
            import llama_chat
            self.wrapper = llama_chat.LlamaChatWrapper(
                model_path, n_ctx, n_batch, n_ubatch, n_threads, flash_attn
            )
            
            # Create additional contexts if configured
            contexts = ['default']
            for cache_cfg in self.cache_configs:
                cache_name = cache_cfg.get('name', 'unnamed')
                cache_n_ctx = cache_cfg.get('n_ctx', 0)
                if cache_name != 'default':
                    self.wrapper.create_context(cache_name, cache_n_ctx)
                    contexts.append(cache_name)
                    logger.info(f"Created context '{cache_name}' with n_ctx={cache_n_ctx}")
            
            # Initialize locks for all known contexts
            for ctx in contexts:
                self._locks[ctx] = asyncio.Lock()
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Available contexts: {self.wrapper.list_contexts()}")
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
    
    async def complete_anthropic(self, request: dict, cache_name: str | None = None) -> str:
        """Handle non-streaming Anthropic request.
        
        Args:
            request: The Anthropic API request
            cache_name: Target cache name for routing (for future multi-context support)
        """
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.anthropic_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        internal_res = await self._generate(internal_req, cache_name=cache_name)
        self._log_request(3, "res_model_raw", internal_res, log_id)
        
        response = self.anthropic_adapter.from_internal(internal_res, request)
        self._log_request(4, "res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def complete_openai(self, request: dict, cache_name: str | None = None) -> str:
        """Handle non-streaming OpenAI request.
        
        Args:
            request: The OpenAI API request
            cache_name: Target cache name for routing (for future multi-context support)
        """
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        internal_res = await self._generate(internal_req, cache_name=cache_name)
        self._log_request(3, "res_model_raw", internal_res, log_id)
        
        response = self.openai_adapter.from_internal(internal_res, request)
        self._log_request(4, "res_client_transformed", response, log_id)
        
        return json.dumps(response, ensure_ascii=False)
    
    async def stream_anthropic(self, request: dict, cache_name: str | None = None) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic request.
        
        Args:
            request: The Anthropic API request
            cache_name: Target cache name for routing (for future multi-context support)
        """
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
        async for chunk in self._stream_generate(internal_req, cache_name=cache_name):
            if "content" in chunk:
                full_content += chunk["content"]
            event = self.anthropic_adapter.chunk_to_sse(chunk, state)
            if event:
                yield event

        # Final log for stream
        self._log_request(3, "res_model_full", {"full_content": full_content}, log_id)
    
    async def stream_openai(self, request: dict, cache_name: str | None = None) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI request.
        
        Args:
            request: The OpenAI API request
            cache_name: Target cache name for routing (for future multi-context support)
        """
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._log_request(1, "req_client_raw", request, log_id)
        
        internal_req = self.openai_adapter.to_internal(request)
        self._log_request(2, "req_model", internal_req, log_id)
        
        full_content = ""
        state = {} # OpenAI is stateless for now
        async for chunk in self._stream_generate(internal_req, cache_name=cache_name):
            if "content" in chunk:
                full_content += chunk["content"]
            event = self.openai_adapter.chunk_to_sse(chunk, state)
            if event:
                yield event
        
        self._log_request(3, "res_model_full", {"full_content": full_content}, log_id)


    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize messages for llama.cpp, handling multi-block content and flattening tool_calls."""
        normalized = []
        for msg in messages:
            n_msg = msg.copy()
            role = n_msg.get("role")
            content = n_msg.get("content")

            # Handle list-style content (Anthropic/OpenAI Multi-modal/Mixed)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif b_type == "image":
                            # Vision not yet supported, notify in logs
                            logger.warning("Vision/Image blocks are not yet supported and will be ignored.")
                        elif b_type == "tool_use":
                            # Already handled in adapter but safe to flatten here if needed
                            if "tool_calls" not in n_msg: n_msg["tool_calls"] = []
                            n_msg["tool_calls"].append({
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}))
                            })
                n_msg["content"] = "\n".join(text_parts)

            # Flatten and normalize tool_calls
            if "tool_calls" in n_msg and n_msg["tool_calls"]:
                n_tcs = []
                for tc in n_msg["tool_calls"]:
                    ntc = tc.copy()
                    # Flatten OpenAI nested structure
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

    def _extract_system_hash(self, messages: List[Dict[str, Any]]) -> str:
        """Calculate hash of all system messages combined."""
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list): # Handle structured content
                    system_content += str(content)
                else:
                    system_content += str(content)
        return hashlib.md5(system_content.encode()).hexdigest()

    def _check_system_prompt(self, internal_req: dict, cache_name: str | None):
        """Check if system prompt changed for the given cache."""
        if not cache_name:
            return
            
        current_hash = self._extract_system_hash(internal_req.get("messages", []))
        
        if cache_name not in self.cache_metadata:
            self.cache_metadata[cache_name] = {"system_hash": current_hash}
        else:
            last_hash = self.cache_metadata[cache_name].get("system_hash")
            if last_hash != current_hash:
                # Only warn if previous hash existed (not first request)
                if last_hash:
                    logger.warning(f"System prompt changed for cache '{cache_name}'. This will cause cache invalidation.")
                self.cache_metadata[cache_name]["system_hash"] = current_hash

    def _prepare_tools(self, internal_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool definitions for llama.cpp."""
        tools = []
        for t in internal_tools:
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
        return tools

    async def _generate(self, internal_req: dict, cache_name: str | None = None) -> dict:
        """Execute non-streaming generation with truncation support."""
        # Check system prompt consistency
        self._check_system_prompt(internal_req, cache_name)

        if self.mock and not self.wrapper:
            return self._mock_generate(internal_req)

        # Select the target context if specified
        if cache_name and self.wrapper:
            try:
                self.wrapper.select_context(cache_name)
            except RuntimeError as e:
                logger.warning(f"Failed to select context '{cache_name}': {e}")

        tools = self._prepare_tools(internal_req.get("tools", []))
        msgs_to_use = internal_req["messages"]
        max_tokens = internal_req.get("max_tokens", 4096)
        
        # 1. Truncation Loop
        raw_response = ""
        max_retries = 10
        t0 = time.time()
        
        for retry in range(max_retries):
            messages = self._normalize_messages(msgs_to_use)
            template_res = self.wrapper.apply_template(messages, tools, True)
            prompt = template_res["prompt"]
            
            try:
                raw_response = self.wrapper.generate(prompt, max_tokens)
                break # Success
            except RuntimeError as e:
                msg = str(e)
                if "ContextLimitExceeded" in msg:
                    match = re.search(r"Request \((\d+) tokens\) exceeds context limit \((\d+)\)", msg)
                    requested = int(match.group(1)) if match else 0
                    limit = int(match.group(2)) if match else 0
                    
                    non_system_indices = [i for i, m in enumerate(msgs_to_use) if m.get("role") != "system"]
                    if len(non_system_indices) > 2:
                        idx_to_drop = non_system_indices[:2]
                        logger.warning(f"Context limit hit ({requested}/{limit}). Truncating oldest message pair at indices {idx_to_drop} and retrying...")
                        msgs_to_use = [m for i, m in enumerate(msgs_to_use) if i not in idx_to_drop]
                        continue
                    else:
                        if match: raise ContextLimitExceededError(limit, requested, cache_name or "unknown")
                raise e
        else:
            # Final fallback: one last attempt with what we have
            raw_response = self.wrapper.generate(prompt, max_tokens)

        t2 = time.time()
        logger.info(f"[Non-Streaming] Generation complete in {t2 - t0:.3f}s (retries={retry}).")
        
        # 3. Parse result
        parsed = self.wrapper.parse_response(raw_response, False)
        usage = self.wrapper.get_usage()
        
        content = parsed.get("content", "")
        reasoning = parsed.get("reasoning_content", "")
        tool_calls = parsed.get("tool_calls", [])
        
        if not tool_calls:
            if "<function=" in raw_response:
                tool_calls = _parse_qwen_tool_calls(raw_response)
            elif "<minimax:tool_call>" in raw_response or "<invoke name=" in raw_response:
                tool_calls = _parse_minimax_tool_calls(raw_response)
        
        # Fallback for reasoning
        if not reasoning:
            active_match = re.search(r"<(thought|think)>(.*?)</\1>(.*)", raw_response, re.DOTALL)
            if active_match:
                reasoning = active_match.group(2).strip()
                content = active_match.group(3).strip()
            else:
                open_match = re.search(r"<(thought|think)>(.*)", raw_response, re.DOTALL)
                if open_match:
                    reasoning = open_match.group(2).strip()
                    content = ""
                else:
                    content = raw_response.strip()

        # Clean up tags from content if tool calls are present
        if tool_calls:
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
            content = re.sub(r'<function=\w+>.*?</function>', '', content, flags=re.DOTALL)
            content = re.sub(r'<minimax:tool_call>.*?</minimax:tool_call>', '', content, flags=re.DOTALL)
            content = re.sub(r'<invoke name="[^"]+">.*?</invoke>', '', content, flags=re.DOTALL)
            content = content.strip()

        return {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"]
            }
        }

    @contextlib.contextmanager
    def _suppress_stderr(self):
        """Suppress stderr if not in debug mode."""
        if not self.debug:
            with open(os.devnull, 'w') as devnull:
                old_stderr = os.dup(sys.stderr.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
                try:
                    yield
                finally:
                    os.dup2(old_stderr, sys.stderr.fileno())
                    os.close(old_stderr)
        else:
            yield

    async def _stream_generate(self, internal_req: dict, cache_name: str | None = None) -> AsyncGenerator[dict, None]:
        """Call llama.cpp for streaming generation.
        
        Args:
            internal_req: Internal request format
            cache_name: Target cache/context name. If None, uses active context.
        """
        # Check system prompt consistency
        self._check_system_prompt(internal_req, cache_name)

        if self.mock and not self.wrapper:
            # Check if the user is asking to list files
            last_user_msg = ""
            for m in reversed(internal_req.get("messages", [])):
                if m["role"] == "user":
                    last_user_msg = m.get("content", "")
                    break
            
            if "list files" in last_user_msg.lower():
                yield {"content": "I have executed the request. Found src/ and other files via ls.", "stop_reason": None}
            else:
                yield {"content": "Mock streaming content.", "stop_reason": None}
                
            yield {"content": "", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 10}}
            return

        # Select the target context and get its lock
        lock = self._locks.get(cache_name or 'default')
        if not lock:
            lock = self._locks.setdefault(cache_name or 'default', asyncio.Lock())

        async with lock:
            if cache_name and self.wrapper:
                try:
                    self.wrapper.select_context(cache_name)
                except RuntimeError as e:
                    logger.warning(f"Failed to select context '{cache_name}': {e}. Using active context.")

            # 1. Prepare and apply template
            tools = self._prepare_tools(internal_req.get("tools", []))
            msgs_to_use = internal_req["messages"]
            max_tokens = internal_req.get("max_tokens", 0) or 0
            
            t0 = time.time()
            
            # 2. Init inference with retry/truncation
            # CRITICAL: Use to_thread to avoid blocking the event loop during heavy prefill!
            if not self.debug:
                logger.info(f"[{cache_name or 'default'}] Prefilling...")
            else:
                logger.info(f"Step 2: [{cache_name or 'default'}] Initializing inference (Prefill)...")
            
            max_retries = 10
            template_res = {}
            for retry in range(max_retries):
                messages = self._normalize_messages(msgs_to_use)
                # Suppress template fallback warnings from C++ if not debug
                with self._suppress_stderr():
                    template_res = self.wrapper.apply_template(messages, tools, True)
                prompt = template_res["prompt"]
                
                try:
                    await asyncio.to_thread(self.wrapper.init_inference, prompt, max_tokens)
                    break # Success
                except RuntimeError as e:
                    msg = str(e)
                    if "ContextLimitExceeded" in msg:
                        match = re.search(r"Request \((\d+) tokens\) exceeds context limit \((\d+)\)", msg)
                        requested = int(match.group(1)) if match else 0
                        limit = int(match.group(2)) if match else 0
                        
                        non_system_indices = [i for i, m in enumerate(msgs_to_use) if m.get("role") != "system"]
                        
                        if len(non_system_indices) > 2:
                            idx_to_drop = non_system_indices[:2]
                            logger.warning(f"Context limit hit ({requested}/{limit}). Truncating oldest message pair and retrying...")
                            msgs_to_use = [m for i, m in enumerate(msgs_to_use) if i not in idx_to_drop]
                            continue 
                        else:
                            if match: raise ContextLimitExceededError(limit, requested, cache_name or "unknown")
                    raise e
            else:
                template_res = self.wrapper.apply_template(messages, tools, True)
                await asyncio.to_thread(self.wrapper.init_inference, template_res["prompt"], max_tokens)

            # Get additional stops from template and request
            all_stops = list(template_res.get("additional_stops", []))
            req_stops = internal_req.get("stop") or []
            if isinstance(req_stops, str): req_stops = [req_stops]
            for s in req_stops:
                if s and s not in all_stops: all_stops.append(s)

            t2 = time.time()
            logger.info(f"[{cache_name or 'default'}] Prefill complete in {t2 - t0:.1f}s. Generating...")
            
            # 3. Stream loop
            full_raw = ""
            from src.utils.scanner import StreamScanner
            scanner = StreamScanner()
            last_yield_time = time.time()
            actual_stop_seq = None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            
            scanner_index = 0
            
            async def process_and_yield_buffer(force: bool = False):
                nonlocal full_raw, scanner_index, last_yield_time
                
                if not force and (time.time() - last_yield_time) < 0.05:
                    return

                new_data = full_raw[scanner_index:]
                if not new_data and not force:
                    return
                
                # Feed new data to scanner
                results = scanner.consume(new_data, is_final=force)
                scanner_index += len(new_data)
                
                for type_, text in results:
                    if type_ == "content":
                        yield {"content": text}
                    elif type_ == "thought":
                        yield {"thought": text}
                        
                last_yield_time = time.time()

            while True:
                # Offload to thread to keep server responsive
                # new_bytes is raw bytes from C++ to handle UTF-8 splits
                new_bytes = await asyncio.to_thread(self.wrapper.get_next_token)
                
                if not new_bytes: # EOS
                    full_raw += decoder.decode(b"", final=True)
                    async for chunk in process_and_yield_buffer(force=True):
                        yield chunk
                    
                    # Final parse (not partial)
                    parsed = self.wrapper.parse_response(full_raw, False)
                    usage = self.wrapper.get_usage()
                    tool_calls = parsed.get("tool_calls", [])
                    
                    if not tool_calls:
                        if "<function=" in full_raw:
                            tool_calls = _parse_qwen_tool_calls(full_raw)
                        elif "<minimax:tool_call>" in full_raw or "<invoke name=" in full_raw:
                            tool_calls = _parse_minimax_tool_calls(full_raw)
                    
                    stop_reason = "end_turn"
                    if tool_calls:
                        stop_reason = "tool_use"
                    elif actual_stop_seq:
                        stop_reason = "stop_sequence"

                    yield {
                        "content": "", 
                        "tool_calls": tool_calls,
                        "stop_reason": stop_reason,
                        "usage": {
                            "input_tokens": usage["prompt_tokens"],
                            "output_tokens": usage["completion_tokens"]
                        }
                    }
                    break
                
                full_raw += decoder.decode(new_bytes, final=False)
                
                # Check for stop sequences in raw stream
                for s in all_stops:
                    if full_raw.endswith(s):
                        actual_stop_seq = s
                        # Trim stop sequence from full_raw
                        full_raw = full_raw[:-len(s)]
                        # Signal EOS to decoder
                        full_raw += decoder.decode(b"", final=True)
                        async for chunk in process_and_yield_buffer(force=True):
                            yield chunk
                        
                        # Finalize
                        parsed = self.wrapper.parse_response(full_raw, False)
                        usage = self.wrapper.get_usage()
                        tool_calls = parsed.get("tool_calls", [])
                        
                        yield {
                            "content": "",
                            "tool_calls": tool_calls,
                            "stop_reason": "stop_sequence",
                            "usage": {
                                "input_tokens": usage["prompt_tokens"],
                                "output_tokens": usage["completion_tokens"]
                            }
                        }
                        return

                async for chunk in process_and_yield_buffer():
                    yield chunk


    def _mock_generate(self, internal_req: dict) -> dict:
        """Generate mock response for testing."""
        # Check if the user is asking to list files
        last_user_msg = ""
        for m in reversed(internal_req.get("messages", [])):
            if m["role"] == "user":
                last_user_msg = m.get("content", "")
                break
        
        if "list files" in last_user_msg.lower():
            return {
                "content": "I have executed the request. Found src and other files via ls.",
                "tool_calls": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 10}
            }

        return {
            "content": "This is a mock response for testing.",
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10}
        }
