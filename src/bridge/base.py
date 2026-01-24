# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

"""
Base bridge logic for request/response handling.
Coordinates with llama.cpp inference.
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

from src.exceptions import ContextLimitExceededError

# Add src/lib (where libllama.dylib is) to DYLD_LIBRARY_PATH dynamically for this process
# Also add src/ to sys.path so llama_chat.so can be imported
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

if sys.platform == "darwin":
    lib_path = os.path.join(_src_dir, "lib")
    if "DYLD_LIBRARY_PATH" not in os.environ:
        os.environ["DYLD_LIBRARY_PATH"] = lib_path
    elif lib_path not in os.environ["DYLD_LIBRARY_PATH"]:
        os.environ["DYLD_LIBRARY_PATH"] = lib_path + ":" + os.environ["DYLD_LIBRARY_PATH"]

logger = logging.getLogger(__name__)


from src.bridge.flavors import get_flavor_for_model
from src.bridge.scanner import StreamScanner

class BridgeBase:
    """
    Base bridge between API formats and llama.cpp inference.
    Provides core inference logic used by protocol-specific bridges.
    """
    
    def __init__(self, model_path: str, debug: bool = False, mock: bool = False, 
                 n_ctx: int = 0, n_batch: int = 0, n_ubatch: int = 0, 
                 n_threads: int = 0, flash_attn: bool = False,
                 cache_configs: list | None = None):
        """Initialize the BridgeBase."""
        self.model_path = model_path
        self.debug = debug
        self.mock = mock
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        self.cache_configs = cache_configs or []
        
        self.wrapper = None
        self.cache_metadata = {}
        self._locks = {} # Per-cache locks for thread safety
        
        # Initialize flavor
        self.flavor = get_flavor_for_model(model_path)
        logger.info(f"Loaded flavor: {self.flavor.__class__.__name__}")
        
        self.check_ready()
        
        if not self.mock:
            try:
                import llama_chat
            except ImportError:
                from src import llama_chat
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
            try:
                import llama_chat
            except ImportError:
                from src import llama_chat
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
        
        # Use .json for dict/list, .txt for everything else (raw model output)
        is_json = isinstance(data, (dict, list))
        ext = "json" if is_json else "txt"
        filepath = log_dir / f"{stage}_{filename}.{ext}"
        
        with open(filepath, "w") as f:
            if is_json:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(data))
        
        logger.debug(f"Logged Step {stage}: {filename}")

    def _log_stream_chunk(self, chunk: dict, log_id: str) -> None:
        """Log a single stream chunk to the stream dump file."""
        if not self.debug:
            return
        
        log_dir = Path("logs") / log_id
        # We assume log_dir exists because _log_request(1...) is called first
        with open(log_dir / "4_stream_dump.jsonl", "a") as f:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

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

    async def _generate(self, internal_req: dict, cache_name: str | None = None, log_id: str | None = None) -> dict:
        """Execute non-streaming generation with truncation support."""
        # Check system prompt consistency
        self._check_system_prompt(internal_req, cache_name)

        if self.mock and not self.wrapper:
            return self._mock_generate(internal_req)

        # Generation uses explicit cache_name
        ctx_to_use = cache_name or 'default'

        tools = self._prepare_tools(internal_req.get("tools", []))
        msgs_to_use = internal_req["messages"]
        max_tokens = internal_req.get("max_tokens", 4096)
        
        # 1. Truncation Loop
        log_id = log_id or str(uuid.uuid4())
        self._log_request(1, "request_internal", internal_req, log_id)
        
        raw_response = ""
        max_retries = 10
        t0 = time.time()
        for retry in range(max_retries):
            messages = self._normalize_messages(msgs_to_use)
            # Authority Stage 1: Try C++ Jinja (llama-server behavior)
            template_res = self.wrapper.apply_template(messages, tools, True)
            prompt = template_res["prompt"]
            
            # If C++ template is somehow missing (legacy GGUF), fallback to Flavor
            if not prompt or len(prompt) < 5:
                flavor_prompt = self.flavor.apply_template(messages, tools)
                if flavor_prompt:
                    prompt = flavor_prompt
                    logger.warning("Falling back to Flavor-based template (Model missing Jinja template)")

            try:
                self._log_request(2, "prompt", prompt, log_id)
                raw_response = self.wrapper.generate(ctx_to_use, prompt, max_tokens)
                self._log_request(3, "raw_output", raw_response, log_id)
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
            # Final fallback
            raw_response = self.wrapper.generate(ctx_to_use, prompt, max_tokens)
        t2 = time.time()
        logger.info(f"[Non-Streaming] Generation complete in {t2 - t0:.3f}s (retries={retry}).")
        
        # Authority Stage 2: Use C++ side common_chat_parse
        parsed = self.wrapper.parse_response(raw_response, False)
        self._log_request(4, "parsed_response", parsed, log_id)
        usage = self.wrapper.get_usage(ctx_to_use)
        
        content = parsed.get("content", "").strip()
        reasoning = parsed.get("reasoning_content", "").strip()
        tool_calls = parsed.get("tool_calls", [])
        
        # Emergency Fallback: If C++ didn't find tools but Flavor knows the model is 'quirky'
        if not tool_calls:
            from .scanner import StreamScanner
            scanner = StreamScanner(protected_tags=self.flavor.protected_tags)
            events = scanner.push(raw_response) + scanner.flush()
            
            curr_block = None
            curr_block_content = ""
            for ev in events:
                if ev.type == "block_start":
                    curr_block = ev.tag.strip("<>").replace("/", "").split("=")[0]
                    curr_block_content = ""
                elif ev.type == "block_content":
                    if curr_block:
                        curr_block_content += ev.data
                    elif not content:
                        # Should we accumulate content here? 
                        # Original code did 'content += edata'
                        # But 'content' is already a string (parsed.get("content")).
                        # If parsed["content"] was empty, we use the scanner's content.
                        pass
                elif ev.type == "block_end":
                    if curr_block:
                        interpretation = self.flavor.interpret_block_complete(curr_block, curr_block_content)
                        if interpretation and interpretation[0] == "tool_use":
                            tool_calls.extend(interpretation[1])
                    curr_block = None
                    curr_block_content = ""
            
            # If after all that we still have no content AND no tools, reconstruct content from events
            if not content and not tool_calls:
                content = "".join([ev.data for ev in events if ev.type == "content"])
        
        if not content and not tool_calls:
            content = raw_response.strip()

        result = {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"]
            },
            "full_raw": raw_response
        }
        self._log_request(5, "final_response", result, log_id)
        return result

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

    async def _stream_generate(self, internal_req: dict, cache_name: str | None = None, log_id: str | None = None) -> AsyncGenerator[dict, None]:
        """Call llama.cpp for streaming generation."""
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
                chunk = {"content": "I have executed the request. Found src/ and other files via ls.", "stop_reason": None}
                self._log_stream_chunk(chunk, log_id)
                yield chunk
            else:
                chunk = {"content": "Mock streaming content.", "stop_reason": None}
                self._log_stream_chunk(chunk, log_id)
                yield chunk
                
            chunk = {"content": "", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 10}}
            self._log_stream_chunk(chunk, log_id)
            yield chunk
            return

        # Select the target context
        ctx_to_use = cache_name or 'default'
        log_id = log_id or str(uuid.uuid4())
        lock = self._locks.get(ctx_to_use)
        if not lock:
            lock = self._locks.setdefault(ctx_to_use, asyncio.Lock())

        async with lock:
            # 1. Prepare and apply template
            self._log_request(1, "request_internal", internal_req, log_id)
            
            tools = self._prepare_tools(internal_req.get("tools", []))
            msgs_to_use = internal_req["messages"]
            max_tokens = internal_req.get("max_tokens", 0) or 0
            
            t0 = time.time()
            
            # 2. Init inference with retry/truncation
            if not self.debug:
                logger.info(f"[{ctx_to_use}] Prefilling...")
            else:
                logger.info(f"Step 2: [{ctx_to_use}] Initializing inference (Prefill)...")
            
            max_retries = 10
            for retry in range(max_retries):
                messages = self._normalize_messages(msgs_to_use)
                
                # Authority Stage 1: Try C++ Jinja (llama-server behavior)
                template_res = self.wrapper.apply_template(messages, tools, True)
                prompt = template_res["prompt"]
                all_stops = list(template_res.get("additional_stops", []))
                
                # If C++ template is somehow missing (legacy GGUF), fallback to Flavor
                if not prompt or len(prompt) < 5:
                    flavor_prompt = self.flavor.apply_template(messages, tools)
                    if flavor_prompt:
                        prompt = flavor_prompt
                        logger.warning("Falling back to Flavor-based template (Model missing Jinja template)")
                
                try:
                    self._log_request(2, "prompt", prompt, log_id)
                    await asyncio.to_thread(self.wrapper.init_inference, ctx_to_use, prompt, max_tokens)
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
                messages = self._normalize_messages(msgs_to_use)
                template_res = self.wrapper.apply_template(messages, tools, True)
                prompt = template_res["prompt"]
                if not prompt or len(prompt) < 5:
                    prompt = self.flavor.apply_template(messages, tools) or prompt
                await asyncio.to_thread(self.wrapper.init_inference, ctx_to_use, prompt, max_tokens)

            req_stops = internal_req.get("stop") or []
            if isinstance(req_stops, str): req_stops = [req_stops]
            for s in req_stops:
                if s and s not in all_stops: all_stops.append(s)

            t2 = time.time()
            logger.info(f"[{ctx_to_use}] Prefill complete in {t2 - t0:.1f}s. Generating...")
            
            # 3. Stream loop
            full_raw = ""
            scanner = StreamScanner(protected_tags=self.flavor.protected_tags)
            
            # --- PREFILL STATE SYNCHRONIZATION ---
            # Use prefill to synchronize scanner state (e.g. if we are already inside a <think> block)
            
            block_content_stack = [] # List of [tag_type, content]
            prefill_source = None
            
            # 1. Try explicit Assistant content from messages (User provided prefill)
            last_msg = internal_req.get("messages", [])[-1] if internal_req.get("messages") else None
            if last_msg and last_msg.get("role") == "assistant" and last_msg.get("content"):
                prefill_source = last_msg["content"]
            
            # 2. If no explicit prefill, check Prompt tail for Implicit Prefill
            # Algorithm: Find the LAST occurrence of any Block Start Tag.
            # If it exists, and is NOT followed by its matching Close Tag, then it is an Open Block.
            if not prefill_source and prompt:
                last_open_tag_pos = -1
                
                for tag_name in self.flavor.block_tags:
                    start_marker = f"<{tag_name}"
                    p = prompt.rfind(start_marker)
                    
                    if p > last_open_tag_pos:
                        # Found a candidate start tag. Check if it is closed.
                        end_marker = f"</{tag_name}>"
                        p_end = prompt.find(end_marker, p)
                        
                        if p_end == -1:
                            # It is NOT closed. This is our winner so far.
                            last_open_tag_pos = p
                
                if last_open_tag_pos != -1:
                    prefill_source = prompt[last_open_tag_pos:]

            if prefill_source:
                # Push prefill text to scanner
                pre_events = scanner.push(prefill_source)
                
                # Process events to update STACK only (suppress output)
                for ev in pre_events:
                    if ev.type == "block_start":
                        block_content_stack.append([ev.tag, ""])
                        # Also append the tag itself to parent blocks if any
                        for item in block_content_stack[:-1]:
                            item[1] += ev.data
                    elif ev.type == "block_end":
                        if block_content_stack:
                            # We don't interpret/emit closed blocks from prefill, just pop stack
                            _, content = block_content_stack.pop()
                            # Append closing tag to parent blocks
                            for item in block_content_stack:
                                item[1] += ev.data
                    elif ev.type == "block_content":
                        # Accumulate in active blocks
                        for item in block_content_stack:
                            item[1] += ev.data
                    # "content" events (plain text in prefill) are ignored/suppressed
                
                # IMPORTANT: Clear the buffer. We don't want any partial text from the prompt 
                # to be emitted as the start of the response.
                scanner.buffer = ""
            
            # Fix: Ensure full_raw includes the prefill so C++ parser sees the full block
            if prefill_source:
                full_raw = prefill_source

            last_yield_time = time.time()
            actual_stop_seq = None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            
            scanner_index = len(full_raw)
            final_tool_calls = []

            async def process_and_yield_buffer(force: bool = False):
                nonlocal full_raw, scanner_index, last_yield_time, final_tool_calls, block_content_stack
                
                new_data = full_raw[scanner_index:]
                if not new_data and not force: return
                
                events = scanner.push(new_data)
                if force:
                    events.extend(scanner.flush())
                
                scanner_index += len(new_data)
                
                for ev in events:
                    if ev.type == "content":
                        yield {"content": ev.data}
                    elif ev.type == "block_start":
                        # Add tag to parent blocks
                        for item in block_content_stack:
                            item[1] += ev.data
                        block_content_stack.append([ev.tag, ""])
                    elif ev.type == "block_content":
                        # Accumulate in all active blocks
                        for item in block_content_stack:
                            item[1] += ev.data
                        
                        if block_content_stack:
                            leaf_tag = block_content_stack[-1][0]
                            interpretation = self.flavor.interpret_block_chunk(leaf_tag, ev.data)
                            if interpretation:
                                yield {interpretation[0]: interpretation[1]}
                    elif ev.type == "block_end":
                        if block_content_stack:
                            tag_type, full_content = block_content_stack.pop()
                            # Add closing tag to parent blocks
                            for item in block_content_stack:
                                item[1] += ev.data
                            
                            interpretation = self.flavor.interpret_block_complete(tag_type, full_content)
                            if interpretation:
                                itype, idata = interpretation
                                if itype == "tool_use":
                                    final_tool_calls.extend(idata)
                                else:
                                    yield {itype: idata}

            while True:
                new_bytes = await asyncio.to_thread(self.wrapper.get_next_token, ctx_to_use)
                
                if not new_bytes: # EOS
                    full_raw += decoder.decode(b"", final=True)
                    async for chunk in process_and_yield_buffer(force=True):
                        self._log_stream_chunk(chunk, log_id)
                        yield chunk
                    
                    # Authority Stage 2 (Final): Re-verify tools with C++ parser
                    parsed = self.wrapper.parse_response(full_raw, False)
                    self._log_request(5, "parsed_response", parsed, log_id)
                    tc = parsed.get("tool_calls", [])
                    if tc and not final_tool_calls:
                        final_tool_calls = tc
                    
                    
                    usage = self.wrapper.get_usage(ctx_to_use)
                    stop_reason = "tool_use" if final_tool_calls else "end_turn"
                    if not final_tool_calls and actual_stop_seq:
                        stop_reason = "stop_sequence"

                    self._log_request(3, "raw_output", full_raw, log_id)
                    final_chunk = {
                        "content": "", 
                        "reasoning_content": parsed.get("reasoning_content", ""),
                        "tool_calls": final_tool_calls,
                        "stop_reason": stop_reason,
                        "usage": {
                            "input_tokens": usage["prompt_tokens"],
                            "output_tokens": usage["completion_tokens"]
                        },
                        "full_raw": full_raw
                    }
                    self._log_stream_chunk(final_chunk, log_id)
                    yield final_chunk
                    break
                
                full_raw += decoder.decode(new_bytes, final=False)
                
                for s in all_stops:
                    if full_raw.endswith(s):
                        actual_stop_seq = s
                        full_raw = full_raw[:-len(s)]
                        full_raw += decoder.decode(b"", final=True)
                        async for chunk in process_and_yield_buffer(force=True):
                            self._log_stream_chunk(chunk, log_id)
                            yield chunk
                        
                        # Authority Stage 2 (Final): Re-verify tools
                        parsed = self.wrapper.parse_response(full_raw, False)
                        self._log_request(5, "parsed_response", parsed, log_id)
                        tc = parsed.get("tool_calls", [])
                        if tc and not final_tool_calls:
                            final_tool_calls = tc

                        usage = self.wrapper.get_usage(ctx_to_use)
                        self._log_request(3, "raw_output", full_raw, log_id)
                        final_chunk = {
                            "content": "",
                            "reasoning_content": parsed.get("reasoning_content", ""),
                            "tool_calls": final_tool_calls,
                            "stop_reason": "stop_sequence",
                            "usage": {
                                "input_tokens": usage["prompt_tokens"],
                                "output_tokens": usage["completion_tokens"]
                            },
                            "full_raw": full_raw
                        }
                        self._log_stream_chunk(final_chunk, log_id)
                        yield final_chunk
                        return

                if (time.time() - last_yield_time) > 0.03:
                    async for chunk in process_and_yield_buffer():
                        self._log_stream_chunk(chunk, log_id)
                        yield chunk
                    last_yield_time = time.time()

    def _mock_generate(self, internal_req: dict) -> dict:
        """Generate mock response for testing."""
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
