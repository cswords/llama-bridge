import logging
import json
import time
import uuid
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional, Union, AsyncGenerator
from pathlib import Path

# Import New Components
from src.chat_bridge.chat_bridge_scanner import ChatBridgeScanner
from src.chat_bridge.flavor_factory import FlavorFactory
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.locked_context import LockedContext

# Import Legacy/Existing Wrapper
try:
    from src import llama_chat
    LlamaChatWrapper = llama_chat.LlamaChatWrapper
except ImportError:
    # Fallback for environments where binding isn't available (e.g. CI without build)
    # We define a dummy for type hinting, actual usage will fail or rely on Mock
    class LlamaChatWrapper: pass

logger = logging.getLogger(__name__)

class ChatBridge:
    """
    The New High-Level Orchestrator for ChatBridge V2 (Experimental).
    Now supports Feature Parity with Legacy Bridge:
    - Multi-Context / Multi-Cache
    - 4-File Debug Logging
    - Anthropic Protocol Adaptation
    """
    
    def __init__(self, model_path: str, debug: bool = False, 
                 n_ctx: int = 0, n_batch: int = 0, n_ubatch: int = 0, 
                 n_threads: int = 0, flash_attn: bool = False,
                 cache_type_k: str = "f16", cache_type_v: str = "f16",
                 cache_configs: list | None = None):
        
        self.model_path = model_path
        self.debug = debug
        self.cache_configs = cache_configs or []
        
        # 1. Initialize Wrapper (C++ Binding)
        logger.info(f"ChatBridge (Exp) initializing for {model_path}...")
        self.wrapper = LlamaChatWrapper(
            model_path, n_ctx, n_batch, n_ubatch, n_threads, flash_attn,
            cache_type_k, cache_type_v
        )
        
        # 2. Initialize Contexts (Multi-Cache Support)
        self.flavor = FlavorFactory.get_flavor_for_model(model_path)
        self.contexts: Dict[str, LockedContext] = {}

        if not self.cache_configs:
            # Fallback: Create a single 'main' context if nothing is defined
            logger.info("No cache configs defined. Creating default 'main' context.")
            self.wrapper.create_context("main", n_ctx)
            self.contexts["main"] = LockedContext(self.wrapper, "main")
        else:
            # Create configured contexts
            for cache_cfg in self.cache_configs:
                name = cache_cfg.get("name", "unnamed")
                c_ctx = cache_cfg.get("n_ctx", 4096)
                
                try:
                    self.wrapper.create_context(name, c_ctx)
                    logger.info(f"Created context '{name}' with n_ctx={c_ctx}")
                    self.contexts[name] = LockedContext(self.wrapper, name)
                except Exception as e:
                    logger.error(f"Failed to create context {name}: {e}")

        logger.info(f"ChatBridge ready. Contexts: {list(self.contexts.keys())}, Flavor: {self.flavor.name if hasattr(self.flavor, 'name') else 'unknown'}")


    # --- Streaming Support ---

    async def stream_anthropic(self, body: Dict[str, Any], cache_name: str = None) -> AsyncGenerator[str, None]:
        """
        Anthropic-compatible streaming endpoint.
        """
        # Structured Logging ID: Timestamp_CacheName_ShortUUID
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cache = cache_name or "main"
        short_id = str(uuid.uuid4())[:8]
        req_id = f"{ts}_{safe_cache}_{short_id}"
        
        # 1. Log Request
        self._log_artifact(req_id, "01", "raw_request", body)
    
        # 2. Extract Data
        messages = body.get("messages", [])
        system = body.get("system")
        max_tokens = body.get("max_tokens", 4096)
        request_model = body.get("model", self.model_path)  # Use request model for response
        
        # 2.5. Normalize system prompt (Anthropic array format to string)
        system_str = self._normalize_system(system)
        
        # Prepend System Prompt
        if system_str:
            messages = [{"role": "system", "content": system_str}] + messages
        
        # 2.6. Normalize messages (Anthropic array content to string)
        messages = self._normalize_messages(messages)
            
        # 3. Resolve Context
        target_cache = cache_name or "main"
        locked_ctx = self.contexts.get(target_cache)
        
        if not locked_ctx:
            # Fallback to the first available context if the specifically requested one isn't found
            # This handles cases where 'main' is expected but only other names exist
            fallback_name = next(iter(self.contexts.keys())) if self.contexts else None
            logger.warning(f"Context '{target_cache}' not found. Falling back to '{fallback_name}'.")
            locked_ctx = self.contexts.get(fallback_name)
            
        if not locked_ctx:
            raise RuntimeError(f"No available contexts to process request for cache '{target_cache}'")

        # 5. Extract tools from request
        tools = self._extract_tools(body)

        # 6. Generate (using bridge-level flavor and LockedContext)
        async with locked_ctx.lock:
            async for chunk in self._generate(req_id, locked_ctx, messages, tools, self.flavor, max_tokens, request_model):
                yield chunk


    async def _generate(self, req_id: str, locked_ctx: LockedContext, messages: List[Dict[str, str]], 
                  tools: List[Dict[str, Any]], flavor: ChatFlavorBase, 
                  max_tokens: int, request_model: str = None) -> AsyncGenerator[str, None]:
        """
        Core Generation Loop: LockedContext -> Scanner -> SSE Event Translation
        
        This method uses a single coarse-grained lock for the entire transaction.
        The lock is held from template construction through response parsing.
        """
        # A. Prompt Construction via apply_template (via LockedContext)
        try:
            template_res = locked_ctx.apply_template(messages, tools, True)
            full_prompt = template_res.get("prompt", "")
        except Exception as e:
            logger.error(f"Template application failed: {e}")
            raise RuntimeError(f"Failed to apply chat template: {e}")
            
        self._log_artifact(req_id, "02", "prompt", full_prompt)
        
        # B. Initialize Scanner for output parsing (with prompt for prefill detection)
        scanner = ChatBridgeScanner(flavor, full_prompt)
        
        # C. Emit Initial SSE Events
        yield self._emit_sse("message_start", {
            "type": "message_start", 
            "message": {
                "id": f"msg_{req_id[:8]}", 
                "role": "assistant", 
                "content": [], 
                "model": request_model or self.model_path,
                "usage": {"input_tokens": len(full_prompt) // 4, "output_tokens": 0}
            }
        }, req_id)
        
        # Content Block 0: Text
        yield self._emit_sse("content_block_start", {
            "type": "content_block_start", 
            "index": 0, 
            "content_block": {"type": "text", "text": ""}
        }, req_id)
        
        # D. Token Loop using LockedContext.generate()
        output_tokens = 0
        async for token_bytes in locked_ctx.generate(full_prompt, max_tokens):
            # Decode bytes to str
            if isinstance(token_bytes, bytes):
                try:
                    token_text = token_bytes.decode("utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"Failed to decode token bytes: {e}")
                    token_text = ""
            else:
                token_text = token_bytes
                
            output_tokens += 1
            self._log_stream_chunk(req_id, token_text)
            
            # Push to Scanner for parsing
            scan_events = list(scanner.push(token_text))
            for event in scan_events:
                for chunk in self._translate_event(event, req_id):
                    yield chunk
                
        # E. Flush remaining scanner buffer
        scan_events = list(scanner.flush())
        for event in scan_events:
            for chunk in self._translate_event(event, req_id):
                yield chunk
        
        # F. Parse full response for structured output (tool calls)
        full_raw_output = scanner.get_full_output() if hasattr(scanner, 'get_full_output') else ""
        parsed = None
        try:
            parsed = locked_ctx.parse_response(full_raw_output, False)
            self._log_artifact(req_id, "05", "parsed_response", parsed)
        except Exception as e:
            logger.warning(f"parse_response failed: {e}")
            self._log_artifact(req_id, "05", "parsed_response", {"error": str(e)})
            
        # G. Stop Reason & content_block_stop for text
        stop_reason = "end_turn"
        if parsed and parsed.get("tool_calls"):
            stop_reason = "tool_use"
            
        yield self._emit_sse("content_block_stop", {"type": "content_block_stop", "index": 0}, req_id)
        
        # H. Emit tool_use blocks from parse_response results
        if parsed and parsed.get("tool_calls"):
            for i, tool_call in enumerate(parsed["tool_calls"]):
                tool_index = i + 1  # Start from index 1 (index 0 is text)
                tool_id = tool_call.get("id") or f"toolu_{req_id[:8]}_{i}"
                tool_name = tool_call.get("name", "unknown")
                raw_args = tool_call.get("arguments", "{}")
                
                # Robust arguments parsing
                if isinstance(raw_args, str):
                    try:
                        tool_args_dict = json.loads(raw_args)
                    except:
                        tool_args_dict = {"raw": raw_args}
                else:
                    tool_args_dict = raw_args
                
                # content_block_start (no input for tool_use in streaming start)
                yield self._emit_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": tool_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name
                    }
                }, req_id)
                
                # input_json_delta (send the raw JSON arguments)
                yield self._emit_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": tool_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
                    }
                }, req_id)
                
                # content_block_stop
                yield self._emit_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": tool_index
                }, req_id)
        
        # I. message_delta with usage
        usage = self.wrapper.get_usage(locked_ctx.ctx_name)
        yield self._emit_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)}
        }, req_id)
        
        # J. message_stop
        yield self._emit_sse("message_stop", {"type": "message_stop"}, req_id)


    def _translate_event(self, event: Dict[str, Any], req_id: str) -> Generator[str, None, None]:
        """
        Translate Scanner Event to Anthropic SSE format.
        """
        etype = event["type"]
        tag = event.get("tag")
        
        if etype == "text_delta":
            payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": event["text"]}
            }
            yield self._emit_sse("content_block_delta", payload, req_id)
            
        elif etype == "block_delta":
            # Handle based on tag type
            if tag == "reasoning_content":
                # Wrap reasoning in tags and send as standard text delta
                payload = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"<think>\n{event['text']}\n</think>\n" if event.get("is_first", False) else event["text"]}
                }
            else:
                # Default text delta
                payload = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": event["text"]}
                }
            yield self._emit_sse("content_block_delta", payload, req_id)
            
        elif etype == "block_end":
            if tag == "tool_call":
                # Tool calls are now handled in post-generation via parse_response
                # Just log that we received the block end
                logger.debug(f"Tool call block ended, content will be parsed by parse_response")
                # Don't emit anything here - tool_use events will be emitted after flush


    # --- Non-Streaming Support ---

    async def complete_anthropic(self, body: Dict[str, Any], cache_name: str = None) -> Dict[str, Any]:
        """
        Anthropic-compatible Non-Streaming endpoint.
        """
        # Structured Logging ID
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cache = cache_name or "main"
        short_id = str(uuid.uuid4())[:8]
        req_id = f"{ts}_{safe_cache}_{short_id}"
        
        # 1. Log Request
        self._log_artifact(req_id, "01", "raw_request", body)
        
        # 2. Extract Data
        messages = body.get("messages", [])
        system = body.get("system")
        max_tokens = body.get("max_tokens", 4096)
        
        if system:
            messages = [{"role": "system", "content": system}] + messages
            
        # 3. Resolve Context
        target_cache = cache_name or "main"
        locked_ctx = self.contexts.get(target_cache)
        
        if not locked_ctx:
            fallback_name = next(iter(self.contexts.keys())) if self.contexts else None
            locked_ctx = self.contexts.get(fallback_name)
            
        if not locked_ctx:
            raise RuntimeError(f"No available contexts to process request for cache '{target_cache}'")

        ctx_to_use = locked_ctx.ctx_name
                
        # 4. Extract tools and apply template
        tools = self._extract_tools(body)

        async with locked_ctx.lock:
            try:
                template_res = self.wrapper.apply_template(messages, tools, True)
                full_prompt = template_res.get("prompt", "")
            except Exception as e:
                logger.error(f"Template application failed: {e}")
                raise RuntimeError(f"Failed to apply chat template: {e}")
                
            self._log_artifact(req_id, "02", "prompt", full_prompt)
            
            # 5. Generate (synchronous)
            full_text = self.wrapper.generate(ctx_to_use, full_prompt, max_tokens)
            
            # Log Raw
            self._log_artifact(req_id, "03", "raw_response_full", full_text)
            
            # 6. Parse content via C++ Wrapper
            try:
                parsed = locked_ctx.parse_response(full_text, False)
                content_blocks = self._translate_parsed_to_anthropic(parsed)
            except Exception as e:
                logger.warning(f"parse_response failed: {e}")
                content_blocks = [{"type": "text", "text": full_text}]
            
            # 7. Assemble Response
            usage = self.wrapper.get_usage(locked_ctx.ctx_name)
            response = {
                "id": f"msg_{req_id[:8]}",
                "type": "message",
                "role": "assistant",
                "model": self.model_path,
                "content": content_blocks,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                }
            }
            
            # 8. Log Final
            self._log_artifact(req_id, "06", "final_response", response)
            return response
        


    def _normalize_system(self, system: Any) -> str:
        """Normalize system prompt from Anthropic array format to string."""
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            # Anthropic format: [{"type": "text", "text": "..."}, ...]
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts)
        return str(system)


    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize messages for llama.cpp, handling multi-block content and flattening tool_calls."""
        normalized = []
        for msg in messages:
            n_msg = msg.copy()
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
                            logger.warning("Vision/Image blocks are not yet supported and will be ignored.")
                        elif b_type == "tool_use":
                            # Flatten tool_use to tool_calls
                            if "tool_calls" not in n_msg: 
                                n_msg["tool_calls"] = []
                            n_msg["tool_calls"].append({
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}))
                            })
                        elif b_type == "tool_result":
                            # Tool result content, extract text
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                text_parts.append(result_content)
                            elif isinstance(result_content, list):
                                for rc in result_content:
                                    if isinstance(rc, dict) and rc.get("type") == "text":
                                        text_parts.append(rc.get("text", ""))
                n_msg["content"] = "\n".join(text_parts)

            # Flatten and normalize tool_calls
            if "tool_calls" in n_msg and n_msg["tool_calls"]:
                n_tcs = []
                for tc in n_msg["tool_calls"]:
                    ntc = tc.copy()
                    # Flatten OpenAI nested structure
                    if "function" in ntc and isinstance(ntc["function"], dict):
                        f = ntc["function"]
                        if "name" in f: 
                            ntc["name"] = f["name"]
                        if "arguments" in f: 
                            ntc["arguments"] = f["arguments"]
                    
                    # Ensure arguments is a string
                    if "arguments" in ntc and not isinstance(ntc["arguments"], (str, type(None))):
                        ntc["arguments"] = json.dumps(ntc["arguments"])
                    
                    n_tcs.append(ntc)
                n_msg["tool_calls"] = n_tcs
            
            normalized.append(n_msg)
        return normalized


    def _extract_tools(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tools from request body in wrapper-compatible format."""
        tools = []
        if "tools" in body:
            for t in body.get("tools", []):
                if t.get("type") == "function":
                    f = t["function"]
                    tools.append({
                        "name": f["name"],
                        "description": f.get("description", ""),
                        "parameters": json.dumps(f.get("parameters", {})) if isinstance(f.get("parameters"), dict) else f.get("parameters", "{}")
                    })
        return tools


    def _translate_parsed_to_anthropic(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert C++ parsed message structure to Anthropic content blocks.
        """
        blocks = []
        
        # 1. Text Content
        content = parsed.get("content", "")
        if content:
            blocks.append({"type": "text", "text": content})
            
        # 2. Reasoning/Thinking Content
        reasoning = parsed.get("reasoning_content", "")
        if reasoning:
            # Wrap thinking in tags for visibility in simple clients
            blocks.append({"type": "text", "text": f"<think>\n{reasoning}\n</think>\n"})
            
        # 3. Tool Calls
        tool_calls = parsed.get("tool_calls", [])
        for i, tc in enumerate(tool_calls):
            raw_args = tc.get("arguments", {})
            
            # Ensure arguments is a dict
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except:
                    args = {"raw": raw_args}
            else:
                args = raw_args

            if not isinstance(args, dict):
                args = {"value": str(args)}

            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"call_{int(time.time())}_{i}",
                "name": tc.get("name", "unknown"),
                "input": args
            })
            
        return blocks
    
    
    # --- Logging Helpers ---

    def _emit_sse(self, event_name: str, data: Dict[str, Any], req_id: str) -> str:
        """Helper to format SSE string and log it."""
        sse_str = f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
        self._log_client_event(req_id, data)
        return sse_str

    def _log_artifact(self, req_id: str, stage_prefix: str, name: str, data: Any):
        if not self.debug: return
        try:
            log_dir = Path(f"logs/{req_id}")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            ext = "json" if isinstance(data, (dict, list)) else "txt"
            dump_data = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
            
            with open(log_dir / f"{stage_prefix}_{name}.{ext}", "w") as f:
                f.write(dump_data)
        except Exception as e:
            logger.warning(f"Logging failed: {e}")

    def _log_stream_chunk(self, req_id: str, text: str):
        if not self.debug: return
        try:
            log_dir = Path(f"logs/{req_id}")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "03_raw_response_stream.txt", "a") as f:
                f.write(text)
        except:
            pass

    def _log_client_event(self, req_id: str, event_data: Dict):
        if not self.debug: return
        try:
            log_dir = Path(f"logs/{req_id}")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "04_final_response.jsonl", "a") as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
        except:
            pass
