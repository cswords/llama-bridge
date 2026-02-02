import re
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase

logger = logging.getLogger(__name__)

class ChatBridgeScanner:
    """
    Generic Scanner for ChatBridge.
    Driven completely by the Flavor's block_tokens definitions.
    Implements a streaming state machine with 'Transparent' (Streaming) 
    and 'Opaque' (Buffering) modes.
    
    Supports prefill detection: if the prompt ends with an unclosed block start tag,
    the scanner enters the correct block state before processing model output.
    """
    
    def __init__(self, flavor: ChatFlavorBase, prompt: str = ""):
        self.flavor = flavor
        self.block_patterns = self._compile_patterns(flavor.block_tokens)
        self.known_prefixes = self._extract_prefixes(flavor.block_tokens)
        
        # State
        self.buffer = ""
        self.full_output = ""  # Accumulator for complete raw output (model only, no prefill)
        self.current_opaque_tag: Optional[str] = None # If set, we are buffering
        self.current_transparent_tag: Optional[str] = None  # For transparent blocks (e.g. think)
        self.end_pattern: Optional[re.Pattern] = None # The pattern that ends the current opaque block
        self.is_opaque_mode = False # Current mode: True=Opaque/Buffering, False=Transparent/Streaming
        
        # Detect and handle prefill from prompt
        if prompt:
            self._detect_and_apply_prefill(prompt)
    
    def _detect_and_apply_prefill(self, prompt: str) -> None:
        """
        Detect if the prompt ends with an unclosed block start tag.
        If so, prime the scanner state without affecting full_output.
        """
        check_window = 1000
        search_segment = prompt[-check_window:] if len(prompt) > check_window else prompt
        
        last_open_tag_pos = -1
        matched_pattern = None
        
        # Find the last unclosed block start tag
        for pat in self.block_patterns:
            start_re = pat["start_re"]
            end_str = pat["end_str"]
            
            # Find all matches of start pattern in search segment
            matches = list(start_re.finditer(search_segment))
            if not matches:
                continue
            
            last_match = matches[-1]
            last_pos = last_match.start()
            
            # Check if this block is closed
            end_pos = search_segment.find(end_str, last_match.end())
            if end_pos == -1:
                # Not closed - this is a prefilled block
                if last_pos > last_open_tag_pos:
                    last_open_tag_pos = last_pos
                    matched_pattern = pat
        
        if matched_pattern:
            # Enter the correct block state
            tag = matched_pattern["tag"]
            is_opaque = matched_pattern["is_opaque"]
            
            logger.debug(f"[Scanner] Prefill detected: entering '{tag}' block (opaque={is_opaque})")
            
            if is_opaque:
                self.current_opaque_tag = tag
                self.end_pattern = matched_pattern["end_re"]
                self.is_opaque_mode = True
            else:
                # Transparent block (e.g. think/reasoning)
                self.current_transparent_tag = tag
                self.end_pattern = matched_pattern["end_re"]
                self.is_opaque_mode = False
    
    def get_full_output(self) -> str:
        """Return the complete accumulated raw output."""
        return self.full_output
        
    def _compile_patterns(self, token_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pre-compile regex patterns for performance.
        """
        compiled = []
        for t in token_defs:
            start_pat = t["start"]
            if not t.get("start_is_regex", False):
                start_pat = re.escape(start_pat)
            
            end_pat_str = re.escape(t["end"])
            
            compiled.append({
                "start_re": re.compile(start_pat),
                "end_str": t["end"],
                "end_re": re.compile(end_pat_str),
                "tag": t["tag"],
                "is_opaque": t["is_opaque"]
            })
        return compiled

    def _extract_prefixes(self, token_defs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract literal start prefixes for smart buffering.
        Only applies to "text_tag" types (as Control Tokens are assumed atomic).
        E.g. regex "<tool_call(?:...)" -> prefix "<tool_call"
        """
        prefixes = []
        for t in token_defs:
            # Only track prefixes for Text Tags. Control tokens are atomic.
            if t.get("match_type") != "text_tag":
                continue
                
            raw_start = t["start"]
            # Heuristic: Cut off at first special regex char if possible
            # Sanitize: Remove regex groups
            prefix = raw_start.split('(')[0].split('[')[0]
            # Unescape if needed? (regex string might have escaped chars)
            # But raw_start in definition is usually python string.
            # Example: "<tool_call(?:\\s+[^>]*)?>" -> "<tool_call"
            
            # For regex chars like `\|`, we need to clean them if they are in the python string.
            # Let's strip backslashes generally.
            prefix = prefix.replace("\\", "")
            if prefix:
                prefixes.append(prefix)
        return prefixes

    def push(self, chunk: str) -> Generator[Dict[str, Any], None, None]:
        """
        Ingest a text chunk and yield events.
        """
        if chunk:
            logger.debug(f"[Scanner] Push chunk: {len(chunk)} chars, buffer_len(before)={len(self.buffer)}")
            self.full_output += chunk  # Accumulate for parse_response
        
        self.buffer += chunk
        
        while self.buffer:
            # Check if we're inside any block (opaque or transparent)
            current_tag = self.current_opaque_tag or self.current_transparent_tag
            
            if current_tag:
                # --- Inside a Block ---
                # We are looking for the end pattern of the current block
                match = self.end_pattern.search(self.buffer)
                
                if match:
                    # Found the end!
                    end_start_idx = match.start()
                    end_end_idx = match.end()
                    
                    content_chunk = self.buffer[:end_start_idx]
                    
                    if self.is_opaque_mode:
                        # Opaque: We buffered everything. Yield the block.
                        logger.debug(f"[Scanner] End Opaque Block: {current_tag}, content_len={len(content_chunk)}")
                        yield {
                            "type": "block_end",
                            "tag": current_tag,
                            "content": content_chunk 
                        }
                    else:
                        # Transparent: Yield the final chunk
                        if content_chunk:
                            yield {
                                "type": "block_delta", 
                                "tag": current_tag, 
                                "text": content_chunk
                            }
                        logger.debug(f"[Scanner] End Transparent Block: {current_tag}")
                        yield {
                            "type": "block_end",
                            "tag": current_tag
                        }
                        
                    # Advance past the end tag
                    self.buffer = self.buffer[end_end_idx:]
                    
                    # Reset State
                    self.current_opaque_tag = None
                    self.current_transparent_tag = None
                    self.end_pattern = None
                    self.is_opaque_mode = False
                    
                else:
                    # No end tag found yet
                    if self.is_opaque_mode:
                        # Opaque: Keep everything in buffer.
                        logger.debug(f"[Scanner] Buffering Opaque... len={len(self.buffer)}")
                        break
                    else:
                        # Transparent: Yield everything we have so far
                        logger.debug(f"[Scanner] Yielding Transparent Delta: {len(self.buffer)} chars")
                        yield {
                            "type": "block_delta",
                            "tag": current_tag,
                            "text": self.buffer
                        }
                        self.buffer = ""
                        break
                        
            else:
                # --- Not inside any block (Scanning) ---
                earliest_match = None
                matched_def = None
                
                for p in self.block_patterns:
                    m = p["start_re"].search(self.buffer)
                    if m:
                        if earliest_match is None or m.start() < earliest_match.start():
                            earliest_match = m
                            matched_def = p
                
                if earliest_match:
                    start_idx = earliest_match.start()
                    end_idx = earliest_match.end()
                    
                    # 1. Yield pre-match text
                    if start_idx > 0:
                        text_val = self.buffer[:start_idx]
                        logger.debug(f"[Scanner] Yielding Pre-Match Text: len={len(text_val)}")
                        yield {"type": "text_delta", "text": text_val}
                    
                    # 2. Start Block
                    tag = matched_def["tag"]
                    is_opaque = matched_def["is_opaque"]
                    
                    logger.info(f"[Scanner] Start Block: tag={tag}, opaque={is_opaque}")
                    yield {"type": "block_start", "tag": tag}
                    
                    # Advance past start tag
                    self.buffer = self.buffer[end_idx:]
                    
                    # Set State
                    self.current_opaque_tag = tag
                    self.end_pattern = matched_def["end_re"]
                    self.is_opaque_mode = is_opaque
                    
                else:
                    # No FULL start tag found in the current buffer.
                    # SMART BUFFERING STRATEGY:
                    # Check if the buffer *could* be the start of a tag.
                    # "Is `buffer` a prefix of any known_prefix?"
                    # OR "Is any known_prefix a prefix of `buffer`?" (e.g. <tool_call id=... but regex failed?)
                    
                    # Note: If `regex` failed but `buffer` starts with a known prefix, 
                    # it implies the buffer is longer than the prefix but didn't match the full regex 
                    # (e.g. incomplete attributes? or regex expects closing `>`).
                    # If regex requires closing `>`, and we have `<tool_call id="...` (no closing),
                    # then regex won't match, but prefix `<tool_call` matches.
                    # So we MUST buffer.
                    
                    should_buffer = False
                    
                    # Check 1: Buffer is a short prefix of a known tag? (e.g. "<t")
                    for kp in self.known_prefixes:
                        if kp.startswith(self.buffer):
                            should_buffer = True
                            break
                    
                    # Check 2: Buffer starts with a known prefix? (e.g. "<tool_call long_attr=...")
                    # If regex failed, we still hold it if it looks like a tag start.
                    if not should_buffer:
                        for kp in self.known_prefixes:
                            if self.buffer.startswith(kp):
                                should_buffer = True
                                break
                    
                    if should_buffer:
                        # Wait for more data
                        break
                    else:
                        # Buffer contains mismatched data.
                        # BUT, verify if the *tail* of the buffer matches a prefix!
                        # E.g. "Content <tool_" -> We want to yield "Content " and keep "<tool_".
                        
                        longest_suffix_len = 0
                        for kp in self.known_prefixes:
                            # We can check overlaps.
                            # Brute force check suffixes for MVP correctnes.
                            # Check if `kp` starts with `buffer[-k:]`
                            # Optimize: iterate k from 1 to len(buffer) or len(kp)
                            
                            # Limit lookup to length of longest prefix to save time? 
                            # Max prefix len is usually small (~20 chars).
                            
                            limit = len(self.buffer)
                            for k in range(1, limit + 1):
                                suffix = self.buffer[-k:]
                                if kp.startswith(suffix):
                                    if k > longest_suffix_len:
                                        longest_suffix_len = k
                                    # Don't break, larger k might exist for other prefixes? 
                                    # No, for same prefix, largest k is naturally found if we iterate backwards?
                                    # Actually, iterating forward 1..limit finds ALL. We want MAX k.
                        
                        if longest_suffix_len > 0:
                            # Yield head, keep tail
                            to_yield = self.buffer[:-longest_suffix_len]
                            next_buffer = self.buffer[-longest_suffix_len:]
                            
                            if to_yield:
                                yield {"type": "text_delta", "text": to_yield}
                            self.buffer = next_buffer
                            break # Wait for more data to complete the prefix
                        else:
                            # No part of the buffer looks like a tag. Yield all.
                            yield {"type": "text_delta", "text": self.buffer}
                            self.buffer = ""
                            break

    def flush(self) -> Generator[Dict[str, Any], None, None]:
        """
        Must be called at the end of the stream to yield any remaining buffer.
        """
        logger.debug(f"[Scanner] Flushing... Buffer len={len(self.buffer)}")
        if self.buffer:
            # If we are in streaming mode, we might have partial content left?
            # Or if we were buffering for a tag match that never completed (e.g. "<tool_ incomplete").
            # We yield it as text.
            
            # Note: if is_opaque_mode is True, we are inside a block. 
            # Ideally we'd close the block or error, but yielding text is a safe fallback.
            
            logger.warning(f"[Scanner] Flush yielding residual buffer: {self.buffer[:20]}...")
            yield {"type": "text_delta", "text": self.buffer}
            self.buffer = ""


    # Correction on Transparent Logic Implementation in loop:
    # The above loop structure needs a slight tweak to support "Streaming Block" (Transparent).
    # I will refine this in the file write directly.
