
from typing import List, Dict, Any, Optional, Tuple, Callable
import dataclasses
import re

@dataclasses.dataclass
class ScanEvent:
    type: str  # "content", "block_start", "block_content", "block_end"
    data: str
    tag: Optional[str] = None

class StreamScanner:
    """
    A stateful scanner that handles LLM token streams with three types of markers:
    1. Skip Tokens (Stateless): Tokens to be silently discarded (e.g., <|start|>).
    2. Block Tokens (Stateful): Markers that start/end blocks (e.g., <tool_call>, </thought>).
       - Determine context boundaries.
       - Events generated for these allow Parsers to switch modes (Streaming vs Buffering).
    
    Implements 'Wait-if-Prefix' logic to correctly handle overlapping tokens (greedy match prevention).
    """
    def __init__(self, block_tokens: List[str], skip_tokens: List[str] = None, end_tokens: List[str] = None):
        """
        block_tokens: List of strings that signify block boundaries.
        skip_tokens: List of tokens to silently discard.
        end_tokens: Explicit list of tokens that act as block closers.
        """
        self.block_tokens = sorted(block_tokens, key=len, reverse=True)
        # Check longer skip tokens first to prevent partial match issues
        self.skip_tokens = sorted(skip_tokens or [], key=len, reverse=True)
        self.end_tokens = set(end_tokens or [])
        self.buffer = ""
        self.block_stack = []  # Stack of (tag, normalized_type)
        
        # We need to account for all tokens matched against buffer
        self.all_tokens = self.block_tokens + self.skip_tokens
        # Re-sort combined list to ensure global longest-match priority
        self.all_tokens.sort(key=len, reverse=True)
        
        self.max_token_len = max([len(t) for t in self.all_tokens]) if self.all_tokens else 0
        
    def push(self, text: str) -> List[ScanEvent]:
        """
        Push new text into the scanner. Returns a list of ScanEvent objects.
        """
        self.buffer += text
        events = []
        
        while self.buffer:
            matched_token = None
            
            # 1. Exact Match Check
            # Find the best matching token at the start (longest match)
            # Since self.all_tokens is sorted by len desc, 
            # the first match is technically the longest among those that match.
            for token in self.all_tokens:
                if self.buffer.startswith(token):
                    matched_token = token
                    break
            
            if matched_token:
                # 2. Ambiguity Check (Lookahead)
                # Even if we matched a token (e.g. "<|start|>"), we must check if the buffer 
                # matches the PREFIX of a LONGER token (e.g. "<|start|>assistant").
                # If so, we must WAIT for more data, unless we exceeded max_token_len.
                
                is_prefix_of_longer = False
                if len(self.buffer) < self.max_token_len:
                    for token in self.all_tokens:
                        # Logic: It's a prefix of 'token', AND 'token' is longer than 'matched_token'
                        if token != matched_token and len(token) > len(matched_token) and token.startswith(self.buffer):
                            is_prefix_of_longer = True
                            break
                
                if is_prefix_of_longer:
                     # Wait for more data
                     break
                
                # 3. Process Logic
                
                # Case A: Skip Token -> Swallow
                if matched_token in self.skip_tokens:
                     self.buffer = self.buffer[len(matched_token):]
                     continue

                # Case B: Block Token -> Emit Event
                # Determine if it's an end token
                # Check explicit list or fallback to standard XML checks
                is_end = matched_token in self.end_tokens or "</" in matched_token
                
                if is_end:
                    # Pop from stack if matches (simplified)
                    tag_to_end = matched_token.replace("/", "") if "</" in matched_token else matched_token
                    # Find the most recent matching start tag if possible, or just pop
                    events.append(ScanEvent("block_end", matched_token, tag=self.current_block_type))
                    if self.block_stack:
                        self.block_stack.pop()
                else:
                    # Extract pure tag name: <function= -> function, <invoke name= -> invoke
                    inner = matched_token.strip("<>").replace("/", "")
                    # Split by = first, then space
                    tag_type = inner.split("=")[0].split()[0]
                    events.append(ScanEvent("block_start", matched_token, tag=tag_type))
                    self.block_stack.append((matched_token, tag_type))
                
                self.buffer = self.buffer[len(matched_token):]
                continue

            # 4. Partial Match Check
            # We must be careful: if it's a prefix, we wait.
            is_any_prefix = False
            for token in self.all_tokens:
                if token.startswith(self.buffer):
                    is_any_prefix = True
                    break
            
            if is_any_prefix:
                # It's a partial match. If we haven't reached max_token_len, wait for more data.
                if len(self.buffer) < self.max_token_len:
                    break

            # 5. Flush Non-Matching Content
            # Not a token, and not a prefix (or we exceeded max length).
            # We should flush as much as possible until the NEXT potential token start.
            
            # Find the first index `i > 0` where `self.buffer[i:]` is a prefix of any token.
            first_potential_start = -1
            for i in range(1, len(self.buffer)):
                sub = self.buffer[i:]
                possible = False
                for token in self.all_tokens:
                    if token.startswith(sub) or sub.startswith(token):
                        possible = True
                        break
                if possible:
                    first_potential_start = i
                    break
            
            if first_potential_start == -1:
                # The entire buffer is "safe" content
                content_to_flush = self.buffer
                self.buffer = ""
            else:
                # Flush up to the point where a new token might start
                content_to_flush = self.buffer[:first_potential_start]
                self.buffer = self.buffer[first_potential_start:]
            
            if content_to_flush:
                if not self.block_stack:
                    # Implicit Block Injection:
                    # If we have content but no active block, inject a default container.
                    # This ensures all content (even from unstructured models) is captured in a block.
                    self.block_stack.append(("<implicit>", "implicit_content"))
                    events.append(ScanEvent("block_start", "", tag="implicit_content"))
                
                events.append(ScanEvent("block_content", content_to_flush, tag=self.current_block_type))
        
        return events

    @property
    def current_block_type(self) -> Optional[str]:
        return self.block_stack[-1][1] if self.block_stack else None

    def flush(self) -> List[ScanEvent]:
        """Final flush at end of stream."""
        # print(f"DEBUG: flush buffer='{self.buffer}'")
        if not self.buffer:
            return []
        etype = "block_content" if self.block_stack else "content"
        ev = ScanEvent(etype, self.buffer, tag=self.current_block_type)
        self.buffer = ""
        return [ev]

    # Legacy compatibility for old consume() API
    def consume(self, chunk: str, is_final: bool = False) -> List[Tuple[str, str, Optional[str]]]:
        events = self.push(chunk)
        if is_final:
            events.extend(self.flush())
        
        res = []
        for ev in events:
            if ev.type == "content":
                res.append(("text", ev.data, None))
            elif ev.type == "block_content":
                res.append(("block_chunk", ev.data, ev.tag))
        return res
