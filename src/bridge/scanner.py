
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
    A stateful scanner that implements preventative buffering.
    It holds back text that MIGHT be a tag until it's confirmed or refuted.
    """
    def __init__(self, protected_tags: List[str]):
        """
        protected_tags: List of strings to buffer and hide if they match.
        Example: ["<thought>", "</thought>", "<tool_code>", "</tool_code>"]
        """
        self.protected_tags = sorted(protected_tags, key=len, reverse=True)
        self.buffer = ""
        self.block_stack = []  # Stack of (tag, normalized_type)
        
        self.max_tag_len = max([len(t) for t in protected_tags]) if protected_tags else 0
        
    def push(self, text: str) -> List[ScanEvent]:
        """
        Push new text into the scanner. Returns a list of ScanEvent objects.
        """
        self.buffer += text
        events = []
        
        while self.buffer:
            # 1. Check if the buffer EXACTLY starts with any protected tag
            matched_tag = None
            for tag in self.protected_tags:
                if self.buffer.startswith(tag):
                    matched_tag = tag
                    break
            
            if matched_tag:
                # We found a tag!
                # print(f"DEBUG: Found tag {matched_tag}")
                
                # Determine if it's an end tag
                is_end = "</" in matched_tag or "_end|>" in matched_tag
                
                if is_end:
                    # Pop from stack if matches (simplified)
                    tag_to_end = matched_tag.replace("/", "") if "</" in matched_tag else matched_tag
                    # Find the most recent matching start tag if possible, or just pop
                    events.append(ScanEvent("block_end", matched_tag, tag=self.current_block_type))
                    if self.block_stack:
                        self.block_stack.pop()
                else:
                    # Extract pure tag name: <function= -> function, <invoke name= -> invoke
                    inner = matched_tag.strip("<>").replace("/", "")
                    # Split by = first, then space
                    tag_type = inner.split("=")[0].split()[0]
                    events.append(ScanEvent("block_start", matched_tag, tag=tag_type))
                    self.block_stack.append((matched_tag, tag_type))
                
                self.buffer = self.buffer[len(matched_tag):]
                continue

            # 2. Check if the buffer is a prefix of any protected tag
            # We must be careful: if it's a prefix, we wait, but ONLY if it could eventually match.
            is_any_prefix = False
            for tag in self.protected_tags:
                if tag.startswith(self.buffer):
                    is_any_prefix = True
                    break
            
            if is_any_prefix:
                # It's a partial match. If we haven't reached max_tag_len, wait for more data.
                if len(self.buffer) < self.max_tag_len:
                    break
                else:
                    # We reached max length and it's still just a prefix?
                    # This shouldn't happen if max_tag_len is correct, 
                    # but if it does, we must flush to avoid deadlocks.
                    pass

            # 3. Not a tag, and not a prefix (or we exceeded max length).
            # We should flush as much as possible until the NEXT potential tag start.
            
            # Find the first index `i > 0` where `self.buffer[i:]` is a prefix of any tag.
            first_potential_start = -1
            for i in range(1, len(self.buffer)):
                sub = self.buffer[i:]
                possible = False
                for tag in self.protected_tags:
                    if tag.startswith(sub) or sub.startswith(tag):
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
                # Flush up to the point where a new tag might start
                content_to_flush = self.buffer[:first_potential_start]
                self.buffer = self.buffer[first_potential_start:]
            
            if content_to_flush:
                etype = "block_content" if self.block_stack else "content"
                events.append(ScanEvent(etype, content_to_flush, tag=self.current_block_type))
        
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
