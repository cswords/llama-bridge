import re
from typing import List, Tuple, Optional

class StreamScanner:
    """
    A robust streaming scanner for LLM output that handles:
    1. Text/Thought separation.
    2. Suppression of tool call blocks (<tool_call>...</tool_call>).
    3. Skipping of self-contained tool tags (<invoke ...>, <function=...>).
    4. Robustness against partial tags and token fragmentation.
    5. 'Word Bubble' protection (max tag length) to avoid hanging on '<' symbols in math/code.
    """
    
    def __init__(self):
        self.buffer = ""
        self.state = "text"  # text, thought, suppress_block, potential_tag
        self.suppress_end_tag = "" 
        self.max_scannable_len = 80 # Max length to buffer before deciding it's not a tag
        
    def consume(self, chunk: str, is_final: bool = False) -> List[Tuple[str, str]]:
        """
        Ingest a chunk of text and return a list of (type, text) tuples.
        types: 'content', 'thought'
        """
        self.buffer += chunk
        events = []
        
        while True:
            # -------------------------------------------------------------
            # STATE: TEXT (Normal output)
            # -------------------------------------------------------------
            if self.state == "text":
                # Look for the start of a tag '<'
                idx = self.buffer.find('<')
                
                if idx == -1:
                    # No tag start found
                    if self.buffer:
                        events.append(("content", self.buffer))
                        self.buffer = ""
                    break
                else:
                    # Tag start found
                    # 1. Yield text before the tag
                    if idx > 0:
                        events.append(("content", self.buffer[:idx]))
                    
                    # 2. Switch to potential_tag mode with the rest
                    self.buffer = self.buffer[idx:]
                    self.state = "potential_tag"
                    continue

            # -------------------------------------------------------------
            # STATE: THOUGHT (Inside <thought> or <think>)
            # -------------------------------------------------------------
            elif self.state == "thought":
                # Look for closing tag </thought> or </think>
                # We need to be careful about partial matches of the closing tag
                
                # Try to find a closing tag
                match = re.search(r'</(?:thought|think)>', self.buffer)
                if match:
                    # Found it. Yield thought content up to tag.
                    content = self.buffer[:match.start()]
                    if content:
                        events.append(("thought", content))
                    
                    # Consume tag and switch back to text
                    self.buffer = self.buffer[match.end():]
                    self.state = "text"
                    continue
                else:
                    # No closing tag yet. 
                    # We can safely yield everything EXCEPT the suffix that might be a partial closing tag.
                    # Partial closing tag starts with <
                    last_lt = self.buffer.rfind('<')
                    if last_lt != -1 and len(self.buffer) - last_lt < 20:
                         # Possible start of </thought>
                         # Check if it partially matches prefixes of expected closing tags
                         partial = self.buffer[last_lt:]
                         
                         # Regex covering:
                         # </
                         # </t, </th, </tho, ... </thought
                         # </thi, ... </think
                         # Also <, </
                         
                         # Valid prefixes for "thought" or "think" after </
                         # t, th, thi, thin, think
                         # tho, thou, thoug, though, thought
                         
                         if re.match(r'^</?(?:t|th|thi|thin|think|tho|thou|thoug|though|thought)?$', partial):
                             # Yield safe part
                             safe_part = self.buffer[:last_lt]
                             if safe_part:
                                 events.append(("thought", safe_part))
                             self.buffer = self.buffer[last_lt:]
                             if is_final:
                                 # End of stream, just yield the rest
                                 events.append(("thought", self.buffer))
                                 self.buffer = ""
                                 break
                             break # Wait for more
                    
                    # No dangerous suffix, yield all
                    events.append(("thought", self.buffer))
                    self.buffer = ""
                    break

            # -------------------------------------------------------------
            # STATE: SUPPRESS_BLOCK (Inside <tool_call>)
            # -------------------------------------------------------------
            elif self.state == "suppress_block":
                # We want to suppress everything until we find self.suppress_end_tag
                # e.g. </tool_call> or </minimax:tool_call>
                
                if not self.suppress_end_tag:
                    # Fallback default
                    self.suppress_end_tag = "</tool_call>"

                idx = self.buffer.find(self.suppress_end_tag)
                if idx != -1:
                    # Found end tag. Skip everything up to the end of it.
                    self.buffer = self.buffer[idx + len(self.suppress_end_tag):]
                    self.state = "text"
                    self.suppress_end_tag = ""
                    continue
                else:
                    # Not found. Keep buffering?
                    # Actually, since we are suppressing, we can just DISCARD safely 
                    # assuming no partial match at end.
                    # We need to preserve partial match of the end tag.
                    
                    # Only keep the last N chars that could be start of end tag
                    keep_len = len(self.suppress_end_tag) + 5
                    if len(self.buffer) > keep_len:
                        self.buffer = self.buffer[-keep_len:]
                    
                    if is_final:
                        self.buffer = ""
                    break

            # -------------------------------------------------------------
            # STATE: POTENTIAL_TAG (Saw '<', deciding what it is)
            # -------------------------------------------------------------
            elif self.state == "potential_tag":
                # We have buffer starting with '<'.
                # We wait for '>' or until buffer gets too long.
                
                # Check for tag closure
                tag_end_idx = self.buffer.find('>')
                
                if tag_end_idx != -1:
                    # We have a complete tag candidate: <... >
                    full_tag = self.buffer[:tag_end_idx+1]
                    
                    # Analyze the tag
                    # Simplify: remove attributes for checking type
                    # e.g. <invoke name="foo"> -> <invoke
                    core_match = re.match(r'^</?([a-zA-Z0-9_:-]+)', full_tag)
                    
                    if core_match:
                        raw_name = core_match.group(1)
                        is_closing = full_tag.startswith("</")
                        
                        # logic
                        if raw_name in ["thought", "think"]:
                            if is_closing:
                                # A stray </thought> in text mode? Ignore it
                                pass
                            else:
                                self.state = "thought"
                        
                        elif raw_name in ["tool_call", "minimax:tool_call"]:
                            if is_closing:
                                pass
                            else:
                                self.state = "suppress_block"
                                self.suppress_end_tag = f"</{raw_name}>"
                        
                        elif raw_name in ["invoke", "function", "parameter"]:
                            # Inline tool tags. We just skip this tag.
                            # Stay in text mode.
                            pass
                            
                        else:
                            # Unknown tag (e.g. <div>, <br>, < 5). 
                            # Treat as content.
                            events.append(("content", full_tag))
                            
                    else:
                        # '<' followed by junk or just space? e.g. '< 5'
                        events.append(("content", full_tag))
                    
                    # Move past this tag
                    self.buffer = self.buffer[tag_end_idx+1:]
                    
                    if self.state == "potential_tag":
                        self.state = "text" # Revert to text if we didn't switch
                    continue

                else:
                    # No '>' yet.
                    # If buffer is too long, assume it's not a tag.
                    if len(self.buffer) > self.max_scannable_len:
                        # Flush the first char '<' as content and retry
                        events.append(("content", "<"))
                        self.buffer = self.buffer[1:]
                        self.state = "text"
                        continue
                    
                    # Wait for more data
                    if is_final:
                         events.append(("content", self.buffer))
                         self.buffer = ""
                    break
                    
        return events
