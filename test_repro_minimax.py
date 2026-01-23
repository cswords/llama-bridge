
import asyncio
import sys
import os
import re

# Simulate the fixed scanner logic from bridge.py
def simulate_bridge_scanner(raw_stream):
    full_raw = ""
    last_pos = 0
    current_mode = "text"
    results = []

    def process_buffer(force=False):
        nonlocal last_pos, current_mode, full_raw
        buffer_results = []
        while last_pos < len(full_raw):
            remaining = full_raw[last_pos:]
            
            if current_mode == "thought":
                end_match = re.search(r'</(?:thought|think)>', remaining)
                if end_match:
                    delta = remaining[:end_match.start()]
                    if delta: buffer_results.append(("thought", delta))
                    last_pos += end_match.end()
                    current_mode = "text"
                else:
                    if not force:
                        last_lt = remaining.rfind('<')
                        if last_lt != -1 and re.match(r'</?(?:th|think|thought).*?$', remaining[last_lt:]):
                            delta = remaining[:last_lt]
                            if delta: buffer_results.append(("thought", delta))
                            last_pos += last_lt
                            break
                    buffer_results.append(("thought", remaining))
                    last_pos += len(remaining)
                    break
            
            elif current_mode == "suppress":
                end_match = re.search(r'</(?:tool_call|minimax:tool_call)>', remaining)
                if end_match:
                    last_pos += end_match.end()
                    current_mode = "text"
                else:
                    last_pos += len(remaining)
                    break
            
            else: # text mode
                combined_pat = r'<(?:thought|think|tool_call|minimax:tool_call|function=|invoke)|</(?:thought|think|tool_call|minimax:tool_call|function|invoke)>'
                match = re.search(combined_pat, remaining)
                if match:
                    delta = remaining[:match.start()]
                    if delta: buffer_results.append(("content", delta))
                    tag = match.group(0)
                    if tag.startswith('</'):
                        last_pos += match.end()
                        continue
                    else:
                        tag_type = tag.strip('<')
                        if tag_type in ["thought", "think"]:
                            current_mode = "thought"
                            last_pos += match.end()
                        elif tag_type in ["tool_call", "minimax:tool_call"]:
                            current_mode = "suppress"
                            last_pos += match.end()
                        else:
                            tag_end = full_raw.find('>', last_pos + match.start())
                            if tag_end != -1:
                                last_pos = tag_end + 1
                            else:
                                last_pos += match.start()
                                return buffer_results
                        continue
                else:
                    if not force:
                        last_lt = remaining.rfind('<')
                        if last_lt != -1 and re.match(r'</?[a-zA-Z0-9_=:]*$', remaining[last_lt:]):
                            delta = remaining[:last_lt]
                            if delta: buffer_results.append(("content", delta))
                            last_pos += last_lt
                            break
                    buffer_results.append(("content", remaining))
                    last_pos += len(remaining)
                    break
        return buffer_results

    for chunk in raw_stream:
        full_raw += chunk
        results.extend(process_buffer())
    results.extend(process_buffer(force=True))
    return results

if __name__ == "__main__":
    # Simulate the exact chaotic output seen by the user
    # Note how tags are broken across chunks to test the "waiting" logic
    chaotic_stream = [
        "⏺ user to the changes. Let check git status what's and the state\n",
        "   the git status staged, commit push>",
        "<minimax:tool_call>",
        "<invoke name=\"git_status\"", # Broken tag
        ">",
        "<parameter name=\"path\">.</parameter>",
        "</invoke>",
        "</minimax:tool_call>",
        "\nDone!"
    ]
    
    print("--- Testing Chaotic MiniMax Stream ---")
    out = simulate_bridge_scanner(chaotic_stream)
    for t, c in out:
        print(f"[{t.upper()}]: {repr(c)}")
    
    # Check if any tag leaked
    full_content = "".join([c for t, c in out if t == "content"])
    if "<name" in full_content or "<invoke" in full_content:
        print("\nFAILURE: Tags leaked into content!")
    else:
        print("\nSUCCESS: No tags leaked!")
