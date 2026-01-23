
import asyncio
import sys
import os
import re

# Simulate the fixed scanner logic from bridge.py
def simulate_bridge_scanner(raw_stream):
    try:
        from src.utils.scanner import StreamScanner
    except ImportError:
        # If running from root without package context
        sys.path.append(os.getcwd())
        from src.utils.scanner import StreamScanner
    
    scanner = StreamScanner()
    results = []

    for chunk in raw_stream:
        results.extend(scanner.consume(chunk))
    # Flush final
    results.extend(scanner.consume("", is_final=True))
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
    
    # User's specific broken case
    # invoke< nameBash<="commandgit --</>
    # name"> staged changesparameter</>
    user_mess_stream = [
        "preamble",
        "invoke< nameBash<=\"commandgit --</>\n",
        "   name\"> staged changesparameter</>\n",
        "  < nameB\">\n"
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
        
    print("\n--- Testing User's Messy Stream (Should passthrough as text) ---")
    # Because these look like garbage, we expect them to be yielded as content, NOT suppressed 
    # (unless they perfectly match strict tags). The user complained that this garbage WAS being output.
    # Wait, the user said "invoke< nameBash...". 
    # If the model outputs garbage, we should just print it. 
    # The user's request was: "cache output stream... to avoid encountered really单独出现的小于号"
    # So if it IS garbage, we print it. If it IS a tool call, we suppress it.
    
    out_mess = simulate_bridge_scanner(user_mess_stream)
    for t, c in out_mess:
        print(f"[{t.upper()}]: {repr(c)}")
