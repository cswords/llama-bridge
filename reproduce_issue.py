
import requests
import time
import sys

def send_request(prompt, tools=None):
    url = "http://127.0.0.1:8000/v1/chat/completions"
    messages = [{"role": "user", "content": prompt}]
    data = {
        "messages": messages, 
        "stream": False, 
        "model": "gpt-4o",
        "max_tokens": 512
    }
    if tools:
        data["tools"] = tools

    print(f"Sending prompt length: {len(prompt)}")
    t0 = time.time()
    try:
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        t1 = time.time()
        print(f"Success in {t1-t0:.2f}s")
        return resp.json()
    except Exception as e:
        print(f"Failed: {e}")
        return None

# 1. Main Chat: Large Request
long_prompt = "Hello, please list 100 random numbers. " * 50
print("--- 1. Main Chat (Long) ---")
send_request(long_prompt)

# 2. Ephemeral Probe: Small Request with Tool
tool_def = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
}]
print("\n--- 2. Ephemeral Probe (Tool Check) ---")
# This should be fast and NOT clear the big cache from #1
send_request("Check tools", tools=tool_def)

# 3. Main Chat: Follow up
print("\n--- 3. Main Chat (Follow up) ---")
# This should reuse the cache from #1
send_request(long_prompt + " And 10 more.")
