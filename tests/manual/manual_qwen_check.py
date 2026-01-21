"""
Manual E2E test for Qwen 2.5 3B Instruct model.
Verifies text generation and tool calling.
"""

import sys
import asyncio
import httpx
import json
from pathlib import Path

# Config
MODEL_PATH = "models/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf"
BASE_URL = "http://127.0.0.1:8001"

async def test_text_gen():
    print(f"Testing Text Generation with {MODEL_PATH}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        try:
            resp = await client.get(f"{BASE_URL}/health")
            print(f"Health: {resp.json()}")
        except Exception as e:
            print(f"Server not up? {e}")
            return

        # 2. Chat completion
        payload = {
            "model": "qwen2.5",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "stream": True
        }
        
        print("\nSending Chat Request:")
        async with client.stream("POST", f"{BASE_URL}/v1/messages", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]": break
                    try:
                        data = json.loads(data_str)
                        if data["type"] == "content_block_delta" and data["delta"]["type"] == "text_delta":
                            print(data["delta"]["text"], end="", flush=True)
                    except:
                        pass
        print("\n\nText Generation Test Complete.")

async def test_tool_call():
    print(f"\nTesting Tool Calling with {MODEL_PATH}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "qwen2.5",
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"}
            ],
            "tools": [{
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }],
            "stream": True # Streaming tool call!
        }
        
        print("\nSending Tool Request:")
        full_tool_input = ""
        async with client.stream("POST", f"{BASE_URL}/v1/messages", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data["type"] == "content_block_start":
                            print(f"\n[Tool Start] {data['content_block']['name']}")
                        elif data["type"] == "content_block_delta" and data["delta"]["type"] == "input_json_delta":
                            fragment = data["delta"]["partial_json"]
                            full_tool_input += fragment
                            print(fragment, end="", flush=True)
                        elif data["type"] == "message_delta":
                            print(f"\n[Stop Reason] {data['delta']['stop_reason']}")
                    except:
                        pass
        
        print(f"\n\nFull Tool Input: {full_tool_input}")
        
    print("\nTool Calling Test Complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tools":
        asyncio.run(test_tool_call())
    else:
        asyncio.run(test_text_gen())
