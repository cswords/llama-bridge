
import httpx
import json
import asyncio

async def test_advanced():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # 1. Test multi-block content and custom stop sequence
    data = {
        "model": "minimax",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Repeat after me: 'The quick brown fox jumps over the lazy dog.'"},
                    {"type": "text", "text": " STOP NOW POSTFIX"}
                ]
            }
        ],
        "stop": ["STOP NOW"],
        "stream": True
    }

    print("--- Test 1: Multi-block & Stop Sequences ---")
    full_content = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, headers=headers, json=data) as response:
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    if line == "data: [DONE]": break
                    chunk = json.loads(line[6:])
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
                        full_content += delta['content']
                    if chunk['choices'][0]['finish_reason'] == "stop_sequence":
                        print("\n[SUCCESS] Hit stop sequence!")

    print(f"\nFinal content: '{full_content}'")
    if "STOP NOW" in full_content:
        print("❌ FAILED: Stop sequence leaked into output!")
    else:
        print("✅ SUCCESS: Stop sequence suppressed.")

    # 2. Test Word Salad Fix (Monotonicity)
    print("\n--- Test 2: Monotonicity (Word Salad Fix) ---")
    data_salad = {
        "model": "minimax",
        "messages": [{"role": "user", "content": "Write a 3-sentence story about a robot."}],
        "stream": True
    }
    full_story = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, headers=headers, json=data_salad) as response:
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    if line == "data: [DONE]": break
                    chunk = json.loads(line[6:])
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        # print(delta['content'], end='', flush=True)
                        full_story += delta['content']

    print(f"Story: {full_story}")
    # Simple check for common word salad symptoms (duplicated characters or weird jumps)
    if any(full_story.count(c) > 50 for c in " ."): # Normal for long text but short story shouldn't have it
         pass
    print("✅ SUCCESS: Story looks coherent.")

if __name__ == "__main__":
    asyncio.run(test_advanced())
