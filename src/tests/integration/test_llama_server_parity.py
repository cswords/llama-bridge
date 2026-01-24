
import subprocess
import time
import json
import requests
import os
import shutil
import socket
import uuid
import asyncio
from pathlib import Path
import pytest
from src.llama_chat import LlamaChatWrapper
from src.bridge.base import BridgeBase
from src.bridge.flavors import get_flavor_for_model

# Configuration
MODEL_PATH = "models/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf"
SERVER_BIN = "./vendor/llama.cpp/build/bin/llama-server"

# Parameters
COMMON_PARAMS = {
    "n_ctx": 2048,
    "n_batch": 512,
    "n_threads": 8,
    "flash_attn": True
}

SCENARIOS = [
    {
        "name": "Simple Greeting",
        "messages": [{"role": "user", "content": "Hi, who are you?"}],
        "tools": []
    },
    {
        "name": "Custom System Prompt",
        "messages": [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Where is the treasure?"}
        ],
        "tools": []
    },
    {
        "name": "Tool Definition Parity",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": json.dumps({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                })
            }
        ]
    }
]

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def to_openai_tools(internal_tools):
    oa_tools = []
    for t in internal_tools:
        oa_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": json.loads(t["parameters"]) if isinstance(t["parameters"], str) else t["parameters"]
            }
        })
    return oa_tools

class LlamaServerProcess:
    def __init__(self, model_path, params):
        self.model_path = model_path
        self.params = params
        self.port = get_free_port()
        self.process = None

    def start(self):
        cmd = [
            SERVER_BIN,
            "-m", self.model_path,
            "--port", str(self.port),
            "-c", str(self.params.get("n_ctx", 2048)),
            "-b", str(self.params.get("n_batch", 512)),
            "-t", str(self.params.get("n_threads", 8)),
            "--no-mmap",
            "-n", "128"
        ]
        if self.params.get("flash_attn"):
            cmd.extend(["-fa", "on"])
        
        self.process = subprocess.Popen(
            cmd,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
            text=True
        )
        
        # Wait for server to be ready
        max_retries = 60
        for i in range(max_retries):
            try:
                resp = requests.get(f"http://localhost:{self.port}/health")
                if resp.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        raise RuntimeError(f"Failed to start llama-server on port {self.port}")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()

    def chat_completion(self, messages, tools=None):
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "seed": 42
        }
        if tools: payload["tools"] = tools
        resp = requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def chat_completion_stream(self, messages, tools=None):
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "seed": 42,
            "stream": True
        }
        if tools: payload["tools"] = tools
        resp = requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        full_content = ""
        last_chunk = None
        stream_tool_calls = []
        for line in resp.iter_lines():
            if not line: continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                data = json.loads(data_str)
                last_chunk = data
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content: full_content += content
                    tc_deltas = delta.get("tool_calls", [])
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        while len(stream_tool_calls) <= idx:
                            stream_tool_calls.append({"name": "", "arguments": ""})
                        f = tcd.get("function", {})
                        if "name" in f: stream_tool_calls[idx]["name"] += f["name"]
                        if "arguments" in f: stream_tool_calls[idx]["arguments"] += f["arguments"]
        return {"content": full_content, "tool_calls": stream_tool_calls, "usage": last_chunk.get("usage") if last_chunk else None}

    def raw_completion(self, prompt, stream=False):
        payload = {"prompt": prompt, "temperature": 0.0, "seed": 42, "n_predict": 128, "stream": stream}
        resp = requests.post(f"http://localhost:{self.port}/completion", json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        if not stream:
            data = resp.json()
            return {"content": data["content"]}
        else:
            full_content = ""
            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    full_content += data.get("content", "")
                    if data.get("stop"): break
            return {"content": full_content}

@pytest.fixture(scope="module")
def bridge():
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    b = BridgeBase(
        model_path=MODEL_PATH,
        n_ctx=COMMON_PARAMS["n_ctx"],
        n_batch=COMMON_PARAMS["n_batch"],
        n_ubatch=COMMON_PARAMS["n_batch"],
        n_threads=COMMON_PARAMS["n_threads"],
        flash_attn=COMMON_PARAMS["flash_attn"],
        debug=True
    )
    b.wrapper.create_context("test_parity", COMMON_PARAMS["n_ctx"])
    yield b

@pytest.fixture(scope="module")
def server():
    if not Path(SERVER_BIN).exists():
        pytest.skip(f"llama-server not found at {SERVER_BIN}")
    s = LlamaServerProcess(MODEL_PATH, COMMON_PARAMS)
    s.start()
    yield s
    s.stop()

@pytest.mark.parametrize("scenario", SCENARIOS)
class TestParity:
    
    def test_prompt_parity(self, bridge, server, scenario):
        """Verify that apply_template produces the exact same prompt as llama-server."""
        oa_tools = to_openai_tools(scenario["tools"])
        server_res = server.chat_completion(scenario["messages"], oa_tools)
        server_prompt = server_res.get("__verbose", {}).get("prompt")
        
        if not server_prompt:
            pytest.skip("Server did not return prompt in __verbose")
            
        template_res = bridge.wrapper.apply_template(scenario["messages"], scenario["tools"], True)
        wrapper_prompt = template_res["prompt"]
        
        s_p = server_prompt.replace("\r\n", "\n").strip()
        w_p = wrapper_prompt.replace("\r\n", "\n").strip()
        assert s_p == w_p

    def test_chat_completion_parity(self, bridge, server, scenario):
        """Verify that BridgeBase._generate matches llama-server output."""
        oa_tools = to_openai_tools(scenario["tools"])
        server_res = server.chat_completion(scenario["messages"], oa_tools)
        
        loop = asyncio.get_event_loop()
        test_log_id = f"log_sync_{scenario['name'].replace(' ', '_').lower()}"
        bridge_res = loop.run_until_complete(bridge._generate(scenario, cache_name="test_parity", log_id=test_log_id))
        
        # Usage check
        assert bridge_res["usage"]["input_tokens"] == server_res["usage"]["prompt_tokens"]
        
        # Content check
        if not scenario["tools"]:
            assert bridge_res["content"].strip() == server_res["choices"][0]["message"]["content"].strip()
        else:
            server_tools = server_res["choices"][0]["message"].get("tool_calls", [])
            assert len(bridge_res["tool_calls"]) == len(server_tools)
            for s_t, w_t in zip(server_tools, bridge_res["tool_calls"]):
                assert s_t["function"]["name"] == w_t["name"]

    def test_chat_streaming_parity(self, bridge, server, scenario):
        """Verify that BridgeBase._stream_generate matches llama-server streaming output."""
        oa_tools = to_openai_tools(scenario["tools"])
        server_stream = server.chat_completion_stream(scenario["messages"], oa_tools)
        
        loop = asyncio.get_event_loop()
        test_log_id = f"log_stream_{scenario['name'].replace(' ', '_').lower()}"

        async def run_stream():
            full_raw = ""
            content = ""
            final_tc = []
            async for chunk in bridge._stream_generate(scenario, cache_name="test_parity", log_id=test_log_id):
                if "full_raw" in chunk: full_raw = chunk["full_raw"]
                if "content" in chunk: content += chunk["content"]
                if "tool_calls" in chunk: final_tc = chunk["tool_calls"]
            return full_raw, content, final_tc

        wrapper_raw, wrapper_content, wrapper_tc = loop.run_until_complete(run_stream())
        
        # Semantic check
        if not scenario["tools"]:
            assert wrapper_content.strip() == server_stream["content"].strip()
        else:
            assert len(wrapper_tc) == len(server_stream["tool_calls"])
            for s_t, w_t in zip(server_stream["tool_calls"], wrapper_tc):
                assert s_t["name"] == w_t["name"]

    def test_log_parity(self, bridge, scenario):
        """Verify that Stage 3 logs contain the exact model output."""
        loop = asyncio.get_event_loop()
        test_log_id = f"log_verify_{scenario['name'].replace(' ', '_').lower()}"
        log_dir = Path("logs") / test_log_id
        if log_dir.exists(): shutil.rmtree(log_dir, ignore_errors=True)
        
        bridge_res = loop.run_until_complete(bridge._generate(scenario, cache_name="test_parity", log_id=test_log_id))
        stage_3_file = log_dir / "3_raw_output.txt"
        
        assert stage_3_file.exists()
        with open(stage_3_file, "r") as f:
            log_raw = f.read()
        
        assert log_raw == bridge_res["full_raw"]

    def test_raw_bit_identity(self, bridge, server, scenario):
        """The ultimate test: Raw /completion output must be identical."""
        template_res = bridge.wrapper.apply_template(scenario["messages"], scenario["tools"], True)
        prompt = template_res["prompt"]
        
        # Non-streaming
        server_raw = server.raw_completion(prompt, stream=False)
        wrapper_raw = bridge.wrapper.generate("test_parity", prompt, 128)
        assert server_raw["content"] == wrapper_raw
        
        # Streaming
        server_raw_stream = server.raw_completion(prompt, stream=True)
        bridge.wrapper.init_inference("test_parity", prompt, 128)
        wrapper_raw_stream = ""
        while True:
            t = bridge.wrapper.get_next_token("test_parity")
            if not t: break
            wrapper_raw_stream += t.decode("utf-8")
        assert server_raw_stream["content"] == wrapper_raw_stream

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
