# Copyright (c) 2026 Llama-Bridge Authors.
from typing import List, Dict, Any, Optional, Tuple
import re
import uuid
import json

class BaseFlavor:
    """Interprets blocks and manages model-specific formatting."""
    
    @property
    def block_tags(self) -> List[str]:
        """Tags that should be captured as blocks."""
        return ["thought", "think"]

    @property
    def protected_tags(self) -> List[str]:
        """Full strings that should be buffered and shielded from direct output."""
        tags = []
        for bt in self.block_tags:
            tags.append(f"<{bt}>")
            tags.append(f"</{bt}>")
        return tags

    @property
    def turn_separators(self) -> List[str]:
        """Strings that mark the end of a turn or message boundary. Used to prevent prefill from greedily capturing history."""
        return []

    def interpret_block_chunk(self, tag: str, chunk: str) -> Optional[Tuple[str, Any]]:
        """Interpret a partial block chunk. Returns (type, data)."""
        if tag in ["thought", "think"]:
            return ("thought", chunk)
        return None

    def interpret_block_complete(self, tag: str, content: str) -> Optional[Tuple[str, Any]]:
        """Interpret a complete block. Returns (type, data)."""
        # We handle thoughts via chunks, so no need to return valid completion data
        # unless we want to emit a "thought_end" event?
        return None

    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        return None

class QwenFlavor(BaseFlavor):
    @property
    def block_tags(self) -> List[str]:
        return super().block_tags + ["tool_call"]

    @property
    def protected_tags(self) -> List[str]:
        return super().protected_tags + ["<function=", "</function>", "<parameter=", "</parameter>"]

    @property
    def turn_separators(self) -> List[str]:
        return ["<|im_end|>", "<|im_start|>"]

    def interpret_block_complete(self, tag: str, content: str) -> Optional[Tuple[str, Any]]:
        res = super().interpret_block_complete(tag, content)
        if res: return res
        
        if tag in ["tool_call", "function"]:
            return ("tool_use", self._parse_qwen_tools(content))
        if tag == "parameter":
            # For Qwen, we don't usually yield parameter chunks directly
            # but we can return them as part of tool_use at the end.
            return None
        return None

    def _parse_qwen_tools(self, raw: str) -> List[Dict[str, Any]]:
        tool_calls = []
        # If StreamScanner already stripped <function=, we might just have the inner content.
        # But for robustness, we handle both cases.
        func_pattern = r'<function=(\w+)>(.*?)</function>'
        param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
        
        # Scenario A: Full content (Fallback/Legacy)
        matches = list(re.finditer(func_pattern, raw, re.DOTALL))
        if matches:
            for match in matches:
                params = {}
                for pm in re.finditer(param_pattern, match.group(2), re.DOTALL):
                    val = pm.group(2).strip()
                    try: params[pm.group(1)] = json.loads(val)
                    except: params[pm.group(1)] = val
                tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "name": match.group(1), "arguments": json.dumps(params)})
        else:
            # Scenario B: Content inside <function=NAME>...</function>
            # It might look like "get_weather><parameter=city>..."
            match_stripped = re.match(r'^(\w+)>(.*)', raw, re.DOTALL)
            if match_stripped:
                fname = match_stripped.group(1)
                inner = match_stripped.group(2)
                params = {}
                for pm in re.finditer(param_pattern, inner, re.DOTALL):
                    val = pm.group(2).strip()
                    try: params[pm.group(1)] = json.loads(val)
                    except: params[pm.group(1)] = val
                tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "name": fname, "arguments": json.dumps(params)})
        
        return tool_calls

    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        # Minimal Qwen 2.5 template fallback
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

class MiniMaxFlavor(BaseFlavor):
    @property
    def block_tags(self) -> List[str]:
        return super().block_tags + ["minimax:tool_call"]

    @property
    def protected_tags(self) -> List[str]:
        return super().protected_tags + ["<invoke name=", "</invoke>", "<parameter name=", "</parameter>"]

    def interpret_block_complete(self, tag: str, content: str) -> Optional[Tuple[str, Any]]:
        res = super().interpret_block_complete(tag, content)
        if res: return res

        if tag in ["minimax:tool_call", "invoke"]:
            return ("tool_use", self._parse_minimax_tools(content))
        if tag == "parameter":
            return None
        return None

    def _parse_minimax_tools(self, raw: str) -> List[Dict[str, Any]]:
        tool_calls = []
        invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
        
        matches = list(re.finditer(invoke_pattern, raw, re.DOTALL))
        if matches:
            for match in matches:
                params = {}
                for pm in re.finditer(param_pattern, match.group(2), re.DOTALL):
                    val = pm.group(2).strip()
                    try: params[pm.group(1)] = json.loads(val)
                    except: params[pm.group(1)] = val
                tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "name": match.group(1), "arguments": json.dumps(params)})
        else:
            # Stripped: name="calculator">... OR "calculator">... (if <invoke name= was consumed)
            match_stripped = re.match(r'^name="([^"]+)"\s*>(.*)', raw, re.DOTALL)
            if not match_stripped:
                # Try just the quoted name: "calculator">...
                match_stripped = re.match(r'^"([^"]+)"\s*>(.*)', raw, re.DOTALL)
            
            if match_stripped:
                fname = match_stripped.group(1)
                inner = match_stripped.group(2)
                params = {}
                for pm in re.finditer(param_pattern, inner, re.DOTALL):
                    val = pm.group(2).strip()
                    try: params[pm.group(1)] = json.loads(val)
                    except: params[pm.group(1)] = val
                tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "name": fname, "arguments": json.dumps(params)})
        return tool_calls

    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        prompt = ""
        msgs = [m.copy() for m in messages]
        if tools:
            tool_desc = f"Available tools:\n{json.dumps(tools, indent=2)}"
            sys_msg = next((m for m in msgs if m["role"] == "system"), None)
            if sys_msg: sys_msg["content"] = f"{sys_msg['content']}\n\n{tool_desc}"
            else: msgs.insert(0, {"role": "system", "content": tool_desc})
        for msg in msgs:
            prompt += f"<|channel|>{msg['role']}<|message|>{msg.get('content', '')}"
        prompt += "<|channel|>assistant<|message|>"
        return prompt

def get_flavor_for_model(model_path: str) -> BaseFlavor:
    path_lower = model_path.lower()
    if "qwen" in path_lower: return QwenFlavor()
    if "minimax" in path_lower: return MiniMaxFlavor()
    return BaseFlavor()
