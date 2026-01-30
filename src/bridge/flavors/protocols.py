from typing import List, Dict, Any, Optional, Tuple
import re
import uuid
import json
from .base import BaseFlavor

class XMLToolProtocol(BaseFlavor):
    """
    Implements Qwen-style XML tool parsing:
    <tool_call><function=NAME><parameter=ARG>VAL</parameter>...</function></tool_call>
    Also supports standard Qwen 2.5 JSON-in-XML: <tool_call>{json}</tool_call>
    """
    @property
    def block_tags(self) -> List[str]:
        return super().block_tags + ["tool_call", "tool_code", "thinking"]

    @property
    def block_tokens(self) -> List[str]:
        return super().block_tokens + ["<function=", "</function>", "<parameter=", "</parameter>"]

    def interpret_block_complete(self, tag: str, content: str, start_tag: Optional[str] = None) -> Optional[Tuple[str, Any]]:
        # Parsing hook for XML tools
        if tag in ["tool_call", "function", "tool_code"]:
            return ("tool_use", self._parse_xml_tools(content))
        if tag == "thinking":
            return ("reasoning_content", content)
        if tag == "parameter":
            return None
        return super().interpret_block_complete(tag, content, start_tag)

    def _parse_xml_tools(self, raw: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        # Strategy 1: Try to parse as JSON (Standard Qwen 2.5)
        # Content might be: {"name": "func", "arguments": {...}}
        # or a list of such objects.
        cleaned = raw.strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict): data = [data]
                for item in data:
                    if "name" in item:
                        # Ensure arguments is a string if the bridge expects it, 
                        # or keep as dict if bridge handles it. 
                        # Bridge usually expects 'arguments' as string for OpenAI compat, 
                        # but let's check item['arguments'].
                        args = item.get("arguments", {})
                        if isinstance(args, dict): args = json.dumps(args)
                        
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:24]}", 
                            "name": item["name"], 
                            "arguments": args
                        })
                if tool_calls: return tool_calls
            except json.JSONDecodeError:
                pass # Fallback to regex

        # Strategy 2: Regex for XML-style (MiMo / Qwen Variants)
        # <function=NAME>...params...</function>
        func_pattern = r'<function=(\w+)>(.*?)</function>'
        param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
        
        # Scenario A: Full content
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
            # Scenario B: Partial/Stripped content (fallback for streaming boundary issues)
            # This is brittle, use with caution.
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

class HarmonyProtocol(BaseFlavor):
    """
    Implements OpenAI Harmony Protocol (used by GPT-OSS, MiniMax).
    Structure: <|start|>role<|channel|>TYPE ... <|message|>BODY...<|end|>/<|call|>/<|return|>
    """
    
    @property
    def skip_tokens(self) -> List[str]:
        # Do not skip start tokens - they are now block containers
        return super().skip_tokens

    @property
    def block_tokens(self) -> List[str]:
        # <|start|> is the container. 
        # We explicitly list specific start tokens to capture them accurately.
        return super().block_tokens + [
            "<|start|>assistant", "<|start|>system", "<|start|>user", "<|start|>developer", "<|start|>",
            "<|end|>", "<|im_end|>", "<|call|>", "<|return|>"
        ]
    
    @property
    def end_tokens(self) -> List[str]:
        return super().end_tokens + ["<|end|>", "<|im_end|>", "<|call|>", "<|return|>"]
        
    def interpret_block_complete(self, tag: str, content: str, start_tag: Optional[str] = None) -> Optional[Tuple[str, Any]]:
        # Tag will roughly be "|start|..."
        if tag.startswith("|start|"):
            # Structure: role<|channel|>type ... <|message|>body
            # Note: content contains everything AFTER the start token.
            # E.g. if token was <|start|>assistant, content starts with <|channel|>...
            # If token was <|start|>, content starts with role...
            
            # 1. Normalize content to remove role prefix if needed
            # (Simplification: we just look for channel/message markers)
            
            parts = content.split("<|message|>", 1)
            if len(parts) != 2: 
                # Fallback: if no message divider, treat whole thing as content??
                # Or maybe it's just a raw message.
                return ("content", content)
            
            header, body = parts[0].strip(), parts[1]
            
            # 2. Check Channel in Header
            # Header might be "<|channel|>final" or "assistant<|channel|>final"
            
            if "final" in header:
                return ("content", body)
            if "analysis" in header:
                return ("reasoning_content", body)
            if "commentary" in header:
                # Header format: "...commentary to=functions.name <|constrain|>json"
                fname = "unknown_tool"
                
                # Extract 'to='
                to_match = re.search(r'to=functions\.([\w_\-\.]+)', header)
                if not to_match:
                    to_match = re.search(r'to=([\w_\-\.]+)', header)
                
                if to_match:
                    fname = to_match.group(1)
                
                return ("tool_use", [{"name": fname, "arguments": body, "id": f"call_{uuid.uuid4().hex[:8]}"}])
                
        return super().interpret_block_complete(tag, content, start_tag)

class LegacyMiniMaxProtocol(BaseFlavor):
    """
    Support for legacy MiniMax tool calls (<invoke>).
    """
    @property
    def block_tags(self) -> List[str]:
        return super().block_tags + ["minimax:tool_call"] # Dummy tag for block identifying logic if needed, but really we use protected tags.

    @property
    def block_tokens(self) -> List[str]:
        return super().block_tokens + ["<invoke name=", "</invoke>", "<parameter name=", "</parameter>"]

    def interpret_block_complete(self, tag: str, content: str, start_tag: Optional[str] = None) -> Optional[Tuple[str, Any]]:
        # This handles the <invoke> parsing if the scanner detects them.
        
        if tag == "invoke":
             # Use start_tag to reconstruct full XML context for regex parsing if available
             full_text = f"{start_tag}{content}</invoke>" if start_tag else content
             return ("tool_use", self._parse_legacy_tools(full_text))
        return super().interpret_block_complete(tag, content, start_tag)

    def _parse_legacy_tools(self, raw: str) -> List[Dict[str, Any]]:
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
             # Fallback for stripped content
             pass 
        return tool_calls
