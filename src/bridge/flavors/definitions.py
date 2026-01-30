from typing import List, Dict, Any, Optional
import json
from .base import BaseFlavor
from .templates import ChatMLTemplate
from .protocols import XMLToolProtocol, HarmonyProtocol, LegacyMiniMaxProtocol

# --- Concrete Flavors ---

class QwenFlavor(XMLToolProtocol, ChatMLTemplate):
    """Qwen/Mimo: ChatML Template + XML Tools."""
    pass

class MiniMaxFlavor(HarmonyProtocol, LegacyMiniMaxProtocol, ChatMLTemplate):
    """MiniMax/GPT-OSS: Harmony Protocol + Legacy <invoke> Support."""
    
    @property
    def turn_separators(self) -> List[str]:
        return super().turn_separators + ["<|channel|>", "<|message|>", "<|call|>", "<|return|>"]
    
    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        # Implementation of full Harmony Template
        # Note: Harmony uses <|start|>role... but ChatMLTemplate uses <|im_start|>role.
        # Ideally we should override to use Harmony native tags if we want perfectly strict adherence.
        # But given we are inheriting ChatMLTemplate, let's stick to <|im_start|> for now unless we define a HarmonyTemplate Mixin.
        
        # However, we MUST format Tool Results correctly for the model to understand.
        prompt = ""
        msgs = [m.copy() for m in messages]
        
        # 1. System/Tools Injection
        if tools:
            tool_desc = f"Available tools:\n{json.dumps(tools, indent=2)}"
            sys_msg = next((m for m in msgs if m["role"] == "system"), None)
            if sys_msg: sys_msg["content"] = f"{sys_msg['content']}\n\n{tool_desc}"
            else: msgs.insert(0, {"role": "system", "content": tool_desc})
            
        # 2. Message Loop
        for msg in msgs:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "tool":
                # Harmony Tool Response: 
                # <|start|>functions.name to=assistant<|channel|>commentary<|message|>RESULT<|end|>
                # But we don't have the function name easily here in standard ChatML dict.
                # Assuming 'name' field exists in tool message (standard OpenAI format).
                tool_name = msg.get("name", "unknown_tool")
                prompt += f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>\n"
            else:
                # Use Harmony tokens: <|start|>role<|message|>content<|end|>
                # Note: 'role' acts as the header here.
                prompt += f"<|start|>{role}<|message|>{content}<|end|>\n"

        # Assistant prompt: start header but wait for channel/message
        # Standard harmony ends prompt with <|start|>assistant
        # The model then generates <|channel|>... or messages.
        prompt += "<|start|>assistant"
        return prompt

class GLMFlavor(BaseFlavor): # Placeholder
    pass
