from typing import List, Dict, Any, Optional
from .base import BaseFlavor

class ChatMLTemplate(BaseFlavor):
    """Provides standard ChatML templating."""
    
    @property
    def turn_separators(self) -> List[str]:
        return super().turn_separators + ["<|im_end|>", "<|im_start|>"]

    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
