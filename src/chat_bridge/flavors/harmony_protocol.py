from typing import List, Dict, Any
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol

class HarmonyProtocol(ChatBridgeProtocol):
    """
    Harmony (GPT-OSS) Control Tokens Protocol.
    
    Based on actual GPT-OSS model output:
    - Analysis/Reasoning: <|channel|>analysis<|message|>...<|end|>
    - Tool Calls: <|channel|>commentary to=functions.ToolName <|constrain|>json<|message|>{...}
    
    This protocol handles the Harmony-style control tokens that differ from XML tags.
    """
    
    def __init__(self):
        """
        HarmonyProtocol is configured for GPT-OSS models.
        No need for tag_map - patterns are fixed based on model format.
        """
        pass

    @property
    def block_tokens(self) -> List[Dict[str, Any]]:
        """
        Define block tokens for Harmony format.
        
        Block structure from samples:
        1. Analysis block: <|channel|>analysis<|message|>...<|end|>
        2. Tool block: <|start|>assistant<|channel|>commentary to=functions.X <|constrain|>json<|message|>{...}
        
        Note: Tool blocks may not have explicit end token in samples.
        """
        return [
            # Analysis/Reasoning Channel (streamed, not buffered)
            {
                "start": r"<\|channel\|>analysis<\|message\|>",
                "end": "<|end|>",
                "tag": "reasoning_content",
                "is_opaque": False,  # Stream reasoning content
                "start_is_regex": True,
                "match_type": "control_token"
            },
            # Tool Call Channel (buffered for complete JSON parsing)
            # Pattern matches: <|channel|>commentary to=functions.X <|constrain|>json<|message|>
            {
                "start": r"<\|channel\|>commentary\s+to=functions\.\w+\s*<\|constrain\|>json<\|message\|>",
                "end": "\n}",  # JSON typically ends with closing brace + newline
                "tag": "tool_call",
                "is_opaque": True,  # Buffer tool JSON completely
                "start_is_regex": True,
                "match_type": "control_token"
            },
        ]
