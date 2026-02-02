from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.xml_tag_protocol import XMLTagProtocol

class MiniMaxFlavor(ChatFlavorBase):
    """
    Flavor for MiniMax models.
    Uses namespaced XML tag: <minimax:tool_call> with <invoke name="..."> subelements.
    Also supports <think> blocks similar to DeepSeek.
    
    Example output format:
        <minimax:tool_call>
        <invoke name="Read">
        <parameter name="file_path">/path/to/file</parameter>
        </invoke>
        </minimax:tool_call>
    """

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [
            XMLTagProtocol(
                tag_map={
                    "minimax:tool_call": "tool_call",  # Namespaced tag maps to standard tool_call
                    "think": "reasoning_content",      # Reasoning support
                },
                opaque_tags=["minimax:tool_call"]  # Tool calls should be buffered
            )
        ]
