from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.xml_tag_protocol import XMLTagProtocol

class QwenFlavor(ChatFlavorBase):
    """
    Flavor for Qwen 2.5/3 models.
    Uses standard XML tags for tool calls.
    Reasoning is usually implicit (content), unless DeepSeek style <think> is used (not default).
    """

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [
            XMLTagProtocol(
                tag_map={
                    "tool_call": "tool_call",  # <tool_call>... maps to tool_call block
                },
                opaque_tags=["tool_call"]  # Tool calls should be buffered
            )
        ]
