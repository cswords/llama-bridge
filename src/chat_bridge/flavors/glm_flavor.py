from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.xml_tag_protocol import XMLTagProtocol

class GLMFlavor(ChatFlavorBase):
    """
    Flavor for GLM-4 models.
    Uses <think> for reasoning and <tool_call> for tools.
    """

    @property
    def name(self) -> str:
        return "glm"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [
            XMLTagProtocol(
                tag_map={
                    "tool_call": "tool_call",
                    "think": "reasoning_content"
                },
                opaque_tags=["tool_call"] # Buffer tools, stream reasoning
            )
        ]
