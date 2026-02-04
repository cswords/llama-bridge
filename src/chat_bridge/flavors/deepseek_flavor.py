from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.xml_tag_protocol import XMLTagProtocol

class DeepSeekFlavor(ChatFlavorBase):
    """
    Flavor for DeepSeek-R1 models.
    Supports <think> tags for reasoning content.
    Reasoning content is streamed transparently.
    """

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [
            XMLTagProtocol(
                tag_map={
                    "think": "reasoning_content"
                },
                opaque_tags=[]  # think sessions should be streamed (transparent)
            )
        ]
