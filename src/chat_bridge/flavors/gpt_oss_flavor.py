from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.harmony_protocol import HarmonyProtocol

class GPTOSSFlavor(ChatFlavorBase):
    """
    Flavor for GPT-OSS (Harmony) models.
    Uses HarmonyProtocol for channel-based control tokens.
    
    Handles:
    - <|channel|>analysis<|message|>...<|end|> for reasoning
    - <|channel|>commentary to=functions.X for tool calls
    """

    @property
    def name(self) -> str:
        return "gpt-oss"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [HarmonyProtocol()]
