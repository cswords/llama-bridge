from typing import List
from src.chat_bridge.flavors.chat_flavor_base import ChatFlavorBase
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol
from src.chat_bridge.flavors.xml_tag_protocol import XMLTagProtocol

class MimoFlavor(ChatFlavorBase):
    """
    Flavor for Mimo models.
    Uses standard <tool_call> tag but with special inner syntax:
        <function=FunctionName>
        <parameter=param_name>value</parameter>
        </function>
    
    Example output format:
        <tool_call>
        <function=Read>
        <parameter=file_path>/path/to/file</parameter>
        </function>
        </tool_call>
    
    Note: The inner function/parameter syntax uses = instead of XML attributes,
    but the outer tool_call tag is standard XML and can be captured by XMLTagProtocol.
    The inner parsing will be handled by _parse_non_stream_content in bridge.
    """

    @property
    def name(self) -> str:
        return "mimo"

    @property
    def protocols(self) -> List[ChatBridgeProtocol]:
        return [
            XMLTagProtocol(
                tag_map={
                    "tool_call": "tool_call",         # Standard tool_call tag
                    "think": "reasoning_content",     # Reasoning support if present
                },
                opaque_tags=["tool_call"]  # Tool calls should be buffered
            )
        ]
