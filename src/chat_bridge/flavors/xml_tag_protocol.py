from typing import List, Dict, Any, Set
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol

class XMLTagProtocol(ChatBridgeProtocol):
    """
    Standard XML-like tagging: <tag>content</tag>
    Also supports arguments with attributes like <tool_call name="foo">
    """
    def __init__(self, tag_map: Dict[str, str], opaque_tags: List[str]):
        """
        tag_map: { "xml_tag_name": "logical_tag_name" }
                 e.g. { "tool_call": "tool_call", "thought": "reasoning" }
        opaque_tags: List of logical_tag_names that are opaque (buffered).
        """
        self.tag_map = tag_map
        self.opaque_tags = set(opaque_tags)

    @property
    def block_tokens(self) -> List[Dict[str, Any]]:
        tokens = []
        for xml_tag, logical_tag in self.tag_map.items():
            # Pattern: <tag( [^>]*)?>
            # We use a pattern that strictly matches the opening bracket and tag name,
            # allowing optional attributes.
            # E.g. <tool_call> or <tool_call name="foo">
            start_pattern = f"<{xml_tag}(?:\\s+[^>]*)?>"
            
            tokens.append({
                "start": start_pattern,
                "end": f"</{xml_tag}>",
                "tag": logical_tag,
                "is_opaque": logical_tag in self.opaque_tags,
                "start_is_regex": True,
                "match_type": "text_tag"
            })
        return tokens
