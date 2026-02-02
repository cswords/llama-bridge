from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ChatBridgeProtocol(ABC):
    """
    Abstract base class for all ChatBridge protocols.
    A Protocol defines HOW to identify blocks (tokens) in a stream.
    """
    
    @property
    @abstractmethod
    def block_tokens(self) -> List[Dict[str, Any]]:
        """
        Returns a list of token definitions for the Scanner.
        Each definition:
        {
            "start": str,           # precise start token string or pattern
            "end": str,             # precise end token string
            "tag": str,             # logical tag name (e.g. "tool_call", "reasoning")
            "is_opaque": bool,      # True=Buffer/Ignore, False=Stream
            "start_is_regex": bool  # (Optional) If True, start is a regex pattern
        }
        """
        pass
