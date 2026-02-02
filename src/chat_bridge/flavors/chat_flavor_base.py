from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from src.chat_bridge.flavors.chat_bridge_protocol import ChatBridgeProtocol

class ChatFlavorBase(ABC):
    """
    Base class for all Chat Flavors.
    A Flavor combines a Protocol with model-specific logic.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def protocols(self) -> List[ChatBridgeProtocol]:
        """
        Returns the list of protocols used by this flavor.
        Scanner will aggregate tokens from all protocols.
        """
        pass
    
    @property
    def block_tokens(self) -> List[Dict[str, Any]]:
        """
        Aggregates block tokens from all protocols.
        """
        all_tokens = []
        for protocol in self.protocols:
            all_tokens.extend(protocol.block_tokens)
        return all_tokens
