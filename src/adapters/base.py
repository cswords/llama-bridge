# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

"""
Base class for all API adapters.
Defines the interface for protocol-specific conversions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncGenerator, Union

class BaseAdapter(ABC):
    """
    Abstract base class for API adapters.
    Each provider (Anthropic, OpenAI, etc.) must implement these methods.
    """
    
    @abstractmethod
    def to_internal(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert provider request to internal format."""
        pass
    
    @abstractmethod
    def from_internal(self, response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal response to provider format."""
        pass
    
    @abstractmethod
    def chunk_to_sse(self, chunk: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Convert internal chunk to provider SSE event."""
        pass
