# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

"""
Main Bridge module combining protocol-specific logic.
"""

from .base import BridgeBase
from .anthropic import AnthropicMixin
from .openai import OpenAIMixin

class Bridge(BridgeBase, AnthropicMixin, OpenAIMixin):
    """
    Main Bridge class that supports Anthropic and OpenAI protocols.
    Inherits core logic from BridgeBase and interface methods from Mixins.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Bridge and its mixins."""
        super().__init__(*args, **kwargs)
        self._init_anthropic()
        self._init_openai()

__all__ = ["Bridge", "BridgeBase"]
