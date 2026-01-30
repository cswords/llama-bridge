# Copyright (c) 2026 Llama-Bridge Authors.
from typing import List, Dict, Any, Optional, Tuple

class BaseFlavor:
    """Base interface for model flavors."""
    
    @property
    def block_tags(self) -> List[str]:
        """Tags that should be captured as blocks (e.g. 'thought')."""
        return ["thought", "think"]

    @property
    def block_tokens(self) -> List[str]:
        """
        [Block Markers]
        Tokens that signify the start or end of a structured block (e.g. <tool_call>, </thought>).
        The Scanner will treat these as signal events, allowing the Flavor to switch modes 
        (e.g., buffering content or streaming chunks).
        Defaults to <tag> and </tag> of block_tags.
        """
        tags = []
        for bt in self.block_tags:
            tags.append(f"<{bt}>")
            tags.append(f"</{bt}>")
        return tags

    @property
    def skip_tokens(self) -> List[str]:
        """
        [Control Tokens]
        Tokens that are implementation details or noise (e.g. <|start|>).
        The Scanner will silently discard these tokens.
        """
        return []

    @property
    def end_tokens(self) -> List[str]:
        """
        tokens that explicitly close a block.
        Must be a subset of block_tokens.
        """
        tags = []
        for bt in self.block_tags:
            tags.append(f"</{bt}>")
        return tags

    @property
    def turn_separators(self) -> List[str]:
        return []

    def interpret_block_chunk(self, tag: str, chunk: str) -> Optional[Tuple[str, Any]]:
        # Common support for reasoning stream
        if tag in ["thought", "think"]:
            return ("thought", chunk)
        return None

    def interpret_block_complete(self, tag: str, content: str, start_tag: Optional[str] = None) -> Optional[Tuple[str, Any]]:
        return None

    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Optional[str]:
        return None
