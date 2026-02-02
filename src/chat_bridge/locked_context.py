"""
LockedContext: Encapsulates a llama.cpp context with its own asyncio.Lock.

This ensures that each context can only be used by one request at a time,
preventing state corruption when multiple requests try to use the same context.
"""

import asyncio
import logging
from typing import Any, List, Dict, AsyncGenerator

logger = logging.getLogger(__name__)


class LockedContext:
    """
    A wrapper around a llama.cpp context that manages its own concurrency lock.
    
    Each context can only process one request at a time. The lock is held from
    init_inference through all get_next_token calls until generation is complete.
    """
    
    def __init__(self, wrapper: Any, ctx_name: str):
        """
        Initialize a LockedContext.
        
        Args:
            wrapper: The LlamaChatWrapper instance
            ctx_name: The name of the context in the wrapper
        """
        self.wrapper = wrapper
        self.ctx_name = ctx_name
        self.lock = asyncio.Lock()
        
    def apply_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], add_generation_prompt: bool) -> Dict[str, Any]:
        """Apply chat template (caller must hold lock)."""
        return self.wrapper.apply_template(messages, tools, add_generation_prompt)

    def parse_response(self, response_text: str, is_partial: bool = False) -> Dict[str, Any]:
        """Parse response (caller must hold lock)."""
        return self.wrapper.parse_response(response_text, is_partial)

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[bytes, None]:
        """
        Initialize inference and generate tokens (caller must hold lock).
        
        Args:
            prompt: The full prompt to send to the model
            max_tokens: Maximum tokens to generate
            
        Yields:
            Raw bytes from get_next_token
        """
        logger.debug(f"[LockedContext:{self.ctx_name}] Starting inference")
        
        # Initialize inference
        self.wrapper.init_inference(
            self.ctx_name, 
            prompt, 
            max_tokens
        )
        
        # Generate tokens
        while True:
            token = self.wrapper.get_next_token(self.ctx_name)
            
            # C++ returns None or empty bytes for EOS/Error
            if token is None or token == b"" or len(token) == 0:
                break
                
            yield token
            
            # Yield control to event loop to keep server responsive
            await asyncio.sleep(0)
            
        logger.debug(f"[LockedContext:{self.ctx_name}] Generation complete")
