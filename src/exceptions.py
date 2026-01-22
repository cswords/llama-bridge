# Copyright (c) 2026 Llama-Bridge Authors.
# This software is released under the GNU General Public License v3.0.
# See the LICENSE file in the project root for full license information.

class ContextLimitExceededError(Exception):
    """Raised when the request exceeds the context window size."""
    def __init__(self, limit: int, requested: int, cache_name: str):
        self.limit = limit
        self.requested = requested
        self.cache_name = cache_name
        super().__init__(f"Request ({requested} tokens) exceeds context limit ({limit} tokens) for cache '{cache_name}'")
