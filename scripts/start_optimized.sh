#!/bin/bash
# Optimized run script for Claude Code integration
# Uses configuration file for multi-cache routing

export GGML_METAL_SHADER_CACHE_DIR=/tmp/llama_cache

# Check if config file exists
if [ -f "configs/claude-code.toml" ]; then
    echo "Using configuration file: configs/claude-code.toml"
    uv run serve \
      --config configs/claude-code.toml \
      --n-batch 512 \
      --n-threads 12 \
      --debug
else
    echo "Config file not found, using legacy mode"
    uv run serve \
      --model unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/ \
      --n-ctx 262144 \
      --n-batch 512 \
      --n-threads 12 \
      --debug
fi
