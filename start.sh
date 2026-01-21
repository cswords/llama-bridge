#!/bin/bash
# Llama-Bridge Startup Script
#
# Usage: ./start.sh [config_name]
# Example: ./start.sh mimo-v2

# 1. Set Metal Shader Cache to avoid recompilation delays
export GGML_METAL_SHADER_CACHE_DIR=/tmp/llama_cache
mkdir -p $GGML_METAL_SHADER_CACHE_DIR

# 2. Determine configuration
CONFIG_NAME=${1:-mimo-v2}
CONFIG_FILE="configs/${CONFIG_NAME}.toml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls configs/*.toml | xargs -n 1 basename | sed 's/.toml//'
    exit 1
fi

echo "🚀 Starting Llama-Bridge with config: $CONFIG_NAME"
echo "   Shader Cache: $GGML_METAL_SHADER_CACHE_DIR"

# 3. Increase system limits (optional, for safety)
ulimit -n 4096

# 4. Run Server
uv run serve --config "$CONFIG_FILE" --debug
