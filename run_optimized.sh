export GGML_METAL_SHADER_CACHE_DIR=/tmp/llama_cache
uv run serve \
  --model unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/ \
  --n-ctx 262144 \
  --n-batch 512 \
  --n-threads 12 \
  --debug
