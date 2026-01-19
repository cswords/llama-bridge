# Llama-Bridge

GGUF 模型推理服务，支持多种 LLM API 格式（Anthropic、OpenAI、Gemini 等）。

## 快速开始

```bash
# 1. 克隆并初始化
git clone --recursive https://github.com/xxx/llama-bridge.git
cd llama-bridge

# 2. 安装依赖
uv sync --all-extras

# 3. 构建
make build

# 4. 下载模型
make hfd REPO=bartowski/Qwen3-Coder-30B-A3B-Instruct-GGUF INCLUDE='*Q5_K_M.gguf'

# 5. 启动服务
uv run serve --model models/bartowski/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf
```

## 特性

- **多 API 格式支持** - 通过 LiteLLM 支持 Anthropic、OpenAI、Gemini 等格式
- **零 HTTP 中转** - FFI 直接调用 llama.cpp
- **真流式输出** - token-by-token 流式响应
- **完整工具调用** - 利用 llama.cpp 的 chat.h 实现

## 开发

```bash
# 安装开发环境
make dev

# 运行测试
make test
```

详细开发规约参见 [INSTRUCTION.md](./INSTRUCTION.md)
