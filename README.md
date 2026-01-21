# 🦙 Llama-Bridge

Llama-Bridge 是一个基于 `llama.cpp` 的高性能 GGUF 模型推理服务。它通过 `pybind11` 直接调用 C++ 推理引擎，实现了**零 HTTP 开销**的架构，并在 Python 层通过 `LiteLLM` 提供通用的 API 接口支持（Anthropic, OpenAI 等）。

## 核心特性

- **🚀 零中转架构** - 摒弃传统的多层 HTTP 代理，Python 通过 FFI 直接调用 `libllama.dylib`，性能损耗接近零。
- **🔌 万能接口** - 原生支持 Anthropic Claude (Messages API) 和 OpenAI Chat Completions 格式。
- **⚡️ 真流式输出** - 基于 `llama.cpp` 的 `chat.h` 实现 Token 级的实时流式响应和工具边界检测。
- **🛠 完整工具链** - 完全兼容模型内置的 Chat Template，支持复杂的 Tool Use 和 Reasoning (Thinking)。
- **📦 一键化管理** - 使用 `uv` 管理依赖，所有操作均可通过 `uv run` 或 `make` 完成。

## 快速开始

### 1. 克隆与初始化

```bash
git clone --recursive https://github.com/your-repo/Llama-Bridge.git
cd Llama-Bridge

# 安装依赖
uv sync --all-extras
```

### 2. 编译 C++ 绑定

项目需要编译 `llama.cpp` 及其 Python 绑定：

```bash
make build
```

### 3. 下载模型

使用内置的 `hfd` 实用程序从 Hugging Face 下载模型：

```bash
# 下载指定版本的 GGUF 模型
uv run hfd bartowski/Llama-3.1-8B-Instruct-GGUF --include "*Q5_K_M.gguf"
```

### 4. 启动推理服务

```bash
# 自动加载模型并启动 FastAPI 服务
uv run serve --model models/bartowski/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-Q5_K_M.gguf --debug
```

## CLI 命令参考

项目提供三个主要的 CLI 命令：

### `uv run serve` - 启动推理服务器

启动 FastAPI 推理服务，支持 Anthropic 和 OpenAI API 格式。

```bash
# 使用配置文件启动（推荐）
uv run serve --config configs/claude-code.toml

# 或使用命令行参数启动
uv run serve --model <path> [options]
```

#### 配置文件优先

当使用 `--config` 指定配置文件时，与模型相关的 CLI 参数（`--model`, `--n-ctx`）将被忽略。
配置文件使用 TOML 格式，支持多缓存路由。参见 `configs/claude-code.toml` 示例。

| 参数 | 说明 |
|------|------|
| `--config <path>` | TOML 配置文件路径。启用多缓存路由，推荐用于 Claude Code。 |

#### 传统参数（如果未使用 --config）

| 参数 | 说明 |
|------|------|
| `--model <path>` | GGUF 模型路径。支持文件路径或目录（自动选择第一个 `.gguf` 文件）。支持相对于 `models/` 的短路径。 |

#### 服务器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host <addr>` | `127.0.0.1` | 服务器监听地址。设为 `0.0.0.0` 可接受外部连接。 |
| `--port <num>` | `8000` | 服务器监听端口。 |
| `--debug` | `false` | 启用调试模式：详细日志 + 4-File Rule 请求追踪。 |
| `--mock` | `false` | Mock 模式：不加载模型，用于测试 API 流程。 |

#### 推理引擎配置

| 参数 | 默认值 | 说明 | 建议设置 |
|------|--------|------|----------|
| `--n-ctx <num>` | `0` (自动) | **上下文窗口大小**（Token 数）。决定模型能处理多长的对话历史。 | 设为模型训练上限（如 MiMo: `262144`）。内存充足时越大越好。 |
| `--n-batch <num>` | `0` (自动) | **逻辑批处理大小**。每次 Prefill 送入模型的最大 Token 数。 | `512` ~ `2048` 是较好的平衡点。 |
| `--n-ubatch <num>` | `0` (自动) | **物理批处理大小**。GPU 实际执行的微批次大小。 | 通常保持 `0`（自动）。 |
| `--n-threads <num>` | `0` (自动) | **CPU 线程数**。用于 CPU 计算部分。 | 设为物理核心数的 50-70%（如 24 核设 `12-16`）。 |
| `--flash-attn` | `false` | **启用 Flash Attention**。减少长上下文的显存占用。 | 如果模型支持，**强烈建议开启**。 |

#### 示例

```bash
# 使用配置文件（推荐，支持多缓存隔离）
uv run serve --config configs/claude-code.toml --debug

# 基础启动
uv run serve --model bartowski/Llama-3.1-8B-Instruct-GGUF

# 高性能配置 (Mac Studio M3 Ultra 512GB)
uv run serve \
  --model unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/ \
  --n-ctx 262144 \
  --n-batch 2048 \
  --n-threads 16 \
  --flash-attn \
  --debug
```

---

### `uv run hfd` - 下载 Hugging Face 模型

从 Hugging Face Hub 下载模型到本地 `models/` 目录。

```bash
uv run hfd <repo_id> [options]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `<repo_id>` | **必需**。Hugging Face 仓库 ID，格式为 `username/model-name`。 |
| `--include "<pattern>"` | 只下载匹配 glob 模式的文件。常用于筛选特定量化版本。 |
| `--exclude "<pattern>"` | 排除匹配模式的文件。 |
| `--local-dir <path>` | 自定义下载目录（默认: `models/<repo_id>`）。 |

#### 示例

```bash
# 下载整个仓库
uv run hfd bartowski/Llama-3.1-8B-Instruct-GGUF

# 只下载 Q5_K_M 量化版本
uv run hfd bartowski/Llama-3.1-8B-Instruct-GGUF --include "*Q5_K_M.gguf"

# 下载特定量化，排除分片
uv run hfd unsloth/MiMo-V2-Flash-GGUF --include "*Q6_K_XL*" --exclude "*00002*"
```

---

### `uv run cc` - 启动 Claude Code CLI

启动 Claude Code CLI，自动配置环境变量以连接本地 Llama-Bridge 服务器。

```bash
uv run cc [claude-args...]
```

#### 自动配置的环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `ANTHROPIC_BASE_URL` | `http://localhost:8000` | 将 Claude Code 的 API 请求重定向到本地服务器。 |
| `ANTHROPIC_API_KEY` | `sk-ant-none` | 占位符 API Key（本地服务器不验证）。 |

#### 示例

```bash
# 启动交互式 Claude Code
uv run cc

# 非交互式执行任务
uv run cc -p "Write a hello world in Python"

# 指定工作目录
uv run cc --cwd /path/to/project
```

> **注意**: 使用 `uv run cc` 前，请确保 Llama-Bridge 服务器已在后台运行（`uv run serve ...`）。

---

## 开发与调试

### 4-File Rule 调试法则

当开启 `--debug` 模式时，Llama-Bridge 会在 `logs/` 目录下为每个请求生成 4 个阶段的快照，用于全链路追踪模型表现。

```bash
# 运行测试
uv run pytest

# 重新构建绑定 (如果 llama.cpp 更新)
make build
```

## 技术架构

1.  **Core**: `llama.cpp` (C++) - 负责张量计算与词表管理。
2.  **Binding**: `pybind11` (C++) - 将 `common/chat.h` 的模板处理与推理逻辑暴露给 Python。
3.  **Adapter**: `LiteLLM` (Python) - 负责请求/响应格式的标准化转换。
4.  **Server**: `FastAPI` (Python) - 提供高性能的异步 Web API。


---

## 性能测试与验证 (Performance & Benchmarks)

我们在 Mac Studio (M3 Ultra) 上对以下模型配置进行了端到端验证（2026-01-21）：

| 模型名称 | 规模 | 量化 | 启动耗时 | Cold | Hot |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 3B Instruct** | 3B | Q4_K_M | ~3s | 0.12s | 0.08s |
| **Qwen3-Coder-30B-A3B** | 30B | Q5_K_M | ~12s | 0.19s | 0.14s |
| **MiMo-V2-Flash** | ~70B | UD-Q6_XL | ~17s | 0.67s | 0.29s |
| **GLM-4.7-REAP-218B** | 218B | Q2_K | ~30s | 0.80s | 0.54s |

> **测试说明**: 
> - 以上数据基于 Mac Studio (M3 Ultra) 及 SSD 环境。
> - **OS Cache**: 启动耗时包含 Page Cache 命中情况。冷启动（刚开机）首个超大模型可能需要 1-2 分钟。
> - **首请求延迟**: 包含 Metal Shader 编译与预热。MiMo 等动态架构模型在首个请求上有显著的编译开销。
> - **内存管理**: `mimo-v2.toml` 默认开启 256K Context，如遇内存压力请在配置中调低 `n_ctx`。

---

详细设计方案与 TDD 进化路线图请参考 [INSTRUCTION.md](./INSTRUCTION.md)。

