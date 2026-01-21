# Llama-Bridge Technical Whitepaper & Developer Guide

Llama-Bridge 是一个高性能的本地 LLM 桥接服务，旨在为 GGUF 模型提供原生级的 API 兼容性体验。通过集成 LiteLLM 和 llama.cpp 的 FFI 绑定，它允许本地模型被 Claude Code、Cursor、OpenAI SDK 等现代开发工具直接调用，而无需复杂的配置或额外的 HTTP 中转。

---

## 1. 核心技术特点 (Key Features)

### 🚀 零 HTTP 中转 (Zero-HTTP Overhead)
Llama-Bridge 不使用 `llama-server` 或其他 HTTP 后端。它通过 pybind11 将 llama.cpp 编译为 Python 的 C++ 扩展，直接在进程内进行 FFI 调用。
- **低延迟**：消除了内部组件间的 HTTP 序列化/反序列化开销。
- **真流式**：Token 生成后立即传递给 Python 层，实现毫秒级首字响应。

### 🔄 统一配置与路由系统 (Unified Config & Routing)
支持通过 TOML 文件定义复杂的路由规则，实现单一端口支持多种业务场景。
- **多模型/多缓存**：支持加载一个大模型（如 Sonnet 级），并为其配置多个隔离的 KV Cache。
- **智能路由**：根据 API Endpoint、模型名称或通配符，自动将请求导向“主对话”（大上下文）或“后台任务”（小上下文）。

### 🧊 多 Context 物理隔离 (Multi-Context Isolation)
彻底解决了传统的“KV Cache 污染”问题。
- **物理隔离**：每个 Cache 对应 C++ 层一个独立的 `llama_context` 实例。
- **状态独立**：不同客户端或不同类型的任务（如代码补全 vs 对话）互不干扰。
- **内存高效**：所有 Context 共享同一份模型权重，仅 KV Cache 占用独立内存。

### �️ 健壮的错误处理 (Robust Error Handling)
- **Context Overflow Protection**：当请求超过模型上下文限制时，不再崩溃或挂起，而是能够捕获 C++ 异常并将其转换为符合 OpenAI/Anthropic 标准的 `400 Bad Request` 错误响应，允许客户端智能处理（如自动截断历史）。
- **友好提示**：错误信息包含详细的 token 统计（请求数 vs 限制数），方便调试。

### 🔍 缓存一致性监测 (Cache Observability)
- **System Prompt Fingerprinting**：自动检测 System Prompt 的变更。如果检测到同一个 Cache Slot 的 System Prompt 发生哈希变化，会在日志中发出警告，提示潜在的 Cache Miss 性能影响。

### �🔌 全协议支持 (Universal Protocol Support)
集成 LiteLLM（库模式），支持几乎所有主流 LLM API 格式：
- **Anthropic Messages API**: 完美支持 Claude Code。
- **OpenAI Chat Completions**: 支持 Cursor、Continue 等工具。
- **工具调用 (Tool Use)**: 利用 llama.cpp 原生的 grammar-constrained generation，实现高可靠性的函数调用。

---

## 2. 系统架构 (Architecture)

```mermaid
graph TD
    Client[Client (Claude Code / Cursor)] -->|HTTP (Anthropic/OpenAI Protocol)| Server[FastAPI Server]
    
    subgraph "Llama-Bridge Process"
        Server --> Router[Router]
        Router -->|Route Request| Bridge[Bridge Core]
        
        subgraph "Python Logic"
            Bridge -->|Convert Protocol| Adapter[Protocol Adapter (LiteLLM)]
            Bridge -->|Select Context| Wrapper[LlamaChatWrapper (C++)]
        end
        
        subgraph "C++ Binding (pybind11)"
            Wrapper -->|FFI Call| LlamaCPP[libllama.dylib]
            Wrapper -->|Checks| Limits[Overflow Guard]
            
            subgraph "Memory Space"
                LlamaCPP -->|Shared Weights| Model[Model Weights]
                LlamaCPP -->|Context A| Cache1[KV Cache (Main)]
                LlamaCPP -->|Context B| Cache2[KV Cache (Fast)]
            end
        end
    end
```

### 数据流向
1. **Request**: 客户端发送 HTTP 请求（如 `/v1/messages`）。
2. **Routing**: `Router` 根据配置（如模型名 `claude-3-5-haiku`）决定使用哪个 Cache（如 `fast`）。
3. **Adaptation**: `Adapter` 将请求转换为标准化的内部格式。
4. **Validation**: `Bridge` 检查 System Prompt 指纹，并在初始化推理时验证 Context 限制。
5. **Inference**: `Bridge` 调用 C++ 绑定，指定目标 Context 进行推理。支持流式（Streaming）智能合并逻辑，处理思考块（Thinking）的抑制或输出。
6. **Streaming**: C++ 层每生成一个 Token，立即回调 Python 层，转换为 SSE 事件推送到客户端。

---

## 3. 实现细节 (Implementation Details)

### 3.1 C++ 绑定 (`bindings/llama_chat_wrapper.cpp`)
我们维护了自己的轻量级绑定，以便直接访问 `common/chat.h` 的高级功能。
- **Context Management**: 实现了 `create_context(name, n_ctx)` 和 `select_context(name)`。
- **Overflow Checks**: 在 `init_inference` 中硬性检查 `n_prompt + max_tokens > n_ctx`，并在溢出时抛出标准 C++ 异常。
- **Chat Templates**: 复用了 llama.cpp 强大的 Jinja2 模板引擎。

### 3.2 Python Bridge (`src/bridge.py`)
连接 Web 服务和 C++ 推理的核心胶水层。
- **适配器模式**: `AnthropicAdapter` 和 `OpenAIAdapter` 处理协议差异。
- **流式处理**: 实现了智能的 `_stream_generate` 循环，能够处理结构化输出（如 `<thought>`），支持从 C++ 解析结果或 Python 正则 fallback 中提取内容。
- **异常桥接**: 将 C++ `RuntimeError` 映射为 Python `ContextLimitExceededError`。

### 3.3 服务层 (`src/server.py`)
- **Global Exception Handlers**: 统一捕获逻辑错误并映射为 API 错误格式（如 `invalid_request_error`），确保客户端不会收到令人困惑的 `500 Internal Server Error`。

---

## 4. 构建与安装 (Build & Setup)

### 前置要求
- Python 3.10+
- CMake 3.10+
- C++ 编译器 (Clang/GCC)
- `uv` (推荐) 或 `pip`

### 构建步骤
1. **初始化子模块** (如果是 git clone):
   ```bash
   git submodule update --init --recursive
   ```

2. **编译 C++ 绑定**:
   ```bash
   make build
   # 这将生成 llama_chat_wrapper.cpython-*.so 并放入 src/ 目录
   ```

3. **安装 Python 依赖**:
   ```bash
   uv sync
   ```

---

## 5. 使用实例 (Usage Guide)

### 5.1 配置文件 (Recommended)

创建 `configs/claude-code.toml`：

```toml
# 定义模型
[models.mimo]
path = "unsloth/MiMo-V2-Flash-GGUF/UD-Q6_K_XL/"

# 定义主缓存 (用于对话)
[caches.main]
model = "mimo"
n_ctx = 32768
description = "主对话缓存"

# 定义快速缓存 (用于后台任务/Haiku)
[caches.fast]
model = "mimo"
n_ctx = 8192
description = "后台任务缓存"

# 路由规则
[[routes]]
match = "*haiku*"   # 所有 Haiku 模型请求去 fast
cache = "fast"

[[routes]]
match = "*"         # 其他默认去 main
cache = "main"
```

### 5.2 启动服务

```bash
# 推荐：使用配置文件
uv run serve --config configs/claude-code.toml
```

### 5.3 客户端连接

**Claude Code**:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
claude
```

**Cursor / OpenAI SDK**:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"
)

# 即使请求过长，现在也会收到清晰的 400 错误
try:
    response = client.chat.completions.create(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "..." * 10000}]
    )
except openai.BadRequestError as e:
    print(f"Context Overflow: {e}")
```

---

## 6. 开发规约 (Development Guidelines)

为了保持项目的高质量和可维护性，所有贡献者（包括 AI Agent）必须遵守以下规则：

### 6.1 目录权限
* **【允许编辑】** `src/`, `tests/`, `bindings/`, `configs/`
* **【只读/禁动】** `models/` (仅通过 hfd.sh 下载), `vendor/` (仅 git update)
* **【禁止触碰】** `.venv/`

### 6.2 测试驱动开发 (TDD)
* **变更流程**：先写测试 -> 运行失败 -> 修改代码 -> 测试通过。
* **回归测试**：确保所有 `tests/unit` 和 `tests/integration` 下的测试（特别是 `test_overflow_handling.py` 和 `test_structured_content.py`）持续通过。

### 6.3 4-File 日志规则
在 `--debug` 模式下，主要请求/响应数据会被记录，用于调试协议转换问题。
