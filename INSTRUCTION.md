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

### 3.2 Python Bridge (`src/bridge/`)
连接 Web 服务和 C++ 推理的核心胶水层。
- **模块化架构**: 该组件被拆分为 `base.py` (通用核心逻辑)、`anthropic.py` (Anthropic 协议集成) 和 `openai.py` (OpenAI 协议集成)，极大地提高了可维护性。
- **适配器模式**: `AnthropicAdapter` 和 `OpenAIAdapter` 处理协议差异。
- **模型翻译层 (Model Translation Layers)**: 对于具有特殊工具调用格式的模型（如 Qwen 的 XML 格式或 MiniMax-M2 的 `<minimax:tool_call>` 格式），我们在 Python 层实现了专门的解析器（`_parse_qwen_tool_calls`, `_parse_minimax_tool_calls`）。
    - **无状态设计**: C++ 包装器保持简单且无状态，不存储模型特定的解析规则。
    - **双层防护**: 流式生成时，Python 层的 `process_and_yield_buffer` 会主动抑制特有的 XML 标签，确保 UI 洁净，同时在生成结束时进行完整解析以提取结构化数据。
- **流式处理**: 实现了智能的 `_stream_generate` 循环，能够处理结构化输出（如 `<thought>`），支持从 C++ 解析结果或 Python 正则 fallback 中提取内容。
- **异常桥接**: 将 C++ `RuntimeError` 映射为 Python `ContextLimitExceededError`。

### 3.3 服务层 (`src/server.py`)
- **Global Exception Handlers**: 统一捕获逻辑错误并映射为 API 错误格式（如 `invalid_request_error`），确保客户端不会收到令人困惑的 `500 Internal Server Error`。

### 3.4 参数优化系统 (Parameter Optimization System)
Llama-Bridge 集成了基于 `llama-bench` 的自动化参数调优功能，通过在目标硬件上实际运行基准测试来寻找最优配置。
- **多维度搜索**：
    - **线程与计算优化**：自动寻找计算线程 (`n_threads`) 的最优平衡点。
    - **批处理优化**：测试 `n_batch` 和 `n_ubatch` 的组合，极大化 Prompt 处理 (PP) 吞吐。
    - **Flash Attention**：验证 `-fa` (Flash Attention) 在当前硬件上的吞吐提升。
    - **KV Cache 压缩**：测试 `f16` vs `q8_0` vs `q4_0` 等 KV Cache 格式，平衡显存占用与推理速度。
- **交互模式**：
    - `uv run optimize --config <path>`：对指定配置文件进行基准测试并以 TOML 格式在大屏输出推荐配置。
    - **自动应用**：使用 `--apply` 参数时，系统会将最优参数（包括 `flash_attn`, `cache_type_k`, `cache_type_v` 等）直接回写至原配置文件。

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
   uv run build
   # 这将自动同步 llama.cpp (git submodule) 并生成绑定文件
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

---

## Appendix A: 协议规范 (Protocol Specifications)

### A.1 OpenAI Harmony / MiniMax 协议

**参考资料 (References):**
- [OpenAI Harmony Cookbook](https://developers.openai.com/cookbook/articles/openai-harmony)
- [Unsloth Chat Templates](https://unsloth.ai/docs/basics/chat-templates)
- [llama.cpp Supported Templates](https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)

**消息结构 (Structure):**
```
<|start|>{header}<|message|>{content}<|end|>
```

**核心组件 (Key Components):**
- **角色 (Roles)**: `system`, `user`, `assistant`, `developer`, `functions.{name}`
- **频道 (Channels/Headers)**: `final` (文本), `analysis` (推理), `commentary` (工具调用).
- **分隔符 (Delimiters)**: `<|message|>` (头体分隔), `<|channel|>` (类型分隔).
- **停止符 (Stop Tokens)**: `<|end|>`, `<|return|>` (生成结束), `<|call|>` (工具调用结束).

**工具调用 (模型输出):**
```
<|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{JSON_ARGS}<|call|>
```
*注意：`<|constrain|>` Token 属于头部信息，**不是** Block 分隔符。*

**工具结果 (模型输入):**
```
<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{JSON_RESULT}<|end|>
```

### A.2 模板解析策略与单一事实来源 (Template Resolution Strategy & Source of Truth)

为了确保使用的控制 Token 与模型量化时完全一致，我们主要依赖 `llama.cpp` 内部的模板应用逻辑，它会直接读取嵌入在 GGUF 模型元数据中的聊天模板定义。

**回退/验证流程 (Fallback/Verification Workflow):**
如果 GGUF 模型缺少元数据或表现异常，请按以下步骤操作：
1.  **检查原始模型**: 在 HuggingFace 上定位未量化的源模型。
2.  **检查配置文件**: 下载源模型的 `tokenizer_config.json` 或 `chat_template.json`。
3.  **参考官方定义**: 对照官方文档（如 OpenAI Harmony 文档、Unsloth 维基）。
4.  **最终验证**: 仅将这些外部来源作为*参考*来验证 `llama.cpp` 的行为，或在自动检测失败时用于构建手动模板。

**核心原则**: `llama.cpp` 是最终的执行引擎；它对 GGUF Header 的解读是 Token 化和模板应用的最高事实来源。

### A.3 流式策略：智能缓冲与守门人逻辑 (Smart Buffering & Gatekeeper Logic)

为了解决 **协议安全性**（防止原始协议文本泄露）与 **用户体验**（最小化首字延迟 Time-To-First-Token）之间的固有冲突，Llama-Bridge 采用了一套基于以下哲学的 **混合缓冲/流式 (Hybrid Buffering/Streaming)** 策略：

**1. Block 容器原则 (The Block Container Principle)**
*   **真正的 Block Starter**：只有那些定义了完整、有效容器的 Token 才是 Block Starter。
    *   **Harmony**: `<|start|>` (开启一个以 `<|end|>` 结尾的完整 Message 容器)。
    *   **XML**: 标准标签如 `<thinking>`, `<tool_code>` (以 `</...>` 结尾)。
*   **分隔符不是 Block**：诸如 `<|channel|>`, `<|message|>` 或 `<|constrain|>` 等 Token 仅仅是 Block 内部的分隔符。它们**不**触发独立的缓冲状态。

**2. 守门人机制 (The Gatekeeper Mechanism)**
*   **缓冲阶段 (Buffering Phase)**：一旦进入 Block（例如遇到 `<|start|>`），Scanner 即进入 `Buffering` 状态。在此阶段 **没有任何内容会被发送给用户**。这确保了不完整的或内部的 Header（如 `assistant<|channel|>commentary`）绝不会泄露。
*   **开闸决策 (The Gate Decision)**：Scanner 监控缓冲区中的特定 "Gate" Token，这些 Token 标志着 Header 到 Body 的过渡。
    *   **Harmony Gate**: `<|message|>`
    *   **XML Gate**: 标签的闭合括号 `>`。
*   **透传模式 (Pass-Through/Streaming Mode)**：如果 Header 表明内容是 **安全且可展示的**（例如 `final` channel 的文本，或 `analysis` channel 的推理），“闸门”打开。Scanner 立即 Flush 缓冲区，并实时流式传输所有后续字符。
*   **拦截模式 (Hold/Parsing Mode)**：如果 Header 表明内容是 **机器可读的**（例如 `commentary` channel 的工具调用，或未知的元数据），“闸门”保持关闭。整个 Block 将被缓冲直到遇到结束 Token（`<|end|>`），然后被送往 Parser 进行结构化处理。

**策略汇总表：**

| 协议 (Protocol) | Block 类型 | Header 特征 | Gate Token | 动作 (Action) |
| :--- | :--- | :--- | :--- | :--- |
| **Harmony** | **文本 (Text)** | `<|channel|>final` | `<|message|>` | **流式 (Stream)** |
| **Harmony** | **推理 (Reasoning)** | `<|channel|>analysis` | `<|message|>` | **流式 (Stream)** (如开启) |
| **Harmony** | **工具调用 (Tool Call)** | `<|channel|>commentary` | `<|message|>` | **缓冲 (Buffer)** (等待解析) |
| **XML** | **文本 (Text)** | (无 Tag / `<text>`) | N/A | **流式 (Stream)** |
| **XML** | **思考 (Thinking)** | `<thinking>` | `>` | **流式 (Stream)** (如开启) |
| **XML** | **工具代码 (Tool Code)** | `<tool_code>` | `>` | **缓冲 (Buffer)** (等待解析) |

这一统一逻辑确保了：
1.  **零泄露**：用户永远不会看到原始协议头或不完整的工具调用。
2.  **低延迟**：普通的文本回复一旦 Header 校验通过，立即开始流式输出。
3.  **鲁棒性**：复杂的多行工具调用在触发任何事件前，都会被被完整捕获并验证。

**3. 隐式 Block 注入 (Implicit Block Injection)**

为了统一处理结构化协议（Harmony/XML）与普通文本输出（如 ChatML, Llama-2），Scanner 引入了“隐式 Block”机制：
*   **触发条件**：当 Scanner 遇到如果不属于 Skip Token 且不是显式 Block Starter 的普通文本时。
*   **动作**：Scanner 自动进入一个虚拟的 **Content Block**（可以视为 `<|implicit|>`）。
*   **属性**：该 Block 默认被标记为 **Safe & Displayable**，立即进入 **透传模式 (Streaming)**。
*   **意义**：这一机制消除了 Scanner 中“游离在 Block 之外的 Raw Text”状态。所有的输出字符，要么被明确丢弃（Skip），要么被封装在某个 Block 中受控，从而实现了逻辑的完全闭环和统一。

### A.4 上下文感知的 EOS 处理 (Context-Aware EOS Handling)

Scanner 在处理 End-Of-String (EOS) 信号时，必须根据当前的生命周期阶段采取截然不同的策略，以区分 **Prefill (预填充)** 和 **Generation (生成)**：

**1. Input Phase (Prefill Check)**
*   **语义**：在此阶段，EOS 仅代表用户输入的结束，是模型生成的**起点**。它意味着“接力棒的交接”，而非任务的完结。
*   **动作**：Scanner **绝对禁止** 自动关闭未闭合的 Block（如 `<|start|>assistant`）。
*   **状态保留**：若 Scanner 在 Input 结束时处于 `Block Open` 状态，则判定为 **存在 Prefill**。该状态必须被完整保留并传递给生成循环，确保模型生成的第一个字符能够无缝接入该 Block。

**2. Output Phase (Generation Flush)**
*   **语义**：在此阶段，EOS 代表模型停止生成（或连接断开）。它意味着“承诺必须兑现”。
*   **动作**：Scanner **必须强制关闭** 所有未闭合的 Block，并尝试进行 Interpret（解析）。
*   **容错**：即使缺少显式的 `<|end|>` 或 `</tag>`，Scanner 也应假设 Block 已结束，并尽最大努力提取已缓冲的内容（如完整的 JSON），避免因截断导致的数据丢失。

**核心原则**：**Input EOS != Close Block**，只有 **Output EOS** 才是强制结算点。
