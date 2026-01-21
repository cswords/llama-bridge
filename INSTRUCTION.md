# Llama-Bridge 项目开发规约

## 1. 项目愿景与自动进化目标

Llama-Bridge 是一个独立项目，旨在为 GGUF 格式模型提供**多种 LLM API 格式**的原生支持，通过集成 **LiteLLM** 实现 API 格式转换层，使本地 GGUF 模型能够被任何兼容主流 LLM API 的客户端调用。

* **核心使命**：Antigravity 应利用此规约，通过"测试-失败-修正"的进化循环，自动完成项目开发
* **进化终点**：实现对 **Claude Code** 及其他主流 LLM 客户端的**完美支持**
* **技术路径**：通过 FFI 直接调用 llama.cpp，零 HTTP 中转

### 核心目标

1. **多 API 格式支持** - 通过 LiteLLM 支持 Anthropic、OpenAI、Gemini、Azure 等多种 API 格式
2. **零 HTTP 中转** - 内部组件通过 FFI 直接调用 llama.cpp，避免额外的 HTTP 开销
3. **自动跟踪 llama.cpp 更新** - 通过 submodule + 重新编译即可支持新模型
4. **完整工具调用支持** - 利用 llama.cpp 的 `common/chat.h` 实现，而非 Python 重写

---

## 2. 核心技术栈

* **环境管理**: uv (依赖管理), nodeenv (在 venv 中集成 nodejs 环境)
  * **初始化**: 执行 `nodeenv -p` 将 node 环境内嵌于 venv
* **推理引擎**: llama.cpp (通过 pybind11 binding)
* **API 服务**: FastAPI & uvicorn (默认端口: 8000)
* **格式转换**: LiteLLM (库模式)
* **模型下载**: hfd.sh (存储于 scripts/)
* **开发模式**: 严格遵循 **TDD (测试驱动开发)**

---

## 3. 核心设计策略

### 3.1 流式传输策略

1. **协议协商**: 综合判定客户端是否接受流式响应。除显式的 stream 字段外，还需识别隐含信号（如 Accept: text/event-stream 标头或特定的协议模式）。若客户端指示可接受流式传输，则优先采用；否则服务必须采用非流式响应。
2. **真流式输出 (True Streaming)**: 由于 llama.cpp 的 `common/chat.h` 在 C++ 层实现了**实时工具调用检测**，本项目支持 **100% 真流式输出**，无需先缓存完整响应再解析。每个 token 生成后立即通过 FFI 传递给 Python 层，并实时转换为目标 API 格式的 SSE 事件。
3. **结构化内容边界检测**: llama.cpp 通过 grammar-constrained generation 在生成过程中即可识别各类结构化内容块（如工具调用、思考过程、代码块等）的开始和结束边界，无需后处理解析。

### 3.2 调试与日志 (4-File Rule)

在 `--debug` 模式下记录：1_req_client_raw.json, 2_req_model.json, 3_res_model_raw.json, 4_res_client_transformed.json。

### 3.3 进化式统一结构化缓存

通过内部"统一结构化缓存"层实现双端（客户端协议与模型协议）解耦。

### 3.4 LiteLLM 库模式

**重要**：我们使用 LiteLLM 作为 **Python 库**，而非独立的代理服务器：

```python
# ❌ 不是这样（代理模式，需要额外 HTTP）
# litellm --config config.yaml

# ✅ 而是这样（库模式，进程内调用）
import litellm.utils
openai_request = litellm.utils.convert_to_openai_format(anthropic_request)
anthropic_response = litellm.utils.convert_to_anthropic_format(openai_response)
```

---

## 4. 目录结构与行为边界（严格遵守）

Antigravity 在操作时必须遵循以下**权限约束**：

* **【允许编辑】** `src/`, `tests/`, `bindings/`
* **【只读/禁动】** `scripts/`: 仅用于存储第三方脚本（如 hfd.sh），在脚本下载完成后**严禁编辑**
* **【只读/禁动】** `models/`:
  * **唯一来源**：无论是开发测试还是真实运行，**必须且只能**使用该目录下的模型
  * **严禁修改**：仅存放 hfd.sh 下载的模型，严禁修改任何内容
* **【只读/禁动】** `vendor/`: llama.cpp submodule，通过 git 命令更新
* **【禁止触碰】** `.venv/`: 由 uv 自动生成，**严禁手动修改内部文件**
* **【绝对禁止】** 项目文件夹以外的任何路径: 严禁访问或操作项目根目录以外的文件系统

### 项目结构

```
llama-bridge/
├── vendor/
│   └── llama.cpp/                  # llama.cpp submodule (pinned to b7775)
├── bindings/
│   ├── CMakeLists.txt              # 构建脚本
│   └── llama_chat_wrapper.cpp      # pybind11 wrapper
├── src/                            # Python 源代码
│   ├── server.py                   # FastAPI server
│   ├── bridge.py                   # Core Bridge logic
│   └── adapters/                   # Protocol Adapters
│       ├── base.py                 # Adapter interface
│       ├── anthropic.py            # Anthropic adapter
│       └── openai.py               # OpenAI adapter
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
│   └── hfd.sh                      # 模型下载工具
├── models/                         # 模型存储（只读）
├── logs/                           # 调试日志
├── pyproject.toml
└── Makefile
```

### models/ 目录规范

```
models/
├── unsloth/
│   └── MiMo-V2-Flash-GGUF/
│       └── UD-Q6_K_XL/*.gguf
├── bartowski/
│   └── Qwen3-Coder-30B-A3B-Instruct-GGUF/
│       └── *.gguf
└── ...
```

**规则**：
- **唯一来源**：所有模型必须通过 `hfd.sh` 下载或用户手动放置
- **严禁修改**：模型文件必须与下载的原始文件严格一致
- **分片模型**：llama.cpp 会自动识别 `-00001-of-` 格式的分片模型

### logs/ 目录规范（4-File Rule）

```
logs/
└── 20260119_104530_123456/
    ├── 1_req_client_raw.json
    ├── 2_req_model.json
    ├── 3_res_model_raw.json
    └── 4_res_client_transformed.json
```

---

## 5. 命令行接口 (uv run)

1. **serve**: 启动服务
   * 命令: `uv run serve --model <path> [--port <port>] [--debug]`
   * 模型路径解析规则：
     * 自动在 `models/` 目录下查找（无需手动添加 `models/` 前缀）
     * 支持目录路径，自动查找其中的 `.gguf` 文件
     * 分片模型自动选择第一个分片 (`-00001-of-`)
   * 示例：
     * `uv run serve --model Qwen/Qwen2.5-3B-Instruct-GGUF`
     * `uv run serve --model unsloth/GLM-4.7-REAP-218B-A32B-GGUF/Q2_K`
2. **cc**: Claude Code Wrapper
   * 命令: `uv run cc` (自动配置 ANTHROPIC_BASE_URL)
3. **hfd**: 下载模型
   * 命令: `uv run hfd <repo> --include <glob>`
   * 示例: `uv run hfd Qwen/Qwen2.5-3B-Instruct-GGUF --include "*q4_k_m.gguf"`

---

## 6. 全维度 TDD 进化路线图

* **Step 0 (引导起点)**: 编写基础 E2E 测试 tests/e2e/test_basic.py
* **Step 1 (基础设施)**: 实现 serve 基础框架并建立全维度测试目录
* **Step 2 (协议转换核心)**: 实现统一结构化缓存映射，通过单元测试验证
* **Step 3 (功能迭代)**: 通过集成测试和 E2E 测试驱动文字生成、工具调用和记忆功能的实现
* **Step 4 (稳态达成)**: 迭代至通过所有严格与宽容模式的测试

---

## 7. 注意事项

1. **功能完整性**: 严禁阉割协议细节
2. **Mock 依据**: 必须基于 logs/ 真实数据
3. **调试辅助**: 参考模型自带的 chat template
4. **下载限制**: 除测试所需模型外，严禁主动下载
5. **模型资产保护与下载限制**:
   * models/ 目录必须保证其中的模型文件与 hfd.sh 下载的原始文件严格一致
   * **严禁主动下载**：Antigravity **严禁主动执行任何非必要的模型下载任务**
6. **功能完整性驱动修正**:
   * 任何未知的请求 Header 或协议细节可能直接影响客户端行为，严禁盲目忽略

---

## 8. 全自动执行授权与闭环工作流 (Autonomous Execution)

为了实现最高效的自动进化，Antigravity 必须遵循以下自动化原则：

* **全自动修改授权**：在 `src/`、`tests/` 和 `bindings/` 目录范围内，Antigravity **被授予无需询问、直接修改文件的完全权限**
* **闭环迭代逻辑**：
  1. **观测**：运行测试并分析失败日志
  2. **决策**：自主判定导致失败的协议缺失或逻辑错误
  3. **执行**：**直接应用代码修改**，无需请求用户确认
  4. **验证**：重新运行测试
* **中断原则**：
  * **禁止询问**：禁止在 Step 内部的循环迭代中询问"我是否可以修改 xxx 文件"
  * **仅限报告**：仅在以下情况停下：
    1. 当前 Step 的所有测试已通过
    2. **连续 5 次修复尝试失败**：在请求决策前，**必须回滚所有导致失败的尝试**
    3. 完成了北极星测试

---


---

## 9. 断点续传与进度自检 (State Check & Resumption)

若开发过程因外部原因停止，Antigravity 在重新启动时应执行以下自检流程：

1. **扫描环境**：检查 src/ 是否已有代码，检查 .venv/ 是否已配置
2. **定位进度**：依次运行 Step 0 至 Step 4 相关的测试，根据第一个失败的测试用例判定当前所在的 Step
3. **恢复上下文**：分析最新的 logs/ 日志，理解中断前的最后一次请求
4. **自主接力**：直接从断点处继续执行工作流

### 9.1 [TODO] 自动上下文切换侦测 (Auto Context Switching)

当前系统使用基于长度和匹配度的启发式算法来区分"临时请求"与"从主对话"。未来应升级为基于 **System Prompt 指纹** 判定：
- 比对新请求头部与主缓存头部
- 若头部（System Prompt）完全不匹配，则判定为**切换客户端/新话题**，自动重置主缓存
- 此功能将彻底解决多客户端切换时的缓存污染问题，无需人工重启

### 9.2 [TODO] 多模型分流 (Multi-Model Routing)

支持根据请求中的 `model` 字段或 API endpoint 将请求路由到不同的模型实例：
- **路由依据**：`payload.model` 字段（如 `claude-3-sonnet` vs `claude-3-haiku`）或 endpoint 路径
- **实现方式**：在 Python 层维护多个 `LlamaChatWrapper` 实例，按模型名映射
- **内存开销**：每个模型独立加载权重，适合异构模型场景（如主力模型 + 轻量探测模型）
- **用例**：模拟云端架构，用大模型处理主对话、小模型处理分类/探测任务

### 9.3 [TODO] 多 Context KV Cache 隔离 (Multi-Context Isolation)

支持单模型多 Context 架构，实现 KV Cache 的物理隔离：
- **核心思想**：`llama_model` 权重共享，但创建多个 `llama_context`，每个 Context 有独立的 KV Cache
- **路由依据**：同 9.2，基于 `model` 字段或 endpoint
- **内存开销**：权重只加载一次，每个 Context 仅需额外的 KV Cache 空间
- **优势**：
  - 彻底消除"缓存污染"问题（物理隔离，无需启发式判断）
  - 比多模型方案更节省内存
  - 对任何客户端都有效，无需针对 Claude Code 特殊优化
- **实现要点**：
  - C++ 层：`LlamaChatWrapper` 支持创建/管理多个 `llama_context`
  - Python 层：Context 池管理（类似数据库连接池）
  - API 层：根据请求路由到对应 Context

### 9.4 [TODO] 上下文溢出友好错误 (Context Overflow Error Handling)

当请求的 Token 数量超过 `n_ctx` 限制时，应返回符合 API 规范的错误响应，而非抛出 RuntimeError 导致 500：
- **Anthropic 格式**：
  ```json
  {
    "type": "error",
    "error": {
      "type": "invalid_request_error",
      "message": "Request exceeds maximum context length. Requested 75612 tokens, but limit is 8192."
    }
  }
  ```
- **OpenAI 格式**：
  ```json
  {
    "error": {
      "message": "This model's maximum context length is 8192 tokens. However, your messages resulted in 75612 tokens.",
      "type": "invalid_request_error",
      "code": "context_length_exceeded"
    }
  }
  ```
- **实现要点**：
  - C++ 层：在 `init_inference` 中检测溢出，抛出特定异常类型（如 `ContextOverflowError`）
  - Python 层：捕获该异常，根据请求来源的 API 格式返回对应的错误 JSON
  - HTTP 状态码：`400 Bad Request`（而非 `500 Internal Server Error`）

### 9.5 [BUG] Ephemeral Sequence Batch ID 不匹配 (2026-01-21 发现)

**现象**：当 Claude Code 持续运行一段时间后，模型突然开始输出无意义的重复字符（如 `!!!!!...`），持续数小时无法停止。

**根本原因**：当前的 Ephemeral 序列切换逻辑存在 Bug。在 `init_inference` 中：
1. 我们正确检测到 Ephemeral 请求并设置 `current_seq_id_ = 1`
2. 但在创建 `llama_batch` 时，**仍然硬编码使用了 Sequence ID 0**：
   ```cpp
   common_batch_add(batch, tokens[i], 
                    n_keep + i,    // <-- 位置基于 Seq 0 的 n_keep
                    {0},           // <-- 序列 ID 硬编码为 0！
                    ...);
   ```
3. 结果：Batch 声称 Token 属于 Seq 0 从位置 Y 开始，但 KV Cache 中 Seq 0 的最后位置是 X，且 `Y ≠ X+1`
4. `llama_decode` 检测到位置不连续，返回 -1（失败）
5. 模型进入错误状态，输出垃圾数据

**服务器日志特征**：
```
DEBUG: Ephemeral request detected (seq_id=1). Not touching main cache.
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 507
 - the tokens for sequence 0 in the input batch have a starting position of Y = 138
 it is required that the sequence positions remain consecutive: Y = X + 1
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
```

**修复方向**：
1. **短期**：修复 `common_batch_add` 调用，使用 `{current_seq_id_}` 而非硬编码 `{0}`，并正确计算 Ephemeral 序列的起始位置
2. **长期**：采用 9.3 节描述的"多 Context KV Cache 隔离"方案，彻底规避 Sequence 切换的复杂性

**影响**：此 Bug 会导致长时间运行的 Claude Code 会话在触发 Ephemeral 检测后完全失效，需要重启服务器。

---

## 10. 测试体系详细规范 (Detailed Testing System)

### 10.1 单元测试 (unit/)

* **对象**: 最小功能模块（如格式转换器、Binding 封装）
* **要求**: 极速运行，**严禁依赖任何服务或网络请求**

### 10.2 集成测试 (integration/)

* **对象**: 内部协议流转换
* **方式**: 模拟客户端向服务发送请求。**必须使用测试脚本自动触发并管理的 Mock 服务**
* **目标**: 确保"统一结构化缓存"能够无损地承载各种 API 协议中的所有元数据

### 10.3 E2E 测试 (e2e/)

所有的 E2E 测试必须覆盖交互模式、非交互模式，以及文本生成、工具调用、上下文记忆三大功能。

#### 10.3.1 测试用例分类（按严格程度）

1. **不宽容的部分 (Strict/Intolerant)**:
   * **行为**: **必须使用由测试脚本自动启动并销毁的 Mock 服务**
   * **目的**: 在受控环境下验证协议字段的 100% 匹配
2. **较宽容的部分 (Lenient)**:
   * **行为**: **优先检测并复用在测试脚本外手动运行的服务**
   * **目的**: 允许在真实模型环境下进行端到端验证

### 10.4 Mock 服务设计

* **主要依据**: llama.cpp 源代码和文档是 Mock 服务行为的权威参考
* **验证方式**: 可对比 `llama-server` 的输出验证 Mock 正确性
* **4-File 日志**: 作为交叉验证来源。若日志与代码/文档不一致，**必须分析原因**，不可忽略

---

## 11. 功能完整性驱动修正

目标是支持功能完整的客户端。任何未知的 Header 若导致异常，必须通过分析 4-file 日志在协议转换层进行精准补全。

---

## 12. 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Llama-Bridge                            │
├─────────────────────────────────────────────────────────────────┤
│  任意客户端 (Claude Code, OpenAI SDK, Cursor, Continue...)     │
│      ↓ HTTP (Anthropic/OpenAI/Gemini/... 任意格式)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   FastAPI (多格式端点)                                   │  │
│  │   ├── /v1/messages         (Anthropic)                   │  │
│  │   ├── /v1/chat/completions (OpenAI)                      │  │
│  │   └── /...                 (其他 LiteLLM 支持的格式)      │  │
│  │       ↓ Python method call                               │  │
│  │   Protocol Adapters (src/adapters/)                      │  │
│  │       ↓ Python method call                               │  │
│  │   llama_chat_bindings (pybind11)                         │  │
│  │       ↓ FFI 调用                                         │  │
│  │   libllama.dylib + common/chat.h                         │  │
│  │       ↓ 原生推理                                         │  │
│  │   GGUF 模型                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ✅ 进程内 FFI 调用，无内部 HTTP 中转                          │
│  ✅ 支持所有 LiteLLM 兼容的 API 格式                           │
│  ✅ 自动工具调用解析（使用 llama.cpp chat.h）                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. 关键设计决策

### 13.1 llama.cpp 作为 Git Submodule（非 Fork）

**理由**：
- 可通过 `git submodule update --remote` 轻松跟踪上游
- 无需维护 fork 分支
- 清晰的代码边界

**更新模型支持流程**：
```bash
cd vendor/llama.cpp
git fetch origin
git checkout <new-tag>
cd ../..
make build  # 重新编译
```

### 13.2 编译 common/ 为可链接库

llama.cpp 的 `common/chat.h` 提供：
- Jinja2 模板解析
- 工具调用格式化
- 工具调用响应解析
- 多种模型格式支持（Qwen, Llama, Hermes, MiMo 等）

### 13.3 pybind11 Wrapper 接口设计

```cpp
// bindings/llama_chat_wrapper.cpp
PYBIND11_MODULE(llama_chat, m) {
    m.def("init_templates", ...);
    m.def("apply_template", ...);
    m.def("parse_response", ...);
}
```

### 13.4 Protocol Adapter Pattern (src/adapters/)

```python
# src/adapters/base.py
class BaseAdapter(ABC):
    @abstractmethod
    def to_internal(self, request: dict) -> dict: ...
    @abstractmethod
    def from_internal(self, response: dict, original_req: dict) -> dict: ...
    @abstractmethod
    def chunk_to_sse(self, chunk: dict) -> str: ...
```

---

## 14. 构建流程

### 前置要求

- macOS 12+ 或 Linux
- Python 3.11+
- CMake 3.20+
- C++17 编译器（clang/gcc）
- uv（Python 包管理）

### 构建步骤

```bash
# 1. 克隆项目（含 submodule）
git clone --recursive https://github.com/xxx/llama-bridge.git
cd llama-bridge

# 2. 编译 C++ 组件
make build

# 3. 初始化环境
uv sync

# 4. 运行服务
uv run serve --model models/<path-to-model>.gguf
```

---

## 15. 里程碑

### Phase 1: 基础框架
- [x] 项目结构搭建
- [x] llama.cpp submodule 配置
- [x] 基础构建脚本

### Phase 2: C++ Bindings
- [x] pybind11 wrapper 实现
- [x] chat.h 关键函数暴露
- [x] 推理 API 暴露

### Phase 3: Python 服务
- [x] FastAPI 服务器
- [x] 格式转换层
- [x] 非流式响应

### Phase 4: 流式支持
- [x] 流式生成 binding
- [x] SSE 响应实现
- [x] 真正的流式工具调用

### Phase 5: 测试与优化
- [ ] 完整测试套件
- [ ] 性能基准测试
- [ ] 文档完善

---

## 16. 参考资源

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [llama.cpp function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
