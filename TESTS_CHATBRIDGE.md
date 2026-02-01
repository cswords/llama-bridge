# ChatBridge 单元测试计划 (TESTS_CHATBRIDGE.md)

本文档详细列出了针对 ChatBridge 模块的单元测试用例。测试重点在于 Scanner 与 Flavor 的联合行为，以及 Bridge 的流式调度逻辑。

**核心方针**:
1.  **Scanner负责切分**: 测试重点是透明块能否正确流式输出，不透明块能否被正确缓冲（忽略）。
2.  **Flavor负责规则**: 测试不同的 Flavor (Qwen, MiniMax, GLM) 是否能正确驱动 Scanner。
3.  **Bridge负责调度**: 测试流式、非流式、混合流的终局解析逻辑。

---

## 1. Scanner & Flavor 联合测试
*(验证 Scanner 在不同 Flavor 规则下的行为)*

## 1. Scanner & Flavor 联合测试
*(验证 Scanner 在不同 Flavor 规则下的行为)*

**测试策略**: 测试数据不再硬编码，而是从实例化的 `flavor` 对象的属性（如 `block_tokens`）中动态生成。这样可以确保测试的是 Flavor 真实定义的行为。

### 1.1 普通 Content 流式能力 (Implicit Content)
*   **目标**: 验证没有任何显式 Block 时，Scanner 能否正确输出 `content` 事件。
*   **覆盖**: `QwenFlavor`, `HarmonyFlavor`, `MiniMaxFlavor`, `MimoFlavor`, `GLMFlashFlavor`
*   **Case**:
    *   **Input**: `["你好", "，", "世界"]`
    *   **Verify**: 收到 `content("你好")`, `content("，")`, `content("世界")`。Block 始终维持在 Implicit Root。

### 1.2 Reasoning Content 流式能力 (Thinking)
*   **目标**: 验证透明思考块能否正确输出 `reasoning` 事件。
*   **覆盖**: 必须使用 Flavor 定义的具体标签。
*   **Case 1: QwenFlavor**
    *   **Features**: `think_start="<think>"`
    *   **Verify**: 输出 `reasoning`。
*   **Case 2: MiniMaxFlavor**
    *   **Features**: `<think>`
*   **Case 3: MimoFlavor**
    *   **Features**: `<think>`
*   **Case 4: GLMFlashFlavor**
    *   **Features**: `<think>`
*   **Case 5: HarmonyFlavor (GPT-OSS)**
    *   **Features**: `think_start="<|channel|>analysis<|message|>"` (基于 gpt-oss-120b)
    *   **Verify**: 识别 Control Tokens 后的内容为 Reasoning。

### 1.3 不透明块缓冲与忽略 (Opaque Block)
*   **目标**: 验证不透明块（如工具调用）是否被完整吞没。
*   **覆盖**: 使用 `flavor.block_tokens` 中被标记为 Opaque 的 Tag。
*   **Case 1: QwenFlavor (Standard JSON)**
    *   Input: `["<tool_call>", "...", "</tool_call>"]`
    *   **Verify**: Scanner 吞没所有内容。
*   **Case 2: HarmonyFlavor (Control Tokens)**
    *   Input: `["<|start|>assistant", " to=functions.foo", "<|channel|>commentary", " json<|message|>", "...", "<|call|>"]`
    *   **Verify**: Scanner 识别起始序列 `<|start|>assistant to=functions` 为 Opaque Block Start，吞没后续直到 `<|call|>` (或者下一个 `<|start|>`?)。
    *   *Note: Harmony 的 Opaque 边界比 XML 更复杂，Flavor 需精准定义 block_start/end。*
*   **Case 3: MiniMaxFlavor (Legacy Invoke)**
    *   Input: `["<minimax:tool_call>", "<invoke...>", "</minimax:tool_call>"]`
*   **Case 4: MimoFlavor (Namespace Tags)**
    *   Input: `["<tool_call>", "...", "</tool_call>"]`
*   **Case 5: GLMFlashFlavor (Flat XML Sequence)**
    *   Input: `["<tool_call>", "...", "</tool_call>"]`

### 1.4 极端碎片化健壮性 (Fragmentation Robustness)
*   **目标**: 验证 Scanner 在极端切分下的稳定性。
*   **覆盖**: 选取最复杂的 Tag `MimoFlavor` 和 `GLMFlavor` 进行压力测试。
    *   `Mimo`: `["<", "minimax", ":", "tool", "_", "call", ">"]`
    *   `GLM`: `["<", "arg", "_", "key", ">"]`
*   **预期**: Scanner 能够正确挂起等待，直到拼出完整 Tag。

### 1.4 混合嵌套与状态切换 (Mixed Blocks)
*   **目标**: 验证透明块与不透明块的切换与嵌套。
*   **覆盖 (All Flavors)**: 验证每种模型在“思考中夹带私货”时的行为。
*   **Case 1: MiniMax/Mimo**
    *   Input: `["<think>", "...", "<invoke>...</invoke>", "..."]`
    *   Verify: Reasoning -> Root (Invoke 被吞) -> Root (回落失败? 不, Invoke 是 Opaque, 不会切断 think 除非 think 结束? 需确认嵌套规则: **Tool 禁止嵌套在 Reasoning 下吗?** 表格说是。则 Scanner 行为应是吞没。)
*   **Case 2: Qwen/GLM**
    *   Input: `["<think>", "...", "<tool_call>...</tool_call>"]`
    *   Verify: 同上，Reasoning 期间 Tool 被吞没，用户只看到思考。

### 1.5 Prefill 抓取测试 (Prefill Initialization)
*   **目标**: 验证 Scanner 根据 Prompt 后缀从正确状态启动。
*   **覆盖 (All Flavors)**: `DeepSeekFlavor` (或其他支持 Prefill 的 Flavor)。
*   **Case 1: DeepSeek**
    *   Prompt: `... <think>` -> Output `Ok` -> Event `reasoning("Ok")`
*   **Case 2: Qwen**
    *   Prompt: `... <think>` -> Output `Ok` -> Event `reasoning("Ok")`
*   **Case 3: MiniMax**
    *   Prompt: `... <think>` -> Output `Ok` -> Event `reasoning("Ok")`

---

## 2. Bridge 单元测试 (Unit Tests)
*(测试 ChatBridge 类的调度逻辑，Mock Wrapper)*

### 2.1 非流式输出 (Non-Streaming)
*   **配置**: `stream=False`
*   **Mock Wrapper**: 直接返回完整字符串。
*   **预期行为**:
    1.  Scanner 不工作（或仅做简单透传）。
    2.  调用 `wrapper.parse_response` 获取全量解析结果。
    3.  返回完整 JSON 响应。

### 2.2 纯 Content 流式 (Content Only)
*   **Mock Stream**: `["Hello", " World"]`
*   **预期行为**:
    1.  Yield `content("Hello")`
    2.  Yield `content(" World")`
    3.  最后 `parse_response` 确认无 Tool，结束。

### 2.3 纯 Reasoning 流式 (Reasoning Only)
*   **Mock Stream**: `["<think>", "Why", "</think>"]`
*   **预期行为**:
    1.  Yield `reasoning("Why")`
    2.  最后 `parse_response` 确认无 Tool，结束。

### 2.4 混合流 + 工具调用 (Reasoning + Content + Tool)
*   **Mock Stream**: `["<think>", "Thinking", "</think>", "Here is content", "<tool_call>", "...json...", "</tool_call>"]`
*   **Mock C++ Parse**: 返回 Tool Object `{"name": "test"}`
*   **预期行为**:
    1.  Yield `reasoning("Thinking")`
    2.  Yield `content("Here is content")`
    3.  **过程中不 Yield Tool**。
    4.  流结束调用 `parse_response`。
    5.  Yield `tool_use({"name": "test"})` 事件。

### 2.5 只有 Tool 的情况 (Tool Only)
*   **Mock Stream**: `["<tool_call>", "...json...", "</tool_call>"]`
*   **预期行为**:
    1.  全程静默（Yields nothing during stream）。
    2.  流结束调用 `parse_response`。
    3.  Yield `tool_use` 事件。

---

**总结**: 这套测试用例覆盖了从底层字符匹配到上层事件调度的所有关键路径，确保 Hybrid Architecture 的稳健性。
