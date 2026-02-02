# ChatBridge 单元测试计划 (TESTS_CHATBRIDGE.md)

本文档描述 ChatBridge 模块的测试覆盖。测试重点在于 Scanner 与 Flavor 的联合行为，以及 Bridge 的接口对齐。

**核心方针**:
1. **Scanner 负责切分**: 透明块流式输出，不透明块缓冲后整体返回
2. **Flavor 负责规则**: 定义 block_tokens，驱动 Scanner 行为
3. **Bridge 负责调度**: 调用 Wrapper 方法，不含硬编码控制符

---

## 当前测试覆盖

**总计**: 25 个测试
- Scanner/Flavor 测试: 21 个 (`test_scanner_flavors.py`)
- Bridge 完成测试: 4 个 (`test_bridge_completion.py`)

---

## 1. Scanner & Flavor 联合测试

文件: `tests/chat_bridge/test_scanner_flavors.py`

### 1.1 QwenFlavor 测试 (3 个)
| 测试名 | 目标 |
|--------|------|
| `test_plain_content_streaming` | 普通文本流式输出 |
| `test_opaque_tool_call_buffering_qwen` | tool_call 块缓冲（不透明） |
| `test_fragmented_tag_detection` | 极端碎片化标签检测 |

### 1.2 GLMFlavor 测试 (2 个)
| 测试名 | 目标 |
|--------|------|
| `test_reasoning_streaming_glm` | think 块流式输出（透明） |
| `test_mixed_reasoning_and_tool_glm` | think + tool_call 混合 |

### 1.3 GPTOSSFlavor / HarmonyProtocol 测试 (4 个)
| 测试名 | 目标 |
|--------|------|
| `test_plain_content_with_harmony` | 普通文本透传 |
| `test_gpt_oss_analysis_channel` | analysis 通道识别为 reasoning_content |
| `test_gpt_oss_commentary_tool_call` | commentary 通道识别为 tool_call |
| `test_gpt_oss_combined_analysis_and_tool` | 完整流程测试 |

### 1.4 MiniMaxFlavor 测试 (1 个)
| 测试名 | 目标 |
|--------|------|
| `test_minimax_tool_call` | minimax:tool_call 命名空间标签 |

### 1.5 MimoFlavor 测试 (1 个)
| 测试名 | 目标 |
|--------|------|
| `test_mimo_tool_call` | function=Name 特殊语法 |

### 1.6 真实样本测试 (2 个)
| 测试名 | 数据来源 |
|--------|----------|
| `test_qwen_real_output` | samples/qwen/ |
| `test_glm_real_output` | samples/glm/ |

### 1.7 边界情况测试 (4 个)
| 测试名 | 目标 |
|--------|------|
| `test_empty_input` | 空输入处理 |
| `test_only_whitespace` | 纯空白字符 |
| `test_incomplete_tag_at_end` | 流末尾不完整标签 |
| `test_nested_angle_brackets` | 内容中的尖括号 |

---

## 2. Bridge 单元测试

文件: `tests/chat_bridge/test_bridge_completion.py`

### 2.1 非流式测试 (4 个)
| 测试名 | 目标 |
|--------|------|
| `test_complete_anthropic_text_only` | 纯文本响应 |
| `test_complete_anthropic_tool_call` | 包含 tool_call 的响应解析 |
| `test_complete_anthropic_malformed_tool` | 格式错误的工具 JSON |
| `test_complete_anthropic_with_tools` | 验证 tools 参数传递给 apply_template |

---

## 3. 支持的模型格式

| Flavor | Protocol | Tool 格式 | Reasoning 格式 |
|--------|----------|-----------|----------------|
| Qwen | XMLTagProtocol | `<tool_call>{json}</tool_call>` | 隐式 (content) |
| GLM | XMLTagProtocol | `<tool_call>...</tool_call>` | `<think>...</think>` |
| GPT-OSS | HarmonyProtocol | `<\|channel\|>commentary` | `<\|channel\|>analysis` |
| MiniMax | XMLTagProtocol | `<minimax:tool_call>` | `<think>` |
| Mimo | XMLTagProtocol | `<tool_call><function=X>` | `<think>` |

---

## 4. 运行测试

```bash
# 运行所有 ChatBridge 测试
python3 -m pytest tests/chat_bridge/ -v

# 运行单个测试文件
python3 -m pytest tests/chat_bridge/test_scanner_flavors.py -v
python3 -m pytest tests/chat_bridge/test_bridge_completion.py -v
```

---

## 5. 测试数据来源

测试使用 `samples/` 目录下的真实模型输出：
- `samples/qwen/` - Qwen 模型
- `samples/glm/` - GLM 模型
- `samples/gpt-oss/` - GPT-OSS (Harmony) 模型
- `samples/minimax/` - MiniMax 模型
- `samples/mimo/` - Mimo 模型
