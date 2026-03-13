# 03 - 工具调用：让 Agent 能行动

> **主维度**：D1 核心组件
> **次维度**：D3 工程实现
> **关键关系**：
> - Function Calling (方法) --是一种--> 工具调用 (方法)：Function Calling 是工具调用的标准实现
> - 工具调用 (方法) --属于--> Agent (架构)：工具调用属于 Agent 核心组件
> - ReAct (方法) --依赖--> 工具调用 (方法)：ReAct 依赖工具调用来执行 Action
>
> **学习路径**：全景概览 → 提示与推理 → **本章（工具调用）** → 记忆 → 规划
>
> **前置知识**：
> - ReAct 框架（参见 `02-prompting-reasoning.md`）
> - JSON 数据格式基础
>
> **参考**：
> - [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
> - [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
> - [Schick et al., 2023 - Toolformer](https://arxiv.org/abs/2302.04761)
> - [Qin et al., 2023 - Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354)

---

## 核心问题

**LLM 只能生成文本，怎么让它"执行"操作？**

上一章讲了 ReAct 的 Action 步骤——Agent 调用工具。但具体怎么实现？LLM 输出的是 token，怎么变成一次函数调用？工具怎么定义？结果怎么返回？

> 可靠程度：Level 1（Function Calling 已是行业标准）

---

## 1. 工具调用的基本机制

### 1.1 核心思路：LLM 生成结构化输出

工具调用的实现原理很简单：

1. 告诉 LLM 有哪些工具可用（工具的名称、功能描述、参数格式）
2. LLM 分析用户请求后，决定是否需要调用工具
3. 如果需要，LLM **生成一段结构化的 JSON**，描述"调用哪个工具、传什么参数"
4. **外部系统**（不是 LLM）解析这段 JSON，执行实际的函数调用
5. 把执行结果返回给 LLM，LLM 根据结果生成最终回答

关键理解：**LLM 自己不执行任何工具**。它只是"说出"要调用什么——实际执行由 Agent 框架中的外部代码完成。就像一个指挥官下达命令，但不亲自上战场。

### 1.2 完整流程图

```
用户输入
  │
  ▼
┌──────────────────────┐
│  LLM 分析            │
│  "需要调用工具吗？"    │
│                      │
│  ├── 不需要 → 直接生成文本回答
│  │
│  └── 需要 → 生成工具调用 JSON
│              {                          │
│                "name": "get_weather",    │
│                "arguments": {           │
│                  "city": "北京"          │
│                }                        │
│              }                          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Agent 框架（外部代码） │
│  解析 JSON            │
│  执行 get_weather("北京") │
│  得到结果: "15°C, 晴"  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM 接收结果         │
│  生成最终回答：         │
│  "北京今天 15°C，晴天" │
└──────────────────────┘
```

---

## 2. Function Calling：行业标准

### 2.1 工具定义

以 OpenAI 的 Function Calling API 为例，工具通过 JSON Schema 定义：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

每个工具定义包含三个要素：
- **name**：函数名，LLM 用这个名字来"调用"工具
- **description**：功能描述，**这是 LLM 决定是否调用该工具的关键依据**
- **parameters**：参数的 JSON Schema，定义参数名、类型、是否必填

### 2.2 调用过程（代码视角）

```python
import openai

# 1. 定义工具
tools = [...]  # 如上

# 2. 发送请求，附带工具定义
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎样？"}],
    tools=tools
)

# 3. 检查 LLM 是否决定调用工具
message = response.choices[0].message

if message.tool_calls:
    # LLM 决定调用工具
    tool_call = message.tool_calls[0]
    print(tool_call.function.name)       # "get_weather"
    print(tool_call.function.arguments)  # '{"city": "北京"}'

    # 4. 执行实际的函数（这是你写的代码，不是 LLM）
    result = get_weather(city="北京")     # → "15°C, 晴"

    # 5. 把结果返回给 LLM
    messages.append(message)  # 先加上 LLM 的工具调用消息
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })

    # 6. LLM 根据结果生成最终回答
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
    print(final_response.choices[0].message.content)
    # "北京今天 15°C，天气晴朗。"
```

注意第 4 步：**执行函数的是你的代码，不是 LLM**。LLM 只负责"决定调用什么"和"理解返回结果"。

### 2.3 并行工具调用

LLM 可以在一次回复中请求**同时调用多个工具**。例如：

```
用户: "北京和上海今天天气怎样？"

LLM 输出两个工具调用：
  tool_calls[0]: get_weather(city="北京")
  tool_calls[1]: get_weather(city="上海")
```

Agent 框架可以并行执行这两个调用，节省等待时间。

> 参考：[OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)

---

## 3. 工具设计原则

### 3.1 好工具 vs 坏工具

工具设计的质量直接影响 Agent 的效果。核心原则：**让 LLM 容易理解、容易正确使用**。

| 原则 | 好的设计 | 坏的设计 |
|------|---------|---------|
| 名称清晰 | `search_web(query)` | `do_thing(x)` |
| 描述精确 | "搜索互联网，返回前 5 条结果的标题和摘要" | "搜索" |
| 参数简单 | `city: string` | `config: {nested: {deeply: {value}}}` |
| 功能单一 | 一个工具做一件事 | 一个工具做很多事，靠参数区分 |
| 错误信息有用 | `"城市 '北精' 不存在，你是否指 '北京'？"` | `"Error 404"` |

### 3.2 工具数量的权衡

工具越多，Agent 能做的事越多。但工具太多会导致：

- **LLM 选择困难**：面对 50 个工具，LLM 可能选错工具或生成错误的参数
- **上下文占用**：每个工具定义都要放进 prompt，占用上下文窗口空间
- **延迟增加**：更长的 prompt 意味着更慢的推理速度

经验法则：**大多数 Agent 使用 5-15 个工具效果最好**。如果需要更多工具，可以用"工具路由"——先让 LLM 选择工具类别，再在类别内选择具体工具。

### 3.3 工具描述的重要性

LLM 决定是否调用工具，主要依据 **description** 字段。一个反直觉的事实：**工具描述往往比工具代码更重要**——同样的功能，描述写得好和写得差，Agent 的成功率可能差 20-30%。

```
❌ 差的描述：
"description": "查询数据库"

✅ 好的描述：
"description": "在用户数据库中按用户名或邮箱搜索用户信息。
               返回用户的姓名、邮箱、注册日期。
               如果未找到匹配用户，返回空列表。"
```

好的描述要回答三个问题：**什么时候该用这个工具？输入是什么？输出是什么？**

---

## 4. 工具调用的底层原理

### 4.1 LLM 怎么"学会"调用工具？

早期的 LLM（如 GPT-3）不会生成结构化的工具调用。Function Calling 能力来自两个层面：

**训练层面**：模型在微调（fine-tuning）阶段用大量的"用户请求 → 工具调用 JSON → 工具结果 → 最终回答"样本训练，学会了在合适时机输出 JSON 格式的调用请求。这和 RLHF 对齐训练（参见 `domains/LLM/notes/07-finetuning-alignment.md`）是类似的过程——通过训练数据改变模型的输出行为。

**推理层面**：API 提供方（如 OpenAI）在底层对 LLM 的输出做了**受限解码**（constrained decoding）——当模型开始输出工具调用时，强制它只能输出符合 JSON Schema 的 token，避免格式错误。

### 4.2 Toolformer：让模型自己学会使用工具

**Toolformer**（Schick et al., 2023）走了另一条路：不是在推理时通过 prompt 告诉模型怎么用工具，而是**在训练时就让模型学会在文本中插入 API 调用**。

1. 在训练数据中，模型学会在合适的位置插入 `[Calculator(3 + 5) → 8]` 这样的标记
2. 推理时，模型自然地在生成文本过程中插入工具调用

Toolformer 是一种研究方法，目前主流的商业 API 都用 Function Calling 方式（不修改模型权重），但 Toolformer 的思路影响了后续的工具训练研究。

> 可靠程度：Level 2-3
>
> 参考：[Schick et al., 2023 - Toolformer](https://arxiv.org/abs/2302.04761)

---

## 5. MCP：模型上下文协议

### 5.1 问题背景

Function Calling 解决了"LLM 怎么调用工具"的问题，但带来了一个新问题：**每个 LLM 提供商的工具调用格式都不一样**。

- OpenAI 用 `tools` 参数和 `tool_calls` 响应
- Anthropic 用 `tools` 参数和 `tool_use` 内容块
- Google 用 `function_declarations` 和 `function_call`

如果你想让同一个工具在不同的 LLM 上工作，要为每家写不同的适配代码。

### 5.2 MCP 是什么

**MCP**（Model Context Protocol，模型上下文协议）是 Anthropic 提出的一个**开放标准**，目标是统一 LLM 与外部工具/数据源之间的通信协议。

类比：MCP 之于 Agent，就像 USB 之于电脑外设——有了统一标准，任何工具只要实现 MCP 接口，就能被任何支持 MCP 的 Agent 使用。

MCP 的核心概念：

- **MCP Server**（服务端）：提供工具的一方。比如一个天气服务、一个数据库查询服务、一个文件操作服务
- **MCP Client**（客户端）：调用工具的一方。通常是 Agent 框架或 IDE（如 Cursor 本身就是一个 MCP Client）
- **Transport**（传输层）：Server 和 Client 之间的通信方式（标准输入/输出，或 HTTP SSE）

```
Agent / IDE（MCP Client）
  │
  ├──→ 天气服务（MCP Server）
  │      提供: get_weather, get_forecast
  │
  ├──→ 数据库服务（MCP Server）
  │      提供: query_db, insert_record
  │
  └──→ 文件服务（MCP Server）
         提供: read_file, write_file, search_files
```

你正在使用的 Cursor 就支持 MCP——你可以配置 MCP Server，让 Cursor 的 Agent 调用自定义工具。

### 5.3 MCP vs Function Calling

| 对比 | Function Calling | MCP |
|------|-----------------|-----|
| 本质 | LLM API 的一个功能 | 工具与 Agent 之间的通信协议 |
| 范围 | 单次 API 调用中的工具定义 | 整个工具生态的标准化 |
| 工具发现 | 开发者手动写工具定义 | Server 自动暴露可用工具列表 |
| 可复用性 | 每个项目重写 | 一个 MCP Server 可被多个 Client 复用 |
| 状态 | 行业标准（各家各异） | 新兴标准（快速普及中） |

简而言之：Function Calling 是"LLM 怎么表达工具调用"，MCP 是"工具怎么被发现和连接"。两者不冲突——MCP Server 最终还是通过 Function Calling 机制被 LLM 调用。

> 可靠程度：Level 3（MCP 是 2024 年底发布的新标准，正在快速被采用，但生态尚在发展中）
>
> 参考：[Anthropic MCP 文档](https://modelcontextprotocol.io/introduction) · [MCP GitHub](https://github.com/modelcontextprotocol)

---

### 公式速查卡

本章无数学公式，核心是工程流程：

| 概念 | 含义 |
|------|------|
| Function Calling | LLM 生成 `{name, arguments}` JSON → 外部执行 → 结果返回 LLM |
| JSON Schema | 描述工具参数的格式标准（类型、必填、枚举值等） |
| 受限解码 | 强制 LLM 只输出符合特定格式的 token，避免 JSON 格式错误 |
| Toolformer | 训练时让模型学会在文本中自然插入 API 调用 |
| MCP | Model Context Protocol，统一工具与 Agent 的通信标准 |
| 工具路由 | 工具过多时，先选类别再选具体工具的分层策略 |

---

## 理解检测

**Q1**：在 Function Calling 中，LLM 的角色是什么？实际执行工具的是谁？为什么这样设计（而不是让 LLM 直接执行代码）？至少给出两个原因。

你的回答：


**Q2**：你正在设计一个 Agent 的工具。以下两个工具描述，哪个更好？为什么？

描述 A："操作数据库"
描述 B："在 PostgreSQL 数据库中执行 SQL 查询。输入为合法的 SQL 语句（SELECT/INSERT/UPDATE），返回查询结果或受影响的行数。不支持 DROP/ALTER 等危险操作。"

你的回答：


**Q3**：MCP 解决了什么问题？如果没有 MCP，一个工具想同时支持 OpenAI、Anthropic 和 Google 的 Agent，需要怎么做？有了 MCP 之后呢？

你的回答：


**Q4**：一个 Agent 有 12 个工具，每个工具定义平均占 150 token（名称 + 描述 + 参数 schema）。Agent 的 system prompt 占 500 token，上下文窗口总共 8192 token。工具定义总共占多少 token？留给对话历史和 LLM 回复的空间还有多少 token？

> 提示：工具总 token = 工具数 × 每个工具的 token 数

你的回答：

