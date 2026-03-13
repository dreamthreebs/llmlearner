# 07 - 工程实现：从概念到代码

> **主维度**：D3 工程实现
> **关键关系**：
> - LangChain (实验) --用于--> Agent (架构)：LangChain 用于构建 Agent
> - LangGraph (实验) --推广了--> LangChain (实验)：LangGraph 推广了 LangChain，支持图结构工作流
> - Function Calling (方法) --用于--> Agent (架构)：Function Calling 用于 Agent 的工具调用
>
> **学习路径**：全景概览 → ... → 架构模式 → **本章（工程实现）** → 评估与调试
>
> **前置知识**：
> - Agent 核心组件和架构模式（参见 `01-06` 章）
> - Python 基础和 API 调用经验
> - Function Calling（参见 `03-tool-use.md`）
>
> **参考**：
> - [LangChain 文档](https://python.langchain.com/docs/)
> - [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
> - [OpenAI Cookbook - How to build an agent](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

---

## 核心问题

**理解了 Agent 的设计原理，怎么用代码把它搭出来？**

前面六章讲的是"设计图"，本章讲"怎么施工"。我们从最底层（直接用 API）到最上层（用框架），逐步构建一个完整的 Agent。

> 可靠程度：Level 1-3（API 层面是行业标准；框架层面变化较快，以当前主流为准）

---

## 1. 最简 Agent：纯 API 实现

### 1.1 不用任何框架的 ReAct Agent

理解 Agent 最好的方式是**从零搭一个**。下面是一个最简化的 ReAct Agent，只用 OpenAI API：

```python
import openai
import json

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"}
                },
                "required": ["expression"]
            }
        }
    }
]

# 工具的实际实现
def execute_tool(name, arguments):
    if name == "search":
        # 实际项目中调用搜索 API
        return f"搜索 '{arguments['query']}' 的结果：..."
    elif name == "calculator":
        return str(eval(arguments["expression"]))

# Agent 主循环
def run_agent(user_message, max_turns=10):
    messages = [
        {"role": "system", "content": "你是一个有用的助手。需要时使用工具获取信息或计算。"},
        {"role": "user", "content": user_message}
    ]

    for turn in range(max_turns):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message
        messages.append(message)

        # 如果 LLM 没有调用工具，说明任务完成
        if not message.tool_calls:
            return message.content

        # 执行所有工具调用
        for tool_call in message.tool_calls:
            result = execute_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "达到最大轮次限制"
```

这段代码就是一个完整的 Agent。**核心只有一个 for 循环**：

1. 把消息发给 LLM
2. 如果 LLM 要调用工具 → 执行工具 → 把结果加入消息列表 → 回到 1
3. 如果 LLM 直接回复文本 → 返回结果

### 1.2 关键设计点

**消息列表（messages）就是短期记忆**。每一轮的 Thought（LLM 的文本输出）、Action（tool_calls）、Observation（tool 结果）都被追加到 messages 中，LLM 在下一轮能看到所有历史。

**max_turns 是安全阀**。防止 Agent 陷入无限循环——如果 LLM 反复调用工具但无法得出结论，到达上限后强制停止。

**tool_call_id 是匹配标识**。当 LLM 同时调用多个工具时，每个结果需要通过 id 对应到正确的调用请求。

---

## 2. 从简单到实用：需要补什么

上面的最简 Agent 能跑，但离实用还差很多。一个生产级 Agent 需要处理的额外问题：

| 问题 | 最简实现 | 生产级需要 |
|------|---------|-----------|
| 错误处理 | 工具报错直接崩溃 | 捕获异常，告诉 LLM 工具执行失败 |
| 上下文管理 | messages 无限增长 | 摘要压缩或滑动窗口 |
| 流式输出 | 等 LLM 完全生成后才返回 | 逐 token 流式输出，用户体验更好 |
| 成本控制 | 不限制 token 使用 | 设置 token 上限，监控 API 费用 |
| 日志追踪 | 无 | 记录每一步的输入、输出、耗时 |
| 并发 | 工具串行执行 | 并行工具调用支持异步执行 |
| 安全 | 任何工具调用都直接执行 | 高风险操作前请求确认 |

### 2.1 错误处理示例

```python
def execute_tool(name, arguments):
    try:
        if name == "search":
            return search_api(arguments["query"])
        elif name == "calculator":
            return str(eval(arguments["expression"]))
    except Exception as e:
        # 不要让 Agent 崩溃，而是告诉 LLM 工具调用失败了
        return f"工具执行失败: {str(e)}。请尝试其他方法。"
```

把错误信息返回给 LLM，让它根据错误调整策略——这本质上就是 ReAct 中的 Observation 反馈。

---

## 3. LangChain：Agent 框架

### 3.1 LangChain 是什么

**LangChain** 是目前最流行的 Python Agent 框架。它把上面那些"从简单到实用"的问题都封装好了：

- 统一的 LLM 接口（支持 OpenAI、Anthropic、本地模型等）
- 工具定义和执行的标准化
- 上下文管理和记忆
- 链式调用（Chains）和 Agent 循环
- 丰富的第三方工具集成

### 3.2 核心概念

**LLM / ChatModel**：封装了各家 LLM API 的统一接口。

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

**Tool**：工具的标准化定义。

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息。"""
    return search_api(query)

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    return str(eval(expression))
```

LangChain 的 `@tool` 装饰器自动从函数签名和 docstring 生成工具定义（名称、描述、参数格式），省去了手写 JSON Schema 的工作。

**Agent**：把 LLM 和工具组合成一个可执行的 Agent。

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search, calculator]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "2024 年诺贝尔物理学奖得主是谁？"}]
})
```

`create_react_agent` 内部实现的就是第 1 节那个 for 循环——但加上了错误处理、流式输出、日志等功能。

### 3.3 什么时候用 LangChain，什么时候自己写

| 场景 | 建议 |
|------|------|
| 学习 Agent 原理 | 自己写（理解底层机制） |
| 快速原型验证 | LangChain（几行代码跑起来） |
| 简单 Agent（1-2 个工具） | 自己写（避免框架开销） |
| 复杂 Agent（多工具、多步骤） | LangChain（省去大量重复代码） |
| 需要深度定制 | 自己写或 LangGraph |

---

## 4. LangGraph：图结构 Agent

### 4.1 为什么需要 LangGraph

LangChain 的基本 Agent（ReAct 循环）是**线性的**——一步接一步。但真实的 Agent 工作流通常是**图结构**：

- 某些步骤可以并行（同时搜索多个来源）
- 某些步骤有条件分支（如果 A 成功走路径 1，失败走路径 2）
- 某些步骤需要循环（反复修改直到通过测试）
- 需要持久化状态（记住之前的决策）

**LangGraph** 是 LangChain 团队的图结构 Agent 框架，用**有向图**来定义 Agent 的工作流。

### 4.2 核心概念

**State**（状态）：在节点之间传递的数据结构。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    plan: list[str]                          # 当前计划
    current_step: int                        # 执行到第几步
```

**Node**（节点）：图中的每个处理步骤，是一个函数。

```python
def planner(state: AgentState) -> AgentState:
    """规划器：生成执行计划"""
    plan = llm.invoke("请为以下任务制定计划：" + state["messages"][-1].content)
    return {"plan": plan.content.split("\n")}

def executor(state: AgentState) -> AgentState:
    """执行器：执行当前步骤"""
    step = state["plan"][state["current_step"]]
    result = agent.invoke(step)
    return {"current_step": state["current_step"] + 1}
```

**Edge**（边）：节点之间的连接，可以是无条件的或有条件的。

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# 添加节点
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reviewer", reviewer)

# 添加边
graph.add_edge("planner", "executor")

# 条件边：执行完后，看是否还有剩余步骤
def should_continue(state):
    if state["current_step"] < len(state["plan"]):
        return "executor"  # 继续执行
    return "reviewer"      # 全部执行完，交给审查

graph.add_conditional_edges("executor", should_continue)
graph.add_edge("reviewer", END)

# 编译并运行
app = graph.compile()
result = app.invoke({"messages": [user_message]})
```

### 4.3 一个 Plan-and-Execute Agent 的图结构

```
START
  │
  ▼
[Planner] ──→ [Executor] ──→ 还有步骤？──→ [Executor]（循环）
                                │
                                ▼ 没有了
                            [Reviewer]
                                │
                          ┌─────┼─────┐
                          │           │
                        通过 ✅     不通过 ❌
                          │           │
                          ▼           ▼
                        [END]    [Planner]（重新规划）
```

这就是 `05-planning.md` 中讲的 Plan-and-Execute + 动态重规划，用 LangGraph 的图结构表达出来。

> 参考：[LangGraph 文档](https://langchain-ai.github.io/langgraph/)

---

## 5. Prompt 工程实践

### 5.1 System Prompt 的结构

一个好的 Agent system prompt 通常包含以下部分：

```
1. 角色定义：你是谁，你的专长是什么
2. 行为规则：什么该做，什么不该做
3. 工具说明：可用工具列表（通常由框架自动注入）
4. 输出格式：期望的回答格式
5. 示例（可选）：1-2 个输入-输出示例
```

### 5.2 常见的 Prompt 技巧

**强制思考**：要求 LLM 在行动前先分析。

```
"在调用任何工具之前，先用 1-2 句话分析为什么选择这个工具。"
```

**限制行为**：明确告诉 LLM 什么不该做。

```
"如果用户的请求涉及删除文件或修改系统配置，必须先确认。
 绝不要编造不确定的信息，如果不知道就说不知道。"
```

**结构化输出**：当需要特定格式的输出时。

```
"请按以下 JSON 格式输出分析结果：
 {"summary": "...", "confidence": 0-1, "sources": [...]}"
```

### 5.3 Prompt 的迭代方法

写 Agent prompt 是一个**实验性过程**，没有一次成功的银弹。推荐的迭代方法：

1. 写一个最简版本的 prompt
2. 用 5-10 个测试用例运行
3. 分析失败案例：Agent 是在哪一步出了问题？
4. 针对性修改 prompt（加规则、加示例、换措辞）
5. 重新测试，确认修改有效且没有引入新问题

---

### 公式速查卡

| 概念 | 含义 |
|------|------|
| Agent 主循环 | `while True: response = LLM(messages); if no tool_calls: return; execute tools; append results` |
| messages 列表 | Agent 的短期记忆，按时间顺序记录所有 user/assistant/tool 消息 |
| max_turns | 安全阀，限制 Agent 最大循环次数，防止无限循环 |
| `@tool` 装饰器 | LangChain 中定义工具的方式，自动从函数签名生成 JSON Schema |
| StateGraph | LangGraph 的核心，用有向图定义 Agent 工作流 |
| 条件边 | 根据当前状态决定下一步走哪个节点 |

---

## 理解检测

**Q1**：看第 1 节的最简 Agent 代码。如果 LLM 在第一轮同时调用了 `search` 和 `calculator` 两个工具，Agent 主循环会怎么处理？messages 列表里会依次追加哪些消息？

你的回答：


**Q2**：LangGraph 的 StateGraph 用"节点"和"边"来定义 Agent 工作流。如果你要实现一个 Reflexion Agent（执行→测试→如果失败则反思并重试，最多重试 3 次），需要哪些节点？条件边的判断逻辑是什么？画出图结构（文字描述即可）。

你的回答：


**Q3**：在最简 Agent 中，`execute_tool` 函数直接用 `eval()` 执行用户传入的数学表达式。这有什么安全风险？如果用户输入 `calculator("__import__('os').system('rm -rf /')")`，会发生什么？你会怎么修复？

你的回答：


**Q4**：一个 Agent 设置 `max_turns=10`，每轮 LLM 调用平均消耗 800 token（输入 + 输出），API 价格是 $0.01 / 1K token。如果 Agent 在第 6 轮完成任务，总 token 消耗是多少？总成本是多少？如果跑满 10 轮呢？

> 提示：总 token = 轮数 × 每轮 token；总成本 = 总 token / 1000 × 单价

你的回答：

