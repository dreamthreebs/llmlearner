# 01 - AI Agent 全景概览

> **主维度**：全部（D1-D4）
> **关键关系**：
> - Agent (架构) --依赖--> LLM (架构)：Agent 依赖 LLM 作为推理引擎
> - 工具调用 (方法) --属于--> Agent (架构)：工具调用属于 Agent 的核心组件
> - ReAct (方法) --用于--> Agent (架构)：ReAct 用于 Agent 的推理-行动循环
>
> **学习路径**：**本章（全景概览）** → 提示与推理 → 工具调用 → 记忆 → 规划 → 架构模式 → 工程实现 → 评估 → 应用
>
> **前置知识**：LLM 的基本概念（Transformer、预训练、对齐，参见 `domains/LLM/`）
>
> **参考**：
> - [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
> - [Yao et al., 2022 - ReAct](https://arxiv.org/abs/2210.03629)
> - [Wang et al., 2023 - A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432)
> - [Wikipedia - Intelligent agent](https://en.wikipedia.org/wiki/Intelligent_agent)

---

## 1. Agent 是什么？

**可靠程度：Level 1-2**

### 1.1 从 LLM 到 Agent

你已经知道 LLM（如 GPT-4、Claude）的核心能力：给一段文本，生成后续文本。经过对齐训练后，它能理解指令、回答问题、写代码。但纯 LLM 有一个根本限制：**它只能生成文本，不能行动**。

- 你问 LLM "今天天气怎样"，它只能根据训练数据猜测，不能实际查天气
- 你让 LLM "帮我订机票"，它只能生成一段订票的流程描述，不能真的去操作订票系统
- 你让 LLM "分析这份数据"，它只能假设数据内容，不能真的读取文件和运行代码

**AI Agent** 解决了这个问题。Agent 以 LLM 为核心，但增加了与外部世界交互的能力：

$$\text{Agent} = \text{LLM（推理引擎）} + \text{规划} + \text{记忆} + \text{工具调用}$$

| 能力 | 纯 LLM | Agent |
|------|--------|-------|
| 理解自然语言 | ✅ | ✅ |
| 推理和分析 | ✅ | ✅ |
| 搜索互联网 | ❌ | ✅（通过搜索工具） |
| 读写文件 | ❌ | ✅（通过文件工具） |
| 执行代码 | ❌ | ✅（通过代码解释器） |
| 调用 API | ❌ | ✅（通过 Function Calling） |
| 多步骤任务 | ❌（单轮生成） | ✅（规划 + 循环执行） |
| 从错误中恢复 | ❌ | ✅（观察结果 → 调整策略） |

### 1.2 Agent 的工作流程

一个典型的 Agent 执行任务的过程：

```
用户指令："帮我找到最近一周 arXiv 上关于 AI Agent 的论文，总结前三篇。"

Agent 的内部过程：
1. [规划] 把任务拆成子步骤：
   - Step 1: 搜索 arXiv
   - Step 2: 筛选最相关的 3 篇
   - Step 3: 阅读并总结

2. [工具调用] 调用搜索 API，搜索 "AI Agent site:arxiv.org"
   → 得到一批论文链接

3. [推理] 分析搜索结果，选出最相关的 3 篇

4. [工具调用] 读取这 3 篇论文的摘要（调用网页读取工具）

5. [推理] 根据摘要内容生成总结

6. [输出] 返回总结给用户
```

关键观察：Agent 不是一次性生成答案，而是在一个**循环**中交替进行"思考"和"行动"，根据每一步的结果决定下一步怎么做。

### 1.3 Agent 不是新概念

"Agent"这个词在 AI 中有很长的历史。在经典 AI 中（Russell & Norvig 的教科书），**智能体**（intelligent agent）的定义是：

> 一个能感知环境（通过传感器）并对环境采取行动（通过执行器）的系统。

经典 Agent 需要手工编写规则（如果看到 X，就做 Y）。LLM-based Agent 的突破在于：**用 LLM 替代手工规则**——LLM 的语言理解和推理能力让 Agent 可以处理开放域的、自然语言描述的任务，而不需要为每种情况预先编程。

> 参考：[Russell & Norvig,《Artificial Intelligence: A Modern Approach》Ch.2](https://aima.cs.berkeley.edu/) · [Wikipedia - Intelligent agent](https://en.wikipedia.org/wiki/Intelligent_agent)

---

## 2. Agent 的四个核心组件

**可靠程度：Level 2（业界广泛采用的分类框架，但不是唯一的分类方式）**

### 2.1 推理引擎（LLM）

LLM 是 Agent 的"大脑"。它负责：
- 理解用户的指令
- 分析当前状态和历史信息
- 决定下一步做什么
- 生成最终的输出

Agent 的能力上限很大程度上取决于底层 LLM 的能力。用 GPT-4 级别的模型做 Agent 和用一个小模型做 Agent，效果差距巨大。

### 2.2 规划（Planning）

**规划** 是把复杂任务拆解为可执行的子步骤。人类做复杂任务时也是这样——你不会直接"写一篇论文"，而是拆成"选题 → 文献调研 → 拟大纲 → 写初稿 → 修改"。

Agent 中常见的规划方式：

- **无显式规划（ReAct 风格）**：每一步只决定当前做什么，不做全局规划。简单但容易在长任务中迷失方向。
- **预先规划（Plan-and-Execute）**：先生成完整计划，再逐步执行。适合结构清晰的任务。
- **迭代规划**：边执行边调整计划。最灵活，但最复杂。

### 2.3 记忆（Memory）

纯 LLM 的"记忆"只有一个：**上下文窗口**（context window）。对话历史超过窗口大小就被遗忘了。Agent 需要更强的记忆系统：

- **短期记忆**：当前任务的上下文（对话历史、中间结果）。通常就是 LLM 的上下文窗口，加上一些摘要压缩策略。
- **长期记忆**：跨任务持久化的知识。通常用**向量数据库**存储，通过 **RAG**（Retrieval-Augmented Generation，检索增强生成）在需要时检索相关信息。

RAG 的基本流程：把文档切成小块 → 用 embedding 模型（即词嵌入，参见 `domains/LLM/notes/03-word-embedding.md`）转成向量 → 存入向量数据库（专门存储和检索高维向量的数据库） → 用户提问时，先检索最相关的文档块 → 把检索到的内容放进 LLM 的上下文中一起回答。

### 2.4 工具调用（Tool Use）

**工具调用** 是 Agent 最核心的能力——它让 LLM 从"只能说"变成"能做"。

工具就是 Agent 可以调用的外部函数。典型的工具包括：
- **搜索引擎**：获取实时信息
- **代码解释器**：执行 Python 代码、做计算
- **文件操作**：读写本地文件
- **API 调用**：和外部服务交互（数据库、邮件、日历……）
- **浏览器**：访问网页、填写表单

实现方式是 **Function Calling**：LLM 不直接执行工具，而是**生成一个结构化的调用请求**（函数名 + 参数），由外部系统执行后把结果返回给 LLM。

```
用户: "北京今天多少度？"

LLM 输出（不是文本，而是函数调用）:
{
  "function": "get_weather",
  "arguments": {"city": "北京"}
}

外部系统执行 get_weather("北京") → 返回 "15°C, 晴"

LLM 收到结果后生成最终回答: "北京今天 15°C，晴天。"
```

> 参考：[Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) · [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

## 3. 核心推理模式

**可靠程度：Level 1-2**

### 3.1 Chain-of-Thought (CoT)

**链式思维**（Chain-of-Thought，Wei et al. 2022）是 Agent 推理的基础。核心思想：让 LLM 在给出答案之前，先一步步写出推理过程。

不用 CoT：
```
Q: 一个农场有 23 头牛，又买了 15 头，卖了 8 头，现在多少头？
A: 30
```

用 CoT：
```
Q: 一个农场有 23 头牛，又买了 15 头，卖了 8 头，现在多少头？
A: 农场开始有 23 头牛。买了 15 头后有 23 + 15 = 38 头。卖了 8 头后有 38 - 8 = 30 头。答案是 30。
```

看起来只是多写了几行字，但 CoT 显著提升了 LLM 在复杂推理任务上的准确率。原因是 LLM 的 Transformer 架构在每个 token 位置只能做有限的计算——把中间步骤写出来，等于给了模型更多的"计算空间"。

> 参考：[Wei et al., 2022 - Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

### 3.2 ReAct：推理 + 行动

**ReAct**（Reasoning + Acting，Yao et al. 2022）是 Agent 最重要的推理框架之一。它把 CoT 推理和工具调用结合在一个循环中：

```
循环：
  1. Thought（思考）：LLM 分析当前状态，决定下一步做什么
  2. Action（行动）：调用某个工具
  3. Observation（观察）：接收工具返回的结果
  → 回到 1，直到任务完成
```

一个 ReAct 的实际例子：

```
问题：谁是《哈利·波特》作者的配偶？

Thought 1: 我需要先找到《哈利·波特》的作者。
Action 1: search("Harry Potter author")
Observation 1: J.K. Rowling

Thought 2: 现在我知道作者是 J.K. Rowling，需要找她的配偶。
Action 2: search("J.K. Rowling spouse")
Observation 2: Neil Murray (married 2001)

Thought 3: 我找到了答案。
Action 3: finish("J.K. Rowling 的配偶是 Neil Murray。")
```

ReAct 的优势：
- **可解释**：每一步的推理过程是透明的
- **可纠错**：如果某一步的观察结果不符合预期，Agent 可以调整策略
- **通用**：适用于各种需要外部信息或行动的任务

ReAct 的局限：
- **没有全局规划**：每一步只看当前状态，可能在复杂任务中迷失方向
- **容易卡在循环中**：如果推理出错，可能反复尝试同一个失败的策略

> 参考：[Yao et al., 2022 - ReAct](https://arxiv.org/abs/2210.03629)

---

## 4. 知识维度与知识地图

### 4.1 知识维度

| 维度 | 含义 | 核心概念 |
|------|------|---------|
| **D1 核心组件** | Agent 的 building blocks | CoT、ReAct、Function Calling、RAG、向量数据库、任务分解 |
| **D2 架构模式** | 不同的 Agent 设计范式 | ReAct 循环、Plan-and-Execute、Reflexion、Multi-Agent、人机协作 |
| **D3 工程实现** | 框架与实战 | LangChain、LangGraph、Prompt 设计、错误处理、评估 |
| **D4 应用与前沿** | 场景与局限 | 代码 Agent、研究 Agent、安全性、Scaling Agent 能力 |

### 4.2 知识地图

```
LLM 基础（已学）
    │
    ▼
提示与推理（CoT → ReAct）
    │
    ├──→ 工具调用（Function Calling → API 设计）
    │
    ├──→ 记忆系统（上下文管理 → RAG → 向量数据库）
    │
    └──→ 规划（任务分解 → Plan-and-Execute → 自我反思）
              │
              ▼
         架构模式（ReAct → Multi-Agent → 人机协作）
              │
              ▼
         工程实现（LangChain/LangGraph → 完整 Agent → 评估）
              │
              ▼
         应用与前沿（代码/研究 Agent → 局限 → 安全）
```

---

## 5. 前沿方向

**可靠程度：Level 3-4**

| 方向 | 说明 | 可靠度 |
|------|------|--------|
| **Agent 能力的 Scaling** | 更强的 LLM 是否自动带来更强的 Agent？初步证据支持但不完全确定 | Level 3 |
| **长期自主运行** | 让 Agent 持续运行数小时甚至数天完成复杂项目，目前可靠性不够 | Level 4 |
| **Multi-Agent 协作** | 多个 Agent 分工协作（如一个写代码、一个做 review），有前景但协调机制不成熟 | Level 3 |
| **Agent 安全** | Agent 能执行操作（删文件、发邮件），如何确保它不做有害操作？核心未解问题 | Level 3-4 |
| **模型内置 Agent 能力** | 训练时就让模型学会使用工具和规划，而不是纯靠提示词。目前是活跃研究方向 | Level 3 |

> 参考：[Wang et al., 2023 - A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432)

---

### 公式速查卡

本章无核心公式，但有几个关键框架：

| 框架 | 含义 |
|------|------|
| Agent = LLM + 规划 + 记忆 + 工具调用 | Agent 的组成定义 |
| CoT：问题 → 推理步骤 → 答案 | 链式思维，提升推理准确率 |
| ReAct：Thought → Action → Observation → 循环 | 推理 + 行动交替循环 |
| Function Calling：LLM 输出 `{function, arguments}` → 外部执行 → 返回结果 | 工具调用的标准流程 |
| RAG：query → 检索相关文档 → 放入上下文 → LLM 生成 | 检索增强生成 |

---

## 理解检测

**Q1**：纯 LLM 和 Agent 的核心区别是什么？如果用户问"帮我查一下苹果公司今天的股价"，纯 LLM 和 Agent 分别会怎么处理？

你的回答：


**Q2**：ReAct 框架的三个步骤是什么？假设用户问"比较 Python 和 Rust 的运行速度"，请写出一个可能的 ReAct 执行过程（至少 2 轮 Thought-Action-Observation）。

你的回答：


**Q3**：Agent 的四个核心组件中，"记忆"解决了 LLM 的什么根本局限？短期记忆和长期记忆分别用什么技术实现？

你的回答：


**Q4**：一个 Agent 执行任务时经历了 4 轮 ReAct 循环（每轮 = 1 次 Thought + 1 次 Action + 1 次 Observation），其中每次 Thought 和 Observation 各消耗约 200 token。Agent 的上下文窗口是 8192 token，system prompt 占 800 token。4 轮循环后，对话历史大约消耗了多少 token？还剩多少 token 给 LLM 生成下一步回复？

> 提示：每轮的 token ≈ Thought token + Action token（很短，约 50）+ Observation token

你的回答：

