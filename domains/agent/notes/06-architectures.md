# 06 - 架构模式：不同的 Agent 设计范式

> **主维度**：D2 架构模式
> **关键关系**：
> - ReAct (方法) --是一种--> Agent (架构)：ReAct 是一种 Agent 架构
> - Multi-Agent (架构) --推广了--> Agent (架构)：Multi-Agent 推广了单 Agent 到多 Agent 协作
> - Plan-and-Execute (方法) --是一种--> Agent (架构)：Plan-and-Execute 是一种 Agent 架构
>
> **学习路径**：全景概览 → 提示与推理 → 工具调用 → 记忆 → 规划 → **本章（架构模式）** → 工程实现
>
> **前置知识**：
> - ReAct 框架（参见 `02-prompting-reasoning.md`）
> - Plan-and-Execute 与 Reflexion（参见 `05-planning.md`）
> - 工具调用（参见 `03-tool-use.md`）
>
> **参考**：
> - [Andrew Ng - AI Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)
> - [Wu et al., 2023 - AutoGen](https://arxiv.org/abs/2308.08155)
> - [Zhou et al., 2023 - LATS](https://arxiv.org/abs/2310.04406)
> - [Significant Gravitas - AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

---

## 核心问题

**前面学了组件（推理、工具、记忆、规划），怎么把它们组装成不同的 Agent 架构？**

就像搭积木——同样的积木可以搭出不同的结构。不同的架构适合不同的任务场景。本章从简单到复杂，介绍四种主要的 Agent 架构模式。

> 可靠程度：Level 2-3（架构分类来自 Andrew Ng 等人的总结，业界广泛接受但不是唯一分类方式）

---

## 1. 四种核心设计模式

Andrew Ng 在 2024 年总结了四种 Agentic Design Patterns，这是目前最被广泛引用的分类框架：

| 模式 | 核心思想 | 适合场景 |
|------|---------|---------|
| **Reflection** | Agent 审查并改进自己的输出 | 代码生成、文本写作 |
| **Tool Use** | Agent 调用外部工具获取信息或执行操作 | 需要外部交互的任务 |
| **Planning** | Agent 生成并执行多步计划 | 复杂、长链任务 |
| **Multi-Agent** | 多个 Agent 分工协作 | 需要多种专业能力的任务 |

前三种在之前的章节已经讲过（Reflection 对应 Reflexion，Tool Use 对应工具调用，Planning 对应 Plan-and-Execute）。本章重点讲 Multi-Agent，以及各模式的组合方式。

---

## 2. 单 Agent 架构的演化

### 2.1 从 ReAct 到 LATS

前面的章节按时间顺序讲了单 Agent 的演化：

```
ReAct（基础循环）
  │
  ├──→ Plan-and-Execute（加入全局规划）
  │
  ├──→ Reflexion（加入自我反思）
  │
  └──→ LATS（加入树状搜索）
```

**LATS**（Language Agent Tree Search，Zhou et al., 2023）结合了 Tree of Thoughts（`02-prompting-reasoning.md` 中的 ToT）和 ReAct：

1. 在每一步生成多个候选行动（而不是只选一个）
2. 对每个候选评估价值（LLM 打分）
3. 用蒙特卡洛树搜索（MCTS）策略选择最优路径
4. 如果某条路径失败，回退到之前的状态重新选择

**蒙特卡洛树搜索（MCTS）** 是一种决策算法，最早用于围棋程序（如 AlphaGo）。核心思路：通过多次模拟来估计每个决策的好坏，优先探索看起来最有前途的方向，同时保留一定的探索随机性以发现意外的好路径。

LATS 在 HumanEval（代码生成）等基准上超过了 ReAct 和 Reflexion，但代价是**LLM 调用次数成倍增加**——每一步多个候选，每个候选需要评估，失败还要回退重试。

> 参考：[Zhou et al., 2023 - LATS](https://arxiv.org/abs/2310.04406)

### 2.2 单 Agent 架构的选择

| 架构 | LLM 调用次数 | 适合场景 | 缺点 |
|------|-------------|---------|------|
| ReAct | 低（每步 1 次） | 简单任务，快速原型 | 长任务容易迷失 |
| Plan-and-Execute | 中（规划 + 执行） | 结构化的长任务 | 计划可能不准确 |
| Reflexion | 中-高（含重试） | 需要迭代改进的任务 | 需要明确的评估标准 |
| LATS | 高（树搜索） | 需要极高准确率 | 成本高，延迟大 |

实际工程中的经验法则：**先用 ReAct 看效果，不够再加 Planning，还不够再加 Reflection**。不要一上来就用最复杂的架构——复杂度带来的调试和成本问题可能比它解决的问题更大。

---

## 3. Multi-Agent：多 Agent 协作

### 3.1 为什么需要多个 Agent

单 Agent 面临一个矛盾：**system prompt 不能无限长**。一个 Agent 如果要处理复杂任务（比如"开发一个 Web 应用"），它的 system prompt 需要包含：

- 代码编写的规范和最佳实践
- 测试的方法论
- 代码审查的标准
- 项目管理的流程
- ……

把所有这些塞进一个 system prompt，LLM 的注意力会被分散——它可能在写代码时忽略了测试规范，在做 review 时忘了项目管理要求。

**Multi-Agent 的思路：让每个 Agent 专注一件事**。就像一个团队——程序员写代码、测试员测试、经理协调——每个人有自己的专业领域。

### 3.2 Multi-Agent 的基本结构

```
用户任务
  │
  ▼
┌─────────────┐
│ 协调者       │  分配子任务、汇总结果
│ (Orchestrator)│
└─────┬───────┘
      │
   ┌──┼──────────┐
   │  │          │
   ▼  ▼          ▼
┌────┐ ┌─────┐ ┌────────┐
│程序员│ │测试员│ │代码审查 │
│Agent│ │Agent│ │Agent   │
└────┘ └─────┘ └────────┘
```

每个 Agent 有自己的 system prompt、工具集和专业领域。它们通过**消息传递**（message passing）进行交互——一个 Agent 的输出成为另一个 Agent 的输入。

### 3.3 Multi-Agent 的常见模式

**对话模式**：Agent 之间像人类一样对话。

```
程序员 Agent: "我写了这段代码来处理用户登录。"
审查 Agent: "这段代码没有做输入验证，可能有 SQL 注入风险。"
程序员 Agent: "好的，我加上参数化查询。"
审查 Agent: "现在没问题了。"
```

**流水线模式**：每个 Agent 负责一个阶段，输出传给下一个。

```
需求分析 Agent → 代码生成 Agent → 测试 Agent → 部署 Agent
```

**竞争模式**：多个 Agent 独立完成同一任务，选最好的结果。

```
Agent A 写方案 1  ─┐
Agent B 写方案 2  ─┼→ 评审 Agent 选最优方案
Agent C 写方案 3  ─┘
```

### 3.4 AutoGen

**AutoGen**（Wu et al., 2023）是微软开源的 Multi-Agent 框架，核心概念：

- **ConversableAgent**：可以相互对话的 Agent 基类
- **AssistantAgent**：基于 LLM 的 Agent，可以写代码、分析问题
- **UserProxyAgent**：代表用户的 Agent，可以执行代码、请求人类输入

一个典型的 AutoGen 配置：

```python
assistant = AssistantAgent(
    name="assistant",
    system_message="你是一个 Python 程序员。"
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"  # 全自动，不请求人类输入
)

# 启动对话
user_proxy.initiate_chat(
    assistant,
    message="写一个函数计算斐波那契数列的第 n 项。"
)

# 对话过程：
# user_proxy → assistant: "写一个函数..."
# assistant → user_proxy: "这是代码: def fib(n): ..."
# user_proxy 自动执行代码，返回结果
# assistant 看到结果，确认正确或修改
```

> 参考：[Wu et al., 2023 - AutoGen](https://arxiv.org/abs/2308.08155)

---

## 4. 人机协作（Human-in-the-Loop）

### 4.1 为什么需要人类参与

完全自主的 Agent 有一个根本问题：**错误会级联放大**。如果 Agent 在第 3 步做了一个错误决策，后面的所有步骤都建立在错误基础上——最终的结果可能完全不可用。

**Human-in-the-Loop**（人机协作）在关键节点引入人类判断：

```
Agent 执行计划
  │
  ├── Step 1: 自动执行 ✅
  ├── Step 2: 自动执行 ✅
  ├── Step 3: ⚠️ 高风险操作 → 请求人类确认
  │           人类: "同意" 或 "修改为..."
  ├── Step 4: 自动执行 ✅
  └── Step 5: 自动执行 ✅
```

### 4.2 什么时候该让人介入

| 场景 | 例子 | 原因 |
|------|------|------|
| **不可逆操作** | 删除文件、发送邮件、部署代码 | 错了无法撤回 |
| **高成本决策** | 大额采购、服务器配置变更 | 错误代价太大 |
| **歧义性判断** | "优化性能"（优化什么指标？可以牺牲什么？） | Agent 缺乏上下文做判断 |
| **安全边界** | 访问敏感数据、执行未知来源的代码 | 安全风险 |

### 4.3 设计原则

好的人机协作设计：
- **默认安全**：Agent 不确定时应该**问**而不是猜
- **信息充分**：请求确认时，告诉人类"我要做什么、为什么、有什么风险"
- **粒度适中**：不要每一步都问（太烦），也不要全部自动（太危险）
- **可回退**：人类应该能撤销 Agent 的操作

你在用的 Cursor 就是一个人机协作的例子——Agent 提出代码修改建议，但需要你接受才会真正应用。

---

## 5. 架构选择的决策树

```
你的任务是什么？
  │
  ├── 简单、2-5 步 → ReAct
  │
  ├── 复杂、需要规划 → Plan-and-Execute
  │   │
  │   └── 需要从错误中学习？ → + Reflexion
  │
  ├── 需要极高准确率？ → LATS
  │
  ├── 需要多种专业能力？ → Multi-Agent
  │
  └── 涉及高风险操作？ → + Human-in-the-Loop
```

实际项目中，这些模式经常**组合使用**。比如一个代码 Agent 可能是：
- Plan-and-Execute（整体架构）
- + ReAct（每个执行步骤内部）
- + Reflexion（代码测试失败时反思）
- + Human-in-the-Loop（部署前请求确认）

---

### 公式速查卡

| 架构 | 含义 |
|------|------|
| ReAct | Thought → Action → Observation 循环，每步临时决策 |
| Plan-and-Execute | 先规划完整计划，再逐步执行，适合长任务 |
| Reflexion | 执行→评估→反思→记住→重试，从错误中显式学习 |
| LATS | Tree Search + ReAct，每步多候选 + 价值评估 + 回退 |
| Multi-Agent | 多个专业 Agent 分工协作，通过消息传递交互 |
| Human-in-the-Loop | 关键节点引入人类判断，防止错误级联 |
| MCTS | 蒙特卡洛树搜索，通过模拟估计每个决策的价值 |

---

## 理解检测

**Q1**：为什么 Multi-Agent 比单 Agent 更适合处理复杂任务？从 system prompt 和注意力的角度解释。

你的回答：


**Q2**：一个代码开发 Agent 需要完成以下流程：理解需求→写代码→运行测试→如果测试失败则修改→部署。请设计一个 Multi-Agent 方案：需要几个 Agent？每个 Agent 的角色是什么？它们之间的交互模式是对话、流水线还是竞争？

你的回答：


**Q3**：Human-in-the-Loop 设计中，哪些操作应该默认请求人类确认？假设你的 Agent 有以下工具：`search_web`、`read_file`、`write_file`、`delete_file`、`send_email`、`run_python_code`。哪些需要确认？哪些可以自动执行？说明理由。

你的回答：


**Q4**：一个 LATS Agent 在每一步生成 3 个候选行动，每个候选需要 1 次 LLM 调用来评估价值。完成一个 5 步任务（每步都选最优候选，无回退），总共需要多少次 LLM 调用？如果是 ReAct 呢？

> 提示：LATS 每步 = 生成候选的调用 + 评估每个候选的调用；ReAct 每步 = 1 次调用

你的回答：

