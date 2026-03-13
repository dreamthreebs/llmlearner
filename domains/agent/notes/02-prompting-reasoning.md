# 02 - 提示与推理：让 LLM 学会"想"

> **主维度**：D1 核心组件
> **次维度**：D2 架构模式
> **关键关系**：
> - CoT (方法) --依赖--> LLM (架构)：CoT 依赖 LLM 的自回归生成能力
> - ReAct (方法) --推广了--> CoT (方法)：ReAct 推广了 CoT，加入了行动和观察
> - Few-shot Prompting (方法) --用于--> Agent (架构)：Few-shot 用于引导 Agent 行为
>
> **学习路径**：全景概览 → **本章（提示与推理）** → 工具调用 → 记忆 → 规划
>
> **前置知识**：
> - LLM 的自回归生成（参见 `domains/LLM/notes/06-pretraining.md`）
> - Agent 的四大组件（参见 `01-overview.md`）
>
> **参考**：
> - [Wei et al., 2022 - Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
> - [Kojima et al., 2022 - Zero-shot CoT](https://arxiv.org/abs/2205.11916)
> - [Yao et al., 2022 - ReAct](https://arxiv.org/abs/2210.03629)
> - [Yao et al., 2023 - Tree of Thoughts](https://arxiv.org/abs/2305.10601)

---

## 核心问题

**怎么通过设计提示词（prompt），让 LLM 从"直觉式回答"变成"逐步推理"？**

上一章提到 Agent 的核心推理模式是 CoT 和 ReAct。本章深入讲解：这些推理模式**为什么有效**，**具体怎么实现**，以及它们之间的**演化关系**。

> 可靠程度：Level 1-2（CoT 和 ReAct 是有大量实验验证的成熟方法）

---

## 1. Prompting 基础：从 Zero-shot 到 Few-shot

### 1.1 三种提示方式

LLM 的行为完全由输入的 prompt 决定。最基础的三种提示方式：

**Zero-shot**（零样本）：直接给任务，不给示例。

```
Q: 一家商店有 48 个苹果，卖了 15 个，又进货 23 个，现在有多少个？
A:
```

LLM 可能直接输出 "56"（正确），也可能出错——因为它在一步之内完成所有计算。

**Few-shot**（少样本）：给几个输入-输出的示例，再给任务。

```
Q: 2 + 3 = ?
A: 5

Q: 10 - 4 = ?
A: 6

Q: 一家商店有 48 个苹果，卖了 15 个，又进货 23 个，现在有多少个？
A:
```

LLM 通过示例"学会"输出格式和推理方式。注意这不是训练（参数没变），而是**上下文学习**（in-context learning）——LLM 从上下文中的示例推断出任务模式。

**System Prompt**（系统提示）：在对话开始前设定 LLM 的角色和行为规则。

```
System: 你是一个数学助手，解答问题时必须写出完整的计算过程。
User: 一家商店有 48 个苹果，卖了 15 个，又进货 23 个，现在有多少个？
```

这是 Agent 最常用的控制方式——通过 system prompt 定义 Agent 的身份、工具列表、输出格式、行为约束。

### 1.2 为什么 Few-shot 有效？

Few-shot 的效果来自 Transformer 的**注意力机制**（参见 `domains/LLM/notes/04-attention-mechanism.md`）。LLM 在生成每个新 token 时，会 attend to 上下文中所有之前的 token。示例中的输入-输出对提供了一种"模式模板"，LLM 在生成回答时会模仿这个模式。

关键限制：Few-shot 受限于**上下文窗口大小**。示例太多会占用宝贵的上下文空间，留给实际任务的空间就少了。

> 参考：[Brown et al., 2020 - Language Models are Few-Shot Learners (GPT-3 论文)](https://arxiv.org/abs/2005.14165)

---

## 2. Chain-of-Thought (CoT)：让推理过程可见

### 2.1 核心思想

**Chain-of-Thought**（链式思维，Wei et al., 2022）的核心思想极其简单：**让 LLM 在给出最终答案之前，先写出中间推理步骤**。

不用 CoT：
```
Q: 餐厅里原来有 23 位客人。午餐时间又来了 14 位，然后有 9 位离开了。
   晚餐时间又来了 31 位。餐厅现在有多少位客人？
A: 59
```

用 CoT（Few-shot CoT）：
```
Q: 自助餐厅有 23 个苹果。如果用了 20 个后又买了 6 个，有多少苹果？
A: 自助餐厅有 23 个苹果。用了 20 个后剩 23 - 20 = 3 个。又买了 6 个后有 3 + 6 = 9 个。答案是 9。

Q: 餐厅里原来有 23 位客人。午餐时间又来了 14 位，然后有 9 位离开了。
   晚餐时间又来了 31 位。餐厅现在有多少位客人？
A: 餐厅开始有 23 位客人。午餐来了 14 位后有 23 + 14 = 37 位。
   9 位离开后有 37 - 9 = 28 位。晚餐又来了 31 位后有 28 + 31 = 59 位。
   答案是 59。
```

看起来只是多了几行文字，但在 GSM8K（小学数学推理）等基准上，CoT 把 GPT-3 的准确率从 ~17% 提升到 ~58%。

### 2.2 为什么 CoT 有效？——计算深度假说

直觉上，写出中间步骤只是"格式变化"，为什么能大幅提升准确率？

Transformer 的一个基本限制：**每生成一个 token，模型只过一遍前向传播**（对于一个 $L$ 层的 Transformer，就是 $L$ 次矩阵运算）。这意味着模型在一个 token 位置上能做的计算量是固定的。

当问题需要多步推理（比如 4 步加减法），一个 token 位置的计算量可能不够。CoT 把中间结果写出来后，每个中间 token 都提供了一次额外的前向传播——等于**用序列长度换计算深度**。

更形式化地说：标准 Transformer（不带 CoT）的计算能力等价于 $\text{TC}^0$（一种有限深度电路的复杂度类，能做的计算有上限）。加了 CoT 之后，计算能力提升到**图灵完备**（理论上能模拟任意计算过程），因为推理链条可以任意长。

这不是一个严格的理论解释，但它给出了正确的直觉：**CoT 的本质是给 LLM 更多的"思考空间"**。

> 可靠程度：Level 2-3（计算深度假说被广泛接受为直觉解释，但严格理论证明仍在研究中）
>
> 参考：[Feng et al., 2023 - Towards Revealing the Mystery behind CoT](https://arxiv.org/abs/2305.15408)

### 2.3 Zero-shot CoT

Few-shot CoT 需要手工写示例，有没有更简单的方式？

Kojima et al. (2022) 发现一个惊人的结果：只需要在 prompt 末尾加一句 **"Let's think step by step"**，LLM 就会自动生成推理链条，且效果接近 Few-shot CoT。

```
Q: 餐厅里原来有 23 位客人。午餐时间又来了 14 位...
A: Let's think step by step.
   首先，餐厅开始有 23 位客人...
```

这被称为 **Zero-shot CoT**。它的有效性暗示：LLM 在预训练中已经"学会"了逐步推理的模式，只是需要一个触发词来激活。

> 参考：[Kojima et al., 2022 - Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)

---

## 3. 从 CoT 到 Agent 推理：ReAct

### 3.1 CoT 的局限

CoT 让 LLM 会"想"了，但它仍然只在**内部推理**——不能获取外部信息，也不能执行操作。

```
CoT 能解决的：数学推理、逻辑判断、代码分析（不需要外部信息）
CoT 不能解决的：需要实时数据、需要查资料、需要操作外部系统的任务
```

### 3.2 ReAct = Reasoning + Acting

**ReAct**（Yao et al., 2022）的核心创新：把 CoT 推理和工具调用**交织**（interleave）在一起。每一步包含：

1. **Thought**（思考）：分析当前状态，决定策略
2. **Action**（行动）：调用一个工具
3. **Observation**（观察）：接收工具返回的结果

这三步不断循环，直到任务完成。

完整示例——一个知识问答任务：

```
Question: 除了苹果遥控器之外，还有什么设备可以控制苹果遥控器
         最初设计来交互的程序？

Thought 1: 我需要搜索苹果遥控器，找到它最初设计用来交互的程序。
Action 1: Search[Apple Remote]
Observation 1: Apple Remote 是苹果公司设计的遥控器，最初与 Front Row
              媒体中心程序配套...

Thought 2: 苹果遥控器最初设计用来控制 Front Row 程序。现在我需要搜索
           Front Row，看看还有什么设备可以控制它。
Action 2: Search[Front Row (software)]
Observation 2: Front Row 是 macOS 的媒体中心软件，可以通过苹果遥控器
              或键盘功能键控制...

Thought 3: Front Row 可以通过苹果遥控器或键盘功能键控制。所以除了苹果
           遥控器，键盘功能键也可以控制 Front Row。
Action 3: Finish[键盘功能键]
```

### 3.3 ReAct vs 纯 CoT vs 纯 Action

| 方法 | 能推理 | 能行动 | 优势 | 劣势 |
|------|--------|--------|------|------|
| 纯 CoT | ✅ | ❌ | 推理过程清晰 | 无法获取外部信息，可能产生幻觉 |
| 纯 Action | ❌ | ✅ | 能调用工具 | 缺乏策略性，不知道什么时候该搜、搜什么 |
| ReAct | ✅ | ✅ | 推理引导行动，行动提供事实 | 比纯 CoT 需要更多 token，成本更高 |

关键洞察：**推理帮助 Agent 决定做什么（策略），行动帮助 Agent 获取事实（信息）**。两者缺一不可——
- 只有推理没有行动：Agent 可能用不准确的训练知识做推理（幻觉）
- 只有行动没有推理：Agent 不知道什么时候该搜索、搜什么、什么时候该停

### 3.4 ReAct 的 Prompt 设计

在实际实现中，ReAct 通过 system prompt 来引导 LLM 的行为：

```
System Prompt:
你是一个助手，可以使用以下工具：
- Search[query]: 搜索互联网
- Lookup[term]: 在最近搜索到的页面中查找内容
- Finish[answer]: 提交最终答案

请按照以下格式回答：
Thought: 你的思考过程
Action: 工具名称[参数]

观察结果会在 Observation: 之后提供给你。
```

实际的 Agent 框架（如 LangChain）会自动处理 Observation 的注入——执行工具后把结果拼接回对话历史，让 LLM 继续生成下一个 Thought。

> 参考：[Yao et al., 2022 - ReAct](https://arxiv.org/abs/2210.03629)

---

## 4. 更高级的推理策略

### 4.1 Self-Consistency（自洽性采样）

**可靠程度：Level 1-2**

CoT 的一个问题是：同一个问题，不同的推理链条可能得出不同的答案。哪个是对的？

**Self-Consistency**（Wang et al., 2022）的思路：用**多数投票**。

1. 对同一个问题，让 LLM 用 CoT 生成 $N$ 条不同的推理链条（通过调高采样温度 temperature）
2. 每条链条得出一个答案
3. 选**出现次数最多**的答案作为最终答案

```
问题：停车场有 3 辆车，又来了 2 辆，现在几辆？

链条 1: 3 + 2 = 5 → 答案 5
链条 2: 3 + 2 = 5 → 答案 5
链条 3: 3 × 2 = 6 → 答案 6（推理出错）

多数投票 → 答案 5 ✅
```

Self-Consistency 在 GSM8K 上把 CoT 的准确率从 ~58% 提升到 ~74%（GPT-3 175B）。代价是需要多次调用 LLM，成本乘以 $N$。

**温度（temperature）** 控制 LLM 输出的随机性：temperature = 0 时每次输出相同的结果（贪心解码），temperature 越高输出越多样。Self-Consistency 需要 temperature > 0 来获得不同的推理链条。

> 参考：[Wang et al., 2022 - Self-Consistency](https://arxiv.org/abs/2203.11171)

### 4.2 Tree of Thoughts (ToT)

**可靠程度：Level 2-3**

CoT 是一条线性链条——一步步往前推。但有些问题需要**探索多条路径**，发现走不通的路要**回退**。

**Tree of Thoughts**（Yao et al., 2023）把推理从链条扩展为**树状结构**：

```
             问题
           /  |  \
        想法1 想法2 想法3
        / \    |
     1a  1b   2a
      |       |
    1a-i    2a-i
      |
    答案 ✅
```

1. **展开**：在每一步生成多个候选想法（类似 CoT，但每步多个分支）
2. **评估**：让 LLM 对每个候选想法打分（"这个方向有多大可能解决问题？"）
3. **搜索**：用 BFS（广度优先搜索）或 DFS（深度优先搜索）遍历树

BFS 和 DFS 的区别：
- **BFS**：逐层探索，先看完所有第一步候选，再看第二步。适合分支少的问题
- **DFS**：沿一条路走到底，走不通再回退。适合需要深度探索的问题

ToT 在需要"搜索 + 回退"的任务（如 24 点游戏、创意写作）上效果显著，但**成本极高**（需要多次 LLM 调用来展开和评估每个节点）。在大多数实际 Agent 中，ReAct 就够用了——ToT 更多出现在需要极高推理准确率的场景。

> 参考：[Yao et al., 2023 - Tree of Thoughts](https://arxiv.org/abs/2305.10601)

---

## 5. 推理模式的演化关系

```
Zero-shot          最简单，直接给任务
  │
  ▼
Few-shot           给示例，引导格式
  │
  ▼
CoT                写出推理步骤，提升复杂推理能力
  │
  ├──→ Zero-shot CoT     "Let's think step by step" 触发
  │
  ├──→ Self-Consistency   多条链条 + 多数投票
  │
  ├──→ Tree of Thoughts   链条 → 树状搜索
  │
  └──→ ReAct              CoT + 工具调用，Agent 的基础框架
         │
         └──→ Reflexion    ReAct + 自我反思（后续规划章节详讲）
```

对于搭 Agent 来说，**最重要的是 ReAct**——它是绝大多数 Agent 框架的底层模式。CoT 是 ReAct 的基础，理解了 CoT 的"为什么有效"，才能在实践中设计好 ReAct 的 Thought 部分。

---

### 公式速查卡

| 框架 | 含义 |
|------|------|
| Zero-shot | 无示例，直接给任务 |
| Few-shot | 给 $k$ 个输入-输出示例，引导 LLM 模仿模式 |
| CoT | 让 LLM 写出中间推理步骤再给答案，本质是用序列长度换计算深度 |
| Zero-shot CoT | 加 "Let's think step by step" 触发推理链条 |
| Self-Consistency | 生成 $N$ 条推理链条，多数投票选答案；成本 $\times N$ |
| ReAct | Thought → Action → Observation 循环；推理引导行动，行动提供事实 |
| ToT | 推理从链条扩展为树状搜索，支持展开、评估、回退 |

---

## 理解检测

**Q1**：CoT 为什么能提升 LLM 的推理准确率？用"计算深度"的角度解释。如果一个 Transformer 有 96 层，面对一个需要 5 步推理的问题，不用 CoT 时模型最多有多少层的计算深度？用 CoT 时呢？

> 提示：每生成一个 token 过一遍完整的前向传播

你的回答：


**Q2**：写一个 ReAct 风格的执行过程：用户问"Python 的 GIL 是什么？它和多线程的关系是什么？"，你可以使用 `Search[query]` 和 `Finish[answer]` 两个工具。至少写 2 轮 Thought-Action-Observation。

你的回答：


**Q3**：Self-Consistency 方法中，生成了以下 5 条推理链条的答案：[42, 42, 38, 42, 38]。最终答案是什么？如果 LLM 的单次调用成本是 $0.01，用 Self-Consistency 回答这个问题的总成本是多少？

> 提示：多数投票选出现次数最多的答案；总成本 = 单次成本 × 调用次数

你的回答：

