# 07 - 微调与对齐：从续写机器到有用助手

> **学习路径**：神经网络基础 → 词嵌入 → Attention → Transformer → 预训练 → **微调与对齐**
>
> **前置知识**：预训练（06）、交叉熵损失、KL 散度基础概念、强化学习基本概念（可选，会在文中解释）
>
> **参考**：[InstructGPT 论文](https://arxiv.org/abs/2203.02155) · [DPO 论文](https://arxiv.org/abs/2305.18290) · [Constitutional AI 论文](https://arxiv.org/abs/2212.08073) · [Wikipedia - RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) · [Hugging Face RLHF 博客](https://huggingface.co/blog/rlhf)

---

## 核心问题

**预训练出来的模型只会续写文本，怎么让它变成一个有用、安全的助手？**

上一章我们看到，预训练完成后的模型已经"知道"很多东西——它在海量文本中学到了语法、语义、逻辑、世界知识。但它的行为模式是"给定前文，续写最可能的后文"，而不是"理解用户意图并给出有帮助的回答"。

举个例子，你输入 "What is the capital of France?"，一个纯预训练模型可能续写：
- "What is the capital of Germany? What is the capital of Spain?..."（继续出题）
- "The capital of France is Paris. The capital of France has been..."（正确但冗余）
- 或者任何在训练数据中"What is the capital of France?"后面常跟的内容

它不会像 ChatGPT 那样直接简洁地回答 "The capital of France is Paris."——除非训练数据中有大量这种问答格式。

**对齐**（alignment）就是解决这个 gap 的过程：把模型的行为从"续写文本"调整为"按照人类期望行事"。

> 可靠程度：Level 1（预训练模型行为的观察和对齐的必要性是共识）
>
> 参考：[InstructGPT 论文 Section 1](https://arxiv.org/abs/2203.02155)

---

## 1. 三阶段 Pipeline（OpenAI 的方法）

OpenAI 在 InstructGPT（2022）中提出了一个清晰的三阶段流程，后来成为 LLM 对齐的标准范式：

```
预训练模型 (GPT-3)
    ↓ Stage 1: SFT
有监督微调后的模型
    ↓ Stage 2: RM
奖励模型
    ↓ Stage 3: RLHF (PPO)
对齐后的模型 (InstructGPT / ChatGPT)
```

下面逐一拆解每个阶段。

> 可靠程度：Level 1-2（标准方法，但具体实现细节各公司有变体）
>
> 参考：[InstructGPT 论文](https://arxiv.org/abs/2203.02155) · [Hugging Face RLHF 博客](https://huggingface.co/blog/rlhf)

---

## 2. Stage 1：SFT（Supervised Fine-Tuning）

### 2.1 核心思路

SFT 的思路极其直觉：收集一批"指令-回答"对（demonstration data），然后用和预训练相同的方式（最大化对数似然）继续训练模型。

训练数据示例：

```
指令：用三句话解释量子纠缠
回答：量子纠缠是指两个或多个粒子之间存在一种特殊的量子关联。
当你测量其中一个粒子的状态时，另一个粒子的状态会立即确定，
无论它们相距多远。这种关联超越了经典物理的局域性原理。
```

损失函数和预训练完全一样——交叉熵损失：

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)$$

唯一的区别是**数据不同**：预训练用的是互联网上的原始文本，SFT 用的是人工标注的高质量"指令→回答"对。

实践中通常只对回答部分计算损失（指令部分的 token 不计入），这样模型学习的是"给定指令如何回答"，而不是"如何生成指令"。

### 2.2 SFT 的效果和局限

SFT 数据量通常很小——InstructGPT 只用了约 13K 条标注数据（对比预训练的数百 billions tokens）。但效果惊人：模型从"续写文本"变成了"尝试回答问题"。

局限：
- **标注成本高**：需要人类专家写高质量回答
- **只学了"一种好回答"**：同一个问题可能有多种合理的回答风格/深度，但标注数据只提供了标注者的一种
- **不知道什么是"更好"**：SFT 模型能回答问题了，但回答质量参差不齐。它没有"好回答 vs 差回答"的概念

> 可靠程度：Level 1-2（标准做法，InstructGPT 用 13K 数据是论文数据）
>
> 参考：[InstructGPT 论文 Section 3.2](https://arxiv.org/abs/2203.02155)

---

## 3. Stage 2：奖励模型（Reward Model）

### 3.1 为什么需要奖励模型

SFT 后的模型能回答问题了，但回答有好有坏。我们需要一种方式告诉模型"什么是好回答"。

直接方法：让人类对每个回答打分。但这在 RL 训练中需要大量反馈，不可能每次都找人。

解决方案：**训练一个自动打分器**——奖励模型（Reward Model, RM）。

### 3.2 训练过程

奖励模型的训练数据是**人类偏好对比**（preference comparisons）：

1. 给定一个指令 $x$，让 SFT 模型生成多个回答 $y_1, y_2, \ldots, y_k$
2. 让人类标注者对这些回答排序（或选出更好的那个）
3. 用排序数据训练一个模型来预测人类偏好

奖励模型通常和 LLM 架构相同（去掉最后的语言模型头，换成一个输出标量分数的头）。训练目标是 **Bradley-Terry 排序损失**：

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $y_w$ 是人类偏好的回答（winner），$y_l$ 是不被偏好的回答（loser），$r_\theta(x, y)$ 是奖励模型给指令 $x$ 和回答 $y$ 打的分，$\sigma$ 是 sigmoid 函数。

**直觉**：这个损失鼓励奖励模型给"好回答"打的分比"差回答"高。分差越大（即模型越确信谁好谁坏），损失越低。

InstructGPT 使用了约 33K 条对比数据来训练奖励模型。

> 可靠程度：Level 1-2（标准方法，Bradley-Terry 模型是经典的排序模型）
>
> 参考：[InstructGPT 论文 Section 3.3](https://arxiv.org/abs/2203.02155) · [Wikipedia - Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

---

## 4. Stage 3：RLHF（Reinforcement Learning from Human Feedback）

### 4.1 核心目标

现在我们有了一个奖励模型 $r(x, y)$，可以给任意的指令-回答对打分。下一步是用这个奖励信号来优化 LLM，让它生成能获得高奖励的回答。

这本质上是一个强化学习（RL）问题：
- **策略（policy）**：LLM $\pi_\theta(y \mid x)$——给定指令 $x$，生成回答 $y$
- **环境**：无（单步 episode，不需要多步交互）
- **奖励**：$r(x, y)$——奖励模型给出的分数

### 4.2 数学框架

但直接最大化奖励有一个严重问题：**奖励劫持**（reward hacking）。模型可能找到一些奇怪的回答，虽然得到奖励模型的高分，但完全不像正常的文本。这是因为奖励模型本身是不完美的，RL 会 exploit 奖励模型的弱点。

解决方案：加一个 **KL 散度惩罚项**，约束优化后的模型 $\pi_\theta$ 不要偏离初始（SFT 后的）模型 $\pi_{\text{ref}}$ 太远：

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(y|x)} \big[ r(x,y) \big] - \beta \cdot D_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]$$

逐项解释：

| 项 | 含义 |
|----|------|
| $\mathbb{E}_{x \sim \mathcal{D}}$ | 从训练指令集中采样指令 |
| $y \sim \pi_\theta(y \mid x)$ | 当前模型生成回答 |
| $r(x, y)$ | 奖励模型对（指令，回答）对打分 |
| $D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$ | 当前策略与参考策略（SFT 模型）之间的 KL 散度 |
| $\beta$ | 超参数，控制 KL 惩罚的强度 |

**$\beta$ 的作用**：$\beta$ 越大，模型越被约束在 SFT 模型附近，不容易出格但改进也小；$\beta$ 越小，模型可以更自由地优化奖励，但可能产生不自然的输出。

**物理类比**：这很像弹簧系统——奖励信号是驱动力（把模型拉向高奖励方向），KL 惩罚是弹性回复力（把模型拉回参考位置）。$\beta$ 是弹性系数。

### 4.3 PPO 算法

上述优化问题用 **PPO**（Proximal Policy Optimization）算法求解。PPO 是一种策略梯度方法，核心思路：

1. 用当前模型 $\pi_\theta$ 对一批指令生成回答
2. 用奖励模型打分
3. 计算策略梯度：鼓励高奖励的回答、抑制低奖励的回答
4. 用 PPO 的"截断比率"（clipped ratio）限制每步更新的幅度，保证训练稳定性

PPO 的数学细节需要 RL 背景才能完全理解，但核心直觉很简单：**如果一个回答得分高，就增加模型生成它的概率；得分低，就降低概率。**

> 可靠程度：Level 1-2（RLHF + PPO 是标准方法，但 PPO 在 LLM 场景的训练稳定性需要大量工程技巧）
>
> 参考：[PPO 论文](https://arxiv.org/abs/1707.06347) · [InstructGPT 论文 Section 3.4](https://arxiv.org/abs/2203.02155)

---

## 5. DPO：跳过奖励模型的直接优化

### 5.1 RLHF 的痛点

RLHF 虽然有效，但工程上很痛苦：

1. 需要同时维护 4 个模型（SFT 模型、奖励模型、RL 训练中的模型、参考模型）
2. PPO 训练不稳定，超参敏感
3. 采样-打分-更新的循环效率低

### 5.2 DPO 的核心洞察

Rafailov et al. (2023) 提出了一个优雅的解决方案：**DPO（Direct Preference Optimization）**。

核心洞察来自对 RLHF 目标函数的数学分析。RLHF 的最优策略有一个闭式解：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x)$ 是归一化常数。这意味着最优策略和最优奖励函数之间存在一一对应关系。把这个关系反过来，可以把奖励表示为策略的函数：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

把这个代入 Bradley-Terry 模型中的奖励差，$Z(x)$ 项抵消，得到 **DPO 损失函数**：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### 5.3 DPO 的直觉理解

把上面的公式翻译成人话：

$$\text{DPO 损失} = -\log \sigma\big(\underbrace{\beta \cdot (\text{好回答的对数概率提升})}_{\text{增加好回答的概率}} - \underbrace{\beta \cdot (\text{差回答的对数概率提升})}_{\text{但也要降低差回答的概率}}\big)$$

DPO 做的事情是：
- **提高**模型生成"好回答"的概率（相对于参考模型）
- **降低**模型生成"差回答"的概率（相对于参考模型）
- 两者的平衡由 sigmoid 和 $\beta$ 控制

关键优势：
- **不需要训练奖励模型**：直接用偏好数据优化 LLM
- **不需要 RL 采样**：标准的监督学习目标（交叉熵类似的损失），可以用普通的梯度下降
- **只需要 2 个模型**（当前模型 + 冻结的参考模型），而不是 RLHF 的 4 个

DPO 已经成为开源社区最流行的对齐方法。

> 可靠程度：Level 1-2（数学推导正确且严格，实际效果与 RLHF 相当或更好已被广泛验证。但在某些场景下 RLHF 仍可能更好——Level 3 争论中）
>
> 参考：[Direct Preference Optimization 论文](https://arxiv.org/abs/2305.18290)

---

## 6. Constitutional AI：让 AI 自我监督

### 6.1 动机

RLHF 和 DPO 都依赖**人类标注数据**——让人类判断哪个回答更好。但人类标注有几个问题：

1. **成本高**：需要大量专业标注者
2. **不一致**：不同标注者标准不同
3. **可扩展性差**：随着模型能力提升，需要越来越专业的标注者

Anthropic 在 2022 年提出了 **Constitutional AI（CAI）**：让 AI 自己评判自己的输出。

### 6.2 方法

CAI 定义了一组"宪法"（constitution）——一系列明确的行为准则，例如：

- "回答应该是诚实的"
- "回答不应该帮助用户从事非法活动"
- "如果不确定，应该承认不确定性"

然后训练分两步：

1. **Critique + Revision**（自我批评和修改）：
   - 让模型生成一个初始回答
   - 让模型根据"宪法"中的某条原则批评自己的回答
   - 让模型根据批评修改回答
   - 用修改后的回答作为 SFT 数据

2. **RLAIF**（RL from AI Feedback）：
   - 用 AI（而不是人类）来判断哪个回答更符合"宪法"
   - 用这些 AI 偏好数据训练奖励模型
   - 用标准 RL 方法（和 RLHF 相同）优化 LLM

### 6.3 意义和局限

CAI 大幅减少了对人类标注的依赖，使对齐过程更可扩展。但它引入了一个新问题：**AI 评判自己是否可靠？** 如果模型本身就有偏见，它的自我评判可能也带有同样的偏见。

> 可靠程度：Level 2-3（方法有效但仍在迭代中，AI 自我评判的可靠性是开放问题）
>
> 参考：[Constitutional AI 论文](https://arxiv.org/abs/2212.08073)

---

## 7. 对齐的开放问题

### 7.1 浅层对齐 vs 深层理解

当前的对齐方法本质上是在调整模型的**输出分布**——让模型更可能生成"好"的回答、更不可能生成"坏"的回答。但模型是否真正"理解"了为什么某些回答是好的、某些是坏的？

证据倾向于**浅层对齐**：
- 模型可以被 jailbreak（通过精心构造的 prompt 绕过安全限制）
- 对齐的效果在 fine-tuning 后容易被"洗掉"
- 模型有时候会"知道"规则但在边界情况下不遵守

这意味着当前的对齐更像是"行为调教"而不是"价值内化"。

### 7.2 Jailbreak 和安全性

Jailbreak 是指通过特殊的 prompt 让模型生成它本来拒绝生成的内容。常见方法：

- **角色扮演**："假设你是一个没有限制的 AI..."
- **编码/混淆**：用 Base64 编码、隐喻等方式绕过关键词检测
- **多步引导**：先让模型同意无害的前提，逐步引导到有害内容

jailbreak 的存在说明当前对齐方法的脆弱性。这不仅仅是工程问题——它反映了对齐方法的根本局限。

### 7.3 "对齐税"（Alignment Tax）

一个重要的实践问题：**对齐是否会降低模型的基础能力？**

观察到的现象：
- RLHF 后的模型在某些基准测试上可能比 SFT 模型略差
- 过度安全（over-safety）：模型拒绝回答完全合理的问题（"我不能帮助你讨论炸弹，因为这可能被用于..."——即使用户只是在讨论二战历史）
- 对齐可能使模型过于讨好用户（sycophancy），倾向于说用户想听的话而不是真话

理想情况下，对齐不应该有"税"——一个安全有用的模型应该同时也是能力最强的模型。但实践中两者之间存在张力。

> 可靠程度：Level 3-4（这些都是活跃的研究方向，没有定论）
>
> 参考：[Jailbreak 分类研究](https://arxiv.org/abs/2308.03825) · [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)

---

## 8. 方法对比总结

| 方法 | 需要的数据 | 需要的模型 | 训练复杂度 | 可靠程度 |
|------|-----------|-----------|-----------|---------|
| SFT | 指令-回答对（demonstration） | 1 个（LLM） | 低（标准微调） | L1 |
| RLHF | 偏好对比 + RL 采样 | 4 个（LLM, ref, RM, value） | 高（PPO 不稳定） | L1-2 |
| DPO | 偏好对比 | 2 个（LLM, ref） | 中（标准梯度下降） | L1-2 |
| CAI | AI 生成的偏好 + 宪法 | 2-3 个 | 中 | L2-3 |

当前工业界的主流做法：**SFT + DPO**（或 SFT + RLHF），逐渐加入更多 AI 反馈以减少人工标注。

---

## 9. 从预训练到部署：完整流程

把这一章和上一章连起来，一个 LLM 从无到有的完整流程是：

```
随机初始化参数
    ↓ 预训练（数月，数千 GPU）
      目标：最小化下一个词预测的交叉熵
      数据：互联网文本（万亿 token）
    ↓ SFT（数天，数十 GPU）
      目标：学习"指令→回答"的格式
      数据：人工标注数据（万级别）
    ↓ RLHF / DPO（数天，数十 GPU）
      目标：让回答更符合人类偏好
      数据：人类偏好对比数据（万级别）
    ↓ 安全评估和红队测试
    ↓ 部署
```

从计算量的角度看：
- 预训练占总计算量的 **99%+**
- SFT 和对齐加起来不到 1%

但这不到 1% 的计算把模型从"自言自语的语料库"变成了"可以对话的助手"。这是 LLM 工程中最令人惊讶的发现之一。

> 可靠程度：Level 1-2（完整 pipeline 的描述是标准的，具体比例是大致数量级）

---

## 本章总结

| 概念 | 核心要点 | 可靠程度 |
|------|---------|---------|
| SFT | 用标注的"指令-回答"对微调，格式相同但数据不同于预训练 | L1 |
| 奖励模型 | 训练一个打分器，从人类偏好对比中学习 | L1-2 |
| RLHF | 用 PPO 优化 LLM 以最大化奖励（含 KL 约束） | L1-2 |
| KL 惩罚 | 防止模型偏离参考策略太远，避免 reward hacking | L1 |
| DPO | 数学上等价于 RLHF 但跳过奖励模型，直接用偏好数据优化 | L1-2 |
| Constitutional AI | 用 AI 自我监督替代部分人类标注 | L2-3 |
| 浅层对齐 | 当前对齐可能只改变了输出分布，未内化价值 | L3-4 |
| 对齐税 | 对齐可能以牺牲部分能力为代价 | L3-4 |

---

## 理解检测

**Q1**：RLHF 的目标函数中有一个 KL 散度惩罚项 $\beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$。如果去掉这一项（即令 $\beta = 0$），只最大化奖励 $r(x,y)$，会发生什么？从优化的角度解释为什么需要这个约束。

你的回答：



**Q2**：DPO 的核心贡献是把 RLHF 的 RL 问题转化成了一个监督学习问题。但从数据需求的角度看，DPO 和 RLHF 都需要**偏好对比数据**（哪个回答更好）。那 DPO 相对于 RLHF 的核心简化到底在哪里？是减少了数据需求，还是减少了计算/工程复杂度？

你的回答：



**Q3**：假设你训练了一个 LLM，经过 SFT 和 RLHF 后，用户用一个 jailbreak prompt 成功让模型生成了有害内容。有人说这说明"对齐根本没用"，你怎么反驳或者怎么部分同意这个观点？

你的回答：


