# 01 - LLM 是什么，为什么要学，要学什么

## LLM 到底是什么

你每天用的 ChatGPT、Claude、Gemini，背后都是 LLM。它的工作原理用一句话说就是：

**给定前面的文字，预测下一个词最可能是什么。**

比如输入"今天天气真"，模型输出一个概率分布：{ "好": 0.6, "差": 0.15, "热": 0.1, ... }。然后从中选一个词（比如"好"），拼上去，再预测下一个词。如此反复，就生成了一段文字。

这听起来极其简单——确实，**LLM 的训练目标就是简单的**。但令人震惊的发现是：当你用足够大的模型（数十亿到数千亿参数）、在足够多的文本（互联网规模）上训练这个任务时，模型被迫学会了远超"填词"的能力——语法、语义理解、逻辑推理、代码编写、数学推导、甚至某种程度的"常识"。

**为什么"预测下一个词"能学到这么多？** 因为要准确预测下一个词，你必须理解上下文。比如要预测"物体的加速度等于力除以\_\_"的下一个词，模型必须"知道"牛顿第二定律。大量文本中隐含了人类知识的各种结构，而预测任务迫使模型把这些结构编码在参数中。

> 参考：[Wikipedia - Large language model](https://en.wikipedia.org/wiki/Large_language_model) · [Andrej Karpathy - The unreasonable effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## LLM 的核心组件

一个现代 LLM 的构建可以分解为几个层次：

```
文字输入
  ↓
Tokenizer：把文字切成"词元"（token），每个 token 对应一个整数编号
  ↓
Embedding：每个 token 编号查表得到一个向量（比如 768 维或 4096 维）
  ↓
Transformer 层 × N（N 从十几到上百）：
  每一层做两件事：
    1. Self-Attention：每个 token "看"其他所有 token，决定关注谁
    2. FFN（前馈网络）：对每个 token 的表示做非线性变换
  ↓
输出层：把最终的向量映射回词表大小的概率分布
  ↓
下一个词的概率
```

整个模型的可训练参数（几十亿到几千亿个浮点数）就分布在 Embedding 层、Attention 层、FFN 层中。训练的过程就是调整这些参数，使得模型对训练数据中"下一个词"的预测尽可能准确。

> 参考：[Wikipedia - Transformer (deep learning architecture)](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) · ["Attention Is All You Need"（Transformer 原始论文）](https://arxiv.org/abs/1706.03762)

## 为什么 LLM 重要

1. **它改变了 AI 的范式**。传统 AI 是"每个任务训一个专用模型"。LLM 是"一个模型，通用能力"——通过自然语言指令就能完成翻译、问答、编程、摘要等各种任务，不需要针对每个任务单独训练。

2. **它是目前最强的自然语言理解和生成系统**。从搜索引擎到编程助手到科学研究，LLM 正在深刻改变各个领域。

3. **它背后的原理非常深刻且仍不完全理解**。为什么规模增大能力就涌现？为什么 Transformer 比其他架构好？模型内部到底学到了什么"知识"？这些都是活跃的研究问题。

## 学 LLM 需要学哪些东西

从你的数学基础出发，到理解 LLM 的核心原理，需要按顺序学习：

| 步骤 | 要学的概念 | 一句话说为什么需要 | 参考 |
|------|-----------|-------------------|------|
| 1 | 神经网络基础 | 神经网络是 LLM 的基础组件，不懂就全是黑箱 | [3Blue1Brown 神经网络视频](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network) |
| 2 | 词嵌入（Word Embedding） | 文字怎么变成数字向量，为什么语义相近的词距离也近 | [Wikipedia - Word2Vec](https://en.wikipedia.org/wiki/Word2vec) · [Jay Alammar 可视化](https://jalammar.github.io/illustrated-word2vec/) |
| 3 | Attention 机制 | LLM 的灵魂——让模型在处理每个词时能"关注"序列中最相关的其他词 | [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [原始论文](https://arxiv.org/abs/1706.03762) |
| 4 | Transformer 架构 | 把 Attention 和 FFN 组装成完整的模型 | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) · [Harvard - The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) |
| 5 | 预训练（Pretraining） | 怎么用海量文本训练出通用的语言能力 | [Wikipedia - GPT](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) · [GPT-3 论文](https://arxiv.org/abs/2005.14165) |
| 6 | 微调与对齐（Alignment） | 怎么把一个"续写机器"变成有用、安全的助手 | [InstructGPT 论文](https://arxiv.org/abs/2203.02155) · [Wikipedia - RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) |

前置知识你基本够用：线性代数（矩阵乘法、向量）、微积分（偏导数、链式法则）、概率论（条件概率、最大似然）。

## LLM 研究的前沿方向

上面 6 步是"经典 LLM 原理"，大部分已经是教科书/教程级共识（Level 1-2）。以下是当前最活跃的前沿方向：

### 一、Scaling Laws（规模定律）

核心发现：模型性能和三个变量的关系可以用幂律描述——模型参数量 $N$、训练数据量 $D$、计算量 $C$。给定总计算预算 $C$，存在最优的 $N$ 和 $D$ 的分配方式。

- **Kaplan et al. (2020)**：首次发现 LLM 的 scaling laws
- **Chinchilla (Hoffmann et al., 2022)**：修正了之前的分配比例，证明大多数模型"过大而训练不足"

这个发现深刻地改变了工业界训练 LLM 的策略。

> 可靠程度：经验规律 Level 2（幂律关系是实验观测，理论解释仍在发展中）
>
> 参考：[Scaling Laws 原始论文](https://arxiv.org/abs/2001.08361) · [Chinchilla 论文](https://arxiv.org/abs/2203.15556)

### 二、推理能力（Reasoning）

LLM 最初被认为只是"统计鹦鹉"——只会模式匹配。但通过 Chain-of-Thought (CoT) prompting 和专门的推理训练，模型展现出了显著的推理能力：

- **Chain-of-Thought**：让模型"一步一步想"，显著提升数学和逻辑推理
- **推理专用模型**：OpenAI o1/o3、DeepSeek-R1 等在数学和编程上达到了竞赛级水平
- **2026 最新**：小模型（15B 参数级别）通过精心设计的训练数据也能达到大模型的推理水平

> 可靠程度：CoT 有效是 Level 1 实验事实。模型是否真正在"推理"还是在做高级模式匹配——Level 4（激烈争论中）
>
> 参考：[Chain-of-Thought 论文](https://arxiv.org/abs/2201.11903) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

### 三、多模态（Multimodal）

从纯文本扩展到图像、音频、视频：

- GPT-4V、Claude 3.5、Gemini 等支持图像输入
- 架构方式：把图像编码器的输出当作"视觉 token"和文本 token 一起送入 Transformer
- 2026 最新：15B 参数的多模态模型可以做数学推理 + 图表理解 + UI 操作

> 可靠程度：技术路线 Level 2（有效但仍在快速演化），理论理解 Level 3-4
>
> 参考：[NVLM 论文](https://arxiv.org/abs/2409.11402) · [Wikipedia - Multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning)

### 四、高效架构（MoE, SSM, Mamba）

Transformer 的计算量和序列长度平方成正比（$O(n^2)$），长文本很贵。替代方案：

- **MoE（Mixture of Experts）**：每次推理只激活部分参数，总参数大但计算量小。Mixtral、DeepSeek-V3 采用
- **SSM / Mamba**：状态空间模型，线性复杂度 $O(n)$，对长序列更高效
- **混合架构**：Mamba + 稀疏 Attention 的组合（如 Hydra），2025-2026 年活跃方向
- **非自回归生成**：并行去噪代替逐词生成（如 Mercury 2），推理速度可快 5 倍

> 可靠程度：MoE 有效是 Level 1-2。SSM 能否全面替代 Transformer 仍是 Level 4
>
> 参考：[Mamba 论文](https://arxiv.org/abs/2312.00752) · [Mixtral 论文](https://arxiv.org/abs/2401.04088) · [Wikipedia - MoE](https://en.wikipedia.org/wiki/Mixture_of_experts)

### 五、对齐与安全（Alignment & Safety）

训练出来的 LLM 默认只会"续写"，不一定有用或安全。对齐就是让模型按人类期望行事：

- **RLHF**（Reinforcement Learning from Human Feedback）：用人类标注训练一个"奖励模型"，再用 RL 优化 LLM 的输出
- **DPO**（Direct Preference Optimization）：跳过奖励模型，直接从人类偏好数据优化 LLM，更简单
- **2025-2026 最新**：发现当前对齐方法只是"浅层对齐"——模型可能学会了表面遵守规则但没有内化安全推理，容易被 jailbreak

> 可靠程度：RLHF/DPO 有效是 Level 1-2。对齐是否足够、模型是否真正"理解"安全约束——Level 4
>
> 参考：[InstructGPT（RLHF 开创论文）](https://arxiv.org/abs/2203.02155) · [DPO 论文](https://arxiv.org/abs/2305.18290) · [Alignment-Weighted DPO](https://arxiv.org/abs/2602.21346)

### 六、长上下文与记忆

标准 Transformer 的上下文窗口有限（早期 2K-4K tokens）。现在的方向：

- **超长上下文**：Claude 支持 200K tokens，Gemini 支持 1M+ tokens
- 技术手段：RoPE 外推、稀疏注意力、分层记忆
- 核心问题：长上下文 ≠ 有效利用所有信息（"大海捞针"问题）

> 可靠程度：技术方案 Level 2-3，长上下文的有效性评估 Level 3
>
> 参考：[RoPE 论文](https://arxiv.org/abs/2104.09864) · [Wikipedia - Context window](https://en.wikipedia.org/wiki/Context_window)

### 前沿方向总结

| 方向 | 核心问题 | 关键工作 | 可靠程度 |
|------|---------|---------|---------|
| Scaling Laws | 模型该多大？数据该多少？ | Kaplan 2020, Chinchilla 2022 | 经验规律 L2 |
| 推理能力 | LLM 能真正推理吗？ | CoT, o1/o3, DeepSeek-R1 | 有效 L1，本质 L4 |
| 多模态 | 如何统一文本和视觉？ | GPT-4V, Gemini, NVLM | 技术 L2，理论 L3-4 |
| 高效架构 | 如何降低计算成本？ | MoE, Mamba, 混合架构 | MoE L1-2，SSM 替代 L4 |
| 对齐与安全 | 如何让模型安全有用？ | RLHF, DPO, SafeDPO | 方法 L1-2，充分性 L4 |
| 长上下文 | 如何处理超长文本？ | RoPE, 稀疏注意力 | 技术 L2-3 |

## 学习建议

- 步骤 1-6 对应 LLM 核心原理（notes 文件 02-07），这些是理解 LLM 的最短路径
- 理解 Attention 机制是最关键的一步——花多少时间都不嫌多
- 前沿方向可以在基础建立之后选择性深入
- 每个文件末尾有检测问题，你在文件里回答
- 如果某个步骤你觉得已经懂了，告诉我跳过

---

## 理解检测

**Q1**：用你自己的话说一下，LLM 的训练目标是什么？为什么这么简单的目标能让模型学到复杂的能力？

你的回答：



**Q2**：Transformer 里的 Self-Attention 解决的核心问题是什么？（不需要知道具体怎么算，说说你觉得它在做什么就行）

你的回答：



**Q3**：上面列了 6 个学习步骤和 6 个前沿方向，你最想先深入哪个？

你的回答：



