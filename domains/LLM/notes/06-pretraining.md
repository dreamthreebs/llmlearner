# 06 - 预训练：怎么从海量文本中学到通用能力

> **学习路径**：神经网络基础 → 词嵌入 → Attention → Transformer → **预训练** → 微调与对齐
>
> **前置知识**：Transformer 架构（04-05）、交叉熵损失、梯度下降、条件概率
>
> **参考**：[GPT-3 论文](https://arxiv.org/abs/2005.14165) · [Chinchilla 论文](https://arxiv.org/abs/2203.15556) · [Wikipedia - Language model](https://en.wikipedia.org/wiki/Language_model) · [Wikipedia - GPT](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) · [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 核心问题

**怎么用"预测下一个词"这个简单的任务，训练出一个什么都会一点的模型？**

在 01-overview 中我们说过：LLM 的训练目标就是预测下一个词。这一章我们要严格地理解这个过程——从数学定义到工程实现，搞清楚预训练到底在做什么、为什么有效、以及它有什么局限。

---

## 1. 语言模型的数学定义

### 1.1 联合概率的链式分解

一个"语言模型"（Language Model, LM）在数学上就是一个定义在文本序列上的概率分布。给定一个 token 序列 $x_1, x_2, \ldots, x_T$，语言模型要给出这个序列出现的概率。

根据概率论的链式法则（chain rule），任何联合概率分布都可以精确分解为：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

这不是近似，是恒等式。**任何联合分布都可以这样分解。**

**物理类比**：这就像经典力学中拉格朗日方程描述系统演化——给定初始条件和规则，逐步确定系统的下一个状态。语言模型给定前文，逐步确定下一个 token 的概率分布。

> 可靠程度：Level 1（概率论基本定理）
>
> 参考：[Wikipedia - Chain rule (probability)](https://en.wikipedia.org/wiki/Chain_rule_(probability))

### 1.2 自回归语言模型

上面的分解方式天然地定义了一种建模方法：**自回归**（autoregressive）。模型在每一步 $t$，接收前 $t-1$ 个 token 作为输入，输出第 $t$ 个 token 在整个词表 $V$ 上的概率分布：

$$P(x_t \mid x_1, \ldots, x_{t-1}) = \text{softmax}(f_\theta(x_1, \ldots, x_{t-1}))_{x_t}$$

其中 $f_\theta$ 是 Transformer 网络，参数为 $\theta$，输出一个 $|V|$ 维向量（$|V|$ 通常为 32K-128K），经过 softmax 变成概率分布。

GPT 系列、LLaMA、Claude 等主流 LLM 都是自回归语言模型。

> 可靠程度：Level 1（标准定义）

---

## 2. 训练目标：最大化对数似然

### 2.1 从最大似然到交叉熵

训练语言模型的目标很直觉：给定训练数据（大量文本），调整参数 $\theta$，使得模型认为这些文本出现的概率尽可能大。

数学上，这就是**最大似然估计**（Maximum Likelihood Estimation, MLE）。对于训练集中的一个文本序列 $x_1, \ldots, x_T$：

$$\theta^* = \arg\max_\theta \prod_{t=1}^{T} P(x_t \mid x_{1:t-1}; \theta)$$

取对数（连乘变连加，数值更稳定）：

$$\theta^* = \arg\max_\theta \sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)$$

在实践中我们取负号变成**最小化损失**，再除以 $T$ 做平均：

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)$$

这个损失函数有一个名字：**交叉熵损失**（cross-entropy loss）。它衡量的是模型的预测分布 $P_\theta$ 和真实分布（训练数据中的真实下一个词）之间的差距。

**直觉理解**：$-\log P(x_t \mid \ldots)$ 是模型对正确答案的"惊讶度"。如果模型认为正确的下一个词的概率是 0.9，惊讶度 $-\log(0.9) \approx 0.105$，很小；如果概率是 0.01，惊讶度 $-\log(0.01) \approx 4.6$，很大。训练就是让模型对训练数据"越来越不惊讶"。

### 2.2 Perplexity：语言模型的通用评价指标

对交叉熵取指数就得到 **perplexity**（困惑度）：

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)\right)$$

Perplexity 可以理解为：模型在每一步平均要从多少个等可能的候选中"猜"下一个词。PPL = 10 意味着模型在每一步大约在 10 个词之间犹豫。PPL 越低，模型越好。

> 可靠程度：Level 1（标准指标和标准推导）
>
> 参考：[Wikipedia - Perplexity](https://en.wikipedia.org/wiki/Perplexity) · [Stanford CS224N - Language Models](https://web.stanford.edu/class/cs224n/)

---

## 3. 训练数据

预训练的数据规模和质量决定了模型能力的上限。

### 3.1 数据来源

| 数据源 | 规模 | 特点 |
|--------|------|------|
| Common Crawl | 数万亿 token | 互联网网页抓取，覆盖面广但噪声多 |
| Wikipedia | 数十亿 token | 高质量、结构化知识，但领域有限 |
| Books（BookCorpus, Gutenberg 等） | 数十亿 token | 长文本、连贯叙事、文学语言 |
| 代码（GitHub） | 数千亿 token | 训练代码生成能力，同时提升逻辑推理 |
| 学术论文（arXiv, PubMed） | 数十亿 token | 专业知识、数学公式 |
| 对话数据（Reddit 等） | 数百亿 token | 对话能力、非正式语言 |

GPT-3 用了约 300B tokens 训练，LLaMA 用了 1.4T tokens，Chinchilla 用了 1.4T tokens，LLaMA-3 用了 15T+ tokens。

### 3.2 数据质量的重要性

关键发现：**数据质量比数量更重要**。常见的数据处理步骤：

1. **去重**（deduplication）：重复数据会让模型过拟合特定文本
2. **质量过滤**：去除低质量页面（广告、乱码、自动生成内容）
3. **有害内容过滤**：减少有毒/有偏见的内容
4. **领域配比**：不同来源数据的混合比例显著影响模型能力

LLaMA 的一个关键洞察：用高质量数据训练足够久的小模型，可以超过用低质量数据训练的大模型。

> 可靠程度：Level 2-3（工程实践共识，但最优的数据配比仍在探索中）
>
> 参考：[LLaMA 论文](https://arxiv.org/abs/2302.13971) · [The Pile 数据集](https://arxiv.org/abs/2101.00027)

---

## 4. 训练过程的关键技术

### 4.1 Tokenization（BPE）

文本不能直接输入模型，需要先切成 **token**。主流方法是 **Byte Pair Encoding (BPE)**：

1. 从单个字符（或字节）开始
2. 统计训练语料中相邻 token 对的出现频率
3. 把最高频的 token 对合并为一个新 token
4. 重复步骤 2-3，直到词表达到目标大小（如 32K、64K、128K）

结果：常见词（如 "the"）是一个 token，罕见词（如 "photosynthesis"）被拆成几个 token（"photo" + "synth" + "esis"）。

这样做的好处：
- 词表大小可控（不像按词分割那样词表无限大）
- 对未见过的词也能处理（拆成子词）
- 在字符级灵活性和词级效率之间取得平衡

> 可靠程度：Level 1（标准方法）
>
> 参考：[Wikipedia - Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) · [SentencePiece](https://arxiv.org/abs/1808.06226)

### 4.2 Batch 构建

训练时不会一个序列一个序列地处理，而是打包成 **batch**：

- 把多个文档拼接在一起，切成固定长度的块（如 2048 或 4096 tokens）
- 一个 batch 包含多个这样的块（batch size 通常为数百到数千个序列）
- 使用 **causal mask**：位置 $t$ 的 token 只能看到位置 $1, \ldots, t-1$，不能看到未来的 token

每个 batch 中，每个位置都是一个训练样本——模型要预测该位置的 token。一个长度为 $L$、batch size 为 $B$ 的 batch 包含 $B \times L$ 个训练样本。

### 4.3 学习率调度

学习率不是固定不变的，而是按照精心设计的 **schedule** 变化：

1. **Warmup 阶段**：从接近 0 线性增大到峰值学习率（如 $3 \times 10^{-4}$）。通常占总训练步数的 0.1%-1%。目的是避免训练初期梯度不稳定导致参数被推到坏的区域。

2. **Cosine decay 阶段**：学习率按余弦函数从峰值衰减到接近 0：
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\pi\right)\right)$$
   直觉：前期大步探索，后期小步精调。

**物理类比**：类似模拟退火——先用高"温度"跳出局部极小值，再用低"温度"精确收敛。

> 可靠程度：Level 2（标准实践，但最优超参仍需调参）
>
> 参考：[Cosine annealing 论文](https://arxiv.org/abs/1608.03983)

### 4.4 混合精度训练

标准浮点数是 32 位（FP32），每个参数占 4 字节。对于一个 70B 参数的模型，仅参数就需要 280GB 显存。

**混合精度训练**的核心思路：

- 前向传播和反向传播用 **FP16**（16 位）或 **BF16**（Brain Float 16）：内存减半，计算速度在现代 GPU 上可快 2-8 倍
- 参数更新时保留一份 **FP32 的主拷贝**（master weights）：避免小梯度在低精度下变成 0
- **Loss scaling**：把损失乘以一个大数再反向传播，防止小梯度在 FP16 下下溢

BF16 vs FP16：BF16 的指数位和 FP32 相同（8 位），不容易溢出，是目前大模型训练的首选。

> 可靠程度：Level 1-2（工业标准实践）
>
> 参考：[Mixed Precision Training 论文](https://arxiv.org/abs/1710.03740) · [NVIDIA AMP 文档](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

### 4.5 分布式训练

单块 GPU（即使是 H100 的 80GB 显存）放不下一个大模型。分布式训练有三种主要并行策略：

| 策略 | 切什么 | 每块 GPU 做什么 | 通信开销 |
|------|--------|----------------|---------|
| **数据并行（Data Parallel, DP）** | 切数据 | 每块 GPU 有完整模型，处理不同的 batch | 同步梯度（AllReduce） |
| **模型并行 / 张量并行（Tensor Parallel, TP）** | 切模型层内的矩阵 | 每块 GPU 持有每层权重的一部分 | 每层计算需要通信（高频） |
| **流水线并行（Pipeline Parallel, PP）** | 切模型的层 | 每块 GPU 持有若干连续层 | 层间传递中间激活 |

实际大模型训练通常**三种并行同时使用**（3D 并行）。例如 LLaMA-3 400B 训练使用了 16384 块 H100 GPU。

此外还有 **ZeRO**（Zero Redundancy Optimizer）：在数据并行的基础上，把优化器状态、梯度、参数分散到不同 GPU 上，大幅减少每块 GPU 的显存占用。

> 可靠程度：Level 1-2（工程标准，但大规模训练的稳定性仍是挑战）
>
> 参考：[Megatron-LM 论文](https://arxiv.org/abs/1909.08053) · [ZeRO 论文](https://arxiv.org/abs/1910.02054) · [Wikipedia - Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism)

---

## 5. Scaling Laws：规模的力量

### 5.1 Kaplan et al. 的发现

OpenAI 在 2020 年发现了一组惊人的经验规律：模型的测试损失（交叉熵）和模型参数量 $N$、数据量 $D$、计算量 $C$ 之间存在**幂律关系**：

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$，$N_c, D_c, C_c$ 是常数。

关键结论：
- 增大模型参数比增加训练数据更划算（$\alpha_N > \alpha_D$ 的倒数关系意味着参数效率更高）
- 这三个变量对性能的影响是**平滑的、可预测的幂律**
- 没有发现明显的"天花板"——在测试的规模范围内，更多的计算总是带来更好的性能

### 5.2 Chinchilla 的修正

DeepMind 在 2022 年发表的 Chinchilla 论文修正了上述结论。他们发现 Kaplan 的实验设计有偏差（训练步数不够），导致低估了数据量的重要性。

Chinchilla 的核心发现：**给定计算预算 $C$，最优的参数量 $N$ 和数据量 $D$ 应该同步增长**。具体来说：

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

也就是说，计算预算翻倍时，参数量和数据量应该各增加约 $\sqrt{2}$ 倍。

这意味着当时大多数模型（如 GPT-3 的 175B 参数）都**过大而训练不足**。Chinchilla 只有 70B 参数但用了更多数据训练，性能反而超过了 Gopher（280B 参数）。

这个发现深刻改变了工业界的策略：LLaMA、Mistral 等后续模型都转向了"较小但训练充分"的方向。

> 可靠程度：Level 2（经验规律，定量细节可能随架构/数据而变，但定性趋势已被广泛验证）
>
> 参考：[Scaling Laws for Neural Language Models（Kaplan et al.）](https://arxiv.org/abs/2001.08361) · [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556)

---

## 6. 涌现能力（Emergent Abilities）

### 6.1 现象

在 scaling laws 描述的平滑改进之外，研究者还观察到了一种更戏剧性的现象：某些能力在小模型中完全不存在，但当模型规模超过某个阈值时突然出现。

典型例子：
- **多步算术推理**：13B 参数的模型几乎完全做不对，但 175B 参数的模型可以做对相当一部分
- **Few-shot 学习中的复杂推理任务**：小模型的表现和随机猜测差不多，大模型突然跳到高准确率
- **代码生成**：小模型生成的代码几乎不能运行，大模型能生成可运行的复杂程序

### 6.2 为什么会涌现？

这是一个**尚未完全解决**的问题，但有几种假说：

1. **评价指标假说**：涌现可能是评价指标造成的假象。如果用"完全正确"作为指标，任务需要多步推理，每步正确率从 80% 提升到 95% 时，整体正确率从 $0.8^5 \approx 33\%$ 跳到 $0.95^5 \approx 77\%$。如果用连续指标（如每步正确率），改进是平滑的。

2. **能力组合假说**：复杂任务需要同时具备多种子能力（语法、语义、逻辑、世界知识等）。每种子能力随规模平滑增长，但只有当所有子能力都超过某个阈值时，模型才能完成整个任务。

3. **表示学习假说**：更大的模型有更丰富的内部表示空间，可以形成更精细的概念结构，从而支持更复杂的推理。

> 可靠程度：涌现现象本身是 Level 1（实验观测）。涌现是否是"真正的相变"还是度量假象——Level 4（激烈争论中）。Schaeffer et al. (2023) 的论文 "Are Emergent Abilities of Large Language Models a Mirage?" 对此提出了重要质疑。
>
> 参考：[Emergent Abilities of Large Language Models（Wei et al., 2022）](https://arxiv.org/abs/2206.07682) · [Are Emergent Abilities a Mirage?（Schaeffer et al., 2023）](https://arxiv.org/abs/2304.15004) · [Wikipedia - Emergence](https://en.wikipedia.org/wiki/Emergence)

---

## 7. 预训练的局限

预训练结束后，我们得到了一个强大的语言模型，但它有根本性的局限：

1. **只会"续写"**：模型学到的是 $P(x_t \mid x_{1:t-1})$，即给定前文预测下一个词。你输入一个问题，它不会尝试"回答"，而是"续写"——它可能会继续写更多问题，或者用一种不自然的方式续写答案。

2. **不一定有用**：模型训练数据中什么都有（包括错误信息、偏见、有害内容），续写出来的内容可能是错误的、有偏见的、甚至有害的。

3. **不遵循指令**：你说"请用中文回答"，模型不一定会听——因为在训练数据中，指令后面跟的不一定是遵循指令的回答。

4. **没有"对齐"**：模型的目标是"预测下一个词"，不是"做一个有用的助手"。这两个目标之间有本质的差距。

这些局限引出了下一章的主题：**微调与对齐**——如何把一个续写机器变成一个有用的助手。

> 可靠程度：Level 1（预训练模型的局限已被广泛认识）
>
> 参考：[GPT-3 论文中关于 few-shot 局限的讨论](https://arxiv.org/abs/2005.14165) · [InstructGPT 论文的动机部分](https://arxiv.org/abs/2203.02155)

---

## 本章总结

| 概念 | 核心要点 | 可靠程度 |
|------|---------|---------|
| 语言模型定义 | 通过链式法则分解为逐步条件概率 | L1 |
| 训练目标 | 最小化交叉熵 = 最大化对数似然 | L1 |
| Perplexity | $e^{\text{交叉熵}}$，越低越好 | L1 |
| BPE 分词 | 从字符出发逐步合并高频对 | L1 |
| 学习率调度 | Warmup + cosine decay | L2 |
| 混合精度 | FP16/BF16 计算 + FP32 更新 | L1-2 |
| 分布式训练 | DP + TP + PP 三维并行 | L1-2 |
| Scaling Laws | 损失与 N、D、C 的幂律关系 | L2 |
| Chinchilla | N 和 D 应同步增长 | L2 |
| 涌现能力 | 大模型突然获得小模型没有的能力 | 现象 L1，解释 L4 |
| 预训练局限 | 只会续写，不遵循指令，不安全 | L1 |

---

## 理解检测

**Q1**：训练语言模型的损失函数 $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}; \theta)$ 中的每一项 $-\log P(x_t \mid \ldots)$ 的值域是什么？当模型对正确答案非常确信时这一项趋向什么值？当完全不确信时趋向什么值？这和信息论中的哪个概念直接相关？

你的回答：



**Q2**：Chinchilla 论文的核心结论是"之前的大模型训练不足"。假设你有固定计算预算 $C$，你只能选择：(A) 200B 参数模型训练 500B tokens，或 (B) 70B 参数模型训练 1.5T tokens。根据 Chinchilla 的发现，哪个选择更好？从 scaling law 的角度解释为什么。

你的回答：



**Q3**：有人说"涌现能力证明了大模型有了质的飞跃，获得了小模型没有的新能力"。也有人说"涌现只是度量方式的假象，实际上能力是平滑增长的"。用"多步推理中每步正确率"的例子，解释第二种观点的逻辑。你觉得哪种解释更有说服力？

你的回答：


