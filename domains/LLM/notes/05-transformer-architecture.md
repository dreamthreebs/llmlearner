# 05 - Transformer 架构——完整拼图

> **学习路径**：神经网络基础 → 词嵌入 → Attention 机制 → **Transformer 架构** → 预训练 → 微调与对齐
>
> **前置知识**：Scaled Dot-Product Attention、Multi-Head Attention（04-attention-mechanism.md）、矩阵乘法、激活函数（ReLU/GELU）、归一化
>
> **参考**：
> - [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（最佳可视化教程）
> - [Harvard NLP - The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)（带代码的完整实现）
> - [Vaswani et al. "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)（原始论文）
> - [Wikipedia - Transformer (deep learning architecture)](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))

---

## 核心问题

**怎么用 Attention 搭建一个完整的语言模型？**

上一章讲了 Attention 机制的数学原理。但 Attention 只是一个组件——它解决"关注哪里"的问题，但不足以构成一个完整模型。你还需要：位置信息编码、非线性变换、残差连接、归一化，以及输入输出的处理。Transformer 就是把这些组件按照一种优雅的方式拼装起来的架构。

---

## 1. 整体结构：Encoder-Decoder vs Decoder-Only

> 可靠程度：Level 1（教科书共识）

原始 Transformer（2017）是 **Encoder-Decoder** 结构，为机器翻译设计：

```
输入（源语言） → [Encoder: N 层] → 中间表示
                                      ↓
输出（目标语言） → [Decoder: N 层] → 预测下一个词
```

- **Encoder**：双向 Self-Attention（每个 token 能看到所有位置），提取输入的完整语义表示
- **Decoder**：因果掩码 Self-Attention（只能看到之前的位置）+ Cross-Attention（关注 Encoder 输出），自回归生成

### 为什么 LLM 大多只用 Decoder-Only？

现代 LLM（GPT 系列、Claude、LLaMA 等）几乎全部采用 **Decoder-Only** 架构——只保留 Decoder 部分，去掉 Encoder 和 Cross-Attention：

| 架构 | 代表模型 | 特点 |
|------|---------|------|
| Encoder-Decoder | T5, BART, 原始 Transformer | 适合翻译、摘要等"输入→输出"任务 |
| Encoder-Only | BERT | 适合理解任务（分类、抽取），不擅长生成 |
| **Decoder-Only** | **GPT, Claude, LLaMA** | **通用生成，统一的"预测下一词"范式** |

Decoder-Only 胜出的原因：

1. **简单统一**：所有任务都可以转化为"给定前文，预测后文"——翻译、问答、摘要都只是不同的 prompt 格式
2. **Scaling 更好**：实验表明在大规模训练下，Decoder-Only 架构的性能和 Encoder-Decoder 相当甚至更好，且更容易扩展
3. **推理高效**：自回归生成天然适合 KV Cache 优化（只需缓存已生成部分的 Key/Value），Encoder-Decoder 需要维护两套

> 可靠程度：Decoder-Only 在实践中胜出是 Level 1 事实。具体原因的理论解释仍在发展中（Level 2-3）。
>
> 参考：[Wang et al. "What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?" (2022)](https://arxiv.org/abs/2204.05832)

---

## 2. 一个 Transformer 层的组成

> 可靠程度：Level 1（教科书共识，直接来自原始论文）

一个 Decoder-Only Transformer 层由以下组件按顺序组成：

```
输入 x
  │
  ├──→ [Multi-Head Self-Attention (Masked)]
  │         │
  │←── + ←──┘   ← 残差连接（Add）
  │
  ├──→ [Layer Normalization]
  │
  ├──→ [Feed-Forward Network (FFN)]
  │         │
  │←── + ←──┘   ← 残差连接（Add）
  │
  ├──→ [Layer Normalization]
  │
  输出
```

逐一解析每个组件。

### 2.1 Multi-Head Self-Attention（已在第 04 章详细推导）

每个 token 通过多头注意力聚合来自序列中其他位置的信息。使用因果掩码确保不看到未来位置。

### 2.2 残差连接（Residual Connection / Add）

$$\text{output} = x + \text{SubLayer}(x)$$

核心思想：让层的输入直接"跳过"该层，和层的输出相加。

**为什么需要残差连接？**

深度网络面临**退化问题**（degradation problem）：随着层数增加，网络不是越来越好，而是训练误差反而增大。理论上更深的网络至少能和浅网络一样好（多出的层做恒等映射即可），但实际训练中梯度难以有效传播到早期层。

残差连接解决了这个问题：
- 梯度可以通过"跳跃连接"直接流回早期层，缓解梯度消失
- 每一层只需要学习**残差** $F(x) = \text{SubLayer}(x)$，即"在输入基础上做多大的修正"，而不是学习完整的变换
- 这使得训练几十甚至上百层的网络成为可能

> 这个思想来自 ResNet（He et al., 2015），被 Transformer 直接采用。
>
> 参考：[He et al. "Deep Residual Learning" (2015)](https://arxiv.org/abs/1512.03385)

### 2.3 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

其中 $\mu$ 和 $\sigma$ 是 $x$ 在**特征维度**上的均值和标准差，$\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数，$\epsilon$ 是防止除零的小常数。

**为什么需要归一化？**

深度网络中，每一层的输入分布会随训练不断变化（internal covariate shift）。归一化将每层的输入重新标准化到均值 0、方差 1 附近，使训练更稳定、收敛更快。

**Layer Norm vs Batch Norm**：Batch Norm 在 batch 维度上归一化（需要足够大的 batch），Layer Norm 在特征维度上归一化（每个样本独立），更适合序列数据和可变长度输入。

**Pre-Norm vs Post-Norm**：原始 Transformer 用 Post-Norm（先 Sublayer 再 Norm），现代 LLM 几乎都用 **Pre-Norm**（先 Norm 再 Sublayer），训练更稳定：

```
Pre-Norm（现代LLM）:   x + SubLayer(LayerNorm(x))
Post-Norm（原始论文）:  LayerNorm(x + SubLayer(x))
```

> 可靠程度：Pre-Norm 更稳定是 Level 1-2（广泛实验验证）。
>
> 参考：[Ba et al. "Layer Normalization" (2016)](https://arxiv.org/abs/1607.06450) · [Xiong et al. "On Layer Normalization in the Transformer Architecture" (2020)](https://arxiv.org/abs/2002.04745)

### 2.4 Feed-Forward Network（FFN）

FFN 是一个简单的两层全连接网络，作用于每个位置（position-wise），各位置参数共享：

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，把 $d_{\text{model}}$ 维映射到 $d_{\text{ff}}$ 维（通常 $d_{\text{ff}} = 4 \times d_{\text{model}}$）
- $\sigma$ 是激活函数（原始论文用 ReLU，现代 LLM 多用 GELU 或 SwiGLU）
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，把 $d_{\text{ff}}$ 维映射回 $d_{\text{model}}$ 维

**FFN 的作用是什么？**

Self-Attention 做的是 token 之间的信息交换和聚合——它擅长"关注什么"，但缺乏对单个 token 表示的非线性变换能力。FFN 弥补了这一点：

- 先升维（$d_{\text{model}} \to d_{\text{ff}}$），在更高维的空间中做非线性变换
- 再降维（$d_{\text{ff}} \to d_{\text{model}}$），压缩回原来的维度

有研究认为 FFN 层充当了"知识存储"的角色——模型在训练中学到的事实性知识（如"巴黎是法国首都"）很大程度上存储在 FFN 的权重中。

> 可靠程度：FFN 作为知识存储是 Level 2-3（有实验支持，但理解不完整）。
>
> 参考：[Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)](https://arxiv.org/abs/2012.14913)

---

## 3. 位置编码（Positional Encoding）

> 可靠程度：正弦编码是 Level 1（原始论文），RoPE 是 Level 1-2（已成为事实标准）

### 为什么需要位置编码？

Self-Attention 的计算本质上是**集合运算**——$QK^T$ 是对所有 token 对做点积，结果和 token 的排列顺序无关。即如果打乱输入序列的顺序，Attention 的输出（在忽略掩码的情况下）不变。

但语言是有顺序的："猫追老鼠"和"老鼠追猫"含义完全不同。所以必须通过某种方式向模型注入位置信息。

### 3.1 正弦位置编码（Sinusoidal PE，原始方案）

原始 Transformer 用固定的正弦/余弦函数生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $pos$ 是 token 位置，$i$ 是维度索引。

这组公式的直觉：每个维度是一个不同频率的正弦波。低维用高频（捕捉局部位置差异），高维用低频（捕捉全局位置关系）。类似于二进制编码中低位变化快、高位变化慢。

位置编码和 token embedding 直接相加：$\text{input} = \text{TokenEmbed}(x) + PE(pos)$。

**优点**：固定函数不需要学习参数；理论上可以外推到训练时未见过的长度。

**缺点**：实践中外推效果有限；加法注入位置信息不够直接。

### 3.2 RoPE（旋转位置编码，现代 LLM 主流方案）

RoPE（Rotary Position Embedding）是 Su et al. (2021) 提出的，被 LLaMA、GPT-NeoX、Qwen 等主流 LLM 广泛采用。

核心思想：不是把位置信息加到 embedding 上，而是在 Attention 计算中，**对 Query 和 Key 向量施加一个和位置相关的旋转**。

对于位置 $m$ 的 Query 向量 $q$ 和位置 $n$ 的 Key 向量 $k$，RoPE 对它们的每两个相邻维度 $(q_{2i}, q_{2i+1})$ 做一个角度为 $m\theta_i$ 的二维旋转：

$$R_m q = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$（和正弦编码类似的频率）。

关键性质：旋转后的 $q$ 和 $k$ 的点积只依赖于**相对位置** $m - n$：

$$(R_m q)^T (R_n k) = q^T R_{m-n}^T k$$

这意味着模型自动学习的是 token 之间的相对距离，而非绝对位置——这更符合语言的性质（"隔两个词"比"在第 5 个位置"更有意义）。

**RoPE 的优势**：
- 天然编码相对位置
- 不增加额外参数
- 通过调节频率基数可以外推到更长的序列（NTK-aware scaling、YaRN 等方法）

> 参考：[Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)](https://arxiv.org/abs/2104.09864)

---

## 4. 从输入到输出的完整数据流

把所有组件串起来，一个 Decoder-Only Transformer LLM 的完整前向传播过程如下：

```
输入文本: "The cat sat on"
      │
      ▼
[1. Tokenizer]
      "The" → 464, "cat" → 2857, "sat" → 3269, "on" → 319
      Token IDs: [464, 2857, 3269, 319]
      │
      ▼
[2. Token Embedding]
      每个 ID 查表得到一个 d_model 维向量
      X ∈ ℝ^{4 × d_model}
      │
      ▼
[3. 位置编码]
      加入位置信息（正弦PE 或 RoPE）
      │
      ▼
[4. Transformer Layer × L]
      重复 L 次（GPT-2: 12次, GPT-3: 96次）:
      ┌─────────────────────────────────┐
      │  LayerNorm                      │
      │  → Multi-Head Self-Attention    │
      │  → Add (残差连接)                │
      │  → LayerNorm                    │
      │  → FFN                          │
      │  → Add (残差连接)                │
      └─────────────────────────────────┘
      │
      ▼
[5. 最终 LayerNorm]
      │
      ▼
[6. Linear (Language Model Head)]
      W_lm ∈ ℝ^{d_model × V}
      把 d_model 维映射到词表大小 V 维
      logits ∈ ℝ^{4 × V}
      │
      ▼
[7. Softmax]
      把 logits 转为概率分布
      P(next_token | context) ∈ ℝ^V
      │
      ▼
      取最后一个位置的概率分布
      → 最可能的下一个词: "the" (0.15), "a" (0.08), ...
```

**关键细节**：

- 实际中步骤 6 的权重 $W_{\text{lm}}$ 通常和 Token Embedding 的权重**共享**（weight tying），减少参数量
- 推理时只需要最后一个 token 位置的输出分布（用来预测下一个词）
- 训练时所有位置的输出都有用——每个位置预测它的下一个词，一次前向传播可以算 $n$ 个预测的 loss

---

## 5. 参数量估算

> 可靠程度：Level 1（直接从架构可推导）

给定层数 $L$、隐藏维度 $d$（即 $d_{\text{model}}$）、FFN 中间维度 $d_{\text{ff}}$、词表大小 $V$，各组件的参数量：

### 每层 Transformer 的参数

**Attention 部分**：$Q$, $K$, $V$ 三个投影矩阵 + 输出投影矩阵 = $4d^2$（忽略 bias）

$$W^Q, W^K, W^V \in \mathbb{R}^{d \times d}, \quad W^O \in \mathbb{R}^{d \times d}$$

$$\text{Attention 参数} = 4d^2$$

**FFN 部分**：两个线性层

$$W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}, \quad W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$$

$$\text{FFN 参数} = 2d \cdot d_{\text{ff}} = 2d \cdot 4d = 8d^2 \quad \text{（通常 } d_{\text{ff}} = 4d \text{）}$$

**LayerNorm**：每层两个 LayerNorm，每个有 $2d$ 个参数（$\gamma$ 和 $\beta$），共 $4d$。相对于 $d^2$ 可忽略。

**每层合计**：$4d^2 + 8d^2 = 12d^2$

### 总参数量

$$\boxed{P \approx 12Ld^2 + Vd}$$

其中 $Vd$ 是 Token Embedding 矩阵的参数。

### 实际验证

| 模型 | $L$ | $d$ | $h$ | $d_{\text{ff}}$ | $V$ | 公式估算 | 实际参数量 |
|------|-----|-----|-----|---------|-----|---------|-----------|
| GPT-2 Small | 12 | 768 | 12 | 3072 | 50257 | $12 \times 12 \times 768^2 + 50257 \times 768 \approx 124\text{M}$ | 124M |
| GPT-2 Large | 36 | 1280 | 20 | 5120 | 50257 | $12 \times 36 \times 1280^2 + 50257 \times 1280 \approx 774\text{M}$ | 774M |
| GPT-3 | 96 | 12288 | 96 | 49152 | 50257 | $12 \times 96 \times 12288^2 + 50257 \times 12288 \approx 175\text{B}$ | 175B |

公式估算和实际值吻合得很好。

**观察**：参数量和 $d^2$ 成正比（主导项），即隐藏维度翻倍 → 参数量翻四倍。$L$ 只是线性增长。所以大模型主要是靠增大 $d$（配合增加 $L$）来做大的。

> 参考：[Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [Brown et al. "Language Models are Few-Shot Learners" (GPT-3)](https://arxiv.org/abs/2005.14165)

---

## 6. GPT 系列的具体配置

> 可靠程度：Level 1（论文中直接给出的数据）

| 配置 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL | GPT-3 |
|------|------------|-------------|------------|---------|-------|
| 层数 $L$ | 12 | 24 | 36 | 48 | 96 |
| 隐藏维度 $d$ | 768 | 1024 | 1280 | 1600 | 12288 |
| 注意力头数 $h$ | 12 | 16 | 20 | 25 | 96 |
| 每头维度 $d_k$ | 64 | 64 | 64 | 64 | 128 |
| FFN 维度 $d_{\text{ff}}$ | 3072 | 4096 | 5120 | 6400 | 49152 |
| 上下文长度 | 1024 | 1024 | 1024 | 1024 | 2048 |
| 词表大小 $V$ | 50257 | 50257 | 50257 | 50257 | 50257 |
| 总参数量 | 124M | 355M | 774M | 1.5B | 175B |

几个规律：

- **$d_k = d / h$**：每个头的维度 = 隐藏维度 / 头数（GPT-2 是 64，GPT-3 是 128）
- **$d_{\text{ff}} = 4d$**：FFN 中间维度是隐藏维度的 4 倍
- **从 GPT-2 到 GPT-3，参数量增长了约 100 倍**，主要来自 $d$ 从 1600 到 12288（约 8 倍，平方后约 60 倍）以及 $L$ 从 48 到 96（2 倍）

更新的模型（LLaMA 2 70B: $L=80$, $d=8192$, $h=64$；LLaMA 3 405B: $L=126$, $d=16384$, $h=128$）延续了类似的设计模式，但在细节上有所变化（如使用 GQA、SwiGLU 等）。

---

## 7. 和 LLM 的联系

Transformer 架构就是 LLM 的骨架。你现在已经理解了一个 LLM 的完整结构：

1. **输入层**：Tokenizer + Embedding + 位置编码 → 把文字变成向量序列
2. **核心计算**：$L$ 个 Transformer 层叠加，每层做 Attention（信息聚合）+ FFN（非线性变换），用残差连接和 LayerNorm 保证训练稳定
3. **输出层**：线性映射 + Softmax → 把向量变回词表上的概率分布

但架构只是骨架——一个随机初始化的 Transformer 输出的是随机的概率分布，不会说话。让它变成一个强大的语言模型，还需要两个关键步骤：

- **预训练**（下一章）：在海量文本上训练"预测下一词"任务，让参数编码语言知识
- **微调与对齐**：通过人类反馈调整模型行为，让它变成有用的助手

> 参考：[Wikipedia - Generative pre-trained transformer](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)

---

## 理解检测

**Q1**：为什么 Transformer 需要残差连接？如果去掉残差连接（只保留 LayerNorm），一个 96 层的模型在训练时会遇到什么困难？

你的回答：



**Q2**：一个 Transformer 层里的 Self-Attention 和 FFN 各自扮演什么角色？如果只保留 Attention 去掉 FFN（或者反过来），模型的能力会怎样受损？

你的回答：



**Q3**：用参数量公式 $P \approx 12Ld^2 + Vd$ 估算：如果你要设计一个 7B 参数的 LLM，$V = 32000$，选择 $L = 32$, $d = 4096$ 是否合理？计算一下。如果要增大到 13B，你会优先增大 $L$ 还是 $d$？为什么？

你的回答：


