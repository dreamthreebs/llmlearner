# 04 - Attention 机制——LLM 的灵魂

> **学习路径**：神经网络基础 → 词嵌入 → **Attention 机制** → Transformer 架构 → 预训练 → 微调与对齐
>
> **前置知识**：线性代数（矩阵乘法、内积）、softmax 函数、神经网络基本结构（线性层、激活函数）
>
> **参考**：
> - [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（最佳可视化教程）
> - [Vaswani et al. "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)（原始论文）
> - [Wikipedia - Attention (machine learning)](https://en.wikipedia.org/wiki/Attention_(machine_learning))
> - [Lilian Weng - Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)（综合博客）

---

## 核心问题

**处理序列时，怎么让模型知道该"关注"哪些位置？**

语言是序列数据——一句话里的每个词的含义依赖于上下文中其他词。比如 "苹果发布了新手机"和"苹果很好吃"中的"苹果"含义完全不同，由后面的词决定。模型必须有一种机制，在处理每个词时动态地选择去参考序列中哪些其他位置的信息。

Attention 机制就是这个问题的答案。它是 Transformer 的核心，也是现代 LLM 之所以强大的关键。

---

## 1. 序列建模的挑战：RNN 的思路和瓶颈

在 Attention 出现之前，处理序列数据的标准方案是 **RNN（循环神经网络）**。

### RNN 的核心思路

RNN 按时间步逐个处理序列中的元素，维护一个"隐藏状态" $h_t$ 来记录已看过的信息：

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

其中 $x_t$ 是第 $t$ 步的输入，$h_{t-1}$ 是上一步的隐藏状态。信息像接力赛一样，从前往后逐步传递。

### 两个致命瓶颈

**瓶颈一：梯度消失/爆炸**

> 可靠程度：Level 1（教科书共识）

当序列很长时，反向传播需要经过很多时间步。梯度在连乘过程中要么指数衰减（消失），要么指数增长（爆炸）。结果是：RNN 很难学习长距离依赖关系。

比如 "我出生在法国，……（省略 200 个词）……所以我的母语是\_\_"——RNN 在预测"法语"时，"法国"这个信息经过了 200 步传递，梯度已经衰减到几乎为零，模型学不到这个依赖。

LSTM 和 GRU 通过门控机制缓解了这个问题，但并没有根本解决——实际有效的依赖距离通常在几百个词左右。

**瓶颈二：无法并行**

RNN 必须按顺序逐步计算——$h_2$ 依赖 $h_1$，$h_3$ 依赖 $h_2$……无法利用 GPU 的并行计算能力。这使得训练大规模模型极其缓慢。

> 参考：[Wikipedia - Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) · [Wikipedia - Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

---

## 2. Attention 的直觉：Query-Key-Value

Attention 机制的核心思想可以用**图书馆查询**来类比：

- 你带着一个**问题（Query）**走进图书馆
- 图书馆里每本书都有一个**索引标签（Key）**
- 你用自己的问题去和每个标签做**匹配**，找出最相关的书
- 然后读取那些书的**内容（Value）**，匹配度越高的书你读得越仔细

映射到数学：

| 概念 | 含义 | 向量 |
|------|------|------|
| Query ($Q$) | "我在找什么信息" | 当前位置的查询向量 |
| Key ($K$) | "我包含什么信息" | 每个位置的索引向量 |
| Value ($V$) | "我能提供什么内容" | 每个位置的内容向量 |

对于序列中的每个位置，模型会：
1. 用它的 Query 和所有位置的 Key 做比较（计算相关度）
2. 用相关度作为权重，对所有位置的 Value 做加权求和
3. 得到一个"综合了相关信息"的输出向量

关键区别于 RNN：**每个位置可以直接访问序列中的任何其他位置**，信息传递的路径长度是 $O(1)$，而不是 RNN 的 $O(n)$。

---

## 3. Scaled Dot-Product Attention：完整推导

> 可靠程度：Level 1（教科书共识，Vaswani et al. 2017 原始定义）

### 3.1 输入准备

假设输入序列有 $n$ 个 token，每个 token 已经被表示为一个 $d_{\text{model}}$ 维的向量（来自 Embedding 层）。把它们排成矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$。

用三个可学习的线性变换（即三个矩阵）把 $X$ 映射为 $Q$, $K$, $V$：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$。

这一步的意义：同一个输入向量扮演 Query、Key、Value 三个不同角色时，可以强调不同的信息维度。$W^Q$, $W^K$, $W^V$ 是通过训练学到的。

### 3.2 计算注意力分数

第一步：计算 Query 和 Key 之间的相似度。

$$\text{score} = QK^T \in \mathbb{R}^{n \times n}$$

$QK^T$ 的第 $(i,j)$ 个元素就是第 $i$ 个 token 的 Query 向量和第 $j$ 个 token 的 Key 向量的**点积**。点积越大，表示两者越"匹配"。

为什么用点积？它是衡量两个向量相似度最简单高效的方式：$q \cdot k = \|q\| \|k\| \cos\theta$，方向相近的向量点积大。

### 3.3 缩放（除以 $\sqrt{d_k}$）

$$\text{scaled\_score} = \frac{QK^T}{\sqrt{d_k}}$$

**为什么要除以 $\sqrt{d_k}$？** 这是一个关键的工程细节：

当 $d_k$ 很大时，$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个随机变量的求和。如果 $q_i, k_i$ 独立同分布，均值为 0，方差为 1，则 $q \cdot k$ 的方差为 $d_k$（各项方差叠加）。

这意味着当 $d_k = 512$ 时，点积的典型值可能达到 $\pm 20$ 量级。Softmax 函数在输入值很大时会趋于 one-hot 分布（一个值接近 1，其余接近 0），梯度会变得极小（饱和区），训练不动。

除以 $\sqrt{d_k}$ 后，缩放后的点积方差变为 1，数值落在 Softmax 的敏感区域，梯度正常传播。

> 这是一个"让数学 works"的归一化技巧，类似于 Batch Normalization 的思想。
>
> 参考：[Vaswani et al. 2017, Section 3.2.1](https://arxiv.org/abs/1706.03762)

### 3.4 Softmax 归一化

$$\alpha = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Softmax 作用于每一行（即对每个 Query，将它和所有 Key 的匹配分数归一化为概率分布）：

$$\alpha_{ij} = \frac{\exp(\text{scaled\_score}_{ij})}{\sum_{l=1}^{n} \exp(\text{scaled\_score}_{il})}$$

$\alpha_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的**注意力权重**。所有权重为非负且每行求和为 1。

### 3.5 加权求和

$$\text{Attention}(Q,K,V) = \alpha V = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$\alpha V$ 的第 $i$ 行 = $\sum_j \alpha_{ij} v_j$，即第 $i$ 个 token 的输出是所有 Value 向量的加权平均，权重就是注意力分数。

### 完整公式总结

$$\boxed{\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

**每一步做了什么**：

| 步骤 | 操作 | 含义 |
|------|------|------|
| $QK^T$ | 点积 | 计算每对 token 的相关度 |
| $\div \sqrt{d_k}$ | 缩放 | 控制数值范围，防止 softmax 饱和 |
| softmax | 归一化 | 转为概率分布（注意力权重） |
| $\times V$ | 加权求和 | 用注意力权重聚合信息 |

---

## 4. Multi-Head Attention：多个"头"

> 可靠程度：Level 1（教科书共识）

### 为什么一个 Attention 头不够？

一个 Attention 头只能学习一种"关注模式"。但语言中的依赖关系是多维度的：

- 语法依赖："猫 追 老鼠" → 主语-谓语-宾语
- 语义依赖："法国 …… 母语" → 远距离语义关联
- 指代依赖："他 说 他 不去" → 代词指代

用多个 Attention 头，让每个头独立学习不同类型的关注模式，然后合并结果。

### 实现方式

Multi-Head Attention 有 $h$ 个头，每个头有自己独立的 $W^Q_i$, $W^K_i$, $W^V_i$：

$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

每个头的维度是 $d_k = d_v = d_{\text{model}} / h$。把所有头的输出拼接起来，再过一个线性层：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

其中 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ 是输出映射矩阵。

**参数量分析**：把 $d_{\text{model}}$ 维拆成 $h$ 个 $d_k$ 维的小空间，总计算量和单个 $d_{\text{model}}$ 维的 Attention 基本相同，但表达能力大幅增强。

> 原论文用了 $h = 8$, $d_{\text{model}} = 512$, $d_k = d_v = 64$。GPT-3 用了 $h = 96$, $d_{\text{model}} = 12288$, $d_k = 128$。
>
> 参考：[Vaswani et al. 2017, Section 3.2.2](https://arxiv.org/abs/1706.03762)

---

## 5. Self-Attention vs Cross-Attention

这两个术语经常出现，区别很简单：

**Self-Attention**：$Q$, $K$, $V$ 都来自**同一个**序列。每个 token 关注同一序列中的其他 token。

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V \quad \text{（X 是同一个输入）}$$

用途：理解一个句子内部各个词之间的关系。**这是 LLM（Decoder-Only Transformer）的唯一注意力类型。**

**Cross-Attention**：$Q$ 来自一个序列，$K$ 和 $V$ 来自**另一个**序列。

$$Q = X_{\text{decoder}} W^Q, \quad K = X_{\text{encoder}} W^K, \quad V = X_{\text{encoder}} W^V$$

用途：翻译任务中，Decoder 在生成目标语言词时，通过 Cross-Attention 关注源语言句子。也用于多模态模型中文本关注图像特征。

> LLM 中基本只用 Self-Attention（因为是 Decoder-Only 架构，没有单独的 Encoder 输入）。Cross-Attention 在 Encoder-Decoder 模型（如 T5、机器翻译模型）和多模态模型中使用。

---

## 6. Masked Attention（因果掩码）

> 可靠程度：Level 1（教科书共识）

### 为什么 LLM 需要掩码？

LLM 的训练目标是"给定前面的词，预测下一个词"——这意味着**在预测第 $t$ 个词时，模型只能看到位置 1 到 $t-1$，不能偷看后面的词**。

但标准 Self-Attention 会让每个位置关注**所有**位置（包括未来）。所以需要一个**因果掩码（causal mask）**来屏蔽未来位置。

### 实现方式

在 softmax 之前，给未来位置的分数加上 $-\infty$（实现中通常用一个很大的负数如 $-10^9$）：

$$\text{MaskedAttention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M$ 是一个上三角矩阵：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \quad \text{（可以看到）} \\ -\infty & \text{if } j > i \quad \text{（不能看到）} \end{cases}$$

加上 $-\infty$ 后，softmax 会把对应位置的权重变成 0（因为 $e^{-\infty} = 0$），从而完全屏蔽未来信息。

**结果**：位置 1 只能看自己；位置 2 能看位置 1 和自己；位置 3 能看 1、2 和自己……以此类推。这正好对应"给定前文预测下一词"的约束。

---

## 7. 计算复杂度分析

> 可靠程度：Level 1（可直接从公式推导）

Self-Attention 的核心计算是 $QK^T$——这是两个 $n \times d_k$ 矩阵的乘法，复杂度是 $O(n^2 d_k)$。

| 操作 | 矩阵维度 | 复杂度 |
|------|---------|--------|
| $QK^T$ | $(n \times d_k) \times (d_k \times n)$ | $O(n^2 d_k)$ |
| softmax | $n \times n$ 矩阵 | $O(n^2)$ |
| $\alpha V$ | $(n \times n) \times (n \times d_v)$ | $O(n^2 d_v)$ |
| **总计** | | $O(n^2 d)$ |

关键：**复杂度和序列长度 $n$ 的平方成正比**。

实际影响：

- $n = 1024$（GPT-2 上下文长度）：注意力矩阵约 $10^6$ 个元素 → 没问题
- $n = 8192$（现代 LLM）：约 $6.7 \times 10^7$ 个元素 → 还能接受
- $n = 128000$（Claude 的上下文）：约 $1.6 \times 10^{10}$ 个元素 → 需要特殊优化（稀疏注意力、Flash Attention 等）

这就是为什么"长上下文"是一个重要研究方向——序列越长，Attention 的计算和内存消耗越大。

> 参考：[Tay et al. "Efficient Transformers: A Survey" (2022)](https://arxiv.org/abs/2009.06732)

---

## 8. 和 LLM 的联系

Self-Attention 就是 Transformer 的核心计算模块。每一层 Transformer 都包含一个 Multi-Head Self-Attention（加上 FFN），一个 LLM 就是几十到上百个这样的层叠在一起。

Attention 给了 LLM 三个关键能力：

1. **灵活的上下文理解**：每个 token 可以根据需要关注不同的位置，形成动态的信息聚合
2. **并行计算**：不像 RNN 需要逐步计算，Attention 的矩阵运算可以完全并行化，充分利用 GPU
3. **长距离依赖**：任意两个 token 之间的信息路径长度为 $O(1)$（直接计算注意力），而 RNN 需要 $O(n)$ 步

正是这些优势让 Transformer 彻底取代了 RNN/LSTM，成为 LLM 的标准架构。

下一章将讲解如何用 Attention 搭建完整的 Transformer 架构。

---

## 理解检测

**Q1**：在 Scaled Dot-Product Attention 中，如果去掉 $\sqrt{d_k}$ 的缩放（即直接用 $QK^T$ 送入 softmax），模型训练会出什么问题？为什么？

你的回答：



**Q2**：假设一个 Self-Attention 层有 $h=8$ 个头，$d_{\text{model}} = 512$。每个头的 $d_k$ 是多少？如果改为 $h=16$ 个头（$d_{\text{model}}$ 不变），每个头的表达能力和整体表达能力各有什么变化？

你的回答：



**Q3**：为什么 LLM 需要 Masked Attention 而不是让每个 token 看到所有位置？如果在训练时不用因果掩码，模型会学到什么"错误"的能力？

你的回答：


