# 03 - 词嵌入：文字怎么变成数字？

> **学习路径**：`数学基础` → `神经网络基础` → **词嵌入** → `Attention` → `Transformer` → `预训练` → `微调与对齐`
>
> **前置知识**：神经网络基础（02 章）、线性代数（向量空间、内积）、概率论（条件概率、最大似然）
>
> **参考**：[Jay Alammar - The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) · [Wikipedia - Word2Vec](https://en.wikipedia.org/wiki/Word2vec) · [Wikipedia - Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)

---

## 核心问题

神经网络只能处理数字（向量、矩阵），但自然语言是离散的符号（"猫"、"dog"、"跑"）。第一个要解决的问题：

**怎么把文字变成数字，而且让这种表示"有意义"？**

"有意义"的要求很关键——我们希望语义相近的词在数学空间中也"距离近"。比如"猫"和"狗"应该比"猫"和"民主"距离更近。

---

## 最朴素的方案：One-Hot 编码

假设词表有 $V$ 个词。给每个词分配一个编号 $i \in \{1, 2, \ldots, V\}$，然后用一个 $V$ 维向量表示它，只有第 $i$ 个位置是 1，其余全是 0：

$$\text{"猫"} \to [0, 0, 1, 0, \ldots, 0]^T \in \mathbb{R}^V$$
$$\text{"狗"} \to [0, 0, 0, 1, \ldots, 0]^T \in \mathbb{R}^V$$

### 三个致命问题

1. **维度灾难**：实际词表大小 $V$ 通常在 30,000–100,000。每个词是一个极高维的稀疏向量。

2. **没有语义信息**：任意两个 one-hot 向量都是正交的：$\mathbf{e}_i^T \mathbf{e}_j = 0 \,(\forall\, i \neq j)$。"猫"和"狗"的距离 = "猫"和"民主"的距离。模型无法从表示本身获得任何语义先验。

3. **无法泛化**：如果模型学会了"猫坐在垫子上"是合理的句子，它无法推断"狗坐在垫子上"也是合理的，因为"猫"和"狗"在 one-hot 空间中没有任何相似性。

我们需要一种**低维、稠密、能编码语义关系**的表示——这就是词嵌入（Word Embedding）。[Level 1]

---

## 分布式假设：一个深刻的语言学洞察

> "You shall know a word by the company it keeps." — J.R. Firth, 1957

**分布式假设**（Distributional Hypothesis）：一个词的含义可以由它出现的上下文来决定。

如果"猫"和"狗"经常出现在相似的上下文中（"\_\_坐在垫子上"、"我养了一只\_\_"、"\_\_在追蝴蝶"），那么它们的语义就是相近的。

这个假设把"语义相似性"这个模糊的概念转化成了一个**可计算的统计量**：共现频率。这是所有词嵌入方法的理论基石。[Level 1]

> 参考：[Wikipedia - Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics) · Harris, Z. S. (1954). "Distributional structure."

---

## Word2Vec：用神经网络学词向量

Word2Vec（Mikolov et al., 2013）是第一个大规模成功的词嵌入方法。核心思想：**训练一个浅层神经网络来预测上下文，作为副产品得到每个词的向量表示。**

它有两种架构：

### CBOW（Continuous Bag of Words）：用上下文预测中心词

给定窗口大小 $m$，上下文词 $\{w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}\}$，预测中心词 $w_t$。

模型结构：

1. 查找每个上下文词的嵌入向量：$\mathbf{v}_{w_i} \in \mathbb{R}^d$
2. 取平均：$\bar{\mathbf{v}} = \frac{1}{2m}\sum_{j \in \text{context}} \mathbf{v}_{w_j}$
3. 用另一个嵌入矩阵 $U$ 计算得分：$\text{score}(w) = \mathbf{u}_w^T \bar{\mathbf{v}}$
4. 通过 softmax 得到概率：

$$P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^T \bar{\mathbf{v}})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w^T \bar{\mathbf{v}})}$$

训练目标是最大化语料库上所有窗口的对数似然：

$$\mathcal{L} = \sum_{t=1}^{T} \log P(w_t \mid w_{t-m}, \ldots, w_{t+m})$$

### Skip-gram：用中心词预测上下文

反过来：给定中心词 $w_t$，预测它的每个上下文词 $w_{t+j}$（$j \in [-m, m], j \neq 0$）。

$$P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^T \mathbf{v}_{w_t})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w^T \mathbf{v}_{w_t})}$$

训练目标：

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{\substack{j=-m \\ j \neq 0}}^{m} \log P(w_{t+j} \mid w_t)$$

### 训练技巧：负采样（Negative Sampling）

上面的 softmax 需要对整个词表 $V$（几万到十万维）求和——计算量太大。解决方案是**负采样**（Mikolov et al., 2013）：

把问题从"预测正确的词"转化为"区分正确的词和随机采样的错误词"。对于每个正样本 $(w_t, w_{t+j})$，随机采 $k$ 个负样本（噪声词），目标变为：

$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_{t+j}}^T \mathbf{v}_{w_t}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_t})\right]$$

其中 $\sigma$ 是 sigmoid 函数，$P_n(w)$ 是噪声分布（通常取词频的 3/4 次幂 $P_n(w) \propto f(w)^{3/4}$）。这把每步训练的计算量从 $O(V)$ 降到了 $O(k)$（$k$ 通常取 5–20）。[Level 1]

### 两个嵌入矩阵

注意 Word2Vec 实际有两个嵌入矩阵：
- $V_{\text{input}} \in \mathbb{R}^{d \times |V|}$（当词作为输入/中心词时使用）
- $U_{\text{output}} \in \mathbb{R}^{d \times |V|}$（当词作为输出/上下文词时使用）

训练完成后，通常取 $V_{\text{input}}$ 作为最终的词向量，或者取两者的平均。

> 参考：Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space." [arXiv:1301.3781](https://arxiv.org/abs/1301.3781) · Mikolov et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)

---

## 词向量的几何性质

训练出来的词向量展现出令人惊讶的结构性质：

### 语义算术

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

这不是巧合。"king - man"提取出了"皇室"的语义方向，加上"woman"就得到了"女性皇室成员"。

其他例子：
- $\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Italy}} \approx \mathbf{v}_{\text{Rome}}$（首都关系）
- $\mathbf{v}_{\text{walking}} - \mathbf{v}_{\text{walk}} + \mathbf{v}_{\text{swim}} \approx \mathbf{v}_{\text{swimming}}$（语法变化）

### 为什么会这样？

直觉解释：Word2Vec 的目标函数迫使向量编码共现统计信息。如果 "king" 和 "queen" 出现在高度相似的上下文中，但分别和 "man"/"woman" 共现，那么 $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{queen}}$ 和 $\mathbf{v}_{\text{man}} - \mathbf{v}_{\text{woman}}$ 就编码了相同的"性别"方向。

更严格地说，Levy & Goldberg (2014) 证明了 Skip-gram 加负采样实际上在隐式分解一个 PMI（逐点互信息）矩阵：

$$\mathbf{v}_w^T \mathbf{u}_c \approx \text{PMI}(w, c) - \log k$$

其中 $\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)P(c)}$ 衡量两个词共现的程度超过独立假设的多少。这把神经网络方法和传统的统计方法联系了起来。[Level 1-2]

> 参考：Levy & Goldberg (2014). "Neural Word Embedding as Implicit Matrix Factorization." [NIPS 2014](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html) · [Wikipedia - Word2Vec: Result](https://en.wikipedia.org/wiki/Word2vec#Result)

---

## 从静态到上下文化：词嵌入的演进

Word2Vec 有一个根本局限：**每个词只有一个向量，无论上下文是什么。**

但自然语言中，同一个词在不同上下文中含义可能完全不同：
- "我去**银行**取钱" → 金融机构
- "他坐在河**岸**（bank）上" → 河岸

Word2Vec 只能给 "bank" 一个向量，混合了所有含义。

### 演进路线

| 阶段 | 代表模型 | 嵌入类型 | 核心思想 |
|------|---------|---------|---------|
| 2013 | Word2Vec, GloVe | 静态嵌入 | 每个词一个固定向量 |
| 2018 | ELMo | 上下文化嵌入 | 用双向 LSTM 生成依赖上下文的向量 |
| 2018 | BERT | 深度上下文化 | 用 Transformer 编码器，双向注意力 |
| 2018+ | GPT 系列 | 深度上下文化 | 用 Transformer 解码器，单向（自回归） |

**ELMo**（Peters et al., 2018）：在大量文本上训练双向 LSTM 语言模型。同一个词在不同句子中会得到不同的向量，因为 LSTM 的隐藏状态编码了上下文信息。[Level 1]

**BERT**（Devlin et al., 2019）：用 Transformer 架构替代 LSTM。训练方式是"遮住一些词，让模型预测被遮住的词"（Masked Language Model）。BERT 的每一层都会输出上下文化的词表示。[Level 1]

**GPT 系列**：也用 Transformer，但训练方式是"预测下一个词"（自回归语言模型）。架构上只用 Transformer 解码器，attention 只能看前面的词（causal attention）。[Level 1]

关键区别：静态嵌入是一个**查表操作**，上下文化嵌入是**整个网络的计算结果**。现代 LLM 中，虽然输入层仍然有一个静态的嵌入查找表（把 token ID 映射到初始向量），但这个初始向量会经过几十层 Transformer 的处理，变成高度依赖上下文的表示。

> 参考：Peters et al. (2018). "Deep contextualized word representations." [arXiv:1802.05365](https://arxiv.org/abs/1802.05365) · Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

---

## Tokenization：怎么切词？

在进入嵌入层之前，还有一个问题：输入文本怎么切成"词元"（token）？

### 为什么不能按空格切词？

- 中文没有空格
- 词表太大（英语的词形变化：run, runs, running, ran…）
- 无法处理新词（OOV, Out-of-Vocabulary）

### BPE（Byte Pair Encoding）

现代 LLM 几乎都用 BPE 或其变体。核心思想：**从字符开始，反复合并最频繁的相邻对，直到词表达到目标大小。**

算法步骤：

1. 初始词表 = 所有单个字符（+ 特殊标记）
2. 统计训练语料中所有相邻 token 对的频率
3. 合并最高频的对，加入词表
4. 重复步骤 2-3，直到词表大小达到预设值（如 32,000 或 50,000）

例子：假设语料中 "l" 和 "o" 经常相邻出现，就合并为 "lo"。然后 "lo" 和 "w" 经常相邻，就合并为 "low"。

**结果**：常用词（如 "the"）会被保留为完整 token，罕见词（如 "tokenization"）会被切成子词片段（如 "token" + "ization"）。这在词表大小和覆盖率之间取得了好的平衡。[Level 1]

GPT 系列使用的 BPE 词表大小通常在 50,000–100,000 之间。

> 参考：Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) · [Wikipedia - Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)

---

## 和 LLM 的联系：Embedding 是 Transformer 的入口

在一个现代 LLM（如 GPT）中，处理一段输入文本的流程是：

```
"The cat sat on the mat"
     ↓ Tokenizer (BPE)
[464, 3797, 3332, 319, 262, 2603]     ← token ID 序列
     ↓ Embedding 查表
[v_464, v_3797, v_3332, v_319, v_262, v_2603]   ← d 维向量序列
     ↓ + 位置编码（下一章）
     ↓ Transformer 层 × N
     ↓ 上下文化的表示
```

嵌入层是一个可训练的矩阵 $E \in \mathbb{R}^{d \times V}$，其中 $d$ 是嵌入维度（如 768, 4096），$V$ 是词表大小。查表操作就是取 $E$ 的某一列。

**这个嵌入矩阵是随机初始化的，和 Word2Vec 不同——它在 LLM 的预训练过程中从零学习。** 但学到的嵌入空间仍然展现出类似的结构性质（语义相近的 token 距离近），因为 Transformer 的训练目标（预测下一个 token）同样蕴含了分布式假设。

LLM 的嵌入矩阵参数量 = $d \times V$。例如 GPT-3 的 $d = 12288$，$V = 50257$，所以仅嵌入层就有约 6 亿参数。但这在 GPT-3 的 1750 亿总参数中只占很小一部分——绝大多数参数在 Transformer 层中。

> 参考：Radford et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2 论文) · [Andrej Karpathy - Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 本章小结

| 概念 | 要点 | 可靠程度 |
|------|------|---------|
| One-hot 编码 | 稀疏、高维、不编码语义 | L1 |
| 分布式假设 | 词的含义由上下文定义 | L1 |
| Word2Vec | 用浅层网络从共现统计学词向量 | L1 |
| 词向量算术 | 语义关系被编码为向量方向 | L1 |
| PMI 联系 | Skip-gram ≈ 隐式 PMI 矩阵分解 | L1-2 |
| 上下文化嵌入 | ELMo/BERT/GPT：同一词在不同上下文中有不同表示 | L1 |
| BPE | 从字符出发合并频繁对，平衡词表大小和覆盖率 | L1 |
| LLM 嵌入层 | 可训练矩阵，token ID → 向量，Transformer 的入口 | L1 |

下一章我们进入 LLM 的灵魂——**Attention 机制**：当你有了一个 token 向量序列，怎么让每个 token "看到"并"关注"序列中最相关的其他 token？

---

## 理解检测

**Q1**：为什么 one-hot 向量不适合作为神经网络的词表示？如果你有一个 50,000 词的词表，用 one-hot 表示每个词需要多少维？用 Word2Vec（嵌入维度 300）呢？除了效率问题，更根本的缺陷是什么？

你的回答：



**Q2**：Skip-gram 的训练目标是"给定中心词，预测上下文词"。假设训练语料中，"猫"和"狗"经常出现在相似的上下文中（"我养了一只\_\_"、"\_\_在院子里跑"）。训练完成后，$\mathbf{v}_{\text{猫}}$ 和 $\mathbf{v}_{\text{狗}}$ 的余弦相似度 $\cos(\mathbf{v}_{\text{猫}}, \mathbf{v}_{\text{狗}})$ 会是高还是低？请用 Skip-gram 的目标函数解释为什么。

你的回答：



**Q3**："bank"这个词有"银行"和"河岸"两个完全不同的含义。Word2Vec 对这种情况会怎么处理？BERT 会怎么处理？两者的根本区别是什么？

你的回答：


