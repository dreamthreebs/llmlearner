# 04 - RNN 与 LSTM：处理序列数据

> **学习路径**：MLP 基础 → CNN 基础 → **本章（RNN / LSTM）** → Transformer → 训练的数学
>
> **前置知识**：反向传播（链式法则）、梯度消失问题的直觉（02 中讲过深层 MLP 的情况）、矩阵乘法与特征值
>
> **参考**：
> - [Colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)（最好的 LSTM 直觉讲解）
> - [Wikipedia - Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
> - [Wikipedia - Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
> - [d2l.ai Ch.9 - Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
> - [d2l.ai Ch.10 - Modern Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/index.html)
> - [Hochreiter & Schmidhuber, 1997 - 原始 LSTM 论文](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## 1. 核心问题：变长输入怎么办？

**可靠程度：Level 1（教科书共识）**

到目前为止你学过的网络——MLP 和 CNN——都有一个共同点：**输入大小是固定的**。MLP 接受一个固定维度的向量，CNN 接受一个固定尺寸的图像（比如 $224 \times 224 \times 3$）。

但现实中有大量数据是**序列（sequence）**：

- **文本**："今天天气不错"是 6 个字，"我昨天去了一趟超市买了很多东西"是 14 个字
- **语音**：一段话可能是 2 秒，也可能是 20 秒
- **时间序列**：股票价格可能看过去 10 天，也可能看过去 100 天

序列数据的两个关键特点：

1. **长度不固定**——不同样本的输入长度不同
2. **顺序重要**——"猫吃鱼"和"鱼吃猫"意思完全不同

你不能简单地把序列截断或补齐到固定长度再喂给 MLP（信息丢失或浪费），也不能忽略元素之间的顺序关系。我们需要一种能"逐步读取"序列、同时"记住"之前读过什么的网络结构。

这就是**循环神经网络（Recurrent Neural Network, RNN）**的设计动机。

---

## 2. RNN 的基本结构

**可靠程度：Level 1（教科书共识）**

### 2.1 核心思想：隐状态

RNN 的核心思想非常简单：网络维护一个**隐状态（hidden state）** $\mathbf{h}_t$，这是一个向量，可以理解为网络到目前为止对序列的"记忆"或"摘要"。

在每个时间步 $t$，网络做两件事：

1. 读入当前的输入 $\mathbf{x}_t$（比如序列中的第 $t$ 个词的向量表示）
2. 结合上一步的隐状态 $\mathbf{h}_{t-1}$，计算新的隐状态 $\mathbf{h}_t$

数学上，最简单的 RNN 的更新公式是：

$$\mathbf{h}_t = \tanh(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$$

其中每个符号的含义：

| 符号 | 含义 | 维度（假设隐状态维度 $d_h$，输入维度 $d_x$） |
|------|------|------|
| $\mathbf{x}_t$ | 时间步 $t$ 的输入向量 | $d_x \times 1$ |
| $\mathbf{h}_t$ | 时间步 $t$ 的隐状态 | $d_h \times 1$ |
| $\mathbf{h}_{t-1}$ | 上一个时间步的隐状态 | $d_h \times 1$ |
| $W_h$ | 隐状态到隐状态的权重矩阵（**每一步共享**） | $d_h \times d_h$ |
| $W_x$ | 输入到隐状态的权重矩阵（**每一步共享**） | $d_h \times d_x$ |
| $\mathbf{b}$ | 偏置向量 | $d_h \times 1$ |
| $\tanh$ | 激活函数，把输出压到 $(-1, 1)$ 之间 | — |

关键点：**$W_h$、$W_x$、$\mathbf{b}$ 在所有时间步都是同一组参数**——这就是"循环"（recurrent）的含义。网络在每一步用相同的规则处理输入，只是隐状态 $\mathbf{h}_t$ 在不断变化。

### 2.2 直觉类比：滚动的笔记本

想象你在读一本小说。你手边有一个**小笔记本**（隐状态 $\mathbf{h}_t$），空间有限。每读完一页（输入 $\mathbf{x}_t$）：

1. 你看一眼笔记本上之前的笔记（$\mathbf{h}_{t-1}$）
2. 结合这一页的新内容（$\mathbf{x}_t$），在笔记本上更新你的笔记（得到 $\mathbf{h}_t$）

因为笔记本空间有限，你不能把每一页都抄下来——你必须提炼、压缩、保留最重要的信息。这就是为什么隐状态是一个固定维度的向量：它是对"到目前为止整个序列"的压缩摘要。

读完整本书后，最终的隐状态 $\mathbf{h}_T$（$T$ 是序列长度）就是你对整本书的理解。你可以用这个最终的隐状态来做分类（比如判断情感）、生成下一个词，等等。

### 2.3 展开图（Unrolled Diagram）

RNN 可以用两种方式画：**折叠**和**展开**。

**折叠视图**——强调参数共享：

```
        ┌──────────────┐
        │              │
        │   ┌──────┐   │
  x_t ──┤──►│ RNN  │───┤──► y_t
        │   │ Cell │   │
        │   └──────┘   │
        │       ▲      │
        │       │      │
        └───h_{t-1}────┘
              h_t
```

**展开视图**——把序列的每个时间步画出来（每一步用**相同**的 $W_h, W_x, \mathbf{b}$）：

```
  x_1         x_2         x_3         x_4
   │           │           │           │
   ▼           ▼           ▼           ▼
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ RNN  │──►│ RNN  │──►│ RNN  │──►│ RNN  │
│ Cell │   │ Cell │   │ Cell │   │ Cell │
└──────┘   └──────┘   └──────┘   └──────┘
   │           │           │           │
   ▼           ▼           ▼           ▼
  y_1         y_2         y_3         y_4

h_0 ──► h_1 ──► h_2 ──► h_3 ──► h_4
```

展开视图揭示了一个重要事实：**展开后的 RNN 看起来就像一个很深的前馈网络**，只是每一层用的是同一组权重。这个观察对理解训练非常关键。

---

## 3. RNN 的训练：BPTT（Back-Propagation Through Time）

**可靠程度：Level 1（教科书共识）**

### 3.1 基本思路

RNN 的训练叫做 **BPTT（Back-Propagation Through Time, 通过时间的反向传播）**。思路其实很直接：

1. 把 RNN 按时间步**展开**成一个深层网络（如上面的展开图）
2. 对这个展开后的网络做标准的反向传播
3. 因为每一步用的是同一组 $W_h, W_x, \mathbf{b}$，所以把所有时间步贡献的梯度**加起来**作为这些参数的总梯度

假设序列长度为 $T$，损失函数是每一步损失的总和 $\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$，那么我们需要计算 $\frac{\partial \mathcal{L}}{\partial W_h}$。

### 3.2 梯度消失与爆炸：老问题的新面孔

这里出现了一个你在学深层 MLP 时已经见过的问题。考虑损失 $\mathcal{L}_T$（最后一步的损失）对 $\mathbf{h}_1$（第一步的隐状态）的梯度。根据链式法则：

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \cdot \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_{T-1}} \cdot \frac{\partial \mathbf{h}_{T-1}}{\partial \mathbf{h}_{T-2}} \cdots \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}$$

每一项 $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ 涉及对 $\tanh(W_h \mathbf{h}_{t-1} + \ldots)$ 求导，结果大致正比于 $W_h$（乘上 $\tanh$ 的导数，$\tanh$ 的导数在 $[0, 1]$ 之间）。

所以梯度中包含了 $W_h$ 的**连乘**：

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} \propto \prod_{t=2}^{T} \text{diag}(\tanh'(\cdot)) \cdot W_h$$

**用特征值理解**：如果 $W_h$ 的最大特征值（的绝对值）记为 $|\lambda_{\max}|$，那么：

- $|\lambda_{\max}| > 1$：连乘 $T-1$ 次 → 梯度**指数爆炸** → 参数更新变得巨大，训练不稳定
- $|\lambda_{\max}| < 1$：连乘 $T-1$ 次 → 梯度**指数消失** → 早期时间步对损失几乎没有影响，网络"记不住"远处的信息

**具体例子**：假设 $|\lambda_{\max}| = 0.9$，序列长度 $T = 100$。那么梯度经过 99 步传播后，衰减到 $0.9^{99} \approx 2.6 \times 10^{-5}$——几乎为零。网络根本无法学到第 1 个词和第 100 个词之间的依赖关系。

反过来，如果 $|\lambda_{\max}| = 1.1$，梯度放大到 $1.1^{99} \approx 1.2 \times 10^{4}$——一万倍！参数更新会变得极不稳定。

这就是为什么**基础 RNN 在实践中无法处理长序列**。你需要一种方法，让梯度能在长时间跨度上稳定传播。

> 参考：[Bengio et al., 1994 - Learning Long-Term Dependencies with Gradient Descent is Difficult](https://ieeexplore.ieee.org/document/279181) · [d2l.ai - BPTT](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)

---

## 4. LSTM（Long Short-Term Memory）

**可靠程度：Level 1（教科书共识）**

### 4.1 核心思想：引入传送带

LSTM 由 Hochreiter & Schmidhuber 在 1997 年提出，目的就是解决基础 RNN 的梯度消失问题。

LSTM 的核心创新：在隐状态 $\mathbf{h}_t$ 之外，引入一个新的向量——**细胞状态（cell state）** $\mathbf{c}_t$。

**直觉类比**：想象一条**传送带**穿过整个序列。传送带上的物品（$\mathbf{c}_t$）可以不经任何修改地从头传到尾。在每个时间步，有三个操作员（三个"门"）控制：

1. **是否扔掉传送带上的一些物品**（遗忘门）
2. **是否往传送带上放新物品**（输入门）
3. **从传送带上取什么物品作为这一步的输出**（输出门）

因为传送带上的物品可以不被修改地长距离传输，所以信息（和梯度）可以跨越很多时间步而不衰减。

### 4.2 三个门的数学

LSTM 中的"门"（gate）是一个向量，每个元素在 $[0, 1]$ 之间，通过 sigmoid 函数 $\sigma$ 生成。门的值为 0 表示"完全阻止"，为 1 表示"完全通过"。门和某个向量做**逐元素乘法**（记为 $\odot$），控制信息的流通。

下面逐一定义三个门。为了简洁，我把 $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ 记为把上一步隐状态和当前输入**拼接**成一个长向量。

#### 遗忘门（Forget Gate）

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

- $W_f$ 是遗忘门的权重矩阵，$\mathbf{b}_f$ 是偏置
- $\sigma$ 是 sigmoid 函数，输出在 $(0, 1)$ 之间
- **作用**：$\mathbf{f}_t$ 的每个元素决定细胞状态 $\mathbf{c}_{t-1}$ 的对应元素保留多少。$f_t^{(i)} = 1$ 表示完全保留，$f_t^{(i)} = 0$ 表示完全丢弃

**例子**：如果网络在做语言模型，读到了一个新的主语，遗忘门可能会"忘掉"之前的主语信息（因为语法上主语已经变了）。

#### 输入门（Input Gate）

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

- $\mathbf{i}_t$：输入门，决定哪些新信息要写入细胞状态
- $\tilde{\mathbf{c}}_t$：**候选细胞状态**，是"想要写入的新信息"，用 $\tanh$ 把值压到 $(-1, 1)$

**两步分工**：$\tilde{\mathbf{c}}_t$ 决定"写什么"，$\mathbf{i}_t$ 决定"写多少"。

#### 细胞状态更新

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

这是 LSTM 最核心的公式。它说的是：

- 先用遗忘门 $\mathbf{f}_t$ 决定旧细胞状态 $\mathbf{c}_{t-1}$ 保留多少（$\mathbf{f}_t \odot \mathbf{c}_{t-1}$）
- 再用输入门 $\mathbf{i}_t$ 决定写入多少新信息（$\mathbf{i}_t \odot \tilde{\mathbf{c}}_t$）
- 两者相加得到新的细胞状态 $\mathbf{c}_t$

注意这里是**加法**操作（而不是 RNN 中的矩阵乘法），这对梯度传播至关重要（下文会解释）。

#### 输出门（Output Gate）

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

- $\mathbf{o}_t$：输出门，决定细胞状态的哪些部分作为这一步的输出
- $\mathbf{h}_t$：当前时间步的隐状态（对外可见的输出），是对细胞状态做 $\tanh$ 后再用输出门筛选

### 4.3 LSTM 完整公式汇总

把所有公式放在一起：

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(遗忘门：保留多少旧信息)}$$

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(输入门：写入多少新信息)}$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(候选新信息)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(更新细胞状态)}$$

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(输出门：输出哪些信息)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(隐状态 / 输出)}$$

### 4.4 为什么 LSTM 能缓解梯度消失？

回到细胞状态的更新公式：

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

考虑 $\mathcal{L}$ 对 $\mathbf{c}_k$（某个早期时间步的细胞状态）的梯度。沿着细胞状态这条路径传播：

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t) + \ldots$$

关键观察：梯度沿细胞状态传播时，乘的是**遗忘门** $\mathbf{f}_t$ 的值（加上一些其他项），而不是 $W_h$。

在基础 RNN 中，梯度必须反复乘以 $W_h$——一个固定的矩阵，其特征值要么 > 1（爆炸）要么 < 1（消失）。

在 LSTM 中，梯度沿细胞状态传播时乘的是 $\mathbf{f}_t$——**这是一个可学习的、可以接近 1 的值**。当网络判断某个信息需要长期保留时，遗忘门会输出接近 1 的值，梯度就能几乎无损地传过去。

用传送带类比：基础 RNN 就像每传一站都要经过一台复印机（$W_h$），复印 99 次后文字要么糊掉（消失）要么放大到看不清（爆炸）。LSTM 的细胞状态就像一条传送带，物品直接传过去，只在需要的时候才做修改。

> 参考：[Colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) · [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## 5. GRU（Gated Recurrent Unit）

**可靠程度：Level 1（教科书共识）**

GRU（Cho et al., 2014）是 LSTM 的简化版本。它的核心思想和 LSTM 一样——用门控制信息流——但做了两个简化：

1. **合并细胞状态和隐状态**：GRU 没有单独的 $\mathbf{c}_t$，只有 $\mathbf{h}_t$
2. **合并遗忘门和输入门**：用一个"更新门" $\mathbf{z}_t$ 同时控制"忘多少"和"记多少"

GRU 的公式：

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) \quad \text{(更新门)}$$

$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) \quad \text{(重置门)}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h) \quad \text{(候选隐状态)}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(更新隐状态)}$$

各符号含义：

| 符号 | 含义 |
|------|------|
| $\mathbf{z}_t$（更新门） | 控制"旧信息保留多少 vs 新信息写入多少"。$z=0$ → 完全保留旧信息；$z=1$ → 完全用新信息替换 |
| $\mathbf{r}_t$（重置门） | 控制"计算候选新信息时，参考多少旧信息"。$r=0$ → 忽略旧信息，从头开始；$r=1$ → 完全参考旧信息 |
| $\tilde{\mathbf{h}}_t$ | 候选隐状态——"如果要写入新信息，写什么" |

注意最后一行的巧妙设计：$(1 - \mathbf{z}_t)$ 和 $\mathbf{z}_t$ 加起来恒等于 1，所以遗忘和输入是互补的——忘得多就记得多，忘得少就记得少。LSTM 的遗忘门和输入门是独立的，更灵活但也更贵。

**GRU vs LSTM 的选择**：在实践中性能通常相当。GRU 参数更少（3 组权重 vs LSTM 的 4 组）、计算更快，在数据量较小时可能更有优势。LSTM 更灵活，在长序列上有时更好。

> 参考：[Cho et al., 2014 - Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) · [Wikipedia - Gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

---

## 6. RNN 的局限与 Transformer 的取代

**可靠程度：Level 1-2**

尽管 LSTM/GRU 解决了梯度消失问题，RNN 架构还有一些根本性的局限：

### 6.1 无法并行计算

RNN 的本质特征是：$\mathbf{h}_t$ 依赖 $\mathbf{h}_{t-1}$，而 $\mathbf{h}_{t-1}$ 又依赖 $\mathbf{h}_{t-2}$……这是一个**严格串行**的依赖链。

```
h_1 → h_2 → h_3 → h_4 → ... → h_T
（必须按顺序计算，不能跳过或并行）
```

在训练时，一个长度为 $T$ 的序列需要 $T$ 步串行计算。当 $T$ 很大（比如几千个词）时，GPU 的并行能力完全浪费了——GPU 最擅长的是同时做大量独立的计算（矩阵乘法），而不是一步一步串行执行。

### 6.2 长距离依赖仍然困难

虽然 LSTM 缓解了梯度消失，但信息仍然需要**逐步传递**。从 $\mathbf{x}_1$ 到 $\mathbf{h}_T$，信息要经过 $T-1$ 步传递——每一步都有丢失的风险。

实践中，LSTM 在序列长度超过几百步后，性能就开始下降。

### 6.3 Transformer 的解决方案

Transformer（Vaswani et al., 2017）用 **Attention 机制**从根本上改变了处理序列的方式：

| 问题 | RNN 的方式 | Transformer 的方式 |
|------|-----------|-------------------|
| 访问远处信息 | 经过 $T-1$ 步传递 | **直接连接**——任意两个位置的注意力距离为 1 |
| 并行计算 | 必须串行 | **完全并行**——所有位置同时计算注意力 |
| 长序列 | 性能随长度退化 | 更稳定（虽然注意力计算量是 $O(T^2)$） |

Transformer 不再维护"隐状态"，而是让序列中的每个位置直接"看到"所有其他位置，通过学习"注意力权重"来决定关注哪里。

这就是为什么 2017 年之后，RNN/LSTM 在大多数序列任务上被 Transformer 取代了。你已经在 LLM 的学习中接触了 Transformer 的细节（LLM/04-05），这里不再重复。

> 参考：[Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Wikipedia - Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))

---

## 7. 总结

| 模型 | 核心机制 | 优点 | 缺点 |
|------|---------|------|------|
| 基础 RNN | 隐状态 $\mathbf{h}_t = f(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t)$ | 概念简单、可处理变长序列 | 梯度消失/爆炸，无法学长依赖 |
| LSTM | 细胞状态 $\mathbf{c}_t$ + 三个门 | 缓解梯度消失，能学较长的依赖 | 参数多、计算串行 |
| GRU | 合并门，取消独立细胞状态 | 比 LSTM 更轻量，性能相当 | 灵活性略低于 LSTM |
| Transformer | Attention 直接连接所有位置 | 可并行，长依赖处理好 | 注意力 $O(T^2)$，需要位置编码 |

**历史线索**：基础 RNN（1980s）→ LSTM（1997）→ GRU（2014）→ Transformer（2017）。每一步的进化都是为了解决前一代的核心缺陷。

---

## 理解检测

**Q1**：基础 RNN 中梯度消失的根本原因是什么？LSTM 是怎么解决这个问题的？请从梯度传播路径上的"乘什么"这个角度来对比两者。

你的回答：



**Q2**：假设你有一个 LSTM，遗忘门在所有时间步都输出 $\mathbf{f}_t = \mathbf{1}$（全 1 向量），输入门都输出 $\mathbf{i}_t = \mathbf{0}$（全 0 向量）。这时候细胞状态 $\mathbf{c}_t$ 会怎么变化？这对应什么实际含义？

你的回答：



**Q3**：RNN 必须串行计算是因为"$\mathbf{h}_t$ 依赖 $\mathbf{h}_{t-1}$"。Transformer 为什么不需要串行？它用什么机制替代了"逐步传递信息"这个过程？如果 Transformer 也需要知道位置信息（第 3 个词 vs 第 100 个词），它是怎么做到的？（提示：回忆 LLM/04-05 中学过的内容）

你的回答：


