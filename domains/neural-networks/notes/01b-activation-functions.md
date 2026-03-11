# 01b - 激活函数：为什么非线性如此重要

> **主维度**：D1 基础架构 + D3 训练数学
>
> **学习路径**：全景概览 → MLP 基础 → **本章（激活函数）** → 深入 MLP → CNN → ...
>
> **前置知识**：神经元的基本结构（加权求和 + 激活函数）、反向传播的直觉
>
> **参考**：
> - [Deep Learning Book §6.3 - Hidden Units](https://www.deeplearningbook.org/contents/mlp.html)
> - [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function)
> - [d2l.ai §5.1 - 多层感知机](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

---

## 1. 为什么需要激活函数

### 1.1 没有激活函数，深度网络等于一层

一个两层线性网络做的事情是：

$$
\mathbf{y} = W_2 (W_1 \mathbf{x}) = (W_2 W_1) \mathbf{x} = W' \mathbf{x}
$$

两个矩阵乘起来还是一个矩阵。无论你堆多少层，没有非线性的话，整个网络等价于一个单层线性变换。100 层线性网络 = 1 层线性网络。

**激活函数的作用**：在每层线性变换之后加一个非线性函数 $\sigma$，打破这个"矩阵连乘 = 一个矩阵"的规律：

$$
\mathbf{h} = \sigma(W \mathbf{x} + \mathbf{b})
$$

有了非线性，每增加一层就能表达更复杂的函数形状。万能近似定理说：只要一个隐藏层配上非线性激活函数，MLP 就能拟合任意连续函数（虽然可能需要无穷多个神经元）。

### 1.2 对激活函数的基本要求

一个好的激活函数需要满足：

| 要求 | 原因 |
|------|------|
| **非线性** | 否则堆层没意义 |
| **处处可导**（或几乎处处可导） | 反向传播需要计算导数 |
| **导数不能太小** | 否则梯度消失 |
| **导数不能太大** | 否则梯度爆炸 |
| **计算简单** | 每个神经元每次前向/反向都要算，必须快 |

下面按历史顺序介绍主要的激活函数，你会看到后面的函数是如何针对前面的缺陷改进的。

> **可靠程度**：Level 1（教科书共识）

---

## 2. Sigmoid

### 2.1 定义与图形

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**输出范围**：$(0, 1)$

**图形直觉**：一条 S 形曲线。$x$ 很负时输出接近 0，$x$ 很正时输出接近 1，$x = 0$ 时输出 0.5。

```
输出
1.0 |                    _______________
    |                 /
    |               /
0.5 |  - - - - - -/- - - - - - - - - -
    |            /
    |          /
0.0 |_________/
    +---------+---------+---------+---→ x
          -5       0        5
```

### 2.2 导数

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

这个导数有一个优美的性质：它可以用 $\sigma(x)$ 自己来表示，不需要重新算 $x$。

**导数的最大值**：在 $x = 0$ 处，$\sigma(0) = 0.5$，所以 $\sigma'(0) = 0.5 \times 0.5 = 0.25$。

**关键问题：最大导数只有 0.25**。这意味着每经过一层 Sigmoid，梯度**至少**缩小到原来的 $\frac{1}{4}$。

### 2.3 为什么 Sigmoid 导致梯度消失

10 层网络，每层 Sigmoid 的导数按最大值算：

$$
0.25^{10} \approx 10^{-6}
$$

梯度缩小了一百万倍。前面的层几乎收不到任何梯度信号，参数不更新，**网络只有最后几层在学习**。

而且实际情况比 0.25 还糟：当 $|x|$ 较大时，Sigmoid 进入**饱和区**（输出接近 0 或 1），导数趋近于 0。训练中很多神经元的输入 $x$ 并不在 0 附近，导数远小于 0.25。

### 2.4 Sigmoid 的用途

虽然 Sigmoid 不适合做隐藏层的激活函数，但它在两个地方仍然常用：

1. **二分类的输出层**：把任意实数映射到 $(0, 1)$，解释为概率
2. **LSTM 中的门控**：遗忘门、输入门、输出门都用 Sigmoid，因为门的值就应该在 $[0, 1]$ 之间（0 = 关，1 = 开）

> **参考**：[Wikipedia - Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)

---

## 3. Tanh

### 3.1 定义

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
$$

**输出范围**：$(-1, 1)$

**和 Sigmoid 的关系**：Tanh 本质上就是 Sigmoid 的缩放平移版本，形状一样，但**以零为中心**。

### 3.2 相比 Sigmoid 的改进

| 性质 | Sigmoid | Tanh |
|------|---------|------|
| 输出范围 | $(0, 1)$ | $(-1, 1)$ |
| 是否零中心 | 否（输出始终 > 0） | **是** |
| 最大导数 | 0.25 | **1.0** |

**零中心为什么重要**：如果激活函数的输出始终为正（Sigmoid），那么传给下一层的梯度要么全正要么全负，参数更新只能沿着"对角线"方向走——效率很低，收敛慢。Tanh 输出有正有负，缓解了这个问题。

**最大导数为 1.0**：比 Sigmoid 的 0.25 好很多。但当 $|x|$ 大时，$\tanh'(x)$ 仍然趋近于 0（饱和问题依然存在）。

### 3.3 梯度消失仍未解决

虽然最大导数是 1，但**只在 $x = 0$ 附近才接近 1**。大多数情况下导数仍然小于 1，经过多层连乘后依然会消失。Tanh 缓解了问题，但没有根本解决。

> **参考**：[Wikipedia - Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)

---

## 4. ReLU（Rectified Linear Unit）

### 4.1 定义

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

**输出范围**：$[0, +\infty)$

**图形**：

```
输出
    |            /
    |          /
    |        /
    |      /
    |    /
    |  /
0   |/_______________
    +---------+---------→ x
         0
```

### 4.2 导数

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

（$x = 0$ 处严格来说不可导，实践中通常定义为 0 或 1，没有影响。）

### 4.3 为什么 ReLU 缓解了梯度消失

对比一下三种激活函数在正区间上的导数：

| 激活函数 | 正区间导数 | 连乘 10 层 |
|---------|-----------|-----------|
| Sigmoid | 最大 0.25，通常更小 | $\leq 0.25^{10} \approx 10^{-6}$ |
| Tanh | 最大 1.0，通常 < 1 | $< 1^{10} = 1$，但实际会衰减 |
| **ReLU** | **恒等于 1** | $1^{10} = 1$，**梯度完全不衰减** |

ReLU 在正半轴上导数恒为 1，梯度传过去不缩不放，完美。这就是 ReLU 成为深度学习默认激活函数的核心原因。

另外，ReLU 计算极其简单：只需要一个 `max` 操作，比 Sigmoid/Tanh 的指数运算快得多。

### 4.4 ReLU 的问题：Dead Neurons

ReLU 有一个缺陷：**当输入 $x < 0$ 时，输出恒为 0，导数也恒为 0**。

如果某个神经元的权重更新后，导致它对所有训练样本的输入都变成了负数，那么：
- 前向：输出永远是 0
- 反向：梯度永远是 0 → 参数再也不更新
- 这个神经元就**"死了"**（dead neuron）

在实践中，如果学习率设得太大，大量神经元可能同时"死亡"，导致网络容量骤降。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Rectifier (neural networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) · [Glorot et al., 2011 - "Deep Sparse Rectifier Neural Networks"](http://proceedings.mlr.press/v15/glorot11a.html)

---

## 5. ReLU 的变体

为了解决 Dead Neuron 问题，人们提出了几种改进：

### 5.1 Leaky ReLU

$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

其中 $\alpha$ 是一个很小的正数（通常 $\alpha = 0.01$）。

**改进点**：负半轴不再输出 0，而是一个很小的负值。这样导数是 $\alpha$（不是 0），神经元不会"死"。

### 5.2 GELU（Gaussian Error Linear Unit）

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

**直觉**：GELU 根据输入值的"大小"来**概率性地**决定是否保留。大的正值几乎完全保留（$\Phi(x) \approx 1$），大的负值几乎完全丢弃（$\Phi(x) \approx 0$），中间区域平滑过渡。

**GELU 是 Transformer/GPT/BERT 的标准激活函数**。它比 ReLU 更平滑（处处可导），在 Transformer 的 FFN 层中表现更好。

### 5.3 Swish / SiLU

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**和 GELU 非常相似**，曲线形状几乎一样。Google Brain 提出，在某些任务上比 ReLU 效果更好。

### 5.4 总结对比

| 激活函数 | 正区间 | 负区间 | 平滑？ | 典型用途 |
|---------|--------|--------|--------|---------|
| Sigmoid | 压缩到 (0,1) | 压缩到 (0,1) | 是 | 二分类输出、LSTM 门控 |
| Tanh | 压缩到 (-1,1) | 压缩到 (-1,1) | 是 | RNN 隐状态 |
| ReLU | $y = x$ | $y = 0$ | 否 | CNN、MLP（默认选择） |
| Leaky ReLU | $y = x$ | $y = 0.01x$ | 否 | CNN（防 dead neuron） |
| GELU | $\approx x$ | $\approx 0$（平滑过渡） | 是 | **Transformer / LLM** |
| Swish | $\approx x$ | 小负值 | 是 | EfficientNet 等 |

> **可靠程度**：Level 1-2（ReLU/GELU 是工业标准，Swish 是 Level 2）
>
> **参考**：
> - GELU：[Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)
> - Swish：[Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)

---

## 6. 激活函数与梯度问题的关系

把激活函数放到梯度传播的大图里看：

反向传播时，每一层贡献的梯度因子大致是：

$$
\frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} = \text{diag}(\sigma'(\mathbf{z}_l)) \cdot W_l
$$

其中 $\sigma'(\mathbf{z}_l)$ 是激活函数的导数。所以整体梯度的"放大/缩小因子"由两部分决定：

1. **$\sigma'$（激活函数的导数）**：Sigmoid 把它压到 $\leq 0.25$，ReLU 让它等于 1
2. **$W_l$（权重矩阵的尺度）**：Xavier/He 初始化控制这一项

### 常见误解：用了 ReLU 就没有梯度问题了？

不是。ReLU 只解决了 $\sigma'$ 这一项（正区间导数恒为 1），但每层梯度因子是 $\sigma' \cdot W_l$ 的**乘积**。$W_l$ 仍然可以让梯度爆炸或消失：

| 情况 | $\sigma'$（ReLU） | $W_l$ | 结果 |
|------|-------------------|-------|------|
| 权重初始化太大 | 1 | 特征值 > 1 | $1 \times (>1)^n$ → **梯度爆炸** |
| 权重初始化太小 | 1 | 特征值 < 1 | $1 \times (<1)^n$ → **梯度消失** |
| 输入落在负半轴 | **0** | 任意 | $0 \times W = 0$ → **dead neuron**（另一种梯度消失） |

所以 ReLU 不是单独出现的，而是和其他技术**成套使用**：

| 技术 | 解决梯度传播中的哪个环节 |
|------|----------------------|
| ReLU | $\sigma'$ 不压缩（正区间 = 1） |
| He 初始化 | 控制 $W_l$ 的尺度，让每层方差稳定 |
| 残差连接 | 给梯度一条不经过连乘的"高速公路"（$+\mathbf{I}$） |
| LayerNorm | 稳定每层的输出分布，防止训练中漂移 |

**四者缺一不可**——这就是为什么现代深层网络（如 Transformer）总是同时用 GELU + 残差连接 + LayerNorm + 精心设计的初始化。

> **可靠程度**：Level 1（教科书共识）

---

## 7. 实际选择指南

如果你在纠结用哪个激活函数：

1. **默认用 ReLU**——简单、快、在绝大多数场景下够用
2. **Transformer 系列用 GELU**——GPT、BERT、ViT 都用 GELU
3. **RNN/LSTM 的隐状态用 Tanh**——需要输出在 $(-1, 1)$ 之间
4. **门控机制用 Sigmoid**——LSTM 的门、注意力权重的归一化
5. **如果发现大量 dead neuron，换 Leaky ReLU**

不需要纠结太久——激活函数的选择对最终性能的影响通常远小于架构设计、数据质量和训练策略。

---

### 公式速查卡

| 函数 | 公式 | 导数 | 导数范围 |
|------|------|------|---------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | $(0, 0.25]$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | $(0, 1]$ |
| ReLU | $\max(0, x)$ | $0 \text{ (x<0)}, 1 \text{ (x>0)}$ | $\{0, 1\}$ |
| Leaky ReLU | $\max(\alpha x, x)$ | $\alpha \text{ (x<0)}, 1 \text{ (x>0)}$ | $\{\alpha, 1\}$ |

**关键数值**：$\sigma(10) \approx 0.9999$，$\sigma'(10) \approx 0.0001$，$\sigma(0) = 0.5$，$\sigma'(0) = 0.25$

---

## 理解检测

**Q1**：假设你有一个 3 层 MLP，每层用 Sigmoid 激活函数，输入是 $x = 10$。请估算第一层的梯度大约会缩小多少倍。

> 提示：先查速查卡 $\sigma'(10) \approx ?$，然后 3 层连乘 = $(\sigma'(10))^3 \approx ?$

你的回答：



**Q2**：ReLU 在 $x = 0$ 处不可导。为什么这在实践中不是问题？

你的回答：



**Q3**：GELU 和 ReLU 的核心区别是什么？为什么 Transformer 倾向于用 GELU 而不是 ReLU？（提示：想想 ReLU 在 0 处导数的突变对梯度流有什么影响。）

你的回答：


