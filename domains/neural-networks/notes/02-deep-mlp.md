# 02 - 深入 MLP：让深层网络能训练起来

> **主维度**：D1 基础架构 + D3 训练数学
> **关键关系**：MLP `is-instance-of` 万能近似器 · BatchNorm/残差连接 `used-for` 解决深层训练困难
>
> **学习路径**：LLM/02 基础 → **本章** → CNN → RNN/LSTM → 训练数学
>
> **前置知识**：神经元、MLP 结构、损失函数、反向传播、梯度下降（LLM/02 已覆盖）
>
> **参考**：
> - [Goodfellow 等《Deep Learning》Ch.6-8](https://www.deeplearningbook.org/)
> - [d2l.ai Ch.5 - 深度学习计算](https://d2l.ai/chapter_deep-learning-computation/index.html)
> - [d2l.ai Ch.12 - 优化算法](https://d2l.ai/chapter_optimization/index.html)

---

## 引言：为什么"堆层"不够？

你已经知道 MLP 的基本结构：输入层 → 若干隐藏层（每层做线性变换 + 非线性激活） → 输出层。理论上，只有一个隐藏层的 MLP 就能拟合任意连续函数（万能近似定理）。但实践中，**更深的网络往往比更宽的网络更高效**——用更少的参数就能表达更复杂的函数。

直觉上可以这样理解：深层网络可以逐层构建越来越抽象的特征。比如识别人脸时，第一层检测边缘，第二层组合边缘成局部纹理，第三层组合纹理成五官，第四层组合五官判断是谁。这种**层次化表示**是深度网络的核心优势。

然而，**直接把 MLP 堆到 50 层、100 层，你会发现网络根本训不动**。损失函数下降极其缓慢甚至完全不动，或者突然飙到无穷大。这不是因为理论错了，而是因为训练过程中出现了几个致命的数值问题。

本章覆盖的就是这些问题和解决方案。它们不是 MLP 独有的——**Transformer 中每一层都用了残差连接和 Layer Normalization，Adam 是训练 LLM 的标准优化器**。理解这些技术是理解现代深度学习的基础。

> **可靠程度**：本章内容大部分是教科书共识（Level 1），BatchNorm 的有效机制仍有争议（Level 2-3）。

---

## 1. 梯度消失与梯度爆炸

### 1.1 问题的根源：链式法则中的连乘

反向传播的数学本质是链式法则。设一个 $L$ 层网络每层的变换是 $\mathbf{h}_l = f_l(\mathbf{h}_{l-1})$，那么损失 $\mathcal{L}$ 对第 $l$ 层参数的梯度涉及这样一个连乘：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \cdot \frac{\partial \mathbf{h}_{L-1}}{\partial \mathbf{h}_{L-2}} \cdots \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l}
$$

其中：
- $\mathbf{h}_l$ 是第 $l$ 层的输出（激活值）
- $\frac{\partial \mathbf{h}_{k+1}}{\partial \mathbf{h}_k}$ 是每一层的**局部雅可比矩阵**（Jacobian），描述第 $k$ 层输出的微小变化如何影响第 $k+1$ 层的输出
- 整个梯度是 $L - l$ 个矩阵连乘的结果

### 1.2 为什么会指数衰减或爆炸

为了建立直觉，先看一个极简的标量例子。假设每层的梯度传播因子是一个常数 $\alpha$（对应雅可比矩阵的"有效乘子"），那么经过 $n$ 层后，梯度变为：

$$
\text{梯度} \propto \alpha^n
$$

| 每层因子 $\alpha$ | 经过 50 层 | 现象 |
|---|---|---|
| $0.9$ | $0.9^{50} \approx 0.005$ | 梯度缩小 200 倍 → **梯度消失** |
| $0.5$ | $0.5^{50} \approx 10^{-15}$ | 梯度几乎为零 → 前面的层完全学不动 |
| $1.1$ | $1.1^{50} \approx 117$ | 梯度增大 100 多倍 → **梯度爆炸** |
| $2.0$ | $2.0^{50} \approx 10^{15}$ | 梯度飞到天文数字 → 参数更新直接 NaN |

只要每层的梯度传播因子不精确等于 1，经过足够多层，梯度要么消失要么爆炸。这就是深层网络难以训练的核心原因。

### 1.3 和激活函数的关系

Sigmoid 激活函数 $\sigma(x) = 1/(1+e^{-x})$ 的导数最大值只有 0.25（在 $x=0$ 处）。这意味着每过一层，梯度至少缩小到原来的 1/4。10 层就缩小到 $0.25^{10} \approx 10^{-6}$。这是早期深层网络训不动的重要原因之一。

**ReLU**（Rectified Linear Unit）$f(x) = \max(0, x)$ 的导数是 0 或 1，不会像 Sigmoid 那样持续压缩梯度，因此在一定程度上缓解了梯度消失。但仅靠换激活函数是不够的——我们还需要更多技术。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Deep Learning Book §6.2 - Gradient-Based Learning](https://www.deeplearningbook.org/contents/mlp.html) · [Wikipedia - Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

---

## 2. 权重初始化

### 2.1 为什么不能全部初始化为 0

如果把所有权重都设为 0，那么每个神经元接收到的输入完全相同（都是 0），输出也完全相同，反向传播时得到的梯度也完全相同。结果：所有神经元在整个训练过程中始终保持一致——它们永远是同一个神经元的 $n$ 份拷贝。

这叫**对称性问题**（symmetry breaking problem）。初始化的第一个要求就是**打破对称性**：让不同神经元从不同的起点开始，这样它们才能分化成不同的特征检测器。

### 2.2 为什么不能用太大或太小的随机数

打破对称性只需要随机初始化。但随机的**尺度**很重要：

- **太大**：每层输出的值域（方差）逐层放大 → 到了深层，激活值变得非常大或非常小 → 梯度爆炸或进入激活函数的饱和区（Sigmoid 输出接近 0 或 1，梯度接近 0）
- **太小**：每层输出的方差逐层缩小 → 到了深层，所有激活值都接近 0 → 网络退化为近似线性，失去非线性表达能力

关键思路是：**让每层输出的方差在逐层传播时保持稳定**。

### 2.3 Xavier 初始化

Xavier 初始化（也叫 Glorot 初始化）就是基于上述思路推导出来的。考虑一层线性变换（先忽略激活函数）：

$$
y = w_1 x_1 + w_2 x_2 + \cdots + w_{n_{\text{in}}} x_{n_{\text{in}}}
$$

其中 $n_{\text{in}}$ 是输入维度，$x_i$ 是输入，$w_i$ 是权重。假设 $x_i$ 和 $w_i$ 都是均值为 0 的独立随机变量，那么输出 $y$ 的方差为：

$$
\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

（这里用到了"独立随机变量乘积的方差 = 各自方差的乘积"这一性质，以及"和的方差 = 方差的和"——因为 $w_i$ 和 $x_i$ 独立且均值为 0。）

要让输出方差等于输入方差（$\text{Var}(y) = \text{Var}(x)$），我们需要：

$$
\boxed{\text{Var}(w) = \frac{1}{n_{\text{in}}}}
$$

实践中，Xavier 初始化从以下分布中采样权重：

$$
w \sim \mathcal{N}\!\left(0,\; \frac{1}{n_{\text{in}}}\right) \quad \text{或} \quad w \sim U\!\left[-\sqrt{\frac{3}{n_{\text{in}}}},\; \sqrt{\frac{3}{n_{\text{in}}}}\right]
$$

（均匀分布的范围是为了让方差等于 $1/n_{\text{in}}$，因为 $U[-a, a]$ 的方差是 $a^2/3$。）

> 更精确的版本同时考虑前向和反向传播，取 $\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$。

### 2.4 He 初始化

Xavier 初始化的推导假设激活函数在 0 附近近似线性（对 tanh 成立）。但 **ReLU** 会把约一半的输出变成 0（负半轴全部截断），这意味着有效方差减半。

He 初始化（以 Kaiming He 命名）对此做了修正：

$$
\boxed{\text{Var}(w) = \frac{2}{n_{\text{in}}}}
$$

多出来的因子 2 正是补偿 ReLU 截断掉一半激活值带来的方差损失。

> **可靠程度**：Level 1（教科书共识，广泛使用）
>
> **参考**：
> - Xavier 初始化：[Glorot & Bengio, 2010 - "Understanding the difficulty of training deep feedforward neural networks"](http://proceedings.mlr.press/v9/glorot10a.html)
> - He 初始化：[He et al., 2015 - "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852)

---

## 3. Batch Normalization

### 3.1 问题：内部协变量偏移

即使初始化做得好，训练过程中每层的参数在不断更新，导致**每层的输入分布在不断变化**。后面的层需要不断适应前面层输出的新分布——这被称为 **internal covariate shift**（内部协变量偏移）。

想象你在学射箭，但靶子每隔几秒就换一个位置——你很难练好。Batch Normalization（简称 BN）的直觉就是：**把靶子钉住**，让每层的输入分布保持稳定。

### 3.2 数学操作

对于一个 mini-batch 的数据 $\{x_1, x_2, \ldots, x_m\}$，BN 对每个特征维度分别做以下操作：

**第一步：标准化（normalize）**

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

其中：
- $\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$ 是 mini-batch 的均值
- $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$ 是 mini-batch 的方差
- $\epsilon$ 是一个很小的常数（如 $10^{-5}$），防止除以零

这一步把分布拉回均值 0、方差 1。

**第二步：可学习的仿射变换（scale and shift）**

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\gamma$（缩放）和 $\beta$（偏移）是**可学习参数**，通过反向传播和其他权重一起更新。

为什么还要加这一步？因为强制让所有层的输出都是均值 0、方差 1 太过约束——有些情况下，网络**需要**某层的输出有特定的均值和方差（比如 Sigmoid 激活前，你可能希望输入集中在非饱和区域）。$\gamma$ 和 $\beta$ 让网络可以自己学出最合适的分布。

> 特殊情况：如果网络学出 $\gamma = \sigma_B$ 和 $\beta = \mu_B$，那么 BN 层就退化为恒等映射，等于什么都没做。所以 BN 不会损失网络的表达能力。

### 3.3 为什么有效

原始论文（Ioffe & Szegedy, 2015）的解释是减少了 internal covariate shift。但后续研究（Santurkar et al., 2018）发现 BN 的效果主要来自**平滑损失面**——让损失函数的梯度变化更平滑，从而允许使用更大的学习率，加速收敛。

目前学术界的共识是：BN 有效，但确切原因仍不完全清楚。

> **可靠程度**：BN 的操作和效果 Level 1-2，BN 为什么有效的理论解释 Level 2-3

### 3.4 Layer Normalization

BN 依赖 mini-batch 的统计量，这有两个问题：
1. batch size 太小时统计量不稳定
2. 在序列模型（如 Transformer）中，不同序列长度不同，batch 的概念不太自然

**Layer Normalization**（简称 LN）的解决方案：不在 batch 维度上统计，而在**特征维度**上统计。也就是说，对单个样本的所有特征求均值和方差，然后做标准化。

| 归一化方式 | 统计维度 | 适用场景 |
|---|---|---|
| Batch Norm | 同一特征跨所有样本 | CNN、图像任务 |
| Layer Norm | 同一样本的所有特征 | Transformer、序列任务 |

**Transformer 中每一层都使用 Layer Normalization**。这是你在 LLM/04-05 中学到的 Transformer 结构的关键组件之一。

> **参考**：
> - BatchNorm：[Ioffe & Szegedy, 2015 - "Batch Normalization: Accelerating Deep Network Training"](https://arxiv.org/abs/1502.03167)
> - BN 有效原因的质疑：[Santurkar et al., 2018 - "How Does Batch Normalization Help Optimization?"](https://arxiv.org/abs/1805.11604)
> - Layer Norm：[Ba et al., 2016 - "Layer Normalization"](https://arxiv.org/abs/1607.06450)

---

## 4. 残差连接（Skip Connection / Residual Connection）

### 4.1 核心思想

残差连接是深度学习中最重要的结构创新之一，来自 He et al. 2015 年的 ResNet 论文。

普通网络中，每一层试图学一个完整的映射 $\mathbf{h} = f(\mathbf{x})$。残差连接的思想是：**让每一层只学输入和输出之间的"差异"（残差）**，然后把输入直接加回来：

$$
\boxed{\mathbf{h} = f(\mathbf{x}) + \mathbf{x}}
$$

其中：
- $\mathbf{x}$ 是这一层的输入
- $f(\mathbf{x})$ 是这一层的变换（比如 "线性 → BN → ReLU → 线性 → BN"）
- $\mathbf{h}$ 是这一层的输出

"残差"这个名字来自 $f(\mathbf{x}) = \mathbf{h} - \mathbf{x}$，即 $f$ 学的是"输出相对于输入还差什么"。

### 4.2 为什么这解决了梯度消失

考虑反向传播时梯度的传播。对于残差块 $\mathbf{h} = f(\mathbf{x}) + \mathbf{x}$，梯度为：

$$
\frac{\partial \mathbf{h}}{\partial \mathbf{x}} = \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{I}
$$

其中 $\mathbf{I}$ 是单位矩阵。关键在于这个 **$+\mathbf{I}$**：即使 $\frac{\partial f}{\partial \mathbf{x}}$ 很小（梯度消失），梯度至少还有 $\mathbf{I}$ 这一项。这为梯度提供了一条**高速公路**——梯度可以直接通过 skip connection "跳过"这一层，一路畅通地传回网络前端。

如果有 $L$ 个残差块，梯度可以通过 $2^L$ 条不同路径传播（每一层可选择"穿过"或"跳过"），其中最短的路径完全由 skip connection 组成，梯度不经过任何变换，不会衰减。

### 4.3 ResNet 的设计思想

ResNet（Residual Network，He et al., 2015）把残差连接应用到 CNN 中，成功训练了 152 层的网络（之前超过 20 层就很难训练了），并赢得了 2015 年 ImageNet 竞赛。

ResNet 的核心观察：**如果额外的层只需要学习恒等映射（什么都不做），那么带残差连接的网络只需要让 $f(\mathbf{x}) = 0$ 就行了**——学"什么都不加"比学"原样输出"简单得多。这意味着加深网络至少不会让性能变差（worst case 是多余的层什么都不做）。

### 4.4 和 Transformer 的联系

在你学过的 Transformer 中，**每个 Attention 子层和每个 FFN 子层都有残差连接**：

$$
\text{output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))
$$

没有残差连接，几十层的 Transformer 根本训不起来。

> **可靠程度**：Level 1（教科书共识，深度学习的基石技术）
>
> **参考**：[He et al., 2015 - "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) · [Wikipedia - Residual neural network](https://en.wikipedia.org/wiki/Residual_neural_network)

---

## 5. 优化器细节

你已经知道**梯度下降**（Gradient Descent）的基本思想：沿着损失函数梯度的反方向更新参数。但最简单的梯度下降（vanilla SGD）在实践中有很多问题：收敛慢、在"峡谷"型损失面上来回振荡、对学习率敏感。本节介绍两个关键改进。

### 5.1 SGD with Momentum

**Momentum**（动量）的物理类比非常直接——你作为物理背景的学习者会很熟悉。

想象一个小球在损失面上滚动。没有动量时，小球在每个位置都严格沿着当前的最陡方向移动（纯梯度）。加上动量后，小球有了**惯性**：它不再每步都急转弯，而是受之前运动方向的影响。这让它在平坦方向上积累速度，在振荡方向上因为方向交替正负而自然抵消。

数学上，引入一个**速度变量** $\mathbf{v}$：

$$
\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_{t-1})
$$

$$
\theta_t = \theta_{t-1} - \mathbf{v}_t
$$

其中：
- $\theta$ 是网络参数
- $\eta$ 是学习率
- $\nabla_\theta \mathcal{L}$ 是损失函数对参数的梯度
- $\mu$ 是动量系数（通常取 0.9），物理上对应"摩擦力"——$\mu = 0$ 无惯性退化回 vanilla SGD，$\mu = 1$ 完全无摩擦永不减速
- $\mathbf{v}_t$ 是"速度"，是历史梯度的**指数加权移动平均**

### 5.2 Adam：自适应学习率

**Adam**（Adaptive Moment Estimation，Kingma & Ba, 2015）是目前最广泛使用的优化器，也是训练 Transformer/LLM 的标准选择。

Adam 的核心思想：**为每个参数单独维护一个自适应的学习率**。具体方法是追踪梯度的一阶矩（均值）和二阶矩（未中心化方差）的指数移动平均。

完整更新公式（每一步 $t$）：

**第一步：更新一阶矩估计**（梯度的指数移动平均，追踪梯度的"方向"）

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**第二步：更新二阶矩估计**（梯度平方的指数移动平均，追踪梯度的"大小"）

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**第三步：偏差修正**（因为 $m_0 = 0, v_0 = 0$，初期估计偏小）

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**第四步：参数更新**

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中：
- $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ 是当前梯度
- $\beta_1 = 0.9$（一阶矩的衰减率，类似 momentum 系数）
- $\beta_2 = 0.999$（二阶矩的衰减率）
- $\eta$ 是学习率（通常 $10^{-3}$ 到 $10^{-4}$）
- $\epsilon \approx 10^{-8}$（防止除以零）

**为什么有效**：
- $\hat{m}_t$ 起到了类似 Momentum 的作用——累积梯度的方向
- 分母 $\sqrt{\hat{v}_t}$ 实现了**自适应**：对于梯度一直很大的参数，$\hat{v}_t$ 大，有效学习率小（不要走太快）；对于梯度一直很小的参数，$\hat{v}_t$ 小，有效学习率大（加速探索）

### 5.3 学习率调度

即使用了 Adam，学习率 $\eta$ 也不应该在整个训练过程中保持不变。现代训练通常使用 **warmup + cosine decay** 调度：

1. **Warmup**（热身）：训练刚开始时，学习率从接近 0 线性增长到目标值。原因是训练初期参数离最优很远，梯度的方向不太可靠，大学习率会导致不稳定。
2. **Cosine decay**（余弦衰减）：warmup 结束后，学习率按余弦函数从最大值平滑下降到接近 0。

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t \cdot \pi}{T}\right)
$$

其中 $T$ 是总训练步数。这比简单的线性衰减更平滑，在训练后期给模型更多时间做精细调整。

> **可靠程度**：Level 1（SGD with Momentum、Adam 是教科书内容，warmup + cosine decay 是 Level 2 工程实践共识）
>
> **参考**：
> - Adam：[Kingma & Ba, 2015 - "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)
> - [Wikipedia - Stochastic gradient descent § Extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)
> - [Deep Learning Book Ch.8 - Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)

---

## 6. 总结：现代深层网络的标准配方

把上面的技术组合在一起，现代深层网络的训练标准配方是：

| 问题 | 解决方案 | 出现年代 |
|---|---|---|
| 对称性破坏 | 随机初始化 | — |
| 前向/反向传播中方差不稳定 | Xavier / He 初始化 | 2010 / 2015 |
| 每层输入分布漂移 | Batch Norm / Layer Norm | 2015 / 2016 |
| 梯度消失 | 残差连接 + ReLU | 2015 |
| 优化困难 | Adam + 学习率调度 | 2015 |

这些技术不是各自独立的——它们相互配合。比如 He 初始化是为 ReLU 设计的，残差连接配合 BN/LN 使用效果最好，Adam 配合 warmup 更稳定。

**Transformer 是这些技术的集大成者**：每一层都有残差连接 + LayerNorm，用 Adam + warmup + cosine decay 训练。理解了本章内容，你就理解了 Transformer 训练基础设施的大部分。

---

## 理解检测

**Q1**：假设你用 Sigmoid 激活函数训练一个 20 层 MLP，发现前几层的权重几乎不更新。你的同事建议"把学习率调大 100 倍"。这个建议好不好？为什么？（提示：想想梯度消失的本质原因和调大学习率可能带来的问题。）

你的回答：



**Q2**：Xavier 初始化要求 $\text{Var}(w) = 1/n_{\text{in}}$，He 初始化要求 $\text{Var}(w) = 2/n_{\text{in}}$。如果你用 Xavier 初始化来训练一个使用 ReLU 激活函数的深层网络，会发生什么？（提示：ReLU 对输出的方差做了什么？）

你的回答：



**Q3**：有人说"残差连接本质上就是让网络可以选择性地跳过某些层"。你同意这个说法吗？请用反向传播的梯度公式来解释残差连接到底在做什么。

你的回答：


