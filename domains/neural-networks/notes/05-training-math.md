# 05 - 训练的数学：损失面、优化、收敛

> **主维度**：D3 训练数学
> **关键关系**：Adam `generalizes` SGD+Momentum · 优化理论 `used-for` 理解所有 D1 架构的训练过程
>
> **学习路径**：MLP 基础 → CNN 基础 → RNN / LSTM → **本章（训练的数学）** → 正则化与泛化 → 实战
>
> **前置知识**：梯度下降的基本概念（损失函数、参数更新 $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$）、矩阵特征值、基本概率论（期望、方差）
>
> **参考**：
> - [Goodfellow et al.《Deep Learning》Ch.8 - Optimization](https://www.deeplearningbook.org/contents/optimization.html)
> - [Wikipedia - Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
> - [Kingma & Ba, 2014 - Adam 论文](https://arxiv.org/abs/1412.6980)
> - [Dauphin et al., 2014 - Identifying and attacking the saddle point problem](https://arxiv.org/abs/1406.2572)
> - [d2l.ai Ch.12 - Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html)

---

## 1. 核心问题：为什么梯度下降居然能工作？

**可靠程度：Level 2-3**

在你学过的反向传播和梯度下降中，我们把训练看成一个简单的下山过程：计算损失函数 $\mathcal{L}(\theta)$ 对参数 $\theta$ 的梯度，然后朝梯度的反方向迈一步。

但是停下来想一想这件事有多不可思议：

- 一个现代神经网络可能有**几十亿个参数**——$\theta$ 是一个几十亿维的向量
- 损失函数 $\mathcal{L}(\theta)$ 是一个极其复杂的**非凸函数**（到处是弯曲、扭转、高原、峡谷）
- 我们每次只用一小批数据（mini-batch）估计梯度，估计本身**充满噪声**
- 我们用的是最简单的一阶方法（只用梯度，不用二阶信息如 Hessian 矩阵）

在经典优化理论中，对这种高维非凸问题，梯度下降不保证找到全局最优解。但实践中，梯度下降几乎总是能找到一个"足够好"的解——为什么？

要回答这个问题，我们需要理解损失面（loss landscape）的结构。

---

## 2. 损失面的结构

**可靠程度：直觉和鞍点论证 Level 2-3；平坦最小值假说 Level 3-4**

### 2.1 什么是损失面？

**损失面（loss landscape / loss surface）**是损失函数 $\mathcal{L}(\theta)$ 在参数空间中的"地形图"。你可以把每一组参数 $\theta$ 想象成地形上的一个点，损失值 $\mathcal{L}(\theta)$ 就是该点的"海拔"。训练的目标是从某个初始点出发，走到海拔最低的地方。

在低维（2D、3D）中，损失面就像一个真实的地形——有山峰、山谷、鞍部。但真实的神经网络参数空间是几百万到几十亿维的，我们的低维直觉需要修正。

### 2.2 高维空间的关键直觉：鞍点比局部最小值多得多

**临界点（critical point）**是梯度为零的点，即 $\nabla \mathcal{L}(\theta) = \mathbf{0}$。临界点有三种：

- **局部最小值（local minimum）**：所有方向都是"上坡"——Hessian 矩阵的特征值全为正
- **局部最大值（local maximum）**：所有方向都是"下坡"——特征值全为负
- **鞍点（saddle point）**：有些方向上坡，有些方向下坡——Hessian 矩阵的特征值有正有负

在 $n$ 维空间中，一个随机的临界点是局部最小值还是鞍点，取决于其 Hessian 矩阵的 $n$ 个特征值是否全为正。如果每个特征值正或负的概率各为 50%，那么全部 $n$ 个特征值都为正的概率是：

$$P(\text{局部最小值}) = \left(\frac{1}{2}\right)^n$$

当 $n$ 是几百万时，这个概率小到可以忽略。**绝大多数临界点是鞍点，不是局部最小值。**

Dauphin et al.（2014）在论文中验证了这一直觉：他们发现在实际的神经网络损失面上，低损失的临界点几乎都是鞍点，真正的局部最小值极其稀少，且损失值通常和全局最优差不多。

**实际含义**：人们常说"梯度下降可能陷入局部最小值"——但在高维空间中，这通常不是真正的问题。更常见的困境是**鞍点**：在鞍点附近，梯度很小（接近零），训练变慢，但它不是"困住了"——沿着下坡方向走就能逃出去。问题在于梯度太小时，逃出鞍点的速度很慢。

> 参考：[Dauphin et al., 2014 - Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572)

### 2.3 平坦最小值 vs 尖锐最小值

并非所有最小值都一样好。考虑两种极端情况：

- **尖锐最小值（sharp minimum）**：损失面在该点附近非常陡峭，参数稍有偏移，损失就急剧增大
- **平坦最小值（flat minimum）**：损失面在该点附近很平坦，参数在一个较大范围内变化，损失变化很小

**为什么平坦最小值泛化更好？** 直觉如下：训练集和测试集的数据分布略有不同，这意味着训练损失面和测试损失面之间有微小的偏移。如果你在一个尖锐最小值上，这个微小偏移就可能让你"滑落"到一个高损失的位置（训练集上好但测试集上差 = 过拟合）。如果你在平坦最小值上，即使损失面有小偏移，你仍然在一个低损失的区域里。

这个观察最早可以追溯到 Hochreiter & Schmidhuber（1997），后来 Keskar et al.（2017）在深度学习的语境下给出了实验证据。一个有趣的发现是：**大 batch SGD 倾向于找到尖锐最小值，而小 batch SGD 倾向于找到平坦最小值**——这是小 batch 可能泛化更好的原因之一（SGD 的噪声帮助"跳过"尖锐的坑）。

> 可靠程度：平坦最小值泛化更好的直觉被广泛接受（Level 2-3），但严格的理论证明仍在发展中（Level 3-4）。
>
> 参考：[Keskar et al., 2017 - On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836) · [Hochreiter & Schmidhuber, 1997 - Flat Minima](https://doi.org/10.1162/neco.1997.9.1.1)

---

## 3. SGD 的收敛性

**可靠程度：凸情况 Level 1（数学严格）；非凸情况 Level 2**

### 3.1 先回忆：GD vs SGD

**梯度下降（Gradient Descent, GD）**在每一步使用**全部**训练数据计算梯度：

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) = \theta_t - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla \ell(\theta_t; x_i, y_i)$$

其中 $N$ 是训练集大小，$\ell(\theta; x_i, y_i)$ 是单个样本的损失。

**随机梯度下降（Stochastic Gradient Descent, SGD）**在每一步只用一个（或一小批）随机采样的样本估计梯度：

$$\theta_{t+1} = \theta_t - \eta \, \mathbf{g}_t, \quad \text{其中 } \mathbf{g}_t = \nabla \ell(\theta_t; x_{i_t}, y_{i_t})$$

$\mathbf{g}_t$ 是**真实梯度的无偏估计**：$\mathbb{E}[\mathbf{g}_t] = \nabla \mathcal{L}(\theta_t)$，但有方差。

**为什么用 SGD 而不是 GD？** 计算效率。GD 每步要遍历全部数据，对几百万数据来说极其昂贵。SGD 每步只用一个 mini-batch（比如 256 个样本），计算量小得多。虽然每步的方向有噪声，但平均下来方向是对的。

### 3.2 凸优化下的收敛保证

为了建立直觉，先看最简单的情况。如果损失函数 $\mathcal{L}(\theta)$ 是**凸函数**（碗形，只有一个全局最小值），SGD 的收敛性有经典结论。

**定理（SGD 在凸函数上的收敛，简化版）**：

设 $\mathcal{L}$ 是凸函数且 Lipschitz 连续（梯度不会无限大），SGD 使用学习率 $\eta_t = \frac{c}{\sqrt{t}}$（逐步衰减），经过 $T$ 步迭代后：

$$\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}(\theta^*) \leq O\left(\frac{1}{\sqrt{T}}\right)$$

其中 $\bar{\theta}_T$ 是前 $T$ 步参数的平均值，$\theta^*$ 是全局最优解。

**这个结论说的是**：SGD 在凸函数上保证收敛到全局最优，但速度是 $O(1/\sqrt{T})$——要把误差减半，步数要翻四倍。这比全梯度下降的 $O(1/T)$ 慢，代价是 SGD 的梯度噪声。

### 3.3 非凸情况（神经网络的实际情况）

神经网络的损失函数几乎从来不是凸的（有多个最小值、鞍点等）。在非凸情况下：

- **不保证找到全局最优**——只能保证收敛到某个**临界点**（梯度为零的点，可能是局部最小值或鞍点）
- 具体来说：SGD 保证 $\min_{t \leq T} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq O(1/\sqrt{T})$，即经过 $T$ 步后，必定经过了某个梯度很小的点

**但实践中这不是问题。** 原因在 2.2 节已经解释过：高维空间中局部最小值很稀少，且损失值通常和全局最优差不多。SGD 找到的"足够好"的解在实际任务上表现良好。

### 3.4 学习率的作用

学习率 $\eta$ 是训练中最重要的超参数。直觉：

- **$\eta$ 太大**：步子太大，可能直接跨过最小值，甚至发散（损失越来越大）
- **$\eta$ 太小**：步子太小，训练极慢，而且更容易被困在鞍点附近
- **合适的 $\eta$**：存在一个"甜区"（Goldilocks zone），既不太大也不太小

一个经典的实验观察是：对固定学习率的 SGD，存在一个**最大稳定学习率** $\eta_{\max} = \frac{2}{\lambda_{\max}(H)}$，其中 $\lambda_{\max}(H)$ 是损失函数 Hessian 矩阵最大特征值。超过这个值就会发散。

> 参考：[Deep Learning Book Ch.8](https://www.deeplearningbook.org/contents/optimization.html) · [Bottou et al., 2018 - Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)

---

## 4. Momentum：让 SGD 跑得更快

**可靠程度：Level 1（教科书共识）**

### 4.1 SGD 的问题：在峡谷中振荡

考虑一个椭圆形的损失面（一个方向很陡、另一个方向很平缓）。SGD 在陡峭方向上梯度大，在平缓方向上梯度小。结果是：SGD 在陡峭方向上来回振荡（步子太大），在平缓方向上缓慢前进（步子太小）——整体呈"之"字形前进，效率很低。

```
             SGD 的路径（之字形）
损失面：     ╲  ╱╲  ╱╲  ╱╲
              ╲╱  ╲╱  ╲╱  ╲
              ────────────── ★ 最小值
平缓方向 →

             Momentum 的路径（更直接）
损失面：     ╲
              ─────────────── ★ 最小值
平缓方向 →
```

### 4.2 物理类比

Momentum（动量）直接借鉴了物理学的概念。想象一个**带摩擦力的小球在势能面上滚动**：

- 没有动量（SGD）：小球在每个点只看当前位置的坡度决定滚哪走。在两壁之间来回弹
- 有动量：小球有惯性。如果它一直在某个方向上受力（梯度方向一致），它就会**加速**；如果它在某个方向上来回受力（梯度方向振荡），正负相消，不会加速

### 4.3 Momentum 的数学

引入一个**速度变量** $\mathbf{v}_t$（类比物理中的速度），更新规则变为：

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta \mathbf{v}_t$$

各符号含义：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $\mathbf{v}_t$ | "速度"——历史梯度的指数加权平均 | — |
| $\beta$ | 动量系数——"摩擦力"的大小。$\beta=0$ 退化为普通 SGD | 0.9 |
| $\eta$ | 学习率 | — |
| $\nabla \mathcal{L}(\theta_t)$ | 当前梯度 | — |

**$\beta$ 的含义**：$\beta$ 越接近 1，"惯性"越大——小球越重、摩擦越小，历史梯度的影响越大。$\beta = 0.9$ 表示速度 $\mathbf{v}_t$ 大致是最近 $\frac{1}{1-\beta} = 10$ 步梯度的加权平均。

**为什么有效？** 展开 $\mathbf{v}_t$：

$$\mathbf{v}_t = \nabla \mathcal{L}(\theta_t) + \beta \nabla \mathcal{L}(\theta_{t-1}) + \beta^2 \nabla \mathcal{L}(\theta_{t-2}) + \cdots$$

- 如果连续多步的梯度方向一致（都指向最小值），它们会累加 → 速度增大 → **加速**
- 如果连续多步的梯度方向交替变化（在峡谷两壁之间弹），正负相消 → 速度被抑制 → **减振**

这正是我们想要的：在平缓方向上加速，在振荡方向上抑制。

> 参考：[Polyak, 1964 - 经典 Momentum 论文] · [d2l.ai - Momentum](https://d2l.ai/chapter_optimization/momentum.html) · [Sutskever et al., 2013 - On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.html)

---

## 5. Adam：自适应学习率优化器

**可靠程度：Level 1（教科书共识，是实践中最常用的优化器之一）**

### 5.1 动机：不同参数需要不同的学习率

Momentum 对所有参数使用相同的学习率 $\eta$。但在实际网络中：

- 有些参数的梯度很大且稳定（比如经常被激活的权重）→ 可以用较小的学习率，稳步前进
- 有些参数的梯度很小且稀疏（比如稀疏特征对应的权重）→ 需要较大的学习率，否则几乎不更新

**Adam（Adaptive Moment Estimation）** 由 Kingma & Ba 在 2014 年提出，它的核心思想是为每个参数**自动调整学习率**，结合了 Momentum（一阶矩）和 RMSProp（二阶矩）的优点。

### 5.2 Adam 的完整推导

Adam 维护两个滑动平均量，分别跟踪梯度的**均值**和**方差**。设 $\mathbf{g}_t = \nabla \mathcal{L}(\theta_t)$ 为第 $t$ 步的梯度。

**第一步：计算一阶矩估计（梯度均值的指数滑动平均）**

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

- $\mathbf{m}_t$ 跟踪的是梯度的"平均方向和大小"——类似 Momentum 的速度
- $\beta_1$ 是一阶矩的衰减系数，典型值 $\beta_1 = 0.9$
- 含义：$\mathbf{m}_t$ 是最近约 $\frac{1}{1-\beta_1} = 10$ 步梯度的加权平均

**第二步：计算二阶矩估计（梯度平方的指数滑动平均）**

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

- $\mathbf{g}_t^2$ 是逐元素平方（不是矩阵乘法）
- $\mathbf{v}_t$ 跟踪的是梯度的"波动幅度"——每个参数方向上梯度有多大
- $\beta_2$ 是二阶矩的衰减系数，典型值 $\beta_2 = 0.999$
- 含义：$\mathbf{v}_t$ 是最近约 $\frac{1}{1-\beta_2} = 1000$ 步梯度平方的加权平均

**第三步：偏差修正**

因为 $\mathbf{m}_0 = \mathbf{0}$、$\mathbf{v}_0 = \mathbf{0}$（初始化为零），在训练早期 $\mathbf{m}_t$ 和 $\mathbf{v}_t$ 会偏向零（偏小）。偏差修正消除这个初始偏差：

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

为什么需要这个？考虑 $t=1$：$\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1 = 0.1 \cdot \mathbf{g}_1$，这只是真实梯度的 10%——严重偏小。除以 $(1 - \beta_1^1) = 0.1$ 后，$\hat{\mathbf{m}}_1 = \mathbf{g}_1$，修正了偏差。随着 $t$ 增大，$\beta_1^t \to 0$，修正因子趋于 1，偏差修正自动失效。

**第四步：参数更新**

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

各项的物理含义：

| 项 | 含义 |
|---|---|
| $\hat{\mathbf{m}}_t$ | 梯度的"平均方向"——指向哪里走（类比 Momentum） |
| $\sqrt{\hat{\mathbf{v}}_t}$ | 梯度的"波动幅度"——这个方向上梯度有多不稳定 |
| $\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}}$ | 相当于"信噪比"：方向明确且稳定 → 大步走；方向不确定或振荡 → 小步走 |
| $\epsilon$ | 一个很小的常数（如 $10^{-8}$），防止除以零 |
| $\eta$ | 全局学习率，典型值 $\eta = 0.001$ |

**关键洞察**：对于梯度一直很大的参数，$\sqrt{\hat{\mathbf{v}}_t}$ 也大，实际步长 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 就小——自动"刹车"。对于梯度一直很小的参数，$\sqrt{\hat{\mathbf{v}}_t}$ 也小，实际步长就大——自动"加油"。这就是"自适应学习率"的含义。

### 5.3 Adam 的默认超参数

Kingma & Ba 推荐的默认值在大多数情况下都不错：

| 超参数 | 推荐值 | 含义 |
|--------|--------|------|
| $\eta$ | $0.001$ | 全局学习率 |
| $\beta_1$ | $0.9$ | 一阶矩衰减（动量） |
| $\beta_2$ | $0.999$ | 二阶矩衰减（梯度波动） |
| $\epsilon$ | $10^{-8}$ | 数值稳定性 |

> 参考：[Kingma & Ba, 2014 - Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) · [d2l.ai - Adam](https://d2l.ai/chapter_optimization/adam.html)

---

## 6. 学习率调度

**可靠程度：Level 1-2（广泛使用的实践经验，部分有理论支持）**

### 6.1 为什么不能一直用固定学习率？

直觉：训练初期离最优解很远，大学习率可以快速接近；训练后期已经接近最优解了，大学习率会导致在最优解附近来回跳，无法精确收敛。

这不只是直觉——SGD 的收敛理论也要求学习率满足 $\sum_t \eta_t = \infty$（保证能走到任意远的地方）且 $\sum_t \eta_t^2 < \infty$（保证最终稳定下来），经典的选择是 $\eta_t \propto 1/\sqrt{t}$。

实践中，人们使用几种常见的**学习率调度（learning rate schedule）**策略：

### 6.2 Step Decay（阶梯衰减）

每隔固定步数（或 epoch 数），把学习率乘一个衰减因子（如 0.1）：

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}$$

其中 $\eta_0$ 是初始学习率，$\gamma$ 是衰减因子（如 0.1），$s$ 是衰减间隔。

**例子**：$\eta_0 = 0.1$，每 30 个 epoch 乘 0.1 → 0.1, 0.01, 0.001, ...

**优点**：简单直接。**缺点**：需要手动选 $\gamma$ 和 $s$，衰减时刻的跳变可能不平滑。

### 6.3 Cosine Annealing（余弦退火）

学习率按余弦曲线从初始值平滑下降到接近零：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

其中 $T$ 是总训练步数，$\eta_{\max}$ 是最大学习率，$\eta_{\min}$ 通常取 0 或很小的值。

**优点**：平滑衰减，无需手动选衰减时间点。在很多现代训练中表现很好。

### 6.4 Warmup + Cosine Decay

这是目前最流行的策略，特别是在训练 Transformer 和大模型时：

1. **Warmup 阶段**（前 $T_w$ 步）：学习率从 0 或很小的值**线性增长**到 $\eta_{\max}$
2. **Cosine 衰减阶段**（$T_w$ 步之后）：按余弦曲线下降

```
学习率
  ↑
  │         ╭──────╮
  │        ╱        ╲
  │       ╱          ╲
  │      ╱            ╲
  │     ╱              ╲
  │    ╱                ╲
  │   ╱                  ╲
  │  ╱                    ╲
  │ ╱                      ╲
  │╱                        ╲
  └──────────────────────────── 步数
    warmup     cosine decay
```

**为什么需要 warmup？** 这和 Adam 的二阶矩估计有关。训练刚开始时：

- $\mathbf{v}_t$（梯度平方的滑动平均）还没有积累够数据，估计很不准
- 不准的 $\hat{\mathbf{v}}_t$ 可能很小，导致 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 异常大 → 参数更新过猛 → 训练不稳定

通过在前几百到几千步使用较小的学习率（warmup），给 Adam 的二阶矩估计一些时间来"热身"，等估计稳定后再用正常的学习率。

> 参考：[Loshchilov & Hutter, 2016 - SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) · [Goyal et al., 2017 - Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

---

## 7. 梯度裁剪（Gradient Clipping）

**可靠程度：Level 1（教科书共识，广泛使用）**

在 04 中我们看到，RNN 训练中梯度容易爆炸（特征值 > 1 的连乘）。即使不是 RNN，在深层网络或大学习率的情况下，也可能偶尔出现异常大的梯度。

**梯度裁剪（gradient clipping）**是一种简单粗暴但非常有效的方法：如果梯度太大了，就把它"剪"到一个最大值。

### 7.1 按范数裁剪（最常用）

设 $\mathbf{g} = \nabla \mathcal{L}(\theta)$ 是当前梯度，$c$ 是裁剪阈值。如果梯度的范数超过了 $c$，就按比例缩小：

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq c \\ c \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > c \end{cases}$$

**直觉**：梯度裁剪不改变梯度的**方向**（方向信息保留），只是限制了梯度的**大小**。就像给汽车的油门加了一个限速器——你仍然往正确的方向走，但不会因为突然的大梯度而失控。

### 7.2 实践建议

- 裁剪阈值 $c$ 通常取 1.0 或 5.0
- 在 RNN/LSTM 训练中几乎是必需的
- 在 Transformer 训练中也常用
- 裁剪应在梯度计算完成后、参数更新之前进行

> 参考：[Pascanu et al., 2013 - On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) · [d2l.ai - Gradient Clipping](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#gradient-clipping)

---

## 8. 各优化器的对比总结

| 优化器 | 更新规则（核心） | 优点 | 缺点/注意 |
|--------|-----------------|------|-----------|
| SGD | $\theta \leftarrow \theta - \eta \mathbf{g}$ | 最简单，理论保证最清楚 | 收敛慢，需要仔细调学习率 |
| SGD + Momentum | 引入速度 $\mathbf{v}_t$，在一致方向上加速 | 比 SGD 快，减少振荡 | 多一个超参数 $\beta$ |
| Adam | 自适应学习率 + 动量 + 偏差修正 | 默认参数就很好用，收敛快 | 有时泛化不如 SGD + Momentum |
| AdamW | Adam + 解耦权重衰减 | 修复了 Adam 的 L2 正则化问题 | 目前大模型的首选 |

**实践选择**：

- 不确定用什么 → 先用 Adam（$\eta = 0.001$），通常都能跑起来
- 追求最佳泛化性能 → SGD + Momentum + 仔细调学习率
- 训练 Transformer / 大模型 → AdamW + warmup + cosine decay

---

## 9. 本章核心要点回顾

1. **损失面结构**：高维空间中鞍点远比局部最小值多，"陷入局部最小值"通常不是真正的问题
2. **SGD 收敛**：凸情况有 $O(1/\sqrt{T})$ 保证；非凸情况只保证到临界点，但实践中"够好"
3. **Momentum**：用指数加权平均平滑梯度，在一致方向加速、振荡方向抑制
4. **Adam**：自适应学习率，为梯度大的参数自动减速、梯度小的参数自动加速
5. **学习率调度**：warmup 解决 Adam 初期估计不准的问题，cosine decay 让后期精细收敛
6. **梯度裁剪**：简单有效地防止梯度爆炸

---

## 理解检测

**Q1**：有人说"神经网络训练的主要困难是陷入局部最小值"。根据本章学到的内容，你认为这个说法准确吗？在高维参数空间中，更常见的困境是什么？

你的回答：



**Q2**：Adam 的更新公式中，$\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}}$ 可以被理解为一种"信噪比"。请解释：如果某个参数方向上的梯度在过去很多步中大小差不多但方向一致，$\hat{\mathbf{m}}_t$ 和 $\sqrt{\hat{\mathbf{v}}_t}$ 分别大约是多少（相对于梯度大小 $g$）？这个参数方向上的实际步长大约是多少？

你的回答：



**Q3**：为什么训练 Transformer 时通常需要 warmup？如果不用 warmup 而直接用大学习率开始训练 Adam，可能会发生什么问题？（提示：想想 Adam 的二阶矩 $\mathbf{v}_t$ 在训练初期的状态。）

你的回答：


