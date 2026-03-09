# 01a - MLP 基础：从神经元到多层网络

> **主维度**：D1 基础架构
>
> **学习路径**：**本章（MLP 基础）** → 激活函数 → 深入 MLP → CNN → ...
>
> **前置知识**：线性代数（向量、矩阵乘法）、微积分（偏导数、链式法则）
>
> **参考**：
> - [3Blue1Brown - Neural Networks 视频系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)（最佳入门视频）
> - [Michael Nielsen 在线书（免费）](http://neuralnetworksanddeeplearning.com/)
> - [Goodfellow 等《Deep Learning》Ch.6](https://www.deeplearningbook.org/)
> - [Wikipedia - Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

---

## 1. 核心问题：我们在做什么

你做物理实验时经常要拟合数据——比如测了一堆 $(x_i, y_i)$ 数据点，然后假设 $y = ax + b$，用最小二乘法求出 $a$ 和 $b$。

这就是"学习"的最简单形式：**你选定一个函数的形式（$y = ax+b$），然后从数据中找出最好的参数（$a, b$）。**

但线性函数太简单了。如果数据的关系是复杂的非线性呢？你可能试过用多项式 $y = ax^2 + bx + c$，甚至更高阶。但如果输入不是一个数而是几百个数呢（比如一张图片有几万个像素值）？人工选函数形式就不现实了。

**神经网络的核心思想**：不人工选函数形式，而是构造一种"可以变成几乎任何函数"的通用结构，然后让数据自动决定它应该长什么样。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Function approximation](https://en.wikipedia.org/wiki/Function_approximation)

---

## 2. 神经元：最小计算单元

名字听着玄乎，但一个神经元就是一个极其简单的数学操作：

$$
y = f(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)
$$

拆开看：

| 步骤 | 数学 | 含义 |
|------|------|------|
| 输入 | $x_1, x_2, ..., x_n$ | 一组数字（比如物理量的测量值） |
| 权重 | $w_1, w_2, ..., w_n$ | 每个输入乘一个系数，表示"这个输入有多重要" |
| 偏置 | $b$ | 一个可调的常数，类似线性回归里的截距 |
| 加权求和 | $z = \mathbf{w} \cdot \mathbf{x} + b$ | 线性组合 |
| 激活函数 | $y = f(z)$ | 非线性变换（下一节解释为什么需要） |

**为什么叫"神经元"？** 1943 年 McCulloch 和 Pitts 提出这个模型时，灵感来自生物神经元：树突接收信号（输入），胞体整合信号（加权求和），超过阈值就放电（激活函数）。但**现在不需要在意生物类比**——把它当成一个简单的数学函数就行。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Perceptron](https://en.wikipedia.org/wiki/Perceptron) · [Wikipedia - Artificial neuron](https://en.wikipedia.org/wiki/Artificial_neuron)

---

## 3. 为什么要激活函数（非线性）

如果没有激活函数 $f$，神经元就是 $y = \mathbf{w} \cdot \mathbf{x} + b$，一个线性函数。

关键问题：**多个线性函数叠加还是线性函数。** 矩阵乘矩阵还是矩阵：

$$
y = W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)
$$

不管叠多少层，最终都可以合并成 $y = W'\mathbf{x} + \mathbf{b}'$——一个单层线性变换。**100 层线性网络 = 1 层线性网络，叠层没有带来任何新的表达能力。**

所以每层之后必须加一个非线性函数 $f$，打断这种"可合并性"。常用的几种：

| 名称 | 公式 | 一句话 |
|------|------|--------|
| **ReLU** | $f(z) = \max(0, z)$ | 负数变 0，正数不变。最简单，用得最多 |
| **Sigmoid** | $f(z) = \frac{1}{1+e^{-z}}$ | 把任意数压缩到 $(0,1)$。用于输出层和门控 |
| **GELU** | $f(z) \approx z \cdot \sigma(1.702z)$ | ReLU 的平滑版，Transformer 中常用 |

> 激活函数的完整讲解见 [01b-激活函数](01b-activation-functions.md)。
>
> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function)

---

## 4. 多层感知机（MLP）

### 4.1 结构

一个神经元只能做很简单的事。把很多神经元连起来，就组成了**多层感知机（Multilayer Perceptron, MLP）**，也叫**前馈网络（Feed-forward Network）**：

```
输入层          隐藏层 1        隐藏层 2         输出层
x₁ ──┐     ┌── h₁⁽¹⁾ ──┐     ┌── h₁⁽²⁾ ──┐     ┌── y₁
x₂ ──┼──→──┤── h₂⁽¹⁾ ──┼──→──┤── h₂⁽²⁾ ──┼──→──┤── y₂
x₃ ──┘     └── h₃⁽¹⁾ ──┘     └── h₃⁽²⁾ ──┘     └── y₃
```

- **输入层**：原始数据（比如 3 个数字）
- **隐藏层**：中间的计算层。每个节点接收上一层**所有**节点的输出，做加权求和 + 激活
- **输出层**：最终结果

### 4.2 每层的数学

每一层做的事情：

$$
\mathbf{h}^{(\ell)} = f\!\left(W^{(\ell)} \cdot \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right)
$$

其中：
- $W^{(\ell)}$ 是第 $\ell$ 层的**权重矩阵**，维度为 $d_{\text{out}} \times d_{\text{in}}$
- $\mathbf{b}^{(\ell)}$ 是第 $\ell$ 层的**偏置向量**
- $f$ 是激活函数
- $\mathbf{h}^{(\ell-1)}$ 是上一层的输出（第 0 层就是输入 $\mathbf{x}$）
- $\mathbf{h}^{(\ell)}$ 是这一层的输出

**整个网络就是这些矩阵乘法和非线性变换的多次叠加。** 网络的"知识"全部存储在权重矩阵 $W^{(1)}, W^{(2)}, \ldots$ 和偏置 $\mathbf{b}^{(1)}, \mathbf{b}^{(2)}, \ldots$ 中。

### 4.3 "全连接"的含义

MLP 也叫"全连接网络"（Fully Connected Network），因为相邻两层之间**每个**神经元都和另一层的**每个**神经元有连接。

**参数量**：如果第 $\ell$ 层有 $d_{\text{in}}$ 个输入、$d_{\text{out}}$ 个输出，那么 $W^{(\ell)}$ 有 $d_{\text{in}} \times d_{\text{out}}$ 个参数，$\mathbf{b}^{(\ell)}$ 有 $d_{\text{out}}$ 个参数。

**例子**：一张 $256 \times 256$ 的灰度图像，展平后是 65536 维向量。如果第一个隐藏层有 1000 个神经元，光 $W^{(1)}$ 就有 $65536 \times 1000 = 6553.6$ 万个参数。这是 MLP 的核心问题——参数量随输入维度线性增长，对高维数据（如图像）非常浪费。CNN 就是为了解决这个问题而发明的（见 [03-cnn.md](03-cnn.md)）。

### 4.4 万能近似定理

一个深刻的数学结论：只要有**一个隐藏层**且神经元足够多，配上非线性激活函数，MLP 可以以任意精度逼近任何连续函数。这叫**万能近似定理**（Universal Approximation Theorem）。

换句话说，这个结构足够"灵活"，可以拟合几乎任何输入-输出关系。

**但这不意味着一层就够了**：定理只说"存在"这样的参数，没说"找得到"。实践中，更深的网络比更宽的网络更容易训练（见 [02-deep-mlp.md](02-deep-mlp.md)）。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Michael Nielsen - 可视化万能近似](http://neuralnetworksanddeeplearning.com/chap4.html) · [Wikipedia - Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

## 5. 损失函数：衡量"学得好不好"

网络有了结构，但权重一开始是随机的，输出是乱的。怎么衡量"当前的输出有多差"？

这就是**损失函数（Loss Function）**的作用——一个数字，告诉你当前参数下模型的预测和真实值差多远。**损失越小，模型越好。**

### 5.1 回归任务：均方误差（MSE）

预测一个连续数值时使用：

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

就是预测值 $\hat{y}_i$ 和真实值 $y_i$ 之差的平方的平均。你在物理实验中用最小二乘法拟合直线，本质上就是在最小化 MSE。

### 5.2 分类任务：交叉熵（Cross-Entropy）

预测"属于哪一类"时使用。假设有 $K$ 类，网络输出每类的概率 $\hat{p}_1, \hat{p}_2, \ldots, \hat{p}_K$（通过 softmax 函数把任意数值转换为概率分布）：

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \quad \text{（softmax）}
$$

如果真实类别是第 $c$ 类，损失为：

$$
\mathcal{L}_{\text{CE}} = -\log \hat{p}_c
$$

**直觉**：如果模型对正确答案很有信心（$\hat{p}_c \approx 1$），$-\log(1) = 0$，损失很小。如果模型对正确答案没信心（$\hat{p}_c \approx 0.01$），$-\log(0.01) \approx 4.6$，损失很大。

| $\hat{p}_c$ | $-\log \hat{p}_c$ | 含义 |
|---|---|---|
| 0.99 | 0.01 | 很有信心 → 损失几乎为零 |
| 0.5 | 0.69 | 半猜 → 损失中等 |
| 0.01 | 4.6 | 几乎猜错 → 损失很大 |

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Loss function](https://en.wikipedia.org/wiki/Loss_function) · [Wikipedia - Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) · [Wikipedia - Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

---

## 6. 梯度下降：怎么找到最好的参数

有了损失函数 $\mathcal{L}(\theta)$（$\theta$ 代表所有权重和偏置），"学习"就变成了一个优化问题：**找到让 $\mathcal{L}$ 最小的 $\theta$。**

你在物理中求过函数极值：令导数等于零。但神经网络有几百万甚至几十亿个参数，解析解不存在。怎么办？

### 6.1 基本思想

**梯度下降（Gradient Descent）**：别试图一步到位，而是一小步一小步地往"下坡方向"走。

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}
$$

- $\frac{\partial \mathcal{L}}{\partial \theta}$：损失对参数的偏导数（**梯度**），指向"上坡"方向
- $\eta$：**学习率**，步长有多大
- 减号：往梯度的反方向走（下坡）

**类比**：你蒙着眼站在一个山谷里，想找到最低点。你能做的就是用脚感受当前位置的坡度（梯度），然后朝着最陡的下坡方向迈一小步。重复很多次，最终走到谷底附近。

### 6.2 学习率的影响

| 学习率 $\eta$ | 效果 |
|---|---|
| 太大 | 步子迈太大，跳过最低点，甚至越走越远（损失飙升） |
| 太小 | 步子太小，走了一万步还没走多远（收敛极慢） |
| 合适 | 稳步下降，最终收敛到一个好的解 |

### 6.3 随机梯度下降（SGD）

实际训练中，不用全部数据算梯度（太慢），而是每次随机取一小批数据（**mini-batch**，通常 32-512 个样本）算一个"近似梯度"，然后更新参数。这叫**随机梯度下降（Stochastic Gradient Descent, SGD）**。

虽然每一步的梯度有噪声（因为只用了部分数据），但平均下来方向是对的，而且训练速度大大加快。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[3Blue1Brown - Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) · [Wikipedia - Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)

---

## 7. 反向传播：高效计算梯度

梯度下降需要 $\frac{\partial \mathcal{L}}{\partial \theta}$，而网络可能有几十亿个参数。对每个参数分别计算偏导数是不现实的。

### 7.1 核心思想

**反向传播（Backpropagation）** 利用微积分的**链式法则**，从输出层往回一层一层地算，只需要一次"前向"（算输出）和一次"反向"（算梯度）就能得到**所有**参数的梯度。

### 7.2 最简单的例子

假设网络只有一层：

$$
z = wx + b \quad \rightarrow \quad y = f(z) \quad \rightarrow \quad \mathcal{L} = (y - y_{\text{true}})^2
$$

要算 $\frac{\partial \mathcal{L}}{\partial w}$，用链式法则一步步来：

$$
\frac{\partial \mathcal{L}}{\partial w} = \underbrace{\frac{\partial \mathcal{L}}{\partial y}}_{2(y - y_{\text{true}})} \cdot \underbrace{\frac{\partial y}{\partial z}}_{f'(z)} \cdot \underbrace{\frac{\partial z}{\partial w}}_{x}
$$

三项都是已知量！乘起来就得到了 $w$ 的梯度。

### 7.3 多层的情况

对于多层网络，原理完全一样——从最后一层开始，逐层往回乘。每一层的"误差信号"传给上一层，上一层再传给更上一层。这就是"反向传播"名字的由来。

**计算效率**：反向传播的计算量大约只是前向传播的 2-3 倍。如果有 10 亿个参数，不用反向传播而是对每个参数做数值微分（微小扰动法），需要前向传播 10 亿次。反向传播只需要一次前向 + 一次反向。

**没有反向传播就没有深度学习。**

### 7.4 和梯度消失/爆炸的联系

反向传播从最后一层算到第一层，每经过一层都要乘上该层的"局部梯度"（包含激活函数的导数和权重矩阵）。如果这些因子一直 < 1，连乘后梯度消失；一直 > 1，连乘后梯度爆炸。这就是 [02-deep-mlp.md](02-deep-mlp.md) 第 1 节讨论的问题。

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[3Blue1Brown - Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) · [Wikipedia - Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) · [Michael Nielsen - Ch.2](http://neuralnetworksanddeeplearning.com/chap2.html)

---

## 8. 过拟合：学"太好"反而不好

你可能遇到过这个情况：一个 10 次多项式可以完美通过 5 个数据点，但在数据点之间疯狂振荡，预测新数据时一塌糊涂。

这叫**过拟合（Overfitting）**：模型把训练数据的噪声也"记住"了，而不是学到了真正的规律。

### 8.1 训练误差 vs 测试误差

| 指标 | 含义 | 过拟合时 |
|------|------|---------|
| 训练误差 | 模型在训练数据上的损失 | 很低（几乎记住了所有数据） |
| 测试误差 | 模型在**从没见过**的数据上的损失 | 很高（在新数据上表现差） |

**我们真正关心的是测试误差**——模型能不能泛化到新数据上。

### 8.2 对策

| 方法 | 做法 | 直觉 |
|------|------|------|
| **正则化** | 在损失中加惩罚项 $\lambda \|\theta\|^2$ | 阻止权重变得太大，偏好"简单"的模型 |
| **Dropout** | 训练时随机关掉一部分神经元 | 迫使网络不依赖任何单个节点 |
| **Early Stopping** | 监控测试误差，开始变差就停止 | 在"刚好学够"的时候停下 |
| **数据增强** | 对训练数据做随机变换（翻转、裁剪等） | 让模型见到更多"变体"，不容易死记 |

> **可靠程度**：Level 1（教科书共识）
>
> **参考**：[Wikipedia - Overfitting](https://en.wikipedia.org/wiki/Overfitting) · [Wikipedia - Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))

---

## 9. 完整训练流程

把上面串起来：

```
1. 定义网络结构（几层、每层多少神经元、什么激活函数）
     ↓
2. 随机初始化所有权重
     ↓
┌──────────────────────────────────────────────┐
│  3. 前向传播：输入一批数据，逐层计算，得到预测输出  │
│       ↓                                        │
│  4. 计算损失：用损失函数衡量预测和真实值的差距      │
│       ↓                                        │
│  5. 反向传播：从后往前算出所有参数的梯度            │
│       ↓                                        │
│  6. 更新参数：沿梯度反方向走一小步                 │
│       ↓                                        │
│  用下一批数据重复（一个 epoch = 遍历一次全部数据）   │
└──────────────────────────────────────────────┘
     ↓
7. 训练结束（达到设定的轮数，或测试误差不再下降）
```

**术语**：

| 术语 | 含义 |
|------|------|
| **Epoch** | 用全部训练数据训练一轮 |
| **Batch** | 一小批数据（如 64 个样本） |
| **Iteration** | 用一个 batch 做一次参数更新 |

如果有 10000 个样本，batch size = 100，那么一个 epoch = 100 个 iteration。

---

## 10. MLP 的优缺点

| 优点 | 缺点 |
|------|------|
| 结构简单，容易理解和实现 | 参数量随输入维度线性增长（对图像等高维数据不友好） |
| 万能近似——理论上能拟合任何函数 | 不利用输入的空间/时间/图结构 |
| 是所有深度网络的基础构件 | 深层 MLP 有梯度消失/爆炸问题（见 [02-deep-mlp.md](02-deep-mlp.md)） |

后续的每种架构（CNN、RNN、Transformer、GNN）都可以看作是在 MLP 基础上**针对特定数据结构做的改进**：

- **CNN**：利用图像的空间局部性，用小卷积核代替全连接 → 参数量大幅减少
- **RNN**：利用序列的时间顺序，用循环结构处理变长输入
- **Transformer**：让任意两个位置直接交互（Attention），不需要逐步传递
- **GNN**：在图结构上做消息传递，适应非规则数据

> **可靠程度**：Level 1（教科书共识）

---

## 理解检测

**Q1**：用你自己的话解释：为什么没有激活函数的多层网络和单层线性网络没有区别？

你的回答：



**Q2**：一张 $256 \times 256$ 的 RGB 图像展平后有多少维？如果第一个隐藏层有 1000 个神经元，权重矩阵 $W^{(1)}$ 有多少个参数？（不算偏置）

你的回答：



**Q3**：梯度下降的"学习率"如果设得太大会怎样？太小会怎样？

你的回答：



**Q4**：假设你在训练一个分类网络，正确答案是第 3 类。模型给第 3 类的概率从 0.01 提升到 0.99，交叉熵损失从多少变到多少？

你的回答：



**Q5**：训练误差很低但测试误差很高，这说明什么？你应该怎么办？

你的回答：


