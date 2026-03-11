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

#### 问题：网络的输出是"裸分"，不是概率

假设你要分类"猫 / 狗 / 鸟"（$K = 3$ 类）。网络最后一层输出 3 个数字——每个类别对应一个，叫做 **logits**（未归一化的分数）：

$$
z_1 = 2.0, \quad z_2 = 1.0, \quad z_3 = -1.0
$$

这些数字可以是任意实数，可以是负数，加起来也不等于 1——它们**不是概率**。但我们需要概率，才能说"模型有多大把握认为这是猫"。

#### Softmax：把裸分变成概率

**Softmax** 做的事情就是：把任意 $K$ 个实数变成 $K$ 个正数，加起来等于 1，可以解释为概率。

做法分两步：

**第一步**：对每个分数取指数 $e^{z_k}$，把任意实数变成正数：

$$
e^{2.0} = 7.39, \quad e^{1.0} = 2.72, \quad e^{-1.0} = 0.37
$$

**第二步**：除以总和，归一化到概率：

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

代入数字：总和 = $7.39 + 2.72 + 0.37 = 10.48$

| 类别 | 裸分 $z_k$ | $e^{z_k}$ | 概率 $\hat{p}_k$ |
|------|-----------|-----------|-----------------|
| 猫 | 2.0 | 7.39 | $7.39 / 10.48 = 0.71$ |
| 狗 | 1.0 | 2.72 | $2.72 / 10.48 = 0.26$ |
| 鸟 | -1.0 | 0.37 | $0.37 / 10.48 = 0.03$ |

三个概率加起来 = 1.00。模型认为 71% 的可能性是猫。

**为什么用 $e^{z}$ 而不是直接用 $z$？** 因为 $z$ 可以是负数（概率不能是负数），而 $e^z$ 永远是正数。并且 $e^z$ 会放大差距——$z$ 大的类别得到更大的概率份额，让模型的"选择"更鲜明。

#### 交叉熵：衡量概率预测有多差

有了概率之后，怎么衡量"猜得好不好"？假设真实答案是"猫"（第 1 类），我们只关心模型给"猫"分配了多少概率：

$$
\mathcal{L}_{\text{CE}} = -\log \hat{p}_c
$$

其中 $c$ 是正确类别的编号（这里 $c = 1$）。

**为什么用 $-\log$？** 看一下它的行为：

| $\hat{p}_c$（给正确答案的概率） | $-\log \hat{p}_c$（损失） | 含义 |
|---|---|---|
| 0.99 | 0.01 | 很有信心且猜对了 → 损失几乎为零 |
| 0.71 | 0.34 | 有信心但不确定 → 损失较小 |
| 0.5 | 0.69 | 半猜半猜 → 损失中等 |
| 0.1 | 2.3 | 基本猜错了 → 损失大 |
| 0.01 | 4.6 | 几乎完全猜错 → 损失很大 |

$-\log$ 有两个好性质：
1. 概率越大，损失越小（我们希望模型给正确答案高概率）
2. 概率接近 0 时损失趋向无穷大（严厉惩罚"完全猜错"）

#### 完整的例子

继续上面的例子。真实类别是"猫"，模型给猫的概率是 $\hat{p}_1 = 0.71$：

$$
\mathcal{L} = -\log(0.71) = 0.34
$$

如果模型再训练一会儿，给猫的概率变成 0.95：

$$
\mathcal{L} = -\log(0.95) = 0.05
$$

损失从 0.34 降到了 0.05——模型在进步。

### 5.3 为什么分成两类任务

回归和分类是最基础的两类任务，区别在于输出的**性质**：

- **回归**：输出是连续数字（如温度 23.5°C），预测值和真实值之间有"距离"概念，所以用差的平方（MSE）衡量
- **分类**：输出是离散类别（如猫/狗/鸟），类别之间没有"远近"——猜成狗和猜成鸟不存在哪个离猫更近，所以用概率 + 交叉熵衡量"你对正确答案有多大信心"

### 5.4 其他任务类型

回归和分类不能覆盖所有任务，但大部分复杂任务最终都是它们的组合：

| 任务类型 | 输出 | 例子 | 本质上是 |
|---------|------|------|---------|
| 回归 | 连续数字 | 预测房价、温度 | — |
| 分类 | K 类中选一个 | 猫/狗/鸟、垃圾邮件 | — |
| 生成（如 GPT） | 一段文本 | 写文章、对话 | 每一步从词表中做分类（softmax + 交叉熵） |
| 目标检测 | 物体位置 + 类别 | 自动驾驶识别行人 | 回归（框坐标）+ 分类（框里是什么） |
| 图像分割 | 每个像素的类别 | 医学影像分割肿瘤 | 对每个像素做分类 |

在 MLP 基础阶段，理解回归和分类就够用了——它们是构建更复杂任务的"积木"。

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

### 7.3 完整数值例子：两层网络的反向传播

下面用一个两层网络，全部用具体数字算一遍，让你看清链式法则的每一环。

**网络结构**：1 个输入 → 1 个隐藏层（1 个神经元，ReLU 激活）→ 1 个输出（无激活）

**参数**：$w_1 = 0.5$，$b_1 = 0.1$，$w_2 = -0.3$，$b_2 = 0.2$

**输入**：$x = 2.0$，**真实值**：$y_{\text{true}} = 1.0$，**损失函数**：MSE = $(y - y_{\text{true}})^2$

#### 前向传播（从左到右算输出）

第一层：

$$z_1 = w_1 \cdot x + b_1 = 0.5 \times 2.0 + 0.1 = 1.1$$

$$h_1 = \text{ReLU}(z_1) = \text{ReLU}(1.1) = 1.1 \quad \text{（正数，原样保留）}$$

第二层：

$$z_2 = w_2 \cdot h_1 + b_2 = -0.3 \times 1.1 + 0.2 = -0.13$$

$$y = z_2 = -0.13 \quad \text{（输出层无激活函数）}$$

损失：

$$\mathcal{L} = (y - y_{\text{true}})^2 = (-0.13 - 1.0)^2 = (-1.13)^2 = 1.2769$$

预测是 $-0.13$，真实值是 $1.0$，差得远，损失 = 1.28。

#### 反向传播（从右到左算梯度）

目标：算出 $\frac{\partial \mathcal{L}}{\partial w_1}$、$\frac{\partial \mathcal{L}}{\partial b_1}$、$\frac{\partial \mathcal{L}}{\partial w_2}$、$\frac{\partial \mathcal{L}}{\partial b_2}$。

**第 1 步：损失对输出**

$$\frac{\partial \mathcal{L}}{\partial y} = 2(y - y_{\text{true}}) = 2(-0.13 - 1.0) = -2.26$$

**第 2 步：过第二层**（$y = z_2 = w_2 \cdot h_1 + b_2$）

对 $w_2$ 求导：$\frac{\partial y}{\partial w_2} = h_1 = 1.1$

$$\frac{\partial \mathcal{L}}{\partial w_2} = -2.26 \times 1.1 = -2.486$$

对 $b_2$ 求导：$\frac{\partial y}{\partial b_2} = 1$

$$\frac{\partial \mathcal{L}}{\partial b_2} = -2.26 \times 1 = -2.26$$

往前传的信号（第一层需要）：$\frac{\partial y}{\partial h_1} = w_2 = -0.3$

$$\frac{\partial \mathcal{L}}{\partial h_1} = -2.26 \times (-0.3) = 0.678$$

**第 3 步：过 ReLU**（$h_1 = \text{ReLU}(z_1)$）

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial h_1} \times \text{ReLU}'(z_1) = 0.678 \times 1 = 0.678$$

$z_1 = 1.1 > 0$，所以 ReLU 导数 = 1，梯度原样通过。**如果 $z_1 < 0$，这里就是 0，后面所有梯度全部归零——这就是 dead neuron。**

**第 4 步：过第一层**（$z_1 = w_1 \cdot x + b_1$）

对 $w_1$ 求导：$\frac{\partial z_1}{\partial w_1} = x = 2.0$

$$\frac{\partial \mathcal{L}}{\partial w_1} = 0.678 \times 2.0 = 1.356$$

对 $b_1$ 求导：$\frac{\partial z_1}{\partial b_1} = 1$

$$\frac{\partial \mathcal{L}}{\partial b_1} = 0.678 \times 1 = 0.678$$

#### 汇总

| 参数 | 梯度 | 含义 |
|------|------|------|
| $w_1 = 0.5$ | $1.356$ | 应该减小 $w_1$ |
| $b_1 = 0.1$ | $0.678$ | 应该减小 $b_1$ |
| $w_2 = -0.3$ | $-2.486$ | 应该增大 $w_2$ |
| $b_2 = 0.2$ | $-2.26$ | 应该增大 $b_2$ |

#### 更新参数（学习率 $\eta = 0.1$）

$$w_1' = 0.5 - 0.1 \times 1.356 = 0.364$$

$$b_1' = 0.1 - 0.1 \times 0.678 = 0.032$$

$$w_2' = -0.3 - 0.1 \times (-2.486) = -0.051$$

$$b_2' = 0.2 - 0.1 \times (-2.26) = 0.426$$

#### 验证：新参数再前向一次

$$z_1 = 0.364 \times 2.0 + 0.032 = 0.760$$

$$h_1 = \text{ReLU}(0.760) = 0.760$$

$$y = -0.051 \times 0.760 + 0.426 = 0.387$$

$$\mathcal{L} = (0.387 - 1.0)^2 = 0.376$$

损失从 **1.28 → 0.38**，一步就降了 70%。

#### 流程总图

```
前向（→）：
x=2.0 →[×w₁+b₁]→ z₁=1.1 →[ReLU]→ h₁=1.1 →[×w₂+b₂]→ y=-0.13 → L=1.28

反向（←）：
∂L/∂y = -2.26
   │×h₁          │×1           │×w₂
   ▼              ▼              ▼
∂L/∂w₂=-2.486  ∂L/∂b₂=-2.26  ∂L/∂h₁=0.678
                                │×ReLU'(z₁)=1
                                ▼
                              ∂L/∂z₁=0.678
                              │×x           │×1
                              ▼              ▼
                          ∂L/∂w₁=1.356  ∂L/∂b₁=0.678
```

每一步就是链式法则的**一环乘一环**，没有别的东西。

### 7.4 一般性理论推导

下面推导 $L$ 层 MLP 的反向传播公式。你会看到 7.3 的数值例子只是这些公式的特殊情况。

#### 符号约定

考虑一个 $L$ 层网络，第 $\ell$ 层（$\ell = 1, 2, \ldots, L$）的前向传播是：

$$\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$

$$\mathbf{h}^{(\ell)} = f^{(\ell)}(\mathbf{z}^{(\ell)})$$

其中：
- $\mathbf{h}^{(0)} = \mathbf{x}$（输入数据）
- $\mathbf{z}^{(\ell)}$ 是第 $\ell$ 层的**预激活值**（线性变换的结果）
- $\mathbf{h}^{(\ell)}$ 是第 $\ell$ 层的**输出**（激活后的结果）
- $f^{(\ell)}$ 是第 $\ell$ 层的激活函数（逐元素作用）
- $W^{(\ell)}$ 维度为 $d_\ell \times d_{\ell-1}$，$\mathbf{b}^{(\ell)}$ 维度为 $d_\ell \times 1$

最终损失为 $\mathcal{L}(\mathbf{h}^{(L)}, \mathbf{y})$，其中 $\mathbf{y}$ 是真实标签。

#### 核心量：误差信号 $\boldsymbol{\delta}^{(\ell)}$

定义第 $\ell$ 层的**误差信号**为损失对预激活值的梯度：

$$\boldsymbol{\delta}^{(\ell)} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$$

这是反向传播的核心量——一旦知道了每层的 $\boldsymbol{\delta}^{(\ell)}$，就能直接算出参数梯度。

#### 第 1 步：输出层的误差信号

最后一层 $\ell = L$ 的误差信号：

$$\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \odot f'^{(L)}(\mathbf{z}^{(L)})$$

其中 $\odot$ 是逐元素乘法，$f'^{(L)}$ 是激活函数的逐元素导数。

**MSE + 无激活输出层的例子**：$\mathcal{L} = \|\mathbf{h}^{(L)} - \mathbf{y}\|^2$，输出层无激活（$f^{(L)}$ 是恒等函数，$f' = 1$），则：

$$\boldsymbol{\delta}^{(L)} = 2(\mathbf{h}^{(L)} - \mathbf{y})$$

在 7.3 的例子中：$\delta^{(2)} = 2(-0.13 - 1.0) = -2.26$，和前面算的一致。

#### 第 2 步：误差信号的逐层回传（核心递推公式）

知道了第 $\ell + 1$ 层的 $\boldsymbol{\delta}^{(\ell+1)}$，怎么算第 $\ell$ 层的 $\boldsymbol{\delta}^{(\ell)}$？

$$\boxed{\boldsymbol{\delta}^{(\ell)} = \left( W^{(\ell+1)\top} \boldsymbol{\delta}^{(\ell+1)} \right) \odot f'^{(\ell)}(\mathbf{z}^{(\ell)})}$$

推导过程（链式法则）：

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell+1)}} \cdot \frac{\partial \mathbf{z}^{(\ell+1)}}{\partial \mathbf{h}^{(\ell)}} \cdot \frac{\partial \mathbf{h}^{(\ell)}}{\partial \mathbf{z}^{(\ell)}}$$

三项分别是：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell+1)}} = \boldsymbol{\delta}^{(\ell+1)}$（上一步已知）
- $\frac{\partial \mathbf{z}^{(\ell+1)}}{\partial \mathbf{h}^{(\ell)}} = W^{(\ell+1)}$（因为 $\mathbf{z}^{(\ell+1)} = W^{(\ell+1)} \mathbf{h}^{(\ell)} + \mathbf{b}^{(\ell+1)}$）
- $\frac{\partial \mathbf{h}^{(\ell)}}{\partial \mathbf{z}^{(\ell)}} = \text{diag}(f'^{(\ell)}(\mathbf{z}^{(\ell)}))$（激活函数逐元素求导）

乘起来就得到了上面的递推公式。

**直觉**：$W^{(\ell+1)\top}$ 把下一层的误差信号"投影回"当前层的维度，$f'$ 决定这层的每个神经元让多少信号通过。

**梯度消失/爆炸的来源**就在这里：从第 $L$ 层回传到第 $1$ 层，$\boldsymbol{\delta}$ 被反复乘以 $W^\top$ 和 $f'$。如果这些因子的模一直 < 1，$\boldsymbol{\delta}$ 指数衰减（消失）；一直 > 1，$\boldsymbol{\delta}$ 指数增长（爆炸）。

#### 第 3 步：从误差信号算参数梯度

有了 $\boldsymbol{\delta}^{(\ell)}$，参数梯度就是一步之遥：

$$\boxed{\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \cdot \mathbf{h}^{(\ell-1)\top}}$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}}$$

推导：因为 $\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}$，所以 $\frac{\partial \mathbf{z}^{(\ell)}}{\partial W^{(\ell)}} = \mathbf{h}^{(\ell-1)\top}$，$\frac{\partial \mathbf{z}^{(\ell)}}{\partial \mathbf{b}^{(\ell)}} = \mathbf{I}$。

注意：$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$ 的计算需要 $\mathbf{h}^{(\ell-1)}$——这是前向传播时保存的中间结果。**这就是为什么训练比推理更吃内存：前向传播时每层的输出都要存着，等反向传播时用。**

#### 算法总结

```
输入：数据 x, 真实标签 y, 学习率 η

前向传播（保存中间结果）：
  h⁽⁰⁾ = x
  for ℓ = 1 to L:
      z⁽ℓ⁾ = W⁽ℓ⁾ h⁽ℓ⁻¹⁾ + b⁽ℓ⁾     ← 保存 z⁽ℓ⁾
      h⁽ℓ⁾ = f⁽ℓ⁾(z⁽ℓ⁾)              ← 保存 h⁽ℓ⁾
  L = Loss(h⁽ᴸ⁾, y)

反向传播（逐层回传误差信号）：
  δ⁽ᴸ⁾ = ∂L/∂h⁽ᴸ⁾ ⊙ f'⁽ᴸ⁾(z⁽ᴸ⁾)     ← 输出层
  for ℓ = L-1 down to 1:
      δ⁽ℓ⁾ = (W⁽ℓ⁺¹⁾ᵀ δ⁽ℓ⁺¹⁾) ⊙ f'⁽ℓ⁾(z⁽ℓ⁾)   ← 核心递推

参数更新：
  for ℓ = 1 to L:
      W⁽ℓ⁾ ← W⁽ℓ⁾ - η · δ⁽ℓ⁾ · h⁽ℓ⁻¹⁾ᵀ
      b⁽ℓ⁾ ← b⁽ℓ⁾ - η · δ⁽ℓ⁾
```

整个算法就三个公式：输出层误差信号、误差信号递推、参数梯度。其余都是符号展开。

#### 标量版本：两层网络的完整推导

上面的公式有矩阵和求和符号，不太直觉。下面把网络固定为**两层、每层一个神经元、所有量都是标量**，把每一步的链式法则写到底，不跳步。

##### 网络定义

```
输入 x → 第一层 → 第二层 → 输出 y → 损失 L
```

写成公式，一共 5 个等式，从左到右：

$$z_1 = w_1 x + b_1 \quad \text{（第一层线性变换）}$$

$$h_1 = f(z_1) \quad \text{（第一层激活函数）}$$

$$z_2 = w_2 h_1 + b_2 \quad \text{（第二层线性变换）}$$

$$y = z_2 \quad \text{（输出层无激活——回归任务的标准做法）}$$

$$\mathcal{L} = (y - y_{\text{true}})^2 \quad \text{（MSE 损失）}$$

> **为什么输出层不加激活函数？** 回归任务要预测任意实数，加了 Sigmoid 会压到 (0,1)，加了 ReLU 不能输出负数。分类任务则不同：二分类用 Sigmoid，多分类用 Softmax。如果输出层也有激活 $g$，推导唯一的区别是上面的 $\frac{\partial y}{\partial z_2} = 1$ 变成 $g'(z_2)$，其余完全一样。

要求的东西：$\frac{\partial \mathcal{L}}{\partial w_1}$，$\frac{\partial \mathcal{L}}{\partial b_1}$，$\frac{\partial \mathcal{L}}{\partial w_2}$，$\frac{\partial \mathcal{L}}{\partial b_2}$。

##### 画出依赖关系

从 $\mathcal{L}$ 出发，往回看每个量依赖谁：

```
L 依赖 y
y 依赖 z₂
z₂ 依赖 w₂, h₁, b₂
h₁ 依赖 z₁
z₁ 依赖 w₁, x, b₁
```

链式法则就是沿着这条链，**从右往左逐段求导，然后乘起来**。

##### 求 $\frac{\partial \mathcal{L}}{\partial w_2}$（第二层权重）

$w_2$ 离 $\mathcal{L}$ 近，链比较短：$\mathcal{L} \leftarrow y \leftarrow z_2 \leftarrow w_2$

$$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2}$$

逐项算：

| 项 | 怎么算 | 结果 |
|---|---|---|
| $\frac{\partial \mathcal{L}}{\partial y}$ | $\mathcal{L} = (y - y_{\text{true}})^2$，对 $y$ 求导 | $2(y - y_{\text{true}})$ |
| $\frac{\partial y}{\partial z_2}$ | $y = z_2$，对 $z_2$ 求导 | $1$ |
| $\frac{\partial z_2}{\partial w_2}$ | $z_2 = w_2 h_1 + b_2$，对 $w_2$ 求导 | $h_1$ |

乘起来：

$$\boxed{\frac{\partial \mathcal{L}}{\partial w_2} = 2(y - y_{\text{true}}) \cdot 1 \cdot h_1}$$

##### 求 $\frac{\partial \mathcal{L}}{\partial b_2}$（第二层偏置）

和上面一样，只是最后一项变了：$z_2 = w_2 h_1 + b_2$，对 $b_2$ 求导 = 1

$$\boxed{\frac{\partial \mathcal{L}}{\partial b_2} = 2(y - y_{\text{true}}) \cdot 1 \cdot 1 = 2(y - y_{\text{true}})}$$

##### 求 $\frac{\partial \mathcal{L}}{\partial w_1}$（第一层权重）

$w_1$ 离 $\mathcal{L}$ 远，链更长：$\mathcal{L} \leftarrow y \leftarrow z_2 \leftarrow h_1 \leftarrow z_1 \leftarrow w_1$

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

逐项算：

| 项 | 怎么算 | 结果 |
|---|---|---|
| $\frac{\partial \mathcal{L}}{\partial y}$ | 同上 | $2(y - y_{\text{true}})$ |
| $\frac{\partial y}{\partial z_2}$ | 同上 | $1$ |
| $\frac{\partial z_2}{\partial h_1}$ | $z_2 = w_2 h_1 + b_2$，对 $h_1$ 求导 | $w_2$ |
| $\frac{\partial h_1}{\partial z_1}$ | $h_1 = f(z_1)$，对 $z_1$ 求导 | $f'(z_1)$ |
| $\frac{\partial z_1}{\partial w_1}$ | $z_1 = w_1 x + b_1$，对 $w_1$ 求导 | $x$ |

乘起来：

$$\boxed{\frac{\partial \mathcal{L}}{\partial w_1} = 2(y - y_{\text{true}}) \cdot 1 \cdot w_2 \cdot f'(z_1) \cdot x}$$

注意中间多出来的 $w_2 \cdot f'(z_1)$——这就是梯度**穿过第二层和激活函数**时被乘上的因子。层数越多，这样的因子越多，连乘后就可能爆炸或消失。

##### 求 $\frac{\partial \mathcal{L}}{\partial b_1}$（第一层偏置）

和 $w_1$ 一样，只是最后一项变了：$\frac{\partial z_1}{\partial b_1} = 1$

$$\boxed{\frac{\partial \mathcal{L}}{\partial b_1} = 2(y - y_{\text{true}}) \cdot 1 \cdot w_2 \cdot f'(z_1) \cdot 1}$$

##### 四个梯度放在一起看

| 参数 | 梯度公式 |
|------|---------|
| $w_2$ | $2(y - y_{\text{true}}) \cdot h_1$ |
| $b_2$ | $2(y - y_{\text{true}})$ |
| $w_1$ | $2(y - y_{\text{true}}) \cdot w_2 \cdot f'(z_1) \cdot x$ |
| $b_1$ | $2(y - y_{\text{true}}) \cdot w_2 \cdot f'(z_1)$ |

**观察**：
- $w_2$ 和 $b_2$ 的梯度只乘了 $2(y - y_{\text{true}})$ 和 $h_1$，很"直接"
- $w_1$ 和 $b_1$ 的梯度**多乘了 $w_2 \cdot f'(z_1)$**——这就是梯度从第二层传到第一层时被"缩放"的因子
- 如果 $f$ 是 Sigmoid，$f'(z_1) \leq 0.25$，第一层的梯度比第二层小至少 4 倍
- 如果 $f$ 是 ReLU 且 $z_1 > 0$，$f'(z_1) = 1$，缩放因子只有 $w_2$

##### 代入 7.3 的数字验证

$w_1=0.5, b_1=0.1, w_2=-0.3, b_2=0.2, x=2.0, y_{\text{true}}=1.0$

前向：$z_1=1.1, h_1=1.1, z_2=-0.13, y=-0.13$

代入公式：

$$\frac{\partial \mathcal{L}}{\partial w_2} = 2(-0.13 - 1.0) \times 1.1 = -2.26 \times 1.1 = -2.486 \quad \checkmark$$

$$\frac{\partial \mathcal{L}}{\partial w_1} = 2(-0.13 - 1.0) \times (-0.3) \times 1 \times 2.0 = -2.26 \times (-0.3) \times 1 \times 2.0 = 1.356 \quad \checkmark$$

和 7.3 数值例子完全一致。公式就是把数字例子里的具体数换成了字母。

### 7.5 和梯度消失/爆炸的联系

从上面两层的推导可以精确看出来源。回顾四个梯度：

| 参数 | 梯度公式 | 离输出层 |
|------|---------|---------|
| $w_2$ | $2(y - y_{\text{true}}) \cdot h_1$ | 近（第二层） |
| $w_1$ | $2(y - y_{\text{true}}) \cdot \boldsymbol{w_2 \cdot f'(z_1)} \cdot x$ | 远（第一层） |

第一层比第二层**多乘了一个 $w_2 \cdot f'(z_1)$**。这个乘积就是梯度"穿过一层"时的缩放因子。

两层只多乘一次，影响不大。但如果有 $N$ 层，第一层的梯度要连乘 $N-1$ 个这样的因子：

$$\underbrace{w_N \cdot f'(z_{N-1})}_{穿过第 N 层} \cdot \underbrace{w_{N-1} \cdot f'(z_{N-2})}_{穿过第 N-1 层} \cdots \underbrace{w_2 \cdot f'(z_1)}_{穿过第 2 层}$$

- **每个因子 < 1**（例如 Sigmoid 的 $f' \leq 0.25$，乘上 $w$ 后仍然 < 1）→ 连乘 $N-1$ 次后趋近于 0 → **梯度消失**
- **每个因子 > 1**（例如 $w$ 初始化过大）→ 连乘 $N-1$ 次后趋近于 $\infty$ → **梯度爆炸**

所以**解决思路都是让每个因子尽量接近 1**：

| 方法 | 怎么让因子 ≈ 1 |
|------|---------------|
| **ReLU** | 正区间 $f'=1$，去掉了 $f'$ 这半边的衰减 |
| **合理初始化**（He/Xavier） | 控制 $w$ 的大小，让 $w \cdot f'$ 的期望 ≈ 1 |
| **残差连接** | 给梯度开一条不经过 $w \cdot f'$ 连乘的直通路 |
| **梯度裁剪** | 暴力截断，防止爆炸（不治本但管用） |
| **Batch Normalization** | 不是专门为此设计，但间接有帮助（见下文） |

#### He/Xavier 初始化：为什么能防梯度爆炸？

核心想法非常简单：**让每一层输出的方差等于输入的方差**。

一层网络：$z_j = \sum_{i=1}^{n} W_{ji} \, h_i$（$n$ 个输入）。假设 $W_{ji}$ 和 $h_i$ 独立、均值为 0：

$$\text{Var}(z_j) = n \cdot \text{Var}(W_{ji}) \cdot \text{Var}(h_i)$$

要让 $\text{Var}(z_j) = \text{Var}(h_i)$（输出方差 = 输入方差）：

$$n \cdot \text{Var}(W_{ji}) = 1 \quad \Rightarrow \quad \boxed{\text{Var}(W_{ji}) = \frac{1}{n}}$$

这就是 **Xavier 初始化**。He 初始化考虑了 ReLU 砍掉负半轴（一半神经元输出为 0），所以方差 ×2：$\text{Var}(W_{ji}) = \frac{2}{n}$。

**为什么这能防梯度爆炸？** 方差 $= 1/n$ 意味着 $W \mathbf{h}$ 的"放大倍数"期望为 1 倍。反向传播中 $W^\top \boldsymbol{\delta}$ 也是类似的。每穿过一层，信号既不放大也不缩小，连乘多少层都不会爆炸或消失。

> **跟特征值的关系**：随机矩阵理论（Marchenko-Pastur 定律）告诉我们，$n \times n$ 随机矩阵每个元素方差为 $1/n$ 时，奇异值会集中在 1 附近。这跟上面的方差分析一致——奇异值 ≈ 1 意味着 $W$ 不放大也不缩小信号。但初始化的设计者（Glorot、He）并没有从特征值出发，他们就是做了上面这个简单的方差计算。

#### Batch Normalization：不是为了防梯度爆炸

BN 经常和梯度爆炸一起被提到，但它的**设计初衷并不是解决梯度问题**。

| 说法 | 状态 |
|------|------|
| BN 减少了 Internal Covariate Shift（每层输入分布漂移） | 原始论文的动机（Ioffe & Szegedy, 2015），但后来发现不是主要原因 |
| BN 平滑了损失地形，让优化更容易 | **真正起作用的原因**（Santurkar et al., NeurIPS 2018） |
| BN 允许更大的学习率 | 正确，是平滑地形的直接后果 |
| BN 缓解梯度消失/爆炸 | 正确，但是**副作用**，不是设计初衷 |

BN 的做法是在激活函数之前把 $z$ 归一化到均值 0、方差 1：

$$\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$$

然后用可学习参数 $\gamma, \beta$ 恢复表达能力：$\tilde{z} = \gamma \hat{z} + \beta$。

它间接帮助梯度的方式是：把 $z$ 拉回 0 附近，$f'(z)$ 不容易跑到极端值（比如 Sigmoid 的饱和区），缩放因子 $w \cdot f'(z)$ 因此更稳定。

> 类比：ReLU / 残差连接 = **专门修路防塌方**；BN = **把整个地形推平了**，路自然也不容易塌。

> **可靠程度**：Level 1（Xavier/He 初始化）· Level 2（BN 的机制解释，学界仍有讨论）
>
> **参考**：[Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a.html) · [He et al. 2015](https://arxiv.org/abs/1502.01852) · [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167) · [Santurkar et al. 2018](https://arxiv.org/abs/1805.11604)

这就是 [02-deep-mlp.md](02-deep-mlp.md) 第 1 节详细讨论的问题。

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


