# 02 - 神经网络基础

> 学习路径：Step 1（LLM 的地基，不懂这个后面全是黑箱）
> 前置知识：线性代数（向量、矩阵乘法）、微积分（偏导数、链式法则）
> 参考：[3Blue1Brown - Neural Networks 视频系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · [Michael Nielsen 在线书（免费）](http://neuralnetworksanddeeplearning.com/) · [Wikipedia - Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

## 核心问题

你做物理实验时经常要拟合数据——比如测了一堆 $(x_i, y_i)$ 数据点，然后假设 $y = ax + b$，用最小二乘法求出 $a$ 和 $b$。

这就是"学习"的最简单形式：**你选定一个函数的形式（$y = ax+b$），然后从数据中找出最好的参数（$a, b$）。**

但线性函数太简单了。如果数据的关系是复杂的非线性呢？你可能试过用多项式 $y = ax^2 + bx + c$，甚至更高阶。但如果输入不是一个数而是几百个数呢（比如一张图片有几万个像素值）？人工选函数形式就不现实了。

**神经网络的核心思想**：不人工选函数形式，而是构造一种"可以变成几乎任何函数"的通用结构，然后让数据自动决定它应该长什么样。

> 参考：[Wikipedia - Function approximation](https://en.wikipedia.org/wiki/Function_approximation)

---

## 第一步：一个"神经元"长什么样

名字听着玄乎，但一个神经元就是一个极其简单的数学操作：

$$y = f(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)$$

拆开看：

1. **输入**：$x_1, x_2, ..., x_n$ — 就是一组数字（比如物理量的测量值）
2. **权重**：$w_1, w_2, ..., w_n$ — 每个输入乘一个系数，表示"这个输入有多重要"
3. **偏置**：$b$ — 一个可调的常数，类似线性回归里的截距
4. **加权求和**：$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$ — 就是线性组合。用向量记号：$z = \mathbf{w} \cdot \mathbf{x} + b$
5. **激活函数**：$y = f(z)$ — 对求和结果做一个非线性变换（后面解释为什么需要）

**为什么叫"神经元"？** 1943 年 McCulloch 和 Pitts 提出这个模型时，灵感来自生物神经元：树突接收信号（输入），胞体整合信号（加权求和），超过阈值就放电（激活函数）。但**现在不需要在意生物类比**——把它当成一个简单的数学函数就行。

> 参考：[Wikipedia - Perceptron](https://en.wikipedia.org/wiki/Perceptron) · [Wikipedia - Artificial neuron](https://en.wikipedia.org/wiki/Artificial_neuron)

---

## 第二步：为什么要"激活函数"（非线性）

如果没有激活函数 $f$，神经元就是 $y = \mathbf{w} \cdot \mathbf{x} + b$，一个线性函数。

关键问题：**多个线性函数叠加还是线性函数。** 你在线性代数里知道，矩阵乘矩阵还是矩阵：

$$y = W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$$

不管叠多少层，最终都可以合并成 $y = W'\mathbf{x} + \mathbf{b}'$——一个单层线性变换。**叠层没有带来任何新的表达能力。**

所以每层之后必须加一个非线性函数 $f$，打断这种"可合并性"。常用的几种：

| 名称 | 公式 | 一句话 |
|------|------|--------|
| **ReLU** | $f(z) = \max(0, z)$ | 负数变 0，正数不变。最简单，用得最多 |
| **Sigmoid** | $f(z) = \frac{1}{1+e^{-z}}$ | 把任意数压缩到 (0,1)。早期流行，现在较少用 |
| **GELU** | $f(z) \approx z \cdot \sigma(1.702z)$ | ReLU 的平滑版，Transformer/LLM 中常用 |

你不需要记住所有激活函数。记住核心思想：**非线性让叠层有意义。没有它，"深度"学习就不存在。**

> 参考：[Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function) · [3Blue1Brown - ReLU 解释](https://www.youtube.com/watch?v=aircAruvnKk&t=720s)

---

## 第三步：把神经元组成"网络"

一个神经元只能做很简单的事。把很多神经元连起来，就组成了**神经网络**。

最基本的结构叫**前馈网络（Feed-forward Network）**或**多层感知机（MLP）**：

```
输入层          隐藏层 1        隐藏层 2         输出层
x₁ ──┐     ┌── h₁⁽¹⁾ ──┐     ┌── h₁⁽²⁾ ──┐     ┌── y₁
x₂ ──┼──→──┤── h₂⁽¹⁾ ──┼──→──┤── h₂⁽²⁾ ──┼──→──┤── y₂
x₃ ──┘     └── h₃⁽¹⁾ ──┘     └── h₃⁽²⁾ ──┘     └── y₃
```

- **输入层**：你的原始数据（比如 3 个数字）
- **隐藏层**：中间的计算层。每个节点接收上一层所有节点的输出，做加权求和+激活
- **输出层**：最终结果

数学上，每一层做的事情：

$$\mathbf{h}^{(\ell)} = f\left(W^{(\ell)} \cdot \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right)$$

其中 $W^{(\ell)}$ 是一个矩阵（存着这一层所有连接的权重），$\mathbf{b}^{(\ell)}$ 是偏置向量，$f$ 是激活函数。

**整个网络就是这些矩阵乘法和非线性变换的多次叠加。** 网络的"知识"全部存储在这些权重矩阵 $W^{(1)}, W^{(2)}, ...$ 和偏置 $\mathbf{b}^{(1)}, \mathbf{b}^{(2)}, ...$ 中。

**一个深刻的数学结论**：只要有一个隐藏层且神经元足够多，前馈网络可以以任意精度逼近任何连续函数。这叫**万能近似定理**（Universal Approximation Theorem）。换句话说，这个结构足够"灵活"，可以拟合几乎任何输入-输出关系。

> 参考：[Michael Nielsen - 可视化万能近似](http://neuralnetworksanddeeplearning.com/chap4.html) · [Wikipedia - Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

## 第四步：怎么衡量"学得好不好" — 损失函数

网络有了结构，但权重一开始是随机的，输出是乱的。怎么衡量"当前的输出有多差"？

这就是**损失函数（Loss Function）**的作用——一个数字，告诉你当前参数下模型的预测和真实值差多远。

**回归任务**（预测一个数值）：用**均方误差（MSE）**

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

就是预测值和真实值之差的平方的平均。你在物理实验中用最小二乘法拟合直线，本质上就是在最小化 MSE。

**分类任务**（预测属于哪一类）：用**交叉熵（Cross-Entropy）**

假设有 $K$ 类，网络输出每类的概率 $\hat{p}_1, \hat{p}_2, ..., \hat{p}_K$（通过 softmax 函数把任意数值转换为概率分布）：

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \quad \text{（softmax）}$$

如果真实类别是第 $c$ 类，损失为：

$$\mathcal{L} = -\log \hat{p}_c$$

直觉：如果模型对正确答案很有信心（$\hat{p}_c \approx 1$），$-\log(1) = 0$，损失很小。如果模型对正确答案没信心（$\hat{p}_c \approx 0.01$），$-\log(0.01) = 4.6$，损失很大。

**和 LLM 的直接联系**：LLM 的训练目标就是交叉熵。给定"今天天气真"，预测下一个词的概率分布，如果正确答案是"好"，损失就是 $-\log P(\text{"好"})$。训练就是让这个损失尽可能小。

> 参考：[Wikipedia - Loss function](https://en.wikipedia.org/wiki/Loss_function) · [Wikipedia - Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) · [Wikipedia - Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

---

## 第五步：怎么找到最好的参数 — 梯度下降

有了损失函数 $\mathcal{L}(\theta)$（$\theta$ 代表所有权重和偏置），"学习"就变成了一个优化问题：**找到让 $\mathcal{L}$ 最小的 $\theta$。**

你在物理中求过函数极值：令导数等于零。但神经网络有几百万甚至几十亿个参数，解析解不存在。怎么办？

**梯度下降（Gradient Descent）**：别试图一步到位，而是一小步一小步地往"下坡方向"走。

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

- $\frac{\partial \mathcal{L}}{\partial \theta}$：损失对参数的偏导数（梯度），指向"上坡"方向
- $\eta$：**学习率**，步长有多大。太大会跳过最低点，太小会太慢
- 减号：往梯度的反方向走（下坡）

**类比**：你蒙着眼站在一个山谷里，想找到最低点。你能做的就是用脚感受当前位置的坡度（梯度），然后朝着最陡的下坡方向迈一小步。重复很多次，最终走到谷底附近。

实际训练中，不用全部数据算梯度（太慢），而是每次随机取一小批数据（**mini-batch**，通常 32-512 个样本）算一个"近似梯度"，然后更新参数。这叫**随机梯度下降（SGD）**。

现代 LLM 训练用的优化器叫 **Adam**，它在 SGD 基础上自动调整每个参数的步长，效果更好。你不需要现在理解 Adam 的细节，只需要知道它是"更聪明的梯度下降"。

> 参考：[3Blue1Brown - Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) · [Wikipedia - Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) · [Wikipedia - Adam optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)

---

## 第六步：怎么高效计算梯度 — 反向传播

梯度下降需要 $\frac{\partial \mathcal{L}}{\partial \theta}$，而网络可能有几十亿个参数。对每个参数分别计算偏导数是不现实的。

**反向传播（Backpropagation）** 利用微积分的**链式法则**，从输出层往回一层一层地算，只需要一次"前向"（算输出）和一次"反向"（算梯度）就能得到所有参数的梯度。

用一个最简单的例子建立直觉。假设网络只有一层：

$$z = wx + b \quad \rightarrow \quad y = f(z) \quad \rightarrow \quad \mathcal{L} = (y - y_{\text{true}})^2$$

要算 $\frac{\partial \mathcal{L}}{\partial w}$，用链式法则一步步来：

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

- $\frac{\partial \mathcal{L}}{\partial y} = 2(y - y_{\text{true}})$ — 损失对输出的导数
- $\frac{\partial y}{\partial z} = f'(z)$ — 激活函数的导数
- $\frac{\partial z}{\partial w} = x$ — 线性部分对权重的导数

三个都是已知量！乘起来就得到了 $w$ 的梯度。

对于多层网络，原理完全一样——从最后一层开始，逐层往回乘。每一层的"误差信号"传给上一层，上一层再传给更上一层。这就是"反向传播"名字的由来。

**关键洞察**：反向传播的计算量大约只是前向传播的 2 倍。对于有 10 亿个参数的 LLM 来说，如果不用反向传播而是对每个参数做数值微分（微小扰动法），需要前向传播 10 亿次。反向传播只需要一次前向+一次反向。**没有反向传播就没有深度学习。**

> 参考：[3Blue1Brown - Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) · [Wikipedia - Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) · [Michael Nielsen - Ch.2](http://neuralnetworksanddeeplearning.com/chap2.html)

---

## 第七步：过拟合 — 学"太好"反而不好

你可能遇到过这个情况：一个 10 次多项式可以完美通过 5 个数据点，但在数据点之间疯狂振荡，预测新数据时一塌糊涂。

这叫**过拟合（Overfitting）**：模型把训练数据的噪声也"记住"了，而不是学到了真正的规律。

神经网络参数越多越容易过拟合。常见的对策：

- **正则化（Regularization）**：在损失函数中加一个惩罚项 $\lambda \|\theta\|^2$，阻止权重变得太大
- **Dropout**：训练时随机关掉一部分神经元，迫使网络不依赖任何单个节点
- **Early Stopping**：监控"测试数据上的表现"，一旦开始变差就停止训练

有趣的是，现代 LLM 因为参数量极大而训练数据也极多，过拟合反而不是主要问题。但理解这个概念对理解训练过程很重要。

> 参考：[Wikipedia - Overfitting](https://en.wikipedia.org/wiki/Overfitting) · [Wikipedia - Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))

---

## 总结：训练神经网络的完整流程

把上面串起来：

```
1. 定义网络结构（几层、每层多少神经元、什么激活函数）
     ↓
2. 随机初始化所有权重
     ↓
3. 前向传播：输入数据，逐层计算，得到预测输出
     ↓
4. 计算损失：用损失函数衡量预测和真实值的差距
     ↓
5. 反向传播：用链式法则从后往前算出所有参数的梯度
     ↓
6. 更新参数：沿梯度反方向走一小步
     ↓
7. 回到第 3 步，用下一批数据重复。重复几万到几百万次。
```

**LLM 的训练和这个流程完全一样**——只是网络结构从简单 MLP 变成了 Transformer，输入从数值变成了文字（需要先转成数字），损失函数是交叉熵（预测下一个词的概率），参数量从几千变成了几十亿。但基本循环一模一样。

---

## 和 LLM 的联系

| 这一章学的 | 在 LLM 里是什么 |
|-----------|----------------|
| 神经元 | Transformer 中的每一步计算 |
| 权重矩阵 | Attention 的 $W_Q, W_K, W_V$ 和 FFN 的权重 |
| 激活函数（GELU） | FFN 中使用 |
| 交叉熵损失 | LLM 的训练目标（预测下一个词） |
| 反向传播 | 计算梯度的方法，完全一样 |
| Adam 优化器 | LLM 训练的标准优化器 |

---

## 理解检测

**Q1**：用你自己的话解释：为什么没有激活函数的多层网络和单层线性网络没有区别？

你的回答：



**Q2**：梯度下降的"学习率"如果设得太大会怎样？太小会怎样？

你的回答：



**Q3**：LLM 的训练目标是"最小化交叉熵"。假设正确答案是"好"，模型给"好"的概率从 0.01 提升到 0.99，损失（$-\log p$）从多少变到多少？这个变化在直觉上合理吗？

你的回答：



