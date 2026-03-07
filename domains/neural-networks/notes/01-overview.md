# 01 - 神经网络：全景概览

## 这个领域是什么

你在 LLM/02 中已经学了神经网络的基本骨架：神经元（加权求和 + 激活函数）→ 多层网络 → 损失函数 → 反向传播 → 梯度下降。那是一个最小化的入门，让你能理解 LLM 的基础。

现在我们把视野打开。神经网络这个领域实际上有三个大方向：

1. **架构设计**：针对不同类型的数据（图像、序列、图），人们设计了不同的网络结构。每种结构都利用了数据的某种特性（空间局部性、时序依赖、图连接），理解"为什么这样设计"是核心。

2. **训练的数学**：为什么梯度下降能找到好的解？为什么参数比数据多几个数量级的网络不过拟合？损失函数的"地形"长什么样？这些是深度学习理论要回答的问题。

3. **实现与工程**：怎么用 PyTorch 把想法变成代码？怎么调参？怎么调试一个不收敛的模型？

> 参考：[Wikipedia - Deep learning](https://en.wikipedia.org/wiki/Deep_learning) · [Goodfellow 等《Deep Learning》（免费在线）](https://www.deeplearningbook.org/) · [Dive into Deep Learning（免费在线）](https://d2l.ai/)

## 学什么、怎么学

### 一、架构

| 步骤 | 架构 | 核心思想 | 适用数据 | 参考 |
|------|------|---------|---------|------|
| 1 | 深入 MLP | 初始化、BatchNorm、残差连接——让深层网络能训练起来 | 表格数据、通用 | [Deep Learning Book Ch.6-8](https://www.deeplearningbook.org/) |
| 2 | CNN（卷积神经网络） | 利用图像的局部性和平移不变性：卷积核只看一小块区域 | 图像、信号 | [Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) · [d2l.ai Ch.7](https://d2l.ai/chapter_convolutional-neural-networks/index.html) |
| 3 | RNN / LSTM | 用循环结构处理变长序列，用门控机制解决梯度消失 | 文本、时间序列 | [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network) · [Colah's blog - Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 4 | Transformer | 用 Attention 替代循环，可以并行处理序列（已在 LLM/04-05 覆盖） | 文本、图像、多模态 | [Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) |
| 5 | AutoEncoder / VAE | 把数据压缩到低维再重建——学到数据的"本质结构" | 生成、降维 | [Wikipedia](https://en.wikipedia.org/wiki/Variational_autoencoder) · [Tutorial by Kingma](https://arxiv.org/abs/1906.02691) |
| 6 | GAN | 两个网络对抗：一个生成假数据，一个判断真假 | 图像生成 | [Wikipedia](https://en.wikipedia.org/wiki/Generative_adversarial_network) · [原始论文](https://arxiv.org/abs/1406.2661) |
| 7 | GNN（图神经网络） | 在图结构（节点+边）上做消息传递 | 分子、社交网络、推荐系统 | [Wikipedia](https://en.wikipedia.org/wiki/Graph_neural_network) · [Distill - A Gentle Intro to GNNs](https://distill.pub/2021/gnn-intro/) |

**架构设计的统一视角**：每种架构都是在回答同一个问题——**如何把数据的结构先验（inductive bias）编码到网络中？**
- MLP：没有任何先验，全连接，最灵活但最低效
- CNN：假设空间局部性和平移不变性
- RNN：假设序列中相邻元素相关
- Transformer：不假设局部性，让模型自己学关注哪里
- GNN：假设数据有图结构

> 参考：[Wikipedia - Inductive bias](https://en.wikipedia.org/wiki/Inductive_bias)

### 二、训练的数学

| 步骤 | 主题 | 核心问题 | 参考 |
|------|------|---------|------|
| 1 | 优化理论 | 损失面长什么样？SGD 为什么能找到好解？鞍点 vs 局部最小值 | [Deep Learning Book Ch.8](https://www.deeplearningbook.org/contents/optimization.html) · [Wikipedia - SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) |
| 2 | 初始化与归一化 | 为什么随机初始化的方式很重要？BatchNorm / LayerNorm 为什么有效？ | [Xavier 初始化论文](http://proceedings.mlr.press/v9/glorot10a.html) · [BatchNorm 论文](https://arxiv.org/abs/1502.03167) |
| 3 | 正则化 | L2、Dropout、数据增强为什么能防止过拟合？ | [Deep Learning Book Ch.7](https://www.deeplearningbook.org/contents/regularization.html) |
| 4 | 泛化理论 | VC 维、PAC 学习、Rademacher 复杂度——经典框架为什么解释不了深度学习？ | [Wikipedia - VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension) · [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530) |
| 5 | 过参数化的谜题 | 为什么参数 >> 数据时模型反而不过拟合？Double descent 现象 | [Deep Double Descent 论文](https://arxiv.org/abs/1912.02292) · [Wikipedia](https://en.wikipedia.org/wiki/Double_descent) |

### 三、实现

| 步骤 | 内容 | 参考 |
|------|------|------|
| 1 | PyTorch 基础：张量、自动微分、nn.Module | [PyTorch 官方教程](https://pytorch.org/tutorials/) · [d2l.ai Ch.2](https://d2l.ai/chapter_preliminaries/index.html) |
| 2 | 手写 MLP（MNIST 手写数字识别） | [d2l.ai Ch.4](https://d2l.ai/chapter_multilayer-perceptrons/index.html) |
| 3 | 手写 CNN（CIFAR-10 图像分类） | [d2l.ai Ch.7](https://d2l.ai/chapter_convolutional-neural-networks/index.html) |
| 4 | 手写简单 Transformer | [Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 5 | 训练技巧：学习率调度、梯度裁剪、混合精度 | [d2l.ai Ch.12](https://d2l.ai/chapter_optimization/index.html) |

## 前沿方向

### 一、Neural Scaling Laws

核心发现：神经网络的测试损失和模型大小、数据量、计算量之间存在幂律关系。这个关系跨越了好几个数量级，几乎像物理定律一样精确。

最新理论（2026）：幂律可能来自数据中模式的 Zipf 分布——常见模式先被学到，稀有模式需要更大的模型。不同的资源瓶颈（参数受限 vs 数据受限）给出不同的 scaling exponent，统一了之前看似矛盾的 Kaplan 和 Chinchilla 结果。

> 可靠程度：经验规律 Level 1-2，理论解释 Level 3-4
>
> 参考：[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) · [Chinchilla 2022](https://arxiv.org/abs/2203.15556) · [Effective Frontiers 2026](https://arxiv.org/abs/2602.02593)

### 二、Double Descent 与泛化谜题

经典统计理论说：模型复杂度增大，先欠拟合再过拟合（U 形曲线）。但深度学习中观察到的是：过了过拟合的峰值后，测试误差**又下降了**——这叫 double descent。最新研究甚至发现了 **triple descent**（在高维 NTK 下）。

这说明我们对泛化的经典理解是不完整的。过参数化网络可能通过某种隐式正则化找到了"简单"的解。

> 可靠程度：现象 Level 1（可复现），理论解释 Level 3-4
>
> 参考：[Deep Double Descent](https://arxiv.org/abs/1912.02292) · [NTK in High Dimensions](https://proceedings.mlr.press/v119/adlam20a.html) · [Wikipedia](https://en.wikipedia.org/wiki/Double_descent)

### 三、Neural Tangent Kernel（NTK）

核心思想：在无穷宽极限下，神经网络的训练动力学等价于一个**核方法**（kernel method）。NTK 提供了一个可以精确分析的框架来理解训练和泛化。

局限：真实网络是有限宽的，NTK 理论无法完全捕捉"特征学习"（网络中间层表示的变化）。2025 年的新工作扩展到了深度方向（depth-induced NTK）。

> 可靠程度：NTK 理论本身 Level 1（数学严格），对真实网络的适用性 Level 3-4
>
> 参考：[NTK 原始论文 - Jacot et al. 2018](https://arxiv.org/abs/1806.07572) · [Wikipedia](https://en.wikipedia.org/wiki/Neural_tangent_kernel) · [Depth-induced NTK 2025](https://arxiv.org/abs/2511.05585)

### 四、可解释性（Mechanistic Interpretability）

神经网络是"黑箱"——它能工作，但我们不知道里面到底学到了什么。Mechanistic interpretability 试图逆向工程网络内部的计算：

- 单个神经元在检测什么特征？
- 网络中是否存在可理解的"电路"（circuits）？
- Anthropic 的 Sparse Autoencoder 方法：把网络中间层分解为可解释的特征

> 可靠程度：方法论 Level 2-3，理解深度 Level 4
>
> 参考：[Anthropic - Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) · [Wikipedia - Explainable AI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)

### 五、架构搜索（NAS）

让算法自动搜索最优的网络架构，而不是人工设计。当前方向：不只优化准确率，同时优化推理效率（速度、内存）。

> 可靠程度：方法有效 Level 2，是否能超越人工设计 Level 3
>
> 参考：[Wikipedia - Neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) · [EfficientNet 论文](https://arxiv.org/abs/1905.11946)

### 前沿方向总结

| 方向 | 核心问题 | 可靠程度 |
|------|---------|---------|
| Scaling Laws | 性能和规模的关系为什么是幂律？ | 经验 L1-2，理论 L3-4 |
| Double Descent | 过参数化为什么不过拟合？ | 现象 L1，理论 L3-4 |
| NTK | 训练动力学能精确描述吗？ | 数学 L1，适用性 L3-4 |
| 可解释性 | 网络内部到底学到了什么？ | 方法 L2-3 |
| NAS | 能自动找到最优架构吗？ | 方法 L2，超越人工 L3 |

## 学习建议

- 架构、数学、实现三条线可以交叉推进，不必严格按顺序
- 每学一个架构，建议同时用 PyTorch 实现一个小例子
- 前沿理论可以在基础建立后选择性深入
- 每个文件末尾有检测问题，在文件里回答

---

## 理解检测

**Q1**：CNN 和 MLP 的核心区别是什么？为什么处理图像时 CNN 比 MLP 好得多？（提示：想想一张 256×256 的图像作为 MLP 的输入需要多少个权重。）

你的回答：



**Q2**：你在 LLM/02 中学到"参数越多越容易过拟合"。但现代 LLM 有几千亿参数，训练数据虽多但参数量远超数据量，为什么没有严重过拟合？你觉得可能的原因是什么？

你的回答：



**Q3**：架构设计的统一视角是"把数据的结构先验编码到网络中"。RNN 假设了什么先验？Transformer 假设了什么（或者说没假设什么）？这和 Transformer 取代 RNN 有什么关系？

你的回答：



