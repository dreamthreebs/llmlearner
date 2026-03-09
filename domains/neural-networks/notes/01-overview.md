# 01 - 神经网络：全景概览

> **主维度**：全局（跨所有维度）
> **参考**：[Wikipedia - Deep learning](https://en.wikipedia.org/wiki/Deep_learning) · [Goodfellow 等《Deep Learning》](https://www.deeplearningbook.org/) · [d2l.ai](https://d2l.ai/)

## 这个领域是什么

你在 LLM/02 中已经学了神经网络的基本骨架：神经元（加权求和 + 激活函数）→ 多层网络 → 损失函数 → 反向传播 → 梯度下降。那是一个最小化的入门。现在我们把视野打开。

## 知识维度

为了不把这个领域的知识搅成一锅粥，我们先识别它的组织坐标轴。

**Step A — 候选抽取**：
1. 研究什么对象？→ 不同的网络结构（MLP、CNN、RNN、Transformer、GNN）
2. 用什么方法研究？→ 不同的训练组织方式（监督、自监督、对抗、变分推断）
3. 依赖什么理论？→ 优化理论、泛化理论、概率论
4. 面向什么应用？→ 分类、生成、序列预测、图推理……
5. 怎么实现？→ PyTorch、调参技巧、工程实践

**Step B — 合并同类项**：3 和 4 的边界清晰，无需合并。

**Step C — 混层检查**：确认所有候选都是"坐标轴"而非具体对象。CNN 是 D1 里的一个元素，不是维度本身。✓

**Step D — 最终维度集**：

| 维度 | 含义 | 核心问题 |
|------|------|---------|
| **D1 基础架构** | 网络的计算结构（building blocks） | 怎么处理不同类型的数据？ |
| **D2 训练范式** | 怎么组织训练过程 | 没有标签 / 想生成数据时怎么训练？ |
| **D3 训练数学** | 优化与泛化的理论 | 为什么能训练？为什么不过拟合？ |
| **D4 工程实现** | 框架与实战技巧 | 怎么把想法变成代码？ |
| **D5 前沿理论** | 尚未完全解决的开放问题 | Scaling Laws、NTK、可解释性 |

> **D1 vs D2 是初学者最容易混淆的一对。** CNN 是架构（D1），GAN 是范式（D2）。它们可以任意组合：DCGAN = CNN（D1） + GAN（D2）。架构定义"怎么计算"，范式定义"怎么训练"。

## 知识地图

### D1 基础架构 — 怎么处理数据

每种架构 = 一种 inductive bias（数据结构先验）：

| 架构 | Inductive Bias | 适用数据 | 参考 |
|------|---------------|---------|------|
| MLP | 无先验（全连接，最灵活最低效） | 通用/表格 | [DL Book Ch.6](https://www.deeplearningbook.org/) |
| CNN | 空间局部性 + 平移不变性 | 图像、信号 | [Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) · [d2l.ai Ch.7](https://d2l.ai/chapter_convolutional-neural-networks/index.html) |
| RNN/LSTM | 序列中相邻元素相关 | 文本、时间序列 | [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network) · [Colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| Transformer | 不假设局部性，自主学习关注哪里 | 文本、图像、多模态 | [Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))（已在 LLM/04-05 覆盖） |
| GNN | 数据有图结构（节点+边） | 分子、社交网络 | [Wikipedia](https://en.wikipedia.org/wiki/Graph_neural_network) · [Distill GNN 入门](https://distill.pub/2021/gnn-intro/) |

关键关系：
- GCN `generalizes` CNN to graphs（从网格卷积推广到图卷积）
- Transformer `contrasts-with` RNN（并行注意力 vs 循环处理）
- MLP `is-instance-of` 万能近似器（无结构先验的基线）

> [Wikipedia - Inductive bias](https://en.wikipedia.org/wiki/Inductive_bias)

### D2 训练范式 — 怎么组织训练

范式定义"用什么目标函数、怎么训练"，可以搭配任何 D1 架构：

| 范式 | 核心机制 | 参考 |
|------|---------|------|
| 监督学习 | 有标签，最小化预测误差 | [DL Book Ch.5](https://www.deeplearningbook.org/) |
| AutoEncoder | 重建：输入→压缩→重建，最小化重建误差 | [Wikipedia](https://en.wikipedia.org/wiki/Autoencoder) |
| VAE | 变分推断：压缩到概率分布，最大化 ELBO | [Kingma & Welling 2014](https://arxiv.org/abs/1312.6114) |
| GAN | 对抗博弈：生成器 vs 判别器的 minimax | [Goodfellow 2014](https://arxiv.org/abs/1406.2661) |
| Diffusion | 去噪：逐步加噪再学习逆向去噪 | [Ho et al. 2020](https://arxiv.org/abs/2006.11239) |

关键关系：
- VAE `generalizes` AutoEncoder（点估计 → 概率分布）
- GAN `contrasts-with` VAE（清晰但不稳定 vs 模糊但稳定）
- Diffusion `contrasts-with` GAN（2022 后在图像生成领域基本取代 GAN）

> **组合示例**：DCGAN = CNN + GAN，VQ-VAE = CNN + VAE + 离散化，GPT = Transformer + 自回归语言模型

### D3 训练数学 — 为什么能训练、为什么不过拟合

| 主题 | 核心问题 | 参考 |
|------|---------|------|
| 优化理论 | 损失面地形、SGD 为什么能找到好解 | [DL Book Ch.8](https://www.deeplearningbook.org/contents/optimization.html) |
| 初始化与归一化 | Xavier/He、BatchNorm/LayerNorm 为什么重要 | [Xavier 论文](http://proceedings.mlr.press/v9/glorot10a.html) · [BN 论文](https://arxiv.org/abs/1502.03167) |
| 正则化 | L2、Dropout 为什么防过拟合 | [DL Book Ch.7](https://www.deeplearningbook.org/contents/regularization.html) |
| 泛化理论 | VC 维、PAC、Rademacher、double descent | [Zhang et al. 2017](https://arxiv.org/abs/1611.03530) · [Double Descent](https://arxiv.org/abs/1912.02292) |

### D4 工程实现

| 内容 | 参考 |
|------|------|
| PyTorch 基础（Tensor、autograd、nn.Module） | [PyTorch 官方教程](https://pytorch.org/tutorials/) |
| 手写 MLP / CNN / Transformer | [d2l.ai](https://d2l.ai/) · [Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 训练技巧（学习率调度、梯度裁剪、混合精度） | [d2l.ai Ch.12](https://d2l.ai/chapter_optimization/index.html) |

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



