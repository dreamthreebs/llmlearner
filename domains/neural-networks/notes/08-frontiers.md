# 08 - 前沿理论：Neural Tangent Kernel、Scaling Laws、可解释性

> **主维度**：D5 前沿理论
> **关键关系**：
> - NTK (理论) --用于--> 泛化理论 (理论)：NTK 用于分析泛化
> - Scaling Laws (理论) --用于--> 泛化理论 (理论)：Scaling Laws 用于理解泛化随规模的变化
> - Double Descent (概念) --属于--> 泛化理论 (理论)：Double Descent 属于泛化理论
>
> **学习路径**：01-overview → 训练数学 → 泛化理论 → **本章**
>
> **前置知识**：梯度下降与反向传播、损失函数、过参数化与泛化的基本概念、MLP/CNN 架构、Transformer 架构（参见 `domains/LLM/notes/04-attention.md` 和 `05-transformer.md`）
>
> **参考**：
> - [Jacot et al. 2018 - Neural Tangent Kernel](https://arxiv.org/abs/1806.07572)
> - [Kaplan et al. 2020 - Scaling Laws](https://arxiv.org/abs/2001.08361)
> - [Hoffmann et al. 2022 - Chinchilla](https://arxiv.org/abs/2203.15556)
> - [Anthropic - Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
> - [Wikipedia - Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel)

---

## 1. Neural Tangent Kernel (NTK)

### 核心问题

训练一个神经网络——用梯度下降不断更新参数——本质上是一个复杂的非线性动力学过程。损失函数关于参数的地形（loss landscape）既不是凸的，也没有简单的解析解。一个自然的问题是：**能不能用数学精确描述神经网络的训练过程？**

NTK 理论给出了一个答案：在特定条件下（网络足够宽），可以。代价是这个精确描述对应的其实是一个"简化版"的网络行为。

### 1.1 核方法快速回顾

在进入 NTK 之前，需要先理解**核方法**（kernel methods）——一类经典的机器学习工具。如果你之前没学过核方法，这里给出最小化的介绍。

**核函数**（kernel function）$K(x, x')$ 是一个接受两个输入、返回一个标量的函数，它衡量的是两个输入在某个（可能非常高维的）特征空间中的相似度。形式上：

$$K(x, x') = \phi(x) \cdot \phi(x')$$

其中 $\phi(x)$ 是一个特征映射（feature map），把输入 $x$ 映射到高维空间。核函数的精妙之处在于：你可以直接计算 $K(x, x')$，**不需要显式地知道或计算 $\phi(x)$**——这叫"核技巧"（kernel trick）。

**核回归**（kernel regression）是用核函数做非参数回归的方法。给定训练数据 $\{(x_i, y_i)\}$，核回归的预测是：

$$\hat{f}(x) = \sum_{i=1}^{n} \alpha_i K(x, x_i)$$

其中系数 $\alpha_i$ 由训练数据决定（通过求解一个线性方程组）。直觉上，预测值是所有训练样本的加权组合，权重取决于新输入 $x$ 和每个训练样本 $x_i$ 的相似度。

核方法的优势是数学上非常干净：它是一个**线性**问题（关于系数 $\alpha_i$ 是线性的），有闭合解，泛化性质可以精确分析。劣势是不能自动学习好的特征表示——核函数 $K$ 是预先选定的，不随训练数据改变。

> 参考：[Wikipedia - Kernel method](https://en.wikipedia.org/wiki/Kernel_method) · [Wikipedia - Kernel regression](https://en.wikipedia.org/wiki/Kernel_regression)

### 1.2 NTK 的核心思想

现在把目光转回神经网络。考虑一个参数为 $\theta$ 的网络 $f(x; \theta)$，它对输入 $x$ 的输出是一个标量（为简化讨论）。当我们用梯度下降更新参数时，网络输出的变化由链式法则决定：

$$df(x; \theta) = \nabla_\theta f(x; \theta) \cdot d\theta$$

这里 $\nabla_\theta f(x; \theta)$ 是输出关于所有参数的梯度向量——它可以看作输入 $x$ 在"参数梯度空间"中的一个特征表示。

**NTK 的定义**：两个输入 $x$ 和 $x'$ 的 Neural Tangent Kernel 定义为：

$$K_{\text{NTK}}(x, x') = \nabla_\theta f(x; \theta) \cdot \nabla_\theta f(x'; \theta) = \sum_{p=1}^{P} \frac{\partial f(x; \theta)}{\partial \theta_p} \frac{\partial f(x'; \theta)}{\partial \theta_p}$$

其中 $P$ 是参数总数，求和遍历所有参数。

**直觉**：NTK 值 $K_{\text{NTK}}(x, x')$ 衡量的是"当我因为训练样本 $x'$ 更新了权重时，输入 $x$ 的输出会变化多少"。如果两个输入的梯度方向相似（NTK 值大），那更新其中一个的权重就会显著影响另一个的输出。

把它和上面的核函数定义 $K(x, x') = \phi(x) \cdot \phi(x')$ 对比：NTK 就是以 $\nabla_\theta f(x; \theta)$ 为特征映射 $\phi$ 的核函数。

### 1.3 Jacot et al. (2018) 的关键定理

Jacot、Gabriel 和 Hongler 在 2018 年证明了一个关键定理（[arXiv:1806.07572](https://arxiv.org/abs/1806.07572)）：

> **定理**（非正式）：对于全连接网络，当每层宽度趋于无穷时：
>
> 1. **初始化时**，NTK 收敛到一个确定性的核 $K^*$（不依赖于随机初始化的具体值）。
> 2. **训练过程中**，NTK **保持不变**——它"冻结"在初始值 $K^*$ 上，不随参数更新而改变。

这个定理的后果是深远的。因为 NTK 不变，训练动力学变成了**线性的**。具体来说，如果我们用连续时间的梯度流（gradient flow，即学习率趋于零的梯度下降），网络在训练数据上的输出向量 $\mathbf{u}(t) = (f(x_1; \theta(t)), \ldots, f(x_n; \theta(t)))$ 的演化满足：

$$\frac{d\mathbf{u}}{dt} = -K^* (\mathbf{u}(t) - \mathbf{y})$$

其中 $\mathbf{y} = (y_1, \ldots, y_n)$ 是目标值，$K^*$ 是 $n \times n$ 的 NTK 矩阵（$(K^*)_{ij} = K_{\text{NTK}}(x_i, x_j)$）。

这是一个**线性常微分方程**，解是：

$$\mathbf{u}(t) = \mathbf{y} - e^{-K^* t}(\mathbf{y} - \mathbf{u}(0))$$

如果 $K^*$ 是正定的（通常在数据量有限时成立），那么 $t \to \infty$ 时 $\mathbf{u}(t) \to \mathbf{y}$，即训练损失趋于零。这给出了"足够宽的网络在梯度下降下一定能拟合训练数据"的保证。

更重要的是，由于整个训练动力学等价于一个核回归，泛化行为可以用核方法的经典理论精确分析。

> 可靠程度：**Level 1**（数学定理，在无穷宽极限下严格成立）
>
> 参考：[Jacot et al. 2018](https://arxiv.org/abs/1806.07572) · [Wikipedia - Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel)

### 1.4 NTK 的局限

NTK 理论提供了一个漂亮的数学框架，但它有根本性的局限：

**（1）无穷宽假设在实际中不成立。** 实际网络的宽度是有限的。研究表明，即使把网络宽度增大一个数量级，经验 NTK 的行为也不会显著接近无穷宽极限（[Chizat & Bach 2019](https://arxiv.org/abs/1812.07956); [Lee et al. 2020](https://arxiv.org/abs/2012.00152)）。NTK 理论需要网络宽度比深度高出几个数量级才能适用，而实际大模型的设计并不满足这一条件。

**（2）NTK 冻结意味着没有特征学习。** 这是最根本的问题。NTK 不变意味着 $\nabla_\theta f(x; \theta)$——也就是网络中间层的表示——在训练过程中不改变。网络只是在固定的特征映射上做线性回归。

但实际深度学习的力量恰恰来自**特征学习**（feature learning）：网络通过训练学会把原始数据转化为越来越好的中间表示。例如 CNN 学会从像素中提取边缘、纹理、物体部件；Transformer 学会在 attention 中捕捉语义关系。这些都是中间层表示在训练过程中发生剧烈变化的结果。

**（3）"Lazy training" vs "feature learning" 的区分。** NTK 理论描述的regime被称为"lazy training"（[Chizat et al. 2019](https://arxiv.org/abs/1812.07956)）——参数变化很小，网络几乎是在初始化点附近做线性近似。相对的是"rich/feature learning regime"——参数变化显著，网络学到了全新的特征表示。

最新研究（2025）发现，支持特征学习的最大模型宽度远小于实际大模型的宽度，这意味着在某些尺度上，网络可能从特征学习 regime 过渡到类 NTK 的 regime（[Beyond Scaling Curves, 2025](https://arxiv.org/abs/2507.05035)）。这个过渡的具体机制仍在研究中。

> 可靠程度：局限性本身 **Level 1**（有充分的理论和实验证据）；lazy training 与 feature learning 的清晰划分 **Level 2-3**（边界在实际网络中不总是清晰的）
>
> 参考：[Chizat et al. 2019 - Lazy Training](https://arxiv.org/abs/1812.07956) · [Lee et al. 2020](https://arxiv.org/abs/2012.00152) · [Beyond Scaling Curves 2025](https://arxiv.org/abs/2507.05035)

---

## 2. Neural Scaling Laws

### 核心问题

如果你训练一个神经网络，然后把模型参数量翻倍，性能会提升多少？把训练数据翻倍呢？计算量翻倍呢？答案惊人地规律：**测试损失和这些量之间存在幂律关系**，而且这个关系跨越了好几个数量级，几乎像物理定律一样精确。

### 2.1 经验发现

Kaplan 等人在 2020 年的实验中（[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)）发现，对于语言模型的交叉熵损失 $L$：

$$L(N) \approx L_{\infty} + \left(\frac{N_0}{N}\right)^{\alpha_N}$$

$$L(D) \approx L_{\infty} + \left(\frac{D_0}{D}\right)^{\alpha_D}$$

其中：
- $N$ 是模型参数量，$D$ 是训练数据量（token 数）
- $L_{\infty}$ 是"不可约损失"（irreducible loss）——即使模型和数据无穷大也无法消除的损失（来自语言本身的随机性）
- $N_0, D_0$ 是尺度常数，$\alpha_N, \alpha_D$ 是幂律指数
- 对语言模型，经验值大约 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$

在双对数坐标下，这些关系呈现为直线——这是幂律的标志性特征，你在物理中应该很熟悉（类比：光学中的 Stefan-Boltzmann 定律 $j \propto T^4$，虽然物理机制完全不同，但数学形式类似）。

更令人惊讶的是，计算量 $C$（以 FLOPs 计，即浮点运算总次数，Floating Point Operations）和损失之间也有幂律关系：

$$L(C) \approx L_{\infty} + \left(\frac{C_0}{C}\right)^{\alpha_C}$$

这意味着：如果你有固定的计算预算，你可以预测能达到的最佳性能。

> 可靠程度：**Level 1-2**（经验上非常稳健，跨多种架构和数据集可复现；具体的指数值取决于实验设置）
>
> 参考：[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)

### 2.2 Kaplan vs Chinchilla：怎么分配资源？

Scaling laws 不只是学术好奇心——它们直接回答了一个工程问题：**给定固定的计算预算，应该训练一个多大的模型？**

**Kaplan et al. (2020)** 的分析认为：参数量的 scaling 指数比数据量的更大，因此在计算预算固定时，应该优先增大模型（用更大的模型训练更少的步数）。这导致了 GPT-3 的训练策略：1750 亿参数，但训练数据只有约 3000 亿 token。

**Hoffmann et al. (2022)，即 Chinchilla 论文**（[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)），修正了这个结论。他们的发现是：

> **最优策略**：参数量 $N$ 和数据量 $D$ 应该**同比例增长**。具体来说，$N \propto C^{0.5}$, $D \propto C^{0.5}$。

这意味着 GPT-3 是"过大且训练不足的"——它的参数量太多，训练数据太少。Chinchilla 用 700 亿参数但 1.4 万亿 token 的数据，在相同计算预算下取得了更好的性能。

这个结论影响了整个行业的模型训练策略（LLaMA、Gemini 等都采用了更高的数据/参数比）。

> 可靠程度：**Level 2**（Chinchilla 的结论在实践中被广泛验证，但最优比例可能取决于具体场景，且最近的研究表明在数据受限和参数受限两种 regime 下最优策略不同）
>
> 参考：[Chinchilla 2022](https://arxiv.org/abs/2203.15556)

### 2.3 理论解释的尝试

幂律关系为什么存在？这是一个开放问题。以下是几个方向：

**（1）Zipf 分布假说**

"Effective Frontiers" 框架（[arXiv:2602.02593](https://arxiv.org/abs/2602.02593)）提出了一个统一的理论图景：

- 数据中的"可学习模式"（learnable patterns）按频率排列服从 **Zipf 分布**——少数模式非常常见，大量模式很稀有。Zipf 分布是一种幂律分布（rank $r$ 的项出现频率 $\propto r^{-s}$），你在物理中可能遇到过类似的长尾分布。
- 模型在训练过程中逐步覆盖这些模式：先学会常见模式，再学稀有模式。
- 定义一个"有效前沿"（effective frontier）$k^*$：模型已学会频率排名前 $k^*$ 的模式，尾部的还没学。损失主要来自尾部。
- $k^*$ 受三种瓶颈限制：模型容量（$N$）、数据覆盖（$D$）、优化计算（$C$）。不同的瓶颈主导时，得到不同的 scaling exponent。
- **统一 Kaplan 和 Chinchilla**：两者的不同结论对应不同的瓶颈 regime。Kaplan 的实验处于参数受限 regime（模型太小），Chinchilla 的实验接近两者平衡的 regime。

**（2）随机图上的 Scaling**

Baek 等人（[arXiv:2601.10684](https://arxiv.org/abs/2601.10684)）发现，即使在随机图上训练 next-token prediction（数据没有明显的 Zipf 结构），scaling laws 仍然出现。这表明幂律可能不完全依赖于数据的统计结构，网络的学习动力学本身可能也会产生 scaling 行为。

**（3）冗余度解释**

另一条路线（[arXiv:2509.20721](https://arxiv.org/abs/2509.20721)）将 scaling laws 解释为"冗余度定律"：scaling 指数取决于数据的冗余度（由协方差谱的多项式尾部决定），而非一个普适常数。

> 可靠程度：**Level 3-4**（这些理论各有实验支持，但尚未形成共识。Zipf 假说目前最有解释力，但可能只是故事的一部分）
>
> 参考：[Effective Frontiers 2026](https://arxiv.org/abs/2602.02593) · [Random Graphs Scaling 2026](https://arxiv.org/abs/2601.10684) · [PNAS - Explaining Neural Scaling Laws 2024](https://www.pnas.org/doi/full/10.1073/pnas.2311878121)

---

## 3. 可解释性（Mechanistic Interpretability）

### 核心问题

一个训练好的神经网络有数十亿个参数。这些参数编码了什么"知识"？网络内部的计算过程能被人类理解吗？

这不只是学术问题。如果我们不理解模型在做什么，就无法判断它是否安全、是否会在关键场景下失败、是否存在系统性偏见。**可解释性**（interpretability）试图打开这个黑箱。

**机械可解释性**（mechanistic interpretability）是可解释性的一个子方向，目标是逆向工程网络内部的具体计算机制——不只是"哪些输入特征重要"（这是传统可解释性的范畴），而是"网络内部的哪些神经元以怎样的方式协作来完成特定任务"。

> 参考：[Wikipedia - Explainable artificial intelligence](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) · [Anthropic Interpretability Research](https://www.anthropic.com/research#interpretability)

### 3.1 单个神经元的特征可视化

最直接的问题：一个神经元到底在"检测"什么？

对于 CNN，研究者找到了清晰的层次结构（[Zeiler & Fergus 2014](https://arxiv.org/abs/1311.1901); [Olah et al. 2017](https://distill.pub/2017/feature-visualization/)）：

- **浅层**（靠近输入）：检测简单的视觉元素——边缘、颜色梯度、纹理
- **中间层**：检测复杂纹理和物体部件——眼睛、轮子、毛皮
- **深层**（靠近输出）：检测高级语义——整只狗、整辆车、特定场景

特征可视化的方法是：固定网络权重，用梯度上升（gradient ascent）找到能最大化某个神经元激活值的输入图像。生成的图像直观地展示了该神经元"关注"什么。

这给出了一个令人振奋的信号：网络内部并非完全不可理解。至少在 CNN 中，单个神经元确实学到了有意义的特征。

> 可靠程度：**Level 1-2**（CNN 的特征层次是可复现的稳健发现；但单个神经元的可解释性在更大网络和 Transformer 中变得模糊——一个神经元可能同时响应多种不相关的特征，这叫"多义性"（polysemanticity））
>
> 参考：[Distill - Feature Visualization](https://distill.pub/2017/feature-visualization/) · [Zeiler & Fergus 2014](https://arxiv.org/abs/1311.1901)

### 3.2 Circuits 假说

单个神经元能告诉我们一些东西，但网络的功能是由**神经元之间的协作**实现的。Anthropic 和 OpenAI 的研究者提出了 **Circuits 假说**（[Olah et al. 2020](https://distill.pub/2020/circuits/zoom-in/)）：

> **Circuits 假说**：神经网络内部存在可理解的"电路"（circuits）——由一组神经元和它们之间的连接构成的子网络，协作实现特定的、可理解的功能。

一个著名的例子是 **Induction Heads**——在 Transformer 中发现的一种电路（[Olah et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)）。

**Induction Heads 做什么？** 它们实现了一种简单的模式匹配：如果序列中之前出现过 `[A][B]`，当后面再次出现 `[A]` 时，induction head 会预测下一个 token 是 `[B]`。例如，如果模型看到过 "Harry Potter" 这个搭配，当后面出现 "Harry" 时，它会预测 "Potter"。

**Induction Heads 怎么实现的？** 它由两层 attention head 协作完成：
1. **"前一个 token" head**（previous token head）：把每个位置的信息复制到下一个位置（位置 $t$ 的 head 关注位置 $t-1$）。
2. **Induction head**：利用第一层复制的信息，在当前位置寻找"前面哪个位置的前一个 token 和当前 token 相同"，然后复制那个位置的 token 作为预测。

关键观察：induction heads 只在至少两层 attention 的模型中出现（因为需要两层协作），而且它们在训练过程中突然涌现（emergence），伴随着 in-context learning 能力的突然出现。

> 可靠程度：**Level 2-3**（在小到中等规模的 Transformer 中有充分验证；在超大模型中电路可能更复杂、更难分离，且 circuits 假说本身——网络是否真的由离散的可分离电路组成——仍有争议）
>
> 参考：[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) · [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) · [Circuits - Zoom In](https://distill.pub/2020/circuits/zoom-in/)

### 3.3 Sparse Autoencoder 方法

单个神经元的可解释性遇到了一个根本问题：**多义性**（polysemanticity）。在实际网络中，一个神经元往往同时编码多个不相关的概念。例如，一个神经元可能同时对"猫"和"汽车引擎盖"有高激活——它不对应任何单一的可理解特征。

为什么会这样？Anthropic 的"叠加假说"（superposition hypothesis）给出了解释（[Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)）：

> **叠加假说**：网络需要表示的"特征"数量远多于神经元数量。为了高效利用有限的神经元，网络把多个特征**叠加**（superpose）到同一组神经元上——类似于在低维空间中用近似正交的方向编码高维信息。

解决方案：用**稀疏自编码器**（Sparse Autoencoder, SAE）把叠加的表示分解回独立的特征。

**稀疏自编码器**是一种特殊的自编码器。自编码器（autoencoder）你可能在架构部分学过：它把输入压缩到低维，再重建。稀疏自编码器反过来——它把输入**扩展**到更高维的空间，但加上稀疏性约束，使得每个输入只激活少数几个高维特征。

具体做法：
1. 取网络某一层的激活向量 $\mathbf{h} \in \mathbb{R}^{d}$（$d$ 是隐藏层维度）
2. 通过编码器映射到高维空间：$\mathbf{z} = \text{ReLU}(W_e \mathbf{h} + \mathbf{b}_e) \in \mathbb{R}^{D}$，其中 $D \gg d$
3. 通过解码器重建：$\hat{\mathbf{h}} = W_d \mathbf{z} + \mathbf{b}_d$
4. 训练目标：最小化重建误差 $\|\mathbf{h} - \hat{\mathbf{h}}\|^2 + \lambda \|\mathbf{z}\|_1$

$\ell_1$ 惩罚项 $\lambda \|\mathbf{z}\|_1$ 鼓励 $\mathbf{z}$ 是稀疏的——大部分分量为零。每个非零分量对应一个被激活的"特征"。

Anthropic 在 2023 年的工作（[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)）展示了这种方法的威力：他们对 Claude 模型的中间层应用 SAE，成功分解出了数千个**单义特征**（monosemantic features）——每个特征对应一个清晰的概念，如"代码中的括号"、"DNA 序列"、"法律文本"等。

**最新进展（2025-2026）**：
- **Step-level SAE**（[arXiv:2603.03031](https://arxiv.org/abs/2603.03031)）：把 SAE 从 token 级别提升到推理步骤级别，分析 LLM 的推理过程。
- **SAE 的局限性**也在被揭示：不同 SAE 架构会隐含地假设特征的几何结构（如线性可分、均匀维度），当特征不满足这些假设时，SAE 无法正确恢复它们（[arXiv:2503.01822](https://arxiv.org/abs/2503.01822)）。
- **Circuit tracing**：Anthropic 在 2025 年开源了电路追踪工具（[open-source circuit tracing](https://anthropic.com/research/open-source-circuit-tracing)），允许研究者追踪特定模型行为对应的内部电路。

> 可靠程度：**Level 2-3**（SAE 方法在实践中有效，能分解出人类可理解的特征；但"分解出的特征是否真正对应网络的内部表示"仍有理论争议，且方法对 SAE 架构选择敏感）
>
> 参考：[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) · [Superposition Hypothesis](https://transformer-circuits.pub/2022/toy_model/index.html) · [SAE Concept Geometry 2025](https://arxiv.org/abs/2503.01822)

### 3.4 可解释性为什么重要

可解释性不只是有趣的学术问题——它与 **AI 安全**直接相关。

- **安全性**：如果我们不理解模型在做什么，就无法判断它是否在"欺骗"我们。一个模型可能在测试中表现良好，但内部学到了不安全的策略（如 sycophancy——迎合用户而非给出诚实回答）。可解释性工具可以帮助发现这类问题。
- **对齐验证**（alignment verification）：当我们试图让 AI 系统与人类价值观对齐时，我们需要验证对齐是否成功。黑箱测试不够——模型可能学会了通过测试但内部策略完全不同。机械可解释性提供了"看内部"的能力。
- **故障诊断**：当模型犯错时，可解释性帮助定位具体是哪个内部机制出了问题，而不只是知道"模型答错了"。

这是 Anthropic 和 OpenAI 等机构大力投资可解释性研究的核心动机。

> 可靠程度：可解释性的重要性 **Level 1**（业界共识）；当前方法是否足以保证安全 **Level 4**（远未解决）

---

## 三个方向的联系

NTK、Scaling Laws、可解释性看起来是三个独立的方向，但它们共享一个深层主题：**理解神经网络为什么工作**。

| 方向 | 问的问题 | 回答的方式 |
|------|---------|-----------|
| NTK | 训练动力学能精确描述吗？ | 在理想化极限下给出精确数学描述 |
| Scaling Laws | 性能如何随规模变化？ | 经验幂律 + 理论解释尝试 |
| 可解释性 | 网络内部学到了什么？ | 逆向工程内部表示和电路 |

NTK 告诉我们"训练在做什么"（至少在极限下），Scaling Laws 告诉我们"做多少能得到多少"，可解释性告诉我们"做完之后里面有什么"。三者合在一起，勾勒出深度学习理论的前沿全景。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $K(x, x') = \phi(x) \cdot \phi(x')$ | 核函数：衡量两输入在特征空间中的相似度 |
| $\hat{f}(x) = \sum_{i=1}^{n} \alpha_i K(x, x_i)$ | 核回归预测 |
| $K_{\text{NTK}}(x, x') = \nabla_\theta f(x; \theta) \cdot \nabla_\theta f(x'; \theta)$ | NTK 定义：输出关于参数的梯度内积 |
| $\frac{d\mathbf{u}}{dt} = -K^* (\mathbf{u}(t) - \mathbf{y})$ | 无穷宽网络训练动力学（梯度流） |
| $L(N) \approx L_{\infty} + (N_0/N)^{\alpha_N}$ | 损失随参数量 $N$ 的 Scaling Law（$\alpha_N \approx 0.076$） |
| $L(D) \approx L_{\infty} + (D_0/D)^{\alpha_D}$ | 损失随数据量 $D$ 的 Scaling Law（$\alpha_D \approx 0.095$） |
| $N \propto C^{0.5}$, $D \propto C^{0.5}$ | Chinchilla 最优策略：参数与数据同比例增长 |
| $\|\mathbf{h} - \hat{\mathbf{h}}\|^2 + \lambda \|\mathbf{z}\|_1$ | SAE 损失：重建误差 + $\ell_1$ 稀疏惩罚 |

---

## 理解检测

**Q1**：NTK 理论说，无穷宽网络的训练动力学等价于核回归。但实际中最强的模型（如 GPT-4、Claude）恰恰不在这个 regime 里。为什么？NTK 冻结（训练过程中 NTK 不变）意味着网络丢失了什么关键能力？

你的回答：



**Q2**：Kaplan (2020) 认为应该优先扩大模型，Chinchilla (2022) 认为参数和数据应该同比例增长。用 "Effective Frontiers" 框架的语言（容量瓶颈 vs 覆盖瓶颈），解释为什么两者的结论不同但都可以是"正确的"。

你的回答：



**Q3**：一个 Transformer 的某个神经元同时对"猫的图片"和"Python 代码中的括号"有高激活。这种现象叫什么？Anthropic 用什么方法把这些混在一起的概念分离出来？这个方法的核心约束是什么？

你的回答：


