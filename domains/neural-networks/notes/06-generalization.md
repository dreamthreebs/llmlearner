# 06 - 正则化与泛化：为什么过参数化反而好

> **主维度**：D3 训练数学
> **关键关系**：Double Descent `contrasts-with` 经典偏差-方差 U 形曲线 · 隐式正则化 `used-for` 解释过参数化泛化
>
> **学习路径**：Step 5 / 7  
> **前置知识**：MLP 基础（01-overview）、CNN/RNN 架构、损失函数与梯度下降、基本优化理论  
> **参考**：  
> - [Deep Learning Book Ch.7 - Regularization](https://www.deeplearningbook.org/contents/regularization.html)  
> - [Wikipedia - VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension)  
> - [Wikipedia - PAC learning](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning)  
> - [Zhang et al. 2017 - Rethinking Generalization](https://arxiv.org/abs/1611.03530)  
> - [Nakkiran et al. 2019 - Deep Double Descent](https://arxiv.org/abs/1912.02292)

---

## 核心问题

你已经知道神经网络的基本训练流程：定义损失函数、反向传播计算梯度、用 SGD/Adam 更新参数。训练的目标是让 **训练损失** 尽可能小。但我们真正关心的是模型在 **没见过的新数据** 上的表现——这叫做 **泛化**（generalization）。

经典统计学有一个直觉：模型的参数越多，就越容易"记住"训练数据的噪声，从而在新数据上表现变差——这叫做 **过拟合**（overfitting）。如果你用一个 100 次多项式去拟合 10 个数据点，它可以完美穿过每个点，但曲线在数据点之间会剧烈震荡，完全没有预测能力。

> **可靠程度：Level 1**（教科书共识）

然而，现代神经网络完全违反了这个直觉。GPT-3 有 1750 亿参数，训练数据量远小于参数量，但它在各种任务上泛化得很好。一个 ResNet 可以有几千万参数，在 ImageNet（约 120 万张图像）上训练后，能很好地识别从未见过的图像。

**为什么过参数化的网络反而不过拟合？** 这是深度学习理论中最大的未解之谜之一，也是本章的核心问题。

---

## 1. 经典泛化理论回顾

在理解现代神经网络的"反常"之前，我们先搞清楚经典理论是怎么说的，这样才能看到它在哪里失效。

### 1.1 偏差-方差权衡（Bias-Variance Tradeoff）

**偏差**（bias）和 **方差**（variance）是两种不同类型的预测误差：

- **偏差**：模型的"平均预测"和真实值之间的差距。偏差大意味着模型太简单，无法捕捉数据的真实规律。比如用一条直线去拟合一个二次函数关系，不管怎么调参，直线都无法完美描述二次曲线——这就是高偏差。
- **方差**：模型的预测对训练数据的敏感程度。方差大意味着换一组训练数据，模型的预测会剧烈变化。比如用 20 次多项式去拟合 10 个数据点，每换一组数据点，拟合出的曲线都完全不同——这就是高方差。

对于一个回归问题，假设真实关系是 $y = f(x) + \epsilon$，其中 $\epsilon$ 是均值为零、方差为 $\sigma^2$ 的噪声。用模型 $\hat{f}(x)$ 来预测 $y$，在某个测试点 $x$ 上的 **期望预测误差** 可以分解为：

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{方差}} + \underbrace{\sigma^2}_{\text{不可约噪声}}$$

这里的期望 $\mathbb{E}$ 是对所有可能的训练集取的——想象你反复从同一个数据分布中抽取不同的训练集，每次训练出一个模型 $\hat{f}$。

经典的直觉是：
- 模型太简单 → 高偏差、低方差（欠拟合）
- 模型太复杂 → 低偏差、高方差（过拟合）
- 最优模型复杂度在中间，总误差呈 **U 形曲线**

这个框架在多项式回归、决策树等经典模型上工作得很好。但正如我们后面会看到的，它无法解释深度学习中的 double descent 现象。

> **可靠程度：Level 1**（教科书共识）  
> 参考：[Wikipedia - Bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) · [Deep Learning Book §5.4.4](https://www.deeplearningbook.org/contents/ml.html)

### 1.2 VC 维（Vapnik-Chervonenkis Dimension）

偏差-方差权衡给了直觉，但我们需要更精确的工具来衡量模型的"复杂度"。**VC 维**（Vapnik-Chervonenkis dimension）就是这样一种工具。

VC 维的核心想法是：衡量一个模型族（hypothesis class）能 **"打碎"（shatter）** 多少个数据点。

**打碎的定义**：给定 $n$ 个数据点，如果对于这些点的 **任意** 二分类标签组合（共 $2^n$ 种），模型族中都存在一个模型能完美分类它们，那么我们说这个模型族能打碎这 $n$ 个点。

**VC 维的定义**：一个模型族的 VC 维 $d_{\text{VC}}$ 是它能打碎的最大点数。

**例子**：考虑二维平面上的线性分类器（一条直线把平面分成两半）。

- 2 个点：不管这 2 个点怎么摆放，直线总能把所有 $2^2 = 4$ 种标签组合都分开。所以线性分类器能打碎 2 个点。
- 3 个点：只要 3 个点不共线，直线就能把所有 $2^3 = 8$ 种标签组合都分开。所以线性分类器能打碎 3 个点。
- 4 个点：存在某些标签组合（如 XOR 模式——对角的两个点同色，另外两个同色）是直线无法分开的。所以线性分类器 **不能** 打碎 4 个点。

因此，二维线性分类器的 VC 维 = 3。一般地，$d$ 维空间中的线性分类器的 VC 维 = $d + 1$。

**VC 维给出的泛化上界**：经典的 VC 不等式告诉我们，对于 VC 维为 $d_{\text{VC}}$ 的模型族，在 $n$ 个训练样本上训练后，测试误差与训练误差的差距（泛化间隙）以高概率满足：

$$\text{测试误差} - \text{训练误差} \leq O\left(\sqrt{\frac{d_{\text{VC}} \log n}{n}}\right)$$

直觉：VC 维越大（模型越复杂）或数据越少，泛化间隙越大。

**对神经网络的问题**：一个简单的两层 ReLU 网络如果有 $p$ 个参数，其 VC 维的量级大约是 $O(p \log p)$。一个有 1 亿参数的 ResNet，VC 维就是 $10^8$ 量级。对于 ImageNet 的 120 万个训练样本，上面的上界变得完全无意义——它给出的泛化间隙可以大于 1（而误差本身就在 0 到 1 之间）。

> **可靠程度：Level 1**（VC 维的定义和数学性质）  
> 参考：[Wikipedia - VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension) · [Understanding Machine Learning, Ch.6](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

### 1.3 PAC 学习框架

**PAC 学习**（Probably Approximately Correct）是一个形式化框架，用来回答这个问题：给定模型族和数据量，我们能以多高的信心（probably）保证模型学到了一个足够好（approximately correct）的解？

形式化地：我们说一个模型族 $\mathcal{H}$ 是 **PAC 可学习的**，如果存在一个学习算法，对于 **任意** 数据分布和 **任意** 精度要求 $\epsilon > 0$、置信度要求 $\delta > 0$，只需要 $n(\epsilon, \delta)$ 个训练样本，就能以至少 $1 - \delta$ 的概率找到一个模型 $h \in \mathcal{H}$，使得其真实误差不超过最优模型的误差加上 $\epsilon$。

所需样本量 $n$ 和 VC 维有直接关系：

$$n = O\left(\frac{d_{\text{VC}} + \log(1/\delta)}{\epsilon^2}\right)$$

含义：模型越复杂（VC 维越大），需要的数据就越多。这就是经典"参数多 → 需要更多数据"思想的形式化版本。

**对神经网络的问题和 VC 维一样**：PAC 框架给出的样本量需求和泛化上界对现代网络来说太松了，实际上神经网络用远少于理论要求的数据就能泛化得很好。

> **可靠程度：Level 1**（PAC 框架的数学定义和结论）  
> 参考：[Wikipedia - PAC learning](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) · [Valiant 1984 原始论文](https://dl.acm.org/doi/10.1145/1968.1972)

### 1.4 Rademacher 复杂度

**Rademacher 复杂度** 是另一种衡量模型族"灵活度"的工具，它的想法比 VC 维更直接：**看模型族能在多大程度上拟合纯随机噪声**。

定义：给定训练集 $\{x_1, \ldots, x_n\}$，生成 $n$ 个随机标签 $\sigma_i \in \{-1, +1\}$（每个等概率取 $\pm 1$，称为 Rademacher 随机变量）。Rademacher 复杂度定义为：

$$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i h(x_i)\right]$$

直觉解读：
- $\sigma_i$ 是完全随机的标签，不含任何真实信息
- $\sup_{h \in \mathcal{H}}$ 是在模型族中找最能"对齐"这些随机标签的模型
- 如果模型族足够灵活，它总能找到某个 $h$ 和随机标签高度对齐（相关性大）
- Rademacher 复杂度越大，模型族越"灵活"，过拟合风险越高

泛化上界：以高概率，

$$\text{测试误差} - \text{训练误差} \leq 2\mathcal{R}_n(\mathcal{H}) + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

**对神经网络的问题**：和 VC 维一样，对有限深度/宽度的神经网络计算 Rademacher 复杂度，得到的上界远远大于实际观察到的泛化间隙。这些经典工具都在说"这么复杂的模型应该过拟合"，但现实是它没有。

> **可靠程度：Level 1**（数学定义和上界）  
> 参考：[Wikipedia - Rademacher complexity](https://en.wikipedia.org/wiki/Rademacher_complexity) · [Understanding Machine Learning, Ch.26](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

### 小结：经典理论的局限

| 框架 | 衡量什么 | 对神经网络的结论 |
|------|---------|----------------|
| 偏差-方差 | 模型复杂度与误差的权衡 | 预测 U 形曲线，实际观察到 double descent |
| VC 维 | 模型能打碎多少点 | 给出的泛化上界太松，无法解释好表现 |
| PAC 学习 | 需要多少数据才能学好 | 理论要求远多于实际所需 |
| Rademacher 复杂度 | 模型拟合随机噪声的能力 | 上界太松 |

这些框架的共同问题是：它们衡量的是模型族的 **最坏情况** 能力（能拟合任何数据的能力），但没有考虑 **训练算法**（SGD）的具体行为。后面我们会看到，SGD 对解的"选择"可能是泛化的关键。

---

## 2. 经典正则化方法的数学

在深入理论谜题之前，先了解实际用来防止过拟合的工程手段。**正则化**（regularization）是一类技术的统称，它们通过对模型施加某种约束或扰动，来阻止模型过度拟合训练数据。

### 2.1 L2 正则化 / Weight Decay

**L2 正则化** 是最经典的正则化方法。它的想法很简单：在原来的损失函数上加一个惩罚项，惩罚权重的大小。

$$\mathcal{L}_{\text{reg}} = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

这里：
- $\mathcal{L}(\theta)$ 是原始损失函数（如交叉熵），$\theta$ 是模型的所有参数
- $\|\theta\|^2 = \sum_i \theta_i^2$ 是参数向量的 L2 范数的平方
- $\lambda > 0$ 是正则化强度，控制我们多看重"参数要小"这个目标
- $\frac{1}{2}$ 只是为了求导时消去系数

**为什么惩罚权重大小有用？** 大的权重意味着模型对输入的微小变化非常敏感（输出会剧烈变化），这正是过拟合的特征——模型在拟合训练数据中的噪声。限制权重大小，就是强迫模型用"温和"的方式拟合数据。

**梯度更新的变化**：加入 L2 正则化后，参数的梯度变为：

$$\nabla_\theta \mathcal{L}_{\text{reg}} = \nabla_\theta \mathcal{L} + \lambda \theta$$

所以每次更新时：

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta \mathcal{L} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}$$

注意 $(1 - \eta\lambda)$ 这个因子：每一步更新时，参数先被乘以一个小于 1 的数，再减去梯度方向。这就是"**权重衰减**"（weight decay）这个名字的来源——每一步参数都在"衰减"。

**贝叶斯视角**：L2 正则化等价于对参数施加均值为零、方差为 $1/\lambda$ 的 **高斯先验**（Gaussian prior）。如果你熟悉贝叶斯统计：

$$p(\theta | \text{data}) \propto p(\text{data} | \theta) \cdot p(\theta)$$

取负对数后，最大化后验概率（MAP 估计）变成最小化：

$$-\log p(\text{data} | \theta) - \log p(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

其中 $-\log p(\theta) = \frac{\lambda}{2}\|\theta\|^2$ 对应高斯先验 $p(\theta) \propto \exp(-\frac{\lambda}{2}\|\theta\|^2)$。

> **可靠程度：Level 1**  
> 参考：[Wikipedia - Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) · [Deep Learning Book §7.1](https://www.deeplearningbook.org/contents/regularization.html)

### 2.2 L1 正则化

**L1 正则化** 使用参数的绝对值之和（L1 范数）作为惩罚：

$$\mathcal{L}_{\text{reg}} = \mathcal{L}(\theta) + \lambda \|\theta\|_1 = \mathcal{L}(\theta) + \lambda \sum_i |\theta_i|$$

**和 L2 的区别**：L1 正则化倾向于产生 **稀疏** 解——很多参数会被推到恰好为零。

直觉：L2 惩罚大的权重更重（平方增长），所以它让所有权重都变小但不为零。L1 惩罚所有权重同样力度（线性增长），对小权重的"推力"和大权重一样强，这使得小权重更容易被推到零。

从几何上看：L1 约束区域（$\|\theta\|_1 \leq c$）在参数空间中是一个菱形（高维超八面体），它的"角"在坐标轴上。损失函数的等高线更可能和这些角相切，而角对应某些参数恰好为零的位置。

**贝叶斯视角**：L1 正则化等价于对参数施加 **拉普拉斯先验**（Laplace prior）：$p(\theta_i) \propto \exp(-\lambda|\theta_i|)$。拉普拉斯分布在零处有一个尖峰，这就是它促进稀疏性的原因。

在实际的深度学习中，L1 正则化用得比 L2 少，因为稀疏性对于神经网络来说不一定是优势——我们通常希望所有神经元都参与计算。但在需要特征选择的场景（如线性模型中的 LASSO 回归），L1 非常有用。

> **可靠程度：Level 1**  
> 参考：[Wikipedia - Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_(statistics)) · [Deep Learning Book §7.1](https://www.deeplearningbook.org/contents/regularization.html)

### 2.3 Dropout

**Dropout** 是 Hinton 等人在 2014 年提出的一种专门针对神经网络的正则化方法。它的做法出奇地简单：

**训练时**：在每次前向传播中，对于每一层的每个神经元，以概率 $p$（通常 $p = 0.5$）将其输出设为零。也就是说，每次训练迭代都随机"关闭"一部分神经元。

**测试时**：使用所有神经元，但将输出乘以 $(1 - p)$（或等价地，训练时将未被关闭的神经元输出除以 $(1 - p)$，称为 inverted dropout，这是 PyTorch 的默认实现）。

数学表达：如果某层的输出是 $\mathbf{h}$，dropout 后变为：

$$\tilde{\mathbf{h}} = \mathbf{m} \odot \mathbf{h}, \quad m_j \sim \text{Bernoulli}(1 - p)$$

其中 $\mathbf{m}$ 是一个掩码向量（mask），每个元素独立地以概率 $(1-p)$ 为 1、以概率 $p$ 为 0。$\odot$ 表示逐元素乘法。

**为什么 Dropout 有效？** 有几种理解方式：

1. **模型集成（Ensemble）视角**：一个有 $n$ 个神经元的网络使用 dropout，等价于训练了 $2^n$ 个不同的子网络（每种 dropout 掩码对应一个子网络），测试时取它们的平均预测。模型集成是减少方差的经典方法。
2. **共适应（Co-adaptation）防止**：没有 dropout 时，某些神经元可能"依赖"其他特定神经元的输出，形成脆弱的共适应关系。Dropout 迫使每个神经元独立地学习有用的特征，因为它不能假设其他神经元一定在。
3. **噪声注入**：Dropout 本质上是向网络中注入乘性噪声，这增加了训练的随机性，使模型对输入的微小变化更鲁棒。

> **可靠程度：Level 1**（方法本身），Level 2（"等价于模型集成"的解释是简化的）  
> 参考：[Srivastava et al. 2014 - Dropout 论文](https://jmlr.org/papers/v15/srivastava14a.html) · [Wikipedia - Dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) · [Deep Learning Book §7.12](https://www.deeplearningbook.org/contents/regularization.html)

### 2.4 数据增强（Data Augmentation）

**数据增强** 是最直观的正则化方法：在不改变标签的前提下，对训练数据做各种变换来"扩展"数据集。

对于图像任务，常见的增强方式包括：
- **水平翻转**：猫翻转后还是猫
- **随机裁剪**：取图像的一部分
- **颜色抖动**：改变亮度、对比度、饱和度
- **旋转**：小角度旋转
- **仿射变换**：缩放、平移、剪切
- **Cutout / Random Erasing**：随机遮挡图像的一部分
- **Mixup**：将两张图像按比例混合，标签也按同样比例混合

数据增强的数学本质是 **利用数据的对称性**。如果我们知道标签对某种变换不变（翻转后猫还是猫），那么这个变换就提供了"免费的"额外训练数据。这等价于告诉模型"你应该对这类变换保持不变"，减少了模型需要自己学习的内容。

数据增强在计算机视觉中效果极好，是最实用的正则化手段之一。在 NLP 中，数据增强更困难（把一句话的词打乱后意思可能完全改变），但也有一些有效方法如同义词替换、反向翻译等。

> **可靠程度：Level 1**  
> 参考：[Wikipedia - Data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) · [d2l.ai §14.1](https://d2l.ai/chapter_computer-vision/image-augmentation.html)

---

## 3. Zhang et al. (2017) 的关键实验

到目前为止，我们学了经典泛化理论（VC 维、PAC 学习等）和经典正则化方法（L2、Dropout 等）。现在来看一个实验，它深刻地动摇了人们对泛化的理解。

2017 年，Zhang 等人在论文 ["Understanding deep learning requires rethinking generalization"](https://arxiv.org/abs/1611.03530) 中做了一个简单但震撼的实验：

### 实验设计

1. 取标准数据集（如 CIFAR-10 或 ImageNet），使用标准的 CNN 架构
2. **把训练集的标签完全随机打乱**——每张图像被随机分配一个标签，标签和图像内容完全无关
3. 用标准训练流程训练网络

### 实验结果

- 网络仍然能把训练损失降到零——也就是说，模型 **完美记住** 了所有的随机标签
- 即使标签是纯噪声，网络照样能 100% 拟合训练集
- 当然，在测试集上的准确率等于随机猜测（约 10%，因为有 10 个类别）

### 这个实验为什么重要

这个结果的含义非常深刻：

1. **网络的容量（capacity）足以记住任何东西**：同一个网络架构，在真实标签上训练能泛化，在随机标签上训练能完美记忆。这说明泛化 **不是** 因为网络的容量不够（不够"灵活"）——它有足够的容量记住一切，包括纯噪声。

2. **经典复杂度度量失效**：VC 维、Rademacher 复杂度等度量只看模型族的"表达能力"。既然同一个网络能拟合真实数据也能拟合随机数据，那这些复杂度度量对于真实数据和随机数据是一样的，它们无法区分"好的泛化"和"完全记忆"。

3. **经典正则化不是泛化的根本原因**：即使去掉所有显式正则化（不用 L2、不用 Dropout、不用数据增强），网络在真实数据上仍然泛化得不错。正则化能提升几个百分点的准确率，但不是泛化的根本原因。

4. **泛化的关键在于数据和算法的交互**：同一个网络在真实数据上泛化、在随机数据上不泛化。区别不在网络本身，而在数据——真实数据有结构（可学习的模式），随机数据没有。而 SGD 在真实数据上似乎能找到利用这种结构的解。

> **可靠程度：实验结果 Level 1**（已被大量复现），理论解释 Level 3-4  
> 参考：[Zhang et al. 2017](https://arxiv.org/abs/1611.03530)

---

## 4. Double Descent 现象

Zhang 等人的实验告诉我们经典理论不够，那实际上发生了什么？2019 年，Nakkiran 等人的论文 ["Deep Double Descent"](https://arxiv.org/abs/1912.02292) 揭示了一个令人惊讶的现象。

### 4.1 经典 U 形曲线

经典偏差-方差框架预测：随着模型复杂度增大，测试误差先降后升，形成 U 形曲线。

```
测试误差
   |
   |\
   | \        /
   |  \      /
   |   \    /
   |    \  /
   |     \/
   |
   +—————————————→ 模型复杂度
   欠拟合  最优  过拟合
```

### 4.2 实际观察：Double Descent

但在深度学习中，实际观察到的曲线是这样的：

```
测试误差
   |
   |\
   | \         |
   |  \       /|\
   |   \     / | \
   |    \   /  |  \
   |     \ /   |   \___________
   |      V    |
   |           |
   +——————————|————————————————→ 模型复杂度
   欠拟合    插值       过参数化
             阈值
```

**三个阶段**：

1. **经典阶段**（模型参数 < 数据量）：行为符合经典 U 形曲线——复杂度增大，先欠拟合再过拟合。
2. **插值阈值**（interpolation threshold）：当参数量大约等于数据量时，模型刚好能完美拟合训练数据（训练误差 → 0）。此时测试误差达到峰值——模型被迫用几乎所有"自由度"来死记硬背数据，没有余力泛化。
3. **过参数化阶段**（模型参数 >> 数据量）：继续增加参数，测试误差反而 **又下降了**。最终往往能达到甚至超过经典最优点的水平。

这就是 **double descent**——测试误差曲线有两次下降，中间有一个峰。

### 4.3 为什么过参数化反而好？

关键直觉：在插值阈值附近，模型被迫"硬记"数据——就像一个刚好够用的行李箱，塞得满满的，非常紧张。而过参数化的模型就像一个远大于行李的行李箱——你可以轻松地把东西放进去，还有很多种放法。

更精确地说：

- 在插值阈值处，能完美拟合训练数据的解只有一个（或很少几个），这个解可能非常复杂、不规则。
- 在过参数化区域，有 **无穷多个** 解都能完美拟合训练数据。在这么多解中，SGD（随机梯度下降）倾向于找到 **"最简单"** 或 **"最平坦"** 的那个。
- "最平坦"的解往往泛化更好——直觉上，如果一个解在参数空间中处于一个宽阔的低谷（flat minimum），那么参数的微小扰动不会大幅改变模型的输出，这意味着模型对训练数据中的噪声不敏感。

### 4.4 Double Descent 不仅仅关于模型大小

Nakkiran 等人还发现，double descent 现象不仅出现在"模型大小"这个轴上，还出现在：

- **训练时间**（epoch-wise double descent）：训练足够久，测试误差先降、再升、再降
- **数据量**（sample-wise double descent）：固定模型，增加数据量，也能观察到类似现象

这暗示 double descent 是一个更普遍的现象，而不是某种架构的特殊性质。

> **可靠程度：现象 Level 1**（多个独立团队复现），机制解释 Level 3-4  
> 参考：[Nakkiran et al. 2019 - Deep Double Descent](https://arxiv.org/abs/1912.02292) · [Belkin et al. 2019 - Reconciling modern ML practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118) · [Wikipedia - Double descent](https://en.wikipedia.org/wiki/Double_descent)

---

## 5. 隐式正则化假说

Zhang 等人的实验和 double descent 现象都指向同一个问题：过参数化网络在真实数据上为什么能泛化？目前最主流的假说是 **隐式正则化**（implicit regularization）——即使没有显式正则化（L2、Dropout 等），训练算法本身就在做某种正则化。

### 5.1 SGD 的噪声偏好平坦最小值

SGD（随机梯度下降）和全批量梯度下降（GD）的区别在于：SGD 每一步用一个随机的小批量（mini-batch）来估计梯度，因此引入了噪声。

这个噪声不是纯粹的坏事。理论和实验都表明，SGD 的噪声倾向于让参数 **逃离尖锐的最小值**（sharp minima），停留在 **平坦的最小值**（flat minima）。

直觉：在尖锐的最小值处，梯度的方向变化很大，SGD 的噪声容易把参数"震"出这个小坑。在平坦的最小值处，即使有噪声，参数也不会偏离太远。

为什么这和泛化有关？一种假说（Hochreiter & Schmidhuber 1997, Keskar et al. 2017）认为，平坦最小值对应的模型泛化更好。想象损失函数在参数空间中的"地形"——如果模型处于一个宽阔的山谷底部，那么稍微改变参数（或者稍微改变数据分布），损失值变化不大，这意味着模型对数据的微小变化不敏感，即泛化好。

SGD 噪声的大小和学习率 $\eta$ 以及批量大小 $B$ 有关：噪声 $\propto \eta / B$。这解释了一个经验观察：太大的批量大小往往导致泛化变差——因为噪声太小，SGD 可能停在尖锐的最小值里出不来。

> **可靠程度：Level 3**（实验证据较强，理论框架仍在发展中）  
> 参考：[Keskar et al. 2017 - On Large-Batch Training](https://arxiv.org/abs/1609.04836) · [Smith & Le 2018 - A Bayesian Perspective on Generalization and SGD](https://arxiv.org/abs/1710.06451)

### 5.2 过参数化网络的低秩偏好

另一个重要的实验观察：虽然过参数化网络有很多参数，但训练出来的权重矩阵往往是 **低秩的**——也就是说，权重矩阵的有效自由度远小于参数总数。

例如，一个 $1000 \times 1000$ 的权重矩阵有 100 万个参数，但它的有效秩可能只有几十。这意味着网络虽然"名义上"有 100 万个参数，但实际上只用了几百到几千个"有效参数"。

这种低秩偏好可能来自：
- **梯度下降的动力学**：梯度下降（尤其是从小随机初始化开始）倾向于先学习数据中最主要的模式（对应权重矩阵的前几个奇异值），然后才慢慢拟合更细节的模式。如果训练早停或有正则化，高阶细节可能永远不会被学到。
- **数据的结构**：真实数据本身就是低维的（自然图像的有效维度远低于像素数），所以网络只需要低秩的权重来捕捉数据的结构。

### 5.3 这仍然是活跃的研究方向

需要强调的是，隐式正则化的完整理论尚未建立。目前的理解是碎片化的：

- SGD 偏好平坦最小值的理论在简单模型上成立，但对复杂深度网络的分析还不完整
- 低秩偏好的机制理解仍在发展中
- 不同的初始化、学习率调度、架构选择如何影响隐式正则化，也没有统一的理论
- 2024-2025 年的一些新工作开始用 **特征学习理论**（feature learning theory，超越 NTK 的"核"框架）来分析这些问题，但这些理论目前只适用于相对简单的架构

> **可靠程度：Level 3-4**（各个假说都有实验支持，但理论尚未统一）  
> 参考：[Neyshabur et al. 2017 - Exploring Generalization in Deep Nets](https://arxiv.org/abs/1706.08947) · [Arora et al. 2019 - Implicit Regularization in Deep Matrix Factorization](https://arxiv.org/abs/1905.13655)

---

## 6. 和 LLM 的联系

大语言模型（LLM）是过参数化的极端案例：

- GPT-3 有 **1750 亿** 参数
- 训练数据虽然有几万亿 token，但考虑到参数量和数据的信息冗余，模型仍然是高度过参数化的
- 尽管如此，LLM 在各种未见过的任务上展现出了惊人的泛化能力（所谓的"涌现能力"）

本章讨论的理论问题——为什么过参数化的网络能泛化——直接适用于理解 LLM。当前对 LLM 泛化的一些思考：

1. **Scaling laws**（我们在 01-overview 中提到的）暗示，增加参数量并不只是增加"记忆容量"，而是让模型能学到更多层次的抽象模式。
2. LLM 的预训练目标（预测下一个 token）可能自带某种正则化效果——模型必须学会语言的统计结构才能预测好，这要求的不是记忆而是理解。
3. LLM 中的 double descent 现象也已被观察到（Nakkiran 等人的论文中包含了对 Transformer 的实验）。

理解这些问题不仅有理论价值，也有实际意义：它影响我们如何选择模型大小、训练数据量和训练策略。

---

## 本章总结

| 主题 | 核心结论 | 可靠程度 |
|------|---------|---------|
| 经典泛化理论 | VC 维、PAC 等给出的上界太松，无法解释深度学习 | 理论本身 L1，"解释不了深度学习"这一判断 L1 |
| L2 / L1 正则化 | 限制权重大小 / 促进稀疏性，等价于贝叶斯先验 | L1 |
| Dropout | 随机关闭神经元，近似于模型集成 | 方法 L1，理论 L2 |
| 数据增强 | 利用数据对称性扩展训练集 | L1 |
| Zhang et al. 2017 | 网络能记住随机标签 → 泛化不能只靠容量限制解释 | L1 |
| Double Descent | 过参数化后测试误差再次下降 | 现象 L1，机制 L3-4 |
| 隐式正则化 | SGD + 过参数化 → 偏好"简单"解 | L3-4 |

---

## 理解检测

**Q1**：Zhang et al. (2017) 用随机标签训练了神经网络。假设你用 L2 正则化 + Dropout + 数据增强训练同一个网络，它还能拟合随机标签吗？如果可以，这对"正则化是泛化的根本原因"这个观点意味着什么？

你的回答：



**Q2**：Double descent 中，在"插值阈值"附近（参数量 ≈ 数据量），测试误差达到峰值。请用你自己的话解释为什么这个点特别糟糕（提示：想想在这个点上有多少个解能完美拟合训练数据，以及这些解的性质）。

你的回答：



**Q3**：一个同学说："我的模型过拟合了，我应该减少参数量。"根据本章的 double descent 理论，你会怎么反驳他？在什么条件下"增加参数量"反而可能减少过拟合？

你的回答：


