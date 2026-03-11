# 10 - GAN：对抗生成网络

> **主维度**：D2 训练范式
> **关键关系**：
> - GAN (方法) --对比--> VAE (方法)：GAN 用对抗训练，VAE 用变分推断
> - Diffusion (方法) --对比--> GAN (方法)：Diffusion 用去噪替代对抗训练
> - DCGAN (方法) --实例--> GAN (方法)：DCGAN 是用 CNN 架构实现的 GAN 实例
>
> **学习路径**：01-overview → 深入 MLP（02） → CNN（03） → AutoEncoder/VAE（09） → **本章** → GNN（11）
>
> **前置知识**：MLP 与反向传播、CNN（卷积/转置卷积）、概率论基础（概率分布、期望）、VAE 的基本思想（09 已覆盖）
>
> **参考**：
> - [Goodfellow et al. 2014 - Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
> - [Arjovsky et al. 2017 - WGAN](https://arxiv.org/abs/1701.07875)
> - [Karras et al. 2019 - StyleGAN](https://arxiv.org/abs/1812.04948)
> - [Wikipedia - Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network)
> - [Lilian Weng - From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
> - [d2l.ai - Generative Adversarial Networks](https://d2l.ai/chapter_generative-adversarial-networks/index.html)

---

## 核心问题

上一章我们学了 VAE——通过学习数据的潜在分布来生成新数据。但 VAE 生成的图像总是偏模糊。

一个更激进的问题：**能不能训练一个网络，直接生成逼真到人眼无法区分的图像、音乐、甚至文本？**

2014 年，Ian Goodfellow 提出了**生成对抗网络**（Generative Adversarial Network，GAN），用一个全新的思路来回答这个问题：不需要显式地建模数据分布，只需要让两个网络**互相对抗**。

> 可靠程度：**Level 1**（教科书共识）

---

## 1. GAN 的核心思想

### 1.1 伪造者与鉴定师

GAN 的直觉可以用一个类比来理解：

> 想象一个**伪造者**（Forger）在学习造假钞，一个**鉴定师**（Detective）在学习辨别真假。伪造者不断改进自己的技术，鉴定师也不断提升自己的鉴别能力。两者互相竞争，不断进步。最终，伪造者造出的假钞好到连鉴定师也分辨不出来。

对应到 GAN 中：

- **生成器**（Generator，$G$）：扮演伪造者。它接收一个随机噪声向量 $z$，输出一个"假数据"样本 $G(z)$。目标是让生成的数据尽可能逼真，骗过判别器。

- **判别器**（Discriminator，$D$）：扮演鉴定师。它接收一个数据样本（可能是真数据，也可能是生成器造的假数据），输出一个概率 $D(x) \in [0, 1]$——表示这个样本是真数据的概率。目标是正确区分真假。

### 1.2 数学形式

生成器从随机噪声出发：

$$z \sim p_z(z) = \mathcal{N}(0, I), \quad \text{假数据} = G(z)$$

其中 $z$ 是从标准正态分布采样的低维噪声向量（比如 100 维），$G$ 是一个神经网络，把 $z$ 映射到数据空间（比如 $64 \times 64 \times 3$ 的图像）。

判别器是一个二分类器：

$$D(x) = \text{输入 } x \text{ 是真数据的概率}$$

$D$ 的输出经过 sigmoid 函数，范围在 $[0, 1]$ 之间。$D(x) = 1$ 表示"完全确定是真数据"，$D(x) = 0$ 表示"完全确定是假数据"。

---

## 2. GAN 的目标函数（Minimax Game）

GAN 的训练目标是一个**极小极大博弈**（minimax game）：

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

这个公式信息量很大，逐项拆解：

### 2.1 判别器的视角（$\max_D$）

判别器要最大化 $V(D,G)$，即：

- **$\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$**：对真数据，$D$ 希望输出接近 1。$\log D(x)$ 在 $D(x) \to 1$ 时趋向 0（最大值），在 $D(x) \to 0$ 时趋向 $-\infty$。所以这一项鼓励 $D$ 给真数据打高分。

- **$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$**：对假数据 $G(z)$，$D$ 希望输出接近 0。$\log(1 - D(G(z)))$ 在 $D(G(z)) \to 0$ 时趋向 0（最大值），在 $D(G(z)) \to 1$ 时趋向 $-\infty$。所以这一项鼓励 $D$ 给假数据打低分。

总结：**判别器想让真数据得分高、假数据得分低。**

### 2.2 生成器的视角（$\min_G$）

生成器要最小化 $V(D,G)$。它只能影响第二项（因为真数据和 $G$ 无关）：

- **$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$**：$G$ 希望 $D(G(z)) \to 1$（骗过判别器），这样 $\log(1 - D(G(z))) \to -\infty$，$V$ 被最小化。

总结：**生成器想让假数据骗过判别器。**

### 2.3 Nash 均衡

博弈论告诉我们，这个游戏的**Nash 均衡**是：

$$p_G = p_{\text{data}}, \quad D(x) = \frac{1}{2} \quad \forall x$$

即生成器学到的分布 $p_G$ 完全等于真实数据分布 $p_{\text{data}}$，判别器对任何输入都输出 0.5——因为它真的无法区分真假了。

Goodfellow 在原始论文中证明了：在理想情况下（$G$ 和 $D$ 都有无限容量、训练充分），GAN 的全局最优解确实是 $p_G = p_{\text{data}}$。

> 可靠程度：**Level 1**（理论证明存在于原始论文；但实践中几乎不可能达到这个理想状态）

---

## 3. 训练过程

### 3.1 交替训练

GAN 的训练不像普通网络那样只优化一个损失函数——它需要**交替训练**两个网络：

1. **固定 $G$，训练 $D$ 几步**：用真数据和 $G$ 生成的假数据训练判别器，让它更好地区分真假
2. **固定 $D$，训练 $G$ 一步**：让生成器生成假数据，通过判别器的反馈更新 $G$，让它更好地骗过 $D$
3. 重复

在实践中，通常每轮先训练 $D$ 1-5 步，再训练 $G$ 1 步。判别器需要"跟上"生成器，否则它给出的反馈信号就没有意义——就像一个完全不懂行的鉴定师，伪造者从它那里学不到任何东西。

### 3.2 训练不稳定性

GAN 的训练出了名地难。两个主要问题：

**模式坍塌（Mode Collapse）**

生成器发现只生成某一种类型的输出就能骗过判别器，于是放弃了生成其他类型。例如在 MNIST 上训练时，生成器可能只生成数字"1"（因为它学到了一种很逼真的"1"的画法），完全不生成其他数字。

为什么会这样？因为生成器的目标只是骗过判别器，而不是覆盖整个数据分布。如果只生成一种模式就足以让 $D(G(z))$ 接近 1，$G$ 就没有动力去探索其他模式。

**训练震荡**

$G$ 和 $D$ 可能陷入"你追我赶"的震荡：$D$ 学会识别某种假数据，$G$ 就换一种方式造假；$D$ 再学会识别新的假数据，$G$ 再变……损失函数上下波动，不收敛。

这和物理中的问题很像：两个耦合的振荡器，如果阻尼不够，就会永远振荡下去。

> 可靠程度：**Level 1**（训练不稳定性是 GAN 的已知核心问题）

---

## 4. GAN 的重要变体

GAN 提出后催生了大量变体，每个都试图解决原始 GAN 的某个问题。以下是最重要的几个：

### 4.1 DCGAN（Deep Convolutional GAN, 2016）

**核心改进**：用卷积网络替代全连接网络。

原始 GAN 使用 MLP，对图像生成效果有限。DCGAN 把生成器和判别器都换成 CNN：生成器使用**转置卷积**（transposed convolution，也叫反卷积）把低维向量逐步上采样成图像；判别器使用标准卷积逐步下采样图像到一个标量。

DCGAN 还总结了一套架构经验法则（如使用 BatchNorm、避免全连接层、生成器用 ReLU、判别器用 LeakyReLU），这些成为后续 GAN 工作的标准实践。

> 参考：[Radford et al. 2016](https://arxiv.org/abs/1511.06434)

### 4.2 WGAN（Wasserstein GAN, 2017）

**核心改进**：用 **Wasserstein 距离**（也叫推土机距离，Earth Mover's Distance）替代原始 GAN 中的 JS 散度，训练更稳定。

原始 GAN 目标函数隐含地最小化生成分布和真实分布之间的 **JS 散度**（Jensen-Shannon divergence）。问题是：当两个分布没有重叠时，JS 散度恒为常数 $\log 2$，梯度为零——判别器给出的信号完全没有用，生成器无法学习。

Wasserstein 距离的直觉是："把分布 $p$ 变成分布 $q$ 需要搬运多少'土'？"它即使在两个分布不重叠时也能给出有意义的梯度，因此训练更稳定、模式坍塌更少。

代价是判别器（在 WGAN 中改叫 **critic**）需要满足 **Lipschitz 约束**——原始论文通过权重裁剪（weight clipping）实现，后来 WGAN-GP 使用梯度惩罚（gradient penalty）效果更好。

> 参考：[Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875) · [WGAN-GP (Gulrajani et al. 2017)](https://arxiv.org/abs/1704.00028)

### 4.3 StyleGAN（2019）

**核心改进**：对生成过程引入**风格控制**，让生成器可以分层控制图像的不同属性。

StyleGAN 的生成器不是简单地从噪声 $z$ 直接生成图像，而是先把 $z$ 映射到一个"风格空间" $w$，然后在网络的每一层通过**自适应实例归一化**（AdaIN）注入风格信息。低层控制整体结构（如人脸形状、姿态），高层控制细节（如头发颜色、皮肤纹理）。

StyleGAN 能生成 1024×1024 的超逼真人脸（[thispersondoesnotexist.com](https://thispersondoesnotexist.com) 就是用 StyleGAN 做的），标志着 GAN 在图像生成质量上的巅峰。

> 参考：[Karras et al. 2019 - StyleGAN](https://arxiv.org/abs/1812.04948) · [StyleGAN2 (2020)](https://arxiv.org/abs/1912.04958)

### 4.4 条件 GAN（cGAN）

**核心改进**：给生成器和判别器额外的**条件信息**（如类别标签）。

原始 GAN 只能生成随机样本，无法控制生成内容。cGAN 在生成器的输入中加入条件 $y$（比如"生成一个数字 7"），判别器也同时接收条件和数据，判断"这个数据在给定条件 $y$ 下是否真实"。

目标函数变为：

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]$$

应用：文字转图像（pix2pix）、图像超分辨率（SRGAN）、图像风格转换。

> 参考：[Mirza & Osindero 2014](https://arxiv.org/abs/1411.1784) · [pix2pix (Isola et al. 2017)](https://arxiv.org/abs/1611.07004)

---

## 5. GAN vs VAE vs Diffusion Model

2020 年代后，图像生成领域出现了三种主流方法。它们各有优劣：

| 维度 | VAE | GAN | Diffusion Model |
|------|-----|-----|-----------------|
| 生成质量 | 模糊 | 清晰锐利 | 最高质量 |
| 训练稳定性 | 稳定 | 不稳定（模式坍塌） | 稳定 |
| 生成多样性 | 高 | 低（容易模式坍塌） | 高 |
| 生成速度 | 快（一次前向传播） | 快（一次前向传播） | 慢（需要迭代去噪） |
| 理论基础 | 变分推断 | 博弈论 | 随机过程（分数匹配） |
| 潜在空间 | 有意义、可插值 | 可控性差 | 有意义 |
| 代表模型 | β-VAE, VQ-VAE | StyleGAN | Stable Diffusion, DALL-E |

**Diffusion Model**（扩散模型）是 2020 年后崛起的生成模型。它的思路和 GAN 完全不同：不是对抗训练，而是学习一个**逐步去噪**的过程——先把真实图片逐步加噪声变成纯噪声，然后训练一个网络学习反转这个过程（从噪声恢复图片）。详细内容超出本章范围，这里只做对比。

> 可靠程度：**Level 2**（三者对比是领域常见总结，具体优劣取决于任务和模型规模）

---

## 6. GAN 的衰落与 Diffusion Model 的崛起

这是一个值得了解的历史趋势。

**GAN 的黄金期（2014-2021）**：从 Goodfellow 提出 GAN 到 StyleGAN3，GAN 一直是图像生成的主流方法。arXiv 上每年有数千篇 GAN 相关论文。

**转折点（2021-2022）**：

- 2020 年，DDPM（Denoising Diffusion Probabilistic Models）展示了 Diffusion Model 可以生成高质量图像。
- 2021 年，Dhariwal & Nichol 的论文 ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233) 直接在标题里宣告了 Diffusion Model 超越 GAN。
- 2022 年，DALL-E 2、Stable Diffusion、Imagen 等模型横空出世，在文字转图像任务上达到了惊人的质量和多样性——全部基于 Diffusion Model，不用 GAN。

**为什么 GAN 被取代了？** 核心原因是 GAN 的训练不稳定性问题从未被根本解决。模式坍塌、训练震荡、超参数敏感等问题使得 GAN 在大规模、高分辨率生成任务上越来越力不从心。Diffusion Model 虽然生成速度慢，但训练稳定、生成质量高、多样性好，这些优势在大规模部署中更重要。

当然，GAN 并没有完全消亡。在一些需要实时生成的场景（如视频游戏中的纹理生成）和对抗样本研究中，GAN 仍然有用。但在图像/视频生成的主流研究中，GAN 已经基本让位给了 Diffusion Model。

> 可靠程度：**Level 2**（趋势准确，但"取代"的程度因子领域而异）

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $z \sim p_z(z) = \mathcal{N}(0, I)$ | 噪声 $z$ 服从标准正态分布 |
| $G(z)$ | 生成器输出：假数据 |
| $D(x) \in [0, 1]$ | 判别器输出：样本为真数据的概率 |
| $\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$ | GAN 极小极大目标 |
| $p_G = p_{\text{data}}, \; D(x) = 1/2$ | Nash 均衡：最优时生成分布等于真实分布 |
| cGAN: $\mathbb{E}[\log D(x\|y)] + \mathbb{E}[\log(1-D(G(z\|y)\|y))]$ | 条件 GAN 目标（含条件 $y$） |

**常用数值**：噪声维度通常 100 维；训练时通常 D 每轮 1–5 步、G 每轮 1 步。

---

## 理解检测

**Q1**：GAN 的目标函数是 $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$。假设判别器已经训练到最优（对于当前的 $G$），那么生成器的最优化目标等价于最小化什么统计量？（提示：Goodfellow 在原始论文中证明了这一点。）现在再想想：当生成分布和真实分布完全不重叠时（这在高维空间中很常见），这个统计量的梯度是多少？这解释了原始 GAN 的什么训练问题？

> 提示：用 GAN 极小极大目标 $V(D,G)$；当 D 最优时，G 的目标等价于最小化 JS 散度；考虑 JS 散度在分布不重叠时的梯度。

你的回答：


**Q2**：模式坍塌（mode collapse）是 GAN 训练中最常见的失败模式。请用具体例子解释：如果用 GAN 生成 MNIST 手写数字，模式坍塌会表现为什么现象？从生成器的损失函数角度解释，为什么生成器"偷懒"只生成一种数字是一个局部最优策略？

你的回答：


**Q3**：VAE 和 GAN 都可以生成新图像，但方式完全不同。假设你需要做一个药物分子生成任务：在潜在空间中搜索满足特定性质（如高药效、低毒性）的分子，然后解码生成新分子。你会选 VAE 还是 GAN？为什么？（提示：想想哪种模型的潜在空间更适合"搜索和优化"。）

你的回答：

