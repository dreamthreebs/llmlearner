# 09 - AutoEncoder 与 VAE：学习数据的压缩表示

> **主维度**：D2 训练范式
> **关键关系**：
> - VAE (方法) --推广了--> AutoEncoder (方法)：VAE 推广了 AutoEncoder 为概率变分模型
> - GAN (方法) --对比--> VAE (方法)：GAN 用对抗训练，VAE 用变分推断
> - AutoEncoder (方法) --用于--> 无监督特征学习 (任务)：AutoEncoder 用于在无标签数据中学习压缩表示
>
> **学习路径**：01-overview → 深入 MLP（02） → CNN（03） → **本章** → GAN（10）
>
> **前置知识**：MLP 结构与反向传播、CNN 基本操作（卷积/池化）、损失函数与梯度下降、概率论基础（高斯分布、贝叶斯公式、KL 散度的直觉）
>
> **参考**：
> - [Kingma & Welling 2014 - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
> - [Tutorial on VAEs by Carl Doersch](https://arxiv.org/abs/1606.05908)
> - [Wikipedia - Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
> - [Wikipedia - Variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder)
> - [d2l.ai - Variational Autoencoders](https://d2l.ai/)
> - [Lilian Weng - From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

---

## 核心问题

到目前为止我们学过的网络——MLP、CNN、RNN——都是**监督学习**：给网络一张图片和一个标签（"这是猫"），让它学会把图片映射到标签。但标注数据很贵，世界上绝大多数数据都是没有标签的。

一个自然的问题：**能不能让网络自己发现数据最重要的特征，不需要人类标签？**

这就是**无监督学习**（unsupervised learning）的核心动机。AutoEncoder 是其中最直觉的方法之一：让网络把数据**压缩**再**还原**，如果还原得好，说明压缩后保留的信息就是数据最重要的特征。

> 可靠程度：**Level 1**（教科书共识）

---

## 1. AutoEncoder 基本结构

### 1.1 编码器-瓶颈-解码器

AutoEncoder（自编码器）的结构非常简单，由三部分组成：

1. **编码器（Encoder）**：一个神经网络 $f_\phi$，把高维输入 $x$ 压缩到低维的**潜在表示**（latent representation）$z$。这里 $\phi$ 是编码器的参数。

$$z = f_\phi(x), \quad x \in \mathbb{R}^D, \quad z \in \mathbb{R}^d, \quad d \ll D$$

2. **瓶颈层（Bottleneck）**：中间那个低维向量 $z$，也叫**编码**（code）或**潜在变量**（latent variable）。它的维度 $d$ 远小于输入维度 $D$，迫使网络只保留最重要的信息。

3. **解码器（Decoder）**：另一个神经网络 $g_\psi$，从低维表示 $z$ 重建原始输入 $\hat{x}$。$\psi$ 是解码器的参数。

$$\hat{x} = g_\psi(z) = g_\psi(f_\phi(x))$$

### 1.2 训练目标

训练的目标就一件事：让重建结果 $\hat{x}$ 尽可能接近原始输入 $x$。最常用的损失函数是**均方误差**（MSE）：

$$\mathcal{L}(\phi, \psi) = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2 = \frac{1}{N}\sum_{i=1}^{N} \|x_i - g_\psi(f_\phi(x_i))\|^2$$

其中 $N$ 是训练样本数。直觉上：

- 如果瓶颈层维度 $d$ 和输入维度 $D$ 一样大，网络可以直接"复制粘贴"，损失为零但毫无意义。
- 正因为 $d \ll D$，网络必须学会**压缩**——丢掉不重要的信息，只保留重建所需的关键特征。

### 1.3 和 PCA 的联系

如果你学过线性代数，**主成分分析**（PCA，Principal Component Analysis）应该不陌生——它把数据投影到方差最大的几个正交方向上，实现降维。

一个重要的事实：**当编码器和解码器都是线性变换（不加激活函数）、损失函数是 MSE 时，AutoEncoder 学到的潜在表示和 PCA 等价。** 具体来说，瓶颈层 $z$ 张成的子空间就是 PCA 前 $d$ 个主成分张成的子空间（虽然不一定是同一组基，但子空间相同）。

这意味着 AutoEncoder 可以看作 PCA 的**非线性推广**：加上激活函数后，编码器和解码器可以学到弯曲的、非线性的降维流形，捕获 PCA 无法捕获的数据结构。

> 可靠程度：**Level 1**（教科书共识。线性 AE 与 PCA 等价的证明见 Goodfellow 等《Deep Learning》第 14 章）

### 1.4 具体例子：MNIST 压缩到 2 维

考虑 MNIST 数据集：每张手写数字图片是 $28 \times 28 = 784$ 维的灰度图像。我们可以构建一个 AutoEncoder，编码器结构为：

```
784 → 256 → 64 → 2  （编码器）
2 → 64 → 256 → 784  （解码器）
```

瓶颈层只有 **2 维**，意味着每张图片被压缩成平面上的一个点 $(z_1, z_2)$。把所有测试图片编码后的 $z$ 画在二维平面上，按数字标签着色，你会看到：

- 不同数字自然地聚成不同的簇（cluster）
- 形状相似的数字（如 3 和 8、4 和 9）在潜在空间中距离更近
- 网络在没有看到任何标签的情况下，自动发现了数字之间的结构

这就是无监督特征学习的魅力：数据本身的结构蕴含了有意义的信息，AutoEncoder 把它提取出来了。

> 参考：[d2l.ai - Autoencoders](https://d2l.ai/) · [Wikipedia - Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)

---

## 2. AutoEncoder 的局限

AutoEncoder 虽然能学到有用的压缩表示，但有一个根本性的问题：**它不能生成新数据。**

为什么？因为 AutoEncoder 的潜在空间没有任何结构保证。编码器只是把每个训练样本映射到潜在空间中的一个**固定点**。如果你在潜在空间中随机选一个点——比如 $(0.3, -1.7)$——把它送进解码器，得到的可能是一张毫无意义的噪声图。原因是潜在空间中训练样本占据的区域可能是不连续的、有很多"空洞"的。

换句话说：AutoEncoder 学到的是一个**查找表**（每个输入对应潜在空间中的一个点），而不是一个**连续的、有意义的分布**。

这引出了 VAE 的核心动机：如果我们让编码器输出的不是一个点，而是一个**概率分布**，并且强制这些分布有某种规律性，那潜在空间就会变得"平滑"——随机采样就能生成有意义的新数据。

> 可靠程度：**Level 1**

---

## 3. 变分自编码器（VAE）

### 3.1 核心思想

**变分自编码器**（Variational AutoEncoder，VAE，Kingma & Welling 2014）的核心改进只有一条：

> **不是把输入压缩到一个固定的点，而是压缩到一个概率分布。**

具体来说：

- AutoEncoder 的编码器：$x \mapsto z$（一个确定的向量）
- VAE 的编码器：$x \mapsto (\mu, \sigma^2)$（一个高斯分布的参数）

VAE 的编码器输出两个向量：**均值** $\mu \in \mathbb{R}^d$ 和**方差** $\sigma^2 \in \mathbb{R}^d$（对角高斯分布），它们定义了一个条件概率分布：

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

这里 $q_\phi(z|x)$ 读作"给定输入 $x$ 时，潜在变量 $z$ 的近似后验分布"。$\phi$ 是编码器的参数。

训练时，从这个分布中**采样**一个 $z$，再送入解码器重建。因为每次采样的 $z$ 不完全一样（带有随机性），解码器必须能从整个分布的"邻域"都重建出合理的输出——这迫使潜在空间变得连续、平滑。

### 3.2 重参数化技巧（Reparameterization Trick）

这里有一个技术难题：训练需要反向传播，但**"从分布中采样"这个操作不可微**。

想象一下：编码器输出 $\mu$ 和 $\sigma$，然后你从 $\mathcal{N}(\mu, \sigma^2)$ 中随机抽一个 $z$。这个采样过程是随机的，梯度没法穿过它——你没法对"抛骰子"这个操作求导。

**重参数化技巧**（Reparameterization Trick）巧妙地解决了这个问题。核心思路：把随机性从"参数相关的分布"中分离出来。

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\odot$ 是逐元素乘法，$\epsilon$ 是从标准正态分布采样的噪声——它和网络参数完全无关。

这样一来：
- $z$ 是 $\mu$ 和 $\sigma$ 的**确定性函数**（加上一个固定的外部噪声 $\epsilon$）
- 梯度可以通过 $\mu$ 和 $\sigma$ 正常回传到编码器
- 随机性完全由外部的 $\epsilon$ 提供，不阻断梯度流

这个技巧看似简单，但它是 VAE 能用梯度下降训练的关键。没有它，整个框架就无法端到端优化。

> 可靠程度：**Level 1**（教科书内容，已被广泛验证）

### 3.3 VAE 的损失函数：ELBO

VAE 的损失函数来自概率论中的**变分推断**（variational inference），但直觉上并不难理解。它由两项组成：

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建损失}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 散度正则项}}$$

逐项解释：

**第一项：重建损失** $-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$

- $p_\theta(x|z)$ 是解码器定义的"给定 $z$ 生成 $x$ 的概率"（$\theta$ 是解码器参数）
- 如果我们假设 $p_\theta(x|z)$ 是高斯分布，这一项就退化为 MSE 重建损失（和普通 AutoEncoder 一样）
- 直觉：**让重建尽可能准确**

**第二项：KL 散度正则项** $D_{KL}(q_\phi(z|x) \| p(z))$

- $q_\phi(z|x)$ 是编码器输出的分布
- $p(z) = \mathcal{N}(0, I)$ 是我们选定的**先验分布**（prior）——标准正态分布
- KL 散度衡量两个分布的"距离"：$D_{KL}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} dz \geq 0$，当且仅当 $q = p$ 时等于零
- 直觉：**让编码器输出的分布不要偏离标准正态太远**

对于对角高斯分布，KL 散度有闭合解（不需要数值积分）：

$$D_{KL}(q_\phi(z|x) \| p(z)) = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

其中 $d$ 是潜在空间维度，$\mu_j$ 和 $\sigma_j^2$ 是编码器输出的第 $j$ 维的均值和方差。

**两项的拉锯**：
- 如果只有重建损失：编码器会把每个输入压缩到一个很窄的分布（$\sigma \to 0$），退化成普通 AutoEncoder
- 如果只有 KL 项：编码器会输出标准正态分布，完全忽略输入——重建质量极差
- 两项共同优化，达到一个**平衡**：潜在分布足够接近正态（使得采样有意义），同时重建足够准确

> **为什么叫 ELBO？** 这个损失函数的负值叫做 **Evidence Lower BOund**（证据下界），因为它是数据对数似然 $\log p(x)$ 的下界。最大化 ELBO 等价于近似最大化数据的对数似然。这来自变分推断的理论框架，详见 [Doersch Tutorial](https://arxiv.org/abs/1606.05908)。

> 可靠程度：**Level 1**

### 3.4 为什么 VAE 可以生成新数据

训练完成后，VAE 的潜在空间有两个关键性质：

1. **连续性**：KL 正则项迫使所有 $q_\phi(z|x)$ 都接近 $\mathcal{N}(0, I)$，所以潜在空间中不会有大片"空洞"——相邻的 $z$ 解码出的数据也相似。

2. **覆盖性**：不同输入的编码分布互相重叠（都被拉向标准正态），所以潜在空间的大部分区域都被"填满"了。

因此，要生成新数据，只需：

$$z \sim \mathcal{N}(0, I) \quad \Rightarrow \quad \hat{x} = g_\psi(z)$$

从标准正态分布随机采一个 $z$，送进解码器，就能得到一个看起来合理的新样本。你甚至可以在两个编码之间**插值**：取两张图片的编码 $z_1$ 和 $z_2$，沿直线取中间的点 $z_t = (1-t)z_1 + tz_2$，解码后会看到从一张图片到另一张图片的平滑过渡。

---

## 4. 应用

- **降维与可视化**：和 PCA 类似，但能捕捉非线性结构。把高维数据压缩到 2-3 维后可视化，观察数据的聚类结构。
- **异常检测**（anomaly detection）：正常数据的重建误差低，异常数据的重建误差高（因为网络没见过类似的数据，压缩后丢失了太多信息）。工业质检、网络入侵检测都有应用。
- **图像生成**：VAE 可以从潜在空间采样生成新图像，虽然质量不如 GAN 和 Diffusion Model，但训练稳定、潜在空间有意义。
- **药物分子设计**：把分子结构编码到潜在空间，在潜在空间中优化想要的性质（如药效、毒性），再解码回分子结构。这是 VAE 在化学/制药领域的重要应用（参见 [Gómez-Bombarelli et al. 2018](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)）。

> 可靠程度：**Level 1-2**（降维、异常检测是成熟应用；药物分子设计是活跃研究方向）

---

## 5. VAE vs GAN

| 维度 | VAE | GAN |
|------|-----|-----|
| 生成质量 | 图像偏模糊（MSE 损失倾向于"平均化"） | 图像清晰锐利 |
| 训练稳定性 | 稳定，有明确的损失函数可监控 | 不稳定，容易模式坍塌 |
| 潜在空间 | 有意义、连续、可插值 | 结构较混乱 |
| 理论基础 | 完整的概率框架（变分推断） | 博弈论（minimax game） |
| 生成多样性 | 高（覆盖整个数据分布） | 容易丢失部分模式 |

VAE 生成图像偏模糊的原因：MSE 重建损失本质上是在计算像素级的平均误差。对于一张人脸，如果有多种合理的生成方式（头发偏左 vs 偏右），VAE 倾向于输出它们的"平均"——结果就是模糊的图像。这个问题在后来的 Diffusion Model 中得到了更好的解决。

> 可靠程度：**Level 2**（VAE/GAN 对比是业界常见总结，具体表现取决于模型规模和训练细节）

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $z = f_\phi(x)$, $\hat{x} = g_\psi(z)$ | 编码器与解码器 |
| $\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2$ | AutoEncoder MSE 重建损失 |
| $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$ | VAE 编码器输出的条件分布 |
| $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ | 重参数化技巧 |
| $\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) \| p(z))$ | VAE 损失：重建 + KL 正则 |
| $D_{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$ | 对角高斯 KL 散度闭合解 |
| $z \sim \mathcal{N}(0, I) \Rightarrow \hat{x} = g_\psi(z)$ | 从潜在空间采样生成新数据 |

---

## 理解检测

**Q1**：AutoEncoder 的编码器把 MNIST 图片压缩到 2 维。如果我们把瓶颈层维度从 2 增加到 784（和输入一样大），会发生什么？损失函数会怎么变？学到的表示还有意义吗？为什么？

你的回答：


**Q2**：VAE 的重参数化技巧写为 $z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。假设我们不用这个技巧，而是直接从 $\mathcal{N}(\mu, \sigma^2)$ 采样 $z$，然后计算损失并反向传播——具体来说，梯度链条在哪里断裂？是 $\frac{\partial \mathcal{L}}{\partial z}$ 没法算，还是 $\frac{\partial z}{\partial \mu}$ 没法算？

你的回答：


**Q3**：VAE 损失函数中的 KL 散度项把 $q_\phi(z|x)$ 拉向标准正态 $\mathcal{N}(0,I)$。如果我们把这一项的权重设为 0（去掉 KL 项），训练出的模型和普通 AutoEncoder 有什么区别？如果把权重设得非常大（远大于重建损失），又会发生什么？

你的回答：

