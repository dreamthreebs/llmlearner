# 04 - 贝叶斯推断 — 先验、后验、可信区间

> **主维度**：D2 参数估计
> **次维度**：D1 概率论基础（贝叶斯定理回顾）、D4 宇宙学应用（实际先验选择）
> **关键关系**：
> - 贝叶斯推断 (理论) --依赖--> 似然函数 (概念)：贝叶斯推断依赖似然函数构建后验
> - 贝叶斯推断 (理论) --依赖--> 贝叶斯定理 (理论)：贝叶斯推断依赖贝叶斯定理
> - MCMC (方法) --依赖--> 贝叶斯推断 (理论)：MCMC 依赖贝叶斯推断提供后验目标分布
>
> **学习路径**：全景概览 → 概率论复习 → 似然与 MLE → **本章（贝叶斯推断）** → MCMC → Fisher 矩阵 → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：02-probability-review（贝叶斯定理、多元高斯分布、全概率公式），03-likelihood-mle（似然函数、MLE、高斯似然矩阵形式、$\chi^2$）
>
> **参考**：
> - [R. Trotta, 2008 - Bayes in the sky, Section 2-4](https://arxiv.org/abs/0803.4089)
> - [D.S. Sivia, Data Analysis: A Bayesian Tutorial, Ch.2-4](https://global.oup.com/academic/product/data-analysis-9780198568322)
> - [Planck 2018 Results VI - Cosmological Parameters, Section 2](https://arxiv.org/abs/1807.06209)

---

## 1. 贝叶斯定理回顾：从概率论到参数估计

**可靠程度：Level 1**

在 02 章我们推导了贝叶斯定理的一般形式。现在把它应用到参数估计的语境中。用 $\boldsymbol{\theta}$ 替换 $A$，用 $\mathbf{d}$ 替换 $B$：

$$\boxed{P(\boldsymbol{\theta} | \mathbf{d}) = \frac{P(\mathbf{d} | \boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})}{P(\mathbf{d})}}$$

| 符号 | 名称 | 角色 |
|------|------|------|
| $P(\boldsymbol{\theta} \| \mathbf{d})$ | 后验分布（posterior） | **目标**：看到数据后对参数的更新信念 |
| $P(\mathbf{d} \| \boldsymbol{\theta})$ | 似然函数（likelihood） | **输入 1**：给定参数时数据出现的概率（03 章） |
| $\pi(\boldsymbol{\theta})$ | 先验分布（prior） | **输入 2**：观测前对参数的信念 |
| $P(\mathbf{d})$ | 证据 / 边缘似然（evidence） | **归一化常数**：$P(\mathbf{d}) = \int P(\mathbf{d} \| \boldsymbol{\theta}) \, \pi(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$ |

在参数估计中，我们通常只关心后验的形状，不关心归一化常数（MCMC 自动处理归一化），所以常写为：

$$P(\boldsymbol{\theta} | \mathbf{d}) \propto \mathcal{L}(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})$$

这是贝叶斯参数估计的核心公式：**后验 $\propto$ 似然 $\times$ 先验**。

---

## 2. 先验分布的选择

**可靠程度：Level 1（基本概念），Level 3（具体先验的选择经验）**

### 2.1 无信息先验

当你"不知道该选什么先验"时，有两种常用选择：

**均匀先验**（flat prior）：

$$\pi(\theta) = \frac{1}{\theta_{\max} - \theta_{\min}}, \quad \theta_{\min} \leq \theta \leq \theta_{\max}$$

- 在给定范围内不偏好任何值
- 范围外 $\pi(\theta) = 0$——先验**总是**有范围的，不存在真正的"无限均匀先验"（因为无法归一化）
- 后验 $\propto \mathcal{L}(\theta)$（似然主导），MLE 和 MAP（见 3.2 节）给出相同结果
- 注意：均匀先验**不是参数化不变的**。如果 $\theta$ 的先验是均匀的，$\theta^2$ 的先验就不是均匀的

**Jeffreys 先验**：

$$\pi(\theta) \propto \sqrt{|F(\theta)|}$$

其中 $F(\theta)$ 是 Fisher 信息矩阵（06 章详述）。

| 符号 | 含义 |
|------|------|
| $F(\theta)$ | Fisher 信息矩阵，是 $\theta$ 的函数 |
| $\|F(\theta)\|$ | Fisher 矩阵的行列式（多参数情况）；单参数时就是 $F(\theta)$ 本身 |

- Jeffreys 先验是**参数化不变的**：不管你用 $\theta$ 还是 $\phi = g(\theta)$，得到的后验一致
- 对于 $\sigma$（标度参数），Jeffreys 先验给出 $\pi(\sigma) \propto 1/\sigma$，即 $\ln \sigma$ 的均匀先验——这符合"我们不知道量级"的直觉

### 2.2 有信息先验

当其他实验已经测量过某个参数时，可以用其结果作为先验。

**高斯先验**（最常见）：

$$\pi(\theta) = \frac{1}{\sqrt{2\pi}\,\sigma_{\text{prior}}} \exp\!\left(-\frac{(\theta - \mu_{\text{prior}})^2}{2\sigma_{\text{prior}}^2}\right)$$

| 符号 | 含义 |
|------|------|
| $\mu_{\text{prior}}$ | 先验的中心值（通常来自其他实验的最佳拟合值） |
| $\sigma_{\text{prior}}$ | 先验的宽度（通常来自其他实验的误差棒） |

**例子**：用 BBN（大爆炸核合成）对 $\Omega_b h^2$ 的约束 $0.0224 \pm 0.0001$ 作为 CMB 分析的先验。

### 2.3 先验敏感性：什么时候先验重要？

**先验不重要**（似然主导）的条件：

- 数据量大、测量精度高 → 似然很窄、很尖锐
- 先验比似然宽得多 → 先验在似然的非零区域内几乎是常数
- 此时后验 $\approx$ 似然，贝叶斯结果和频率学派结果接近

**先验重要**（先验显著影响后验）的条件：

- 数据约束力弱 → 似然很宽、很平坦
- 先验和似然宽度可比 → 两者共同决定后验的形状和位置
- 参数接近先验边界 → 先验的截断效应显著

**具体例子**：

- $H_0$ 的 Planck 约束：$\sigma \approx 0.5$ km/s/Mpc，先验范围 $[20, 100]$。先验几乎不影响结果，因为似然峰远离边界且非常窄。
- 光学深度 $\tau$ 的约束：$\tau$ 受先验 $\tau > 0$（物理要求）影响较大，因为后验峰靠近下界，截断效应不可忽略。

---

## 3. 后验分布

**可靠程度：Level 1**

### 3.1 后验的含义

后验分布 $P(\boldsymbol{\theta} | \mathbf{d})$ 包含了我们对参数的**全部知识**——它综合了数据（通过似然）和先前知识（通过先验）。从后验分布中可以提取：

- **点估计**：后验的峰值（MAP）、均值（posterior mean）、中位数
- **不确定性**：可信区间、标准差
- **参数间的关联**：联合后验的等高线图

### 3.2 最大后验估计（MAP）vs MLE

**MAP**（Maximum A Posteriori）：

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \, P(\boldsymbol{\theta} | \mathbf{d}) = \arg\max_{\boldsymbol{\theta}} \left[\ln \mathcal{L}(\boldsymbol{\theta}) + \ln \pi(\boldsymbol{\theta})\right]$$

| 比较项 | MLE | MAP |
|--------|-----|-----|
| 最大化目标 | $\ln \mathcal{L}(\boldsymbol{\theta})$ | $\ln \mathcal{L}(\boldsymbol{\theta}) + \ln \pi(\boldsymbol{\theta})$ |
| 先验 | 不使用先验 | 使用先验 |
| 均匀先验下 | — | MAP = MLE |
| 高斯先验下 | — | MAP 被"拉向"先验中心 |

**具体数字例子**：

假设你对某个参数 $\theta$ 有：
- 似然：$\ln \mathcal{L} = -\frac{(\theta - 5.0)^2}{2 \times 0.5^2}$（数据给出 $\hat{\theta}_{\text{MLE}} = 5.0$，$\sigma_{\text{data}} = 0.5$）
- 高斯先验：$\ln \pi = -\frac{(\theta - 4.0)^2}{2 \times 1.0^2}$（先验中心 $\mu_{\text{prior}} = 4.0$，$\sigma_{\text{prior}} = 1.0$）

后验（取对数）：

$$\ln P(\theta | d) = -\frac{(\theta - 5.0)^2}{2 \times 0.25} - \frac{(\theta - 4.0)^2}{2 \times 1.0} + \text{const}$$

$$= -2(\theta - 5.0)^2 - 0.5(\theta - 4.0)^2 + \text{const}$$

对 $\theta$ 求导令其为零：

$$-4(\theta - 5.0) - 1.0(\theta - 4.0) = 0$$

$$-4\theta + 20 - \theta + 4 = 0$$

$$5\theta = 24 \implies \hat{\theta}_{\text{MAP}} = 4.8$$

**解读**：MLE = 5.0（纯数据），MAP = 4.8（被先验向 4.0 拉了一点）。因为数据的精度（$\sigma = 0.5$）比先验（$\sigma = 1.0$）高，所以数据占主导，MAP 只被拉了 0.2。如果先验更窄（比如 $\sigma_{\text{prior}} = 0.3$），MAP 会更接近先验中心。

后验的方差为：

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_{\text{data}}^2} + \frac{1}{\sigma_{\text{prior}}^2} = \frac{1}{0.25} + \frac{1}{1.0} = 4 + 1 = 5$$

$$\sigma_{\text{post}} = \frac{1}{\sqrt{5}} \approx 0.447$$

后验的宽度**总是小于**似然和先验中更窄的那个——加入更多信息总是缩小不确定性。

### 3.3 后验的数值计算困难

对于 $p$ 个参数，后验是 $p$ 维函数。要完整描述它需要：

- **网格法**：在每个参数方向上取 $K$ 个格点，总共 $K^p$ 个点。ΛCDM 有 6 个参数，每个方向取 100 个点，就是 $100^6 = 10^{12}$ 个点——不可行
- **MCMC**：不需要评估整个空间，只在高概率区域采样，计算量和维度近似线性增长——这就是为什么宇宙学使用 MCMC（05 章）

---

## 4. 边缘化（Marginalization）

**可靠程度：Level 1**

### 4.1 定义

假设参数向量 $\boldsymbol{\theta} = (\theta_1, \theta_2)$，你只关心 $\theta_1$，$\theta_2$ 是**干扰参数**（nuisance parameter，在 01-overview 中已介绍）。$\theta_1$ 的**边缘后验**通过对 $\theta_2$ 积分得到：

$$P(\theta_1 | \mathbf{d}) = \int P(\theta_1, \theta_2 | \mathbf{d}) \, d\theta_2$$

| 符号 | 含义 |
|------|------|
| $P(\theta_1, \theta_2 \| \mathbf{d})$ | 联合后验分布 |
| $P(\theta_1 \| \mathbf{d})$ | $\theta_1$ 的边缘后验分布（已把 $\theta_2$ 积分掉） |

直觉：边缘化就是**把不关心的参数"投影"掉**。二维联合分布投影到一维，就像把 2D 等高线图压扁到某个轴上。

### 4.2 和频率学派的区别

在频率学派方法中，处理 nuisance 参数通常用 **profile likelihood**：

$$\mathcal{L}_{\text{prof}}(\theta_1) = \max_{\theta_2} \mathcal{L}(\theta_1, \theta_2)$$

也就是对每个 $\theta_1$ 值，找到使似然最大的 $\theta_2$。这和贝叶斯的积分（加权平均）在概念上完全不同：

- **边缘化**（贝叶斯）：对 $\theta_2$ 的所有可能值加权平均
- **Profile**（频率学派）：只取 $\theta_2$ 的最优值

当后验接近高斯时，两种方法给出近似相同的结果。当后验有明显的非高斯特征（如多峰、长尾）时，结果可能不同。

### 4.3 具体例子：2 参数高斯后验的边缘化

**场景**：两个参数 $\theta_1, \theta_2$ 的联合后验是 2 维高斯分布：

$$P(\theta_1, \theta_2 | \mathbf{d}) \propto \exp\!\left(-\frac{1}{2} \begin{pmatrix} \theta_1 - \mu_1 \\ \theta_2 - \mu_2 \end{pmatrix}^T \mathbf{C}^{-1}_{\text{post}} \begin{pmatrix} \theta_1 - \mu_1 \\ \theta_2 - \mu_2 \end{pmatrix}\right)$$

设 $\boldsymbol{\mu} = \begin{pmatrix} 2.0 \\ 0.5 \end{pmatrix}$，后验协方差矩阵 $\mathbf{C}_{\text{post}} = \begin{pmatrix} 1.0 & 0.6 \\ 0.6 & 0.5 \end{pmatrix}$。

**边缘化 $\theta_2$**：对多元高斯分布，边缘化的结果仍然是高斯分布，边缘后验的均值和方差直接从协方差矩阵中读取：

$$\theta_1 | \mathbf{d} \sim \mathcal{N}(\mu_1, C_{11}) = \mathcal{N}(2.0, 1.0)$$

即 $\theta_1$ 的边缘后验均值为 2.0，标准差为 $\sqrt{1.0} = 1.0$。

**对比条件分布**（固定 $\theta_2 = \mu_2$ 时 $\theta_1$ 的分布）：

$$\sigma_{\theta_1 | \theta_2}^2 = C_{11} - \frac{C_{12}^2}{C_{22}} = 1.0 - \frac{0.36}{0.5} = 1.0 - 0.72 = 0.28$$

条件标准差为 $\sqrt{0.28} \approx 0.529$，**小于**边缘标准差 1.0。

**物理解读**：如果你"知道" $\theta_2$ 的精确值（条件分布），$\theta_1$ 的不确定性更小（0.53 vs 1.0）。边缘化把 $\theta_2$ 的不确定性也"传播"到了 $\theta_1$ 上，所以边缘后验更宽。这就是为什么有参数退化（correlation）时，边缘误差比条件误差大。

---

## 5. 可信区间 vs 置信区间

**可靠程度：Level 1**

### 5.1 定义区别

| | 可信区间（Credible Interval, CI） | 置信区间（Confidence Interval） |
|---|---|---|
| 学派 | 贝叶斯 | 频率学派 |
| 定义 | $\theta$ 有 X% 的概率在此区间内 | 如果重复实验无穷次，X% 的区间包含真值 |
| 计算来源 | 后验分布 $P(\theta \| \mathbf{d})$ | 抽样分布 |
| 直觉 | "我相信参数在这里" | "我的方法长期来看是对的" |

宇宙学论文中写的 "68% CL" 通常是贝叶斯可信区间（从 MCMC 后验得到），虽然用了 "CL"（confidence level）这个频率学派术语。

### 5.2 两种可信区间

**等尾区间**（Equal-tail interval）：取后验分布的第 16% 和第 84% 分位数，使得上下尾各有 16% 的概率。

**最高后验密度区间**（Highest Posterior Density, HPD）：包含后验密度最高的区域，使得区间内的总概率为 68%。

| 性质 | 等尾区间 | HPD 区间 |
|------|---------|---------|
| 计算方法 | 分位数 | 找密度阈值 |
| 对称后验 | 两者相同 | 两者相同 |
| 偏斜后验 | 更宽 | 更窄（最短区间） |
| 唯一性 | 唯一 | 唯一 |

当后验接近高斯（对称）时，两种区间给出相同的结果。宇宙学中，大部分参数的后验近似高斯，所以区别不大。但有些参数的后验明显偏斜（如 $\tau$，因为有 $\tau > 0$ 的截断），此时应该注意使用哪种区间。

### 5.3 在宇宙学中的实际意义

论文中的 "$H_0 = 67.36 \pm 0.54$ (68% CL)" 意味着：

- 这是 MCMC 后验的中心值和等尾区间（或后验均值 $\pm$ 标准差）
- 严格来说是贝叶斯可信区间："根据 Planck 数据和 ΛCDM 模型，$H_0$ 有 68% 的概率在 $[66.82, 67.90]$ 区间内"
- 因为 $H_0$ 的后验非常接近高斯，可信区间和置信区间的数值几乎相同

---

## 6. 宇宙学中的实际先验选择

**可靠程度：Level 3**

### 6.1 ΛCDM 六参数的典型先验

Planck 分析中 ΛCDM 基本参数的常用先验范围：

| 参数 | 含义 | 先验类型 | 先验范围 |
|------|------|---------|---------|
| $\Omega_b h^2$ | 重子物质密度 | 均匀 | $[0.005, 0.1]$ |
| $\Omega_c h^2$ | 暗物质密度 | 均匀 | $[0.001, 0.99]$ |
| $100\theta_{\text{MC}}$ | 声学视界角 | 均匀 | $[0.5, 10.0]$ |
| $\tau$ | 再电离光学深度 | 均匀 | $[0.01, 0.8]$ |
| $\ln(10^{10} A_s)$ | 原初功率谱振幅 | 均匀 | $[1.61, 3.91]$ |
| $n_s$ | 标量谱指数 | 均匀 | $[0.8, 1.2]$ |

这些先验被设得足够宽，使得后验峰远离边界，结果不依赖先验范围。但 $\tau$ 的下界 $0.01$ 对后验有一定影响（因为后验峰在 $\tau \approx 0.054$，离下界相对较近）。

### 6.2 联合约束：多个数据集的组合

贝叶斯框架的一个强大优势是可以自然地组合多个独立数据集。如果有两个独立实验的数据 $\mathbf{d}_1$（如 Planck CMB）和 $\mathbf{d}_2$（如 BAO，重子声学振荡巡天），各自给出似然 $\mathcal{L}_1$ 和 $\mathcal{L}_2$：

$$P(\boldsymbol{\theta} | \mathbf{d}_1, \mathbf{d}_2) \propto \mathcal{L}_1(\boldsymbol{\theta}) \, \mathcal{L}_2(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})$$

即联合后验 $\propto$ 各似然的乘积 $\times$ 先验。等价地：

$$\ln P \propto \ln \mathcal{L}_1 + \ln \mathcal{L}_2 + \ln \pi$$

| 符号 | 含义 |
|------|------|
| $\mathcal{L}_1(\boldsymbol{\theta})$ | 第一个数据集的似然 |
| $\mathcal{L}_2(\boldsymbol{\theta})$ | 第二个数据集的似然（独立于第一个） |
| $\pi(\boldsymbol{\theta})$ | 共享的先验（只用一次，不重复乘） |

也可以理解为**串联更新**：先用 $\mathbf{d}_1$ 更新先验得到后验 $P_1$，再把 $P_1$ 当作新先验，用 $\mathbf{d}_2$ 再更新，结果完全相同。

**实际操作**：在 CosmoMC/Cobaya 中，你在参数文件里列出多个数据集（如 `planck_2018_lowl.TT` + `planck_2018_highl_plik.TTTEEE` + `bao_data`），程序自动把它们的对数似然加起来。你在 kSZ 分析中也是这样做的——kSZ 似然加上其他约束。

**注意**：组合不同数据集时必须确保它们真的**独立**。如果两个数据集使用了相同的原始数据（比如同一片天区的 CMB 数据），直接相乘会重复计数信息，导致误差被低估。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $P(\boldsymbol{\theta} \| \mathbf{d}) \propto \mathcal{L}(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})$ | 贝叶斯推断核心：后验 ∝ 似然 × 先验 |
| $\hat{\theta}_{\text{MAP}} = \arg\max [\ln \mathcal{L} + \ln \pi]$ | 最大后验估计 |
| $\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_{\text{data}}^2} + \frac{1}{\sigma_{\text{prior}}^2}$ | 高斯后验方差（似然和先验都是高斯时） |
| $P(\theta_1 \| \mathbf{d}) = \int P(\theta_1, \theta_2 \| \mathbf{d}) \, d\theta_2$ | 边缘化：对 nuisance 参数积分 |
| $P(\boldsymbol{\theta} \| \mathbf{d}_1, \mathbf{d}_2) \propto \mathcal{L}_1 \mathcal{L}_2 \, \pi$ | 联合约束：独立数据集的似然相乘 |
| $\sigma_{\theta_1|\theta_2}^2 = C_{11} - C_{12}^2 / C_{22}$ | 条件方差（固定一个参数后的方差） |
| $\pi_{\text{Jeffreys}} \propto \sqrt{\|F(\theta)\|}$ | Jeffreys 先验（参数化不变） |

---

## 理解检测

**Q1**（概念）：MAP 和 MLE 的区别是什么？写出两者的优化目标。在什么条件下 MAP = MLE？为什么宇宙学中大部分参数的 MAP 和 MLE 差别不大？

你的回答：


**Q2**（计算）：某参数的似然给出 $\hat{\theta}_{\text{MLE}} = 70.0$，$\sigma_{\text{data}} = 2.0$。你加了一个高斯先验 $\mu_{\text{prior}} = 66.0$，$\sigma_{\text{prior}} = 3.0$。(a) 计算 MAP 估计值 $\hat{\theta}_{\text{MAP}}$；(b) 计算后验标准差 $\sigma_{\text{post}}$；(c) MAP 更接近 MLE 还是先验中心？为什么？

> 提示：用 $\hat{\theta}_{\text{MAP}} = \frac{\hat{\theta}/\sigma_{\text{data}}^2 + \mu_{\text{prior}}/\sigma_{\text{prior}}^2}{1/\sigma_{\text{data}}^2 + 1/\sigma_{\text{prior}}^2}$，$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{\sigma_{\text{data}}^2} + \frac{1}{\sigma_{\text{prior}}^2}$

你的回答：


**Q3**（概念 + 计算）：一个 2 参数高斯后验的协方差矩阵为 $\mathbf{C} = \begin{pmatrix} 4.0 & 2.0 \\ 2.0 & 3.0 \end{pmatrix}$。(a) 边缘化掉 $\theta_2$ 后，$\theta_1$ 的标准差是多少？(b) 如果固定 $\theta_2$（条件分布），$\theta_1$ 的标准差是多少？(c) 哪个更小？为什么？

> 提示：边缘标准差 $= \sqrt{C_{11}}$，条件方差 $= C_{11} - C_{12}^2 / C_{22}$

你的回答：


**Q4**（概念）：你有两个独立实验分别测量 $H_0$：实验 A 得到 $H_0 = 67 \pm 1$，实验 B 得到 $H_0 = 73 \pm 1$。(a) 如果直接组合（联合后验），结果会是什么？(b) 这个结果在物理上有什么问题？(c) 这种情况在宇宙学中被称为什么？

你的回答：
