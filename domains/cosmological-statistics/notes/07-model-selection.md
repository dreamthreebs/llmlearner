# 07 - 模型选择 — 超越参数估计

> **主维度**：D3 模型选择与检验
> **次维度**：D2 参数估计（模型选择建立在参数估计基础上）
> **关键关系**：
> - 贝叶斯证据 (概念) --依赖--> 贝叶斯推断 (理论)：证据是后验归一化常数
> - AIC / BIC (方法) --用于--> 模型选择 (任务)：信息准则用于模型比较
> - 贝叶斯证据 (概念) --对比--> AIC / BIC (方法)：精确证据 vs 近似准则
>
> **学习路径**：全景概览 → 概率论复习 → 似然与 MLE → 贝叶斯推断 → MCMC → Fisher 矩阵 → **本章（模型选择）** → 宇宙学似然 → 结果解读
>
> **前置知识**：似然函数、对数似然、MLE、$\chi^2$（03-likelihood-mle）；贝叶斯定理、先验、后验、边缘化（04-bayesian-inference）；MCMC 基本原理（05-mcmc）
>
> **参考**：
> - [A.R. Liddle, 2007 - Information criteria for astrophysical model selection](https://arxiv.org/abs/0701113)
> - [R. Trotta, 2008 - Bayes in the sky, Section 6](https://arxiv.org/abs/0803.4089)
> - [R. Trotta, 2007 - Applications of Bayesian model selection to cosmological parameters](https://arxiv.org/abs/0803.4089)
> - [H. Jeffreys, 1961 - Theory of Probability (3rd ed.)]

---

## 1. 参数估计 vs 模型选择

**可靠程度：Level 1**

到目前为止，我们学的方法（MLE、贝叶斯推断、MCMC、Fisher 矩阵）都在回答同一类问题：

> **参数估计**：给定一个模型 $M$，参数 $\boldsymbol{\theta}$ 的最佳值和不确定性是什么？

但这默认了模型本身是正确的。还有一类完全不同的问题：

> **模型选择**：在多个候选模型 $M_1, M_2, \ldots$ 之间，数据更支持哪一个？

**宇宙学中的典型例子**：

| 问题 | 类型 |
|------|------|
| $H_0 = ?$ | 参数估计 |
| ΛCDM（$w = -1$）vs wCDM（$w$ 自由）哪个更好？ | 模型选择 |
| Planck 数据是否与 ΛCDM 一致？ | 拟合优度（模型检验） |
| $H_0$ 的 Planck 值与 SH0ES 值是否有矛盾？ | 张力诊断 |

模型选择的核心困难：**更复杂的模型总能更好地拟合数据**（至少不会更差），但更多参数意味着过拟合的风险。需要一个原则来**平衡拟合质量和模型复杂度**。

---

## 2. 拟合优度：$\chi^2$ 检验

**可靠程度：Level 1**

### 2.1 $\chi^2$ 统计量回顾

在高斯似然下（参见 03 章），似然和 $\chi^2$ 直接相关：

$$\chi^2(\boldsymbol{\theta}) = (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta}))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta})) = -2 \ln \mathcal{L}(\boldsymbol{\theta}) + \text{const}$$

- $\mathbf{d}$：数据向量
- $\mathbf{m}(\boldsymbol{\theta})$：理论模型预测
- $\mathbf{C}$：数据协方差矩阵
- 在最佳拟合参数 $\hat{\boldsymbol{\theta}}$ 处得到 $\chi^2_\text{min}$

### 2.2 约化 $\chi^2$

**约化卡方**（reduced chi-squared）：

$$\chi^2_\text{red} = \frac{\chi^2_\text{min}}{\nu}$$

- $\chi^2_\text{min}$：最佳拟合参数处的 $\chi^2$ 值
- $\nu = N_\text{data} - k$：**自由度**（degrees of freedom, dof）
- $N_\text{data}$：数据点的数量
- $k$：自由参数的数量

**解读**：

| $\chi^2_\text{red}$ | 含义 |
|-----------------------|------|
| $\approx 1$ | 好的拟合——数据与模型的偏差在统计误差范围内 |
| $\gg 1$ | 差的拟合——模型不能解释数据，或误差被低估 |
| $\ll 1$ | 过拟合——模型过于灵活，或误差被高估 |

### 2.3 具体数字例子

假设你用 ΛCDM（$k = 6$ 个自由参数）拟合 Planck CMB 功率谱数据（$N_\text{data} = 2500$ 个 $C_\ell$ 数据点）：

- $\nu = 2500 - 6 = 2494$
- 如果 $\chi^2_\text{min} = 2530$，则 $\chi^2_\text{red} = 2530 / 2494 = 1.014$ → 拟合良好
- 如果 $\chi^2_\text{min} = 3500$，则 $\chi^2_\text{red} = 3500 / 2494 = 1.40$ → 拟合显著偏差

$\chi^2_\text{red}$ 的期望值是 1，标准差约为 $\sqrt{2/\nu}$。当 $\nu = 2494$ 时，$\sqrt{2/2494} \approx 0.028$。所以 $\chi^2_\text{red} = 1.014$ 距期望值不到 1 个标准差，完全正常。

### 2.4 PTE：Probability To Exceed

比 $\chi^2_\text{red}$ 更精确的拟合优度指标是 **PTE**（Probability To Exceed）：

$$\text{PTE} = P(\chi^2 > \chi^2_\text{min} \mid \nu) = \int_{\chi^2_\text{min}}^{\infty} f_{\chi^2}(x; \nu) \, dx$$

其中 $f_{\chi^2}(x; \nu)$ 是自由度为 $\nu$ 的 $\chi^2$ 分布的概率密度函数。

**直觉**：如果模型是正确的，你重复实验很多次，PTE 就是"得到比这次更差的拟合的概率"。

| PTE 值 | 含义 |
|--------|------|
| $\sim 0.5$ | 拟合正常——数据和模型偏差在典型范围内 |
| $< 0.05$ | 拟合差——数据与模型有显著不一致（相当于 $> 2\sigma$） |
| $< 0.01$ | 拟合很差——模型可能错误，或误差被低估 |
| $> 0.95$ | 过度拟合——$\chi^2$ 太小，可能误差被高估 |

**具体例子**：用上面 Planck 的数据，$\chi^2_\text{min} = 2530$, $\nu = 2494$。

$$\text{PTE} = P(\chi^2 > 2530 \mid 2494) \approx 0.27$$

PTE = 0.27，意味着"如果 ΛCDM 是对的，有 27% 的概率得到这么大或更大的 $\chi^2$"——完全正常，没有理由拒绝模型。

PTE 是 CMB 论文（如 Planck 系列论文）中报告拟合优度的标准方式。在 Planck 的似然论文中，你会看到类似"PTE = 0.15 for the TT spectrum"这样的表述。

> 参考：[Verde et al., 2003 - Goodness-of-fit statistics and CMB data sets](https://www.aanda.org/articles/aa/abs/2003/40/aa1934/aa1934.html)

### 2.5 $\chi^2$ 检验的局限

$\chi^2$ 检验只能告诉你"**这个模型拟合好不好**"，不能直接告诉你"**哪个模型更好**"。如果 wCDM 的 $\chi^2$ 比 ΛCDM 小了 3，这算显著改进吗？$\chi^2$ 本身没有考虑 wCDM 多了一个参数。需要更系统的方法。

---

## 3. 贝叶斯证据

**可靠程度：Level 1**

### 3.1 从贝叶斯定理到模型比较

回顾贝叶斯定理。之前我们用它来做参数估计——现在把模型本身也作为一个"参数"。对于模型 $M$：

$$P(M | \mathbf{d}) = \frac{P(\mathbf{d} | M) \, P(M)}{P(\mathbf{d})}$$

- $P(M | \mathbf{d})$：看到数据后模型 $M$ 的概率
- $P(\mathbf{d} | M)$：**贝叶斯证据**——数据在模型 $M$ 下的总概率
- $P(M)$：模型的先验概率

**贝叶斯证据**（Bayesian evidence / marginal likelihood）就是之前一直被当作"归一化常数"而忽略的那个量：

$$\mathcal{Z} = P(\mathbf{d} | M) = \int \mathcal{L}(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$$

- $\mathcal{Z}$：贝叶斯证据
- $\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} | \boldsymbol{\theta}, M)$：似然函数
- $\pi(\boldsymbol{\theta})$：参数的先验分布
- 积分遍历参数 $\boldsymbol{\theta}$ 的**整个先验范围**

在参数估计中，$\mathcal{Z}$ 不影响后验的形状，可以忽略。但在模型选择中，$\mathcal{Z}$ 是核心——它是数据对模型的**总评分**。

### 3.2 贝叶斯证据与 Occam's Razor

为什么 $\mathcal{Z}$ 能自动惩罚复杂模型？直觉：

考虑两个模型：
- 简单模型 $M_1$：1 个参数，先验范围 $[0, 10]$
- 复杂模型 $M_2$：2 个参数，先验范围各 $[0, 10]$

$M_2$ 的参数空间体积是 $M_1$ 的 10 倍。**证据 $\mathcal{Z}$ 是似然在先验上的平均值**——如果多出来的参数空间里似然都很低，$M_2$ 的证据就会被"稀释"。

$$\mathcal{Z} \sim \mathcal{L}_\text{max} \times \frac{\text{后验体积}}{\text{先验体积}}$$

- $\mathcal{L}_\text{max}$：最大似然值——模型能拟合得多好
- 后验体积 / 先验体积：先验中有多少比例被数据"支持"——衡量模型的预测力

复杂模型的 $\mathcal{L}_\text{max}$ 可能更高（拟合更好），但先验体积更大导致比值更小。只有当拟合改进**足以补偿**先验体积的增大时，复杂模型才会获胜。这正是**定量版本的 Occam's Razor**。

### 3.3 贝叶斯因子

比较两个模型 $M_1$ 和 $M_2$ 时，使用**贝叶斯因子**（Bayes factor）：

$$B_{12} = \frac{\mathcal{Z}_1}{\mathcal{Z}_2} = \frac{P(\mathbf{d} | M_1)}{P(\mathbf{d} | M_2)}$$

- $B_{12}$：模型 $M_1$ 相对于 $M_2$ 的贝叶斯因子
- $\mathcal{Z}_1, \mathcal{Z}_2$：两个模型各自的贝叶斯证据
- $B_{12} > 1$：数据更支持 $M_1$
- $B_{12} < 1$：数据更支持 $M_2$

### 3.4 Jeffreys 判据

Harold Jeffreys 提出了一个经验性的判读尺度，用 $|\ln B_{12}|$ 来量化证据的强度：

| $|\ln B_{12}|$ | 证据强度 | 含义 |
|-----------------|---------|------|
| $< 1$ | 不值一提（inconclusive） | 数据不足以区分两个模型 |
| $1 \sim 2.5$ | 实质性（substantial） | 有一定偏好，但不是决定性的 |
| $2.5 \sim 5$ | 强烈（strong） | 明确支持某个模型 |
| $> 5$ | 决定性（decisive） | 几乎确定 |

**具体数字例子**：ΛCDM vs wCDM

假设 $\ln B_{12} = \ln(\mathcal{Z}_\text{ΛCDM} / \mathcal{Z}_\text{wCDM}) = 2.8$。

按 Jeffreys 判据，$|\ln B_{12}| = 2.8$ 处于"强烈"区间——数据强烈支持 ΛCDM（$w = -1$），不需要额外的 $w$ 参数。换算成概率：如果先验上两个模型等概率，则 $P(M_1 | \mathbf{d}) / P(M_2 | \mathbf{d}) = e^{2.8} \approx 16$，ΛCDM 的后验概率是 wCDM 的 16 倍。

### 3.5 计算证据的困难

$\mathcal{Z} = \int \mathcal{L}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta}$ 是一个高维积分，比 MCMC 还难算——MCMC 只需要后验的**形状**，而证据需要**绝对值**。

常用的计算方法：
- **嵌套采样**（Nested Sampling）：专门为计算证据设计的算法（MultiNest、PolyChord）
- **热力学积分**（Thermodynamic Integration）：逐步从先验过渡到后验
- **信息准则**（AIC/BIC）：用简单公式近似证据——见下一节

---

## 4. 信息准则

**可靠程度：Level 1（公式与定义）+ Level 2（AIC/BIC 的渐近推导是近似的）**

信息准则是对贝叶斯证据的**快速近似**——不需要计算高维积分，只需要最佳拟合的 $\chi^2_\text{min}$ 和参数个数。

### 4.1 赤池信息准则（AIC）

$$\text{AIC} = \chi^2_\text{min} + 2k$$

- $\chi^2_\text{min} = -2 \ln \mathcal{L}_\text{max}$：最佳拟合处的 $\chi^2$（拟合优度项）
- $k$：模型的自由参数个数（复杂度惩罚项）
- **AIC 越小越好**

两项的含义：
- $\chi^2_\text{min}$：模型拟合得好 → 小 → AIC 下降
- $2k$：参数越多 → 大 → AIC 上升

**比较规则**：$\Delta \text{AIC} = \text{AIC}_2 - \text{AIC}_1$
- $\Delta \text{AIC} > 0$：$M_1$ 更好
- $|\Delta \text{AIC}| < 2$：两模型差不多
- $|\Delta \text{AIC}| > 10$：差别很大

### 4.2 贝叶斯信息准则（BIC）

$$\text{BIC} = \chi^2_\text{min} + k \ln N$$

- $\chi^2_\text{min}$：最佳拟合处的 $\chi^2$
- $k$：自由参数个数
- $N$：数据点的个数
- $\ln N$：当 $N > e^2 \approx 7.4$ 时（几乎总是），$\ln N > 2$，所以 **BIC 比 AIC 对参数数量的惩罚更重**

### 4.3 AIC vs BIC

| 特性 | AIC | BIC |
|------|-----|-----|
| 复杂度惩罚 | $2k$ | $k \ln N$ |
| 惩罚强度 | 较轻 | 较重（$N > 8$ 时） |
| 哲学基础 | 信息论（最小化预测误差） | 贝叶斯（近似贝叶斯证据） |
| 倾向于 | 选择能更好预测的模型 | 选择更简洁的模型 |

**具体数字例子**：

ΛCDM（$k_1 = 6$）vs wCDM（$k_2 = 7$），$N = 2500$ 个数据点。

假设 $\chi^2_{\text{min},1} = 2530$（ΛCDM），$\chi^2_{\text{min},2} = 2527$（wCDM，多一个参数 $w$，拟合稍好）。

AIC：
- $\text{AIC}_1 = 2530 + 2 \times 6 = 2542$
- $\text{AIC}_2 = 2527 + 2 \times 7 = 2541$
- $\Delta \text{AIC} = 2541 - 2542 = -1$ → 差不多，略微偏好 wCDM

BIC：
- $\text{BIC}_1 = 2530 + 6 \ln 2500 = 2530 + 6 \times 7.82 = 2576.9$
- $\text{BIC}_2 = 2527 + 7 \ln 2500 = 2527 + 7 \times 7.82 = 2581.7$
- $\Delta \text{BIC} = 2581.7 - 2576.9 = +4.8$ → 偏好 ΛCDM

**AIC 和 BIC 给出了不同的答案！** AIC 认为两者差不多（因为惩罚轻），BIC 认为应该选 ΛCDM（因为惩罚重）。$\chi^2$ 只改善了 3，额外参数 $w$ 带来的改进不够显著。

### 4.4 与贝叶斯证据的关系

BIC 可以看作贝叶斯证据 $\ln \mathcal{Z}$ 的一个粗略近似：

$$-2 \ln \mathcal{Z} \approx \text{BIC} + O(1)$$

贝叶斯因子的近似：$\ln B_{12} \approx -\frac{1}{2}(\text{BIC}_1 - \text{BIC}_2)$

在上面的例子中：$\ln B_{12} \approx -\frac{1}{2}(-4.8) = 2.4$，接近 Jeffreys 判据的"实质性"到"强烈"边界——与完整的贝叶斯证据计算通常给出一致的结论。

---

## 5. 频率学派 vs 贝叶斯的模型选择

**可靠程度：Level 1**

### 5.1 频率学派方法

**嵌套模型**（nested models）：$M_1$ 是 $M_2$ 的特殊情况（如 ΛCDM 是 wCDM 在 $w = -1$ 时的特殊情况）。

**似然比检验**（Likelihood Ratio Test, LRT）：

$$\Lambda = -2 \ln \frac{\mathcal{L}_\text{max}(M_1)}{\mathcal{L}_\text{max}(M_2)} = \chi^2_{\text{min},1} - \chi^2_{\text{min},2} = \Delta \chi^2$$

- $\Lambda$：似然比统计量
- $\mathcal{L}_\text{max}(M_1)$、$\mathcal{L}_\text{max}(M_2)$：两个模型各自的最大似然值
- $\Delta \chi^2$：两模型最佳拟合 $\chi^2$ 的差

**Wilks 定理**（Level 1）：在一定正则条件下，$\Lambda$ 服从 $\chi^2$ 分布，自由度为两个模型的参数个数之差 $\Delta k = k_2 - k_1$。

例如 ΛCDM vs wCDM：$\Delta k = 7 - 6 = 1$。如果 $\Delta \chi^2 = 3$，查 $\chi^2$ 分布表（$\Delta k = 1$），$p = 0.083$ → 不到 $2\sigma$，不显著。

**非嵌套模型**：似然比检验不适用。频率学派没有一个统一的方法来比较非嵌套模型——这是贝叶斯方法的一个重要优势。

### 5.2 对比总结

| 方法 | 适用范围 | 惩罚复杂度？ | 需要先验？ |
|------|---------|------------|----------|
| 似然比检验 | 嵌套模型 | 通过 $\Delta k$ | 不需要 |
| AIC | 任意模型 | $2k$ | 不需要 |
| BIC | 任意模型 | $k \ln N$ | 不需要 |
| 贝叶斯证据 | 任意模型 | 自动（Occam） | 需要 |

贝叶斯证据的优势：**统一处理嵌套和非嵌套模型**，自动实现 Occam's Razor。代价是需要指定先验，且计算困难。

---

## 6. 宇宙学中的应用

**可靠程度：Level 1（基本框架）+ Level 3（具体例子的数值）**

### 6.1 ΛCDM vs wCDM

标准宇宙学的核心模型选择问题：暗能量的状态方程 $w$ 是否需要作为自由参数？

- **ΛCDM**：$w = -1$（宇宙学常数），6 个参数
- **wCDM**：$w$ 是自由参数，7 个参数

到目前为止（Planck + BAO + SNe），所有模型选择准则都支持 ΛCDM：

- $w$ 的最佳拟合值接近 $-1$（如 $w = -1.03 \pm 0.05$）
- $\Delta \chi^2$ 很小（通常 $< 2$）
- AIC、BIC 和贝叶斯证据都偏好 ΛCDM

这并不意味着 $w$ 一定等于 $-1$，而是说**现有数据精度不足以探测偏离**。

### 6.2 张力（Tension）的量化

近年来宇宙学面临的重要问题：不同实验测量同一参数给出不一致的结果。

**$H_0$ 张力**：
- Planck（CMB）：$H_0 = 67.4 \pm 0.5$ km/s/Mpc
- SH0ES（距离阶梯法）：$H_0 = 73.0 \pm 1.0$ km/s/Mpc

两者差异 $\Delta H_0 = 5.6$，合并误差 $\sigma_\text{comb} = \sqrt{0.5^2 + 1.0^2} \approx 1.12$

$$\text{张力} = \frac{\Delta H_0}{\sigma_\text{comb}} = \frac{5.6}{1.12} \approx 5.0\sigma$$

- $\Delta H_0$：两个测量值的差
- $\sigma_\text{comb}$：合并标准差（两个独立测量的误差相加）
- $5\sigma$：在高斯分布下，偶然出现的概率约为 $3 \times 10^{-7}$

这个 $5\sigma$ 的含义是**频率学派**的："如果两个实验测量的是同一个真值，由统计涨落产生 $\geq 5\sigma$ 差异的概率极低"。它**不等于**"模型是错的概率是 $3 \times 10^{-7}$"——那是对 $p$ 值的常见误读。

**贝叶斯方法处理张力**：可以计算"两组数据一致"vs"两组数据不一致"这两个假设的贝叶斯因子。这是更严格但计算更复杂的方法。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $\chi^2_\text{red} = \chi^2_\text{min} / \nu$ | 约化卡方：$\nu = N_\text{data} - k$（自由度），$\approx 1$ 表示好的拟合 |
| $\text{PTE} = P(\chi^2 > \chi^2_\text{min} \mid \nu)$ | Probability To Exceed：模型正确时得到更差拟合的概率，$\sim 0.5$ 正常，$< 0.05$ 拟合差 |
| $\mathcal{Z} = \int \mathcal{L}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta}$ | 贝叶斯证据：似然在先验上的平均值 |
| $B_{12} = \mathcal{Z}_1 / \mathcal{Z}_2$ | 贝叶斯因子：两个模型证据的比值 |
| $\text{AIC} = \chi^2_\text{min} + 2k$ | 赤池信息准则：$k$ = 参数数 |
| $\text{BIC} = \chi^2_\text{min} + k \ln N$ | 贝叶斯信息准则：$N$ = 数据点数，惩罚更重 |
| $\Lambda = \chi^2_{\text{min},1} - \chi^2_{\text{min},2}$ | 似然比统计量（嵌套模型），服从 $\chi^2_{\Delta k}$ 分布 |
| $\text{张力} = \Delta\theta / \sigma_\text{comb}$ | 两实验的参数差异 / 合并标准差 |

---

## 理解检测

**Q1**：参数估计和模型选择分别回答什么问题？为什么 $\chi^2_\text{min}$ 更小的模型不一定是"更好"的模型？

你的回答：


**Q2**：贝叶斯证据 $\mathcal{Z} = \int \mathcal{L}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta}$ 为什么能自动惩罚复杂模型？解释 Occam's Razor 在这个公式中如何体现。

你的回答：


**Q3（计算题）**：模型 A 有 $k_A = 5$ 个参数，$\chi^2_{\text{min},A} = 120$。模型 B 有 $k_B = 8$ 个参数，$\chi^2_{\text{min},B} = 112$。数据点数 $N = 200$。

(a) 计算两个模型的 AIC 和 BIC。
(b) 根据 AIC 和 BIC，哪个模型更好？两个准则是否一致？

> 提示：$\text{AIC} = \chi^2_\text{min} + 2k$，$\text{BIC} = \chi^2_\text{min} + k \ln N$，$\ln 200 \approx 5.30$

你的回答：


**Q4（计算题）**：实验 X 测得 $\Omega_m = 0.310 \pm 0.008$，实验 Y 测得 $\Omega_m = 0.340 \pm 0.015$。假设两个实验独立，计算两者之间的张力（以 $\sigma$ 为单位）。

> 提示：$\text{张力} = \Delta\theta / \sigma_\text{comb}$，其中 $\sigma_\text{comb} = \sqrt{\sigma_X^2 + \sigma_Y^2}$

你的回答：


**Q5**：Jeffreys 判据中，$|\ln B_{12}| = 3$ 表示什么级别的证据？如果两个模型先验等概率（$P(M_1) = P(M_2) = 0.5$），此时 $P(M_1 | \mathbf{d})$ 大约是多少？

> 提示：$P(M_1|\mathbf{d}) / P(M_2|\mathbf{d}) = B_{12} = e^{\ln B_{12}}$，再用 $P(M_1|\mathbf{d}) + P(M_2|\mathbf{d}) = 1$

你的回答：
