# 08 - 宇宙学似然 — 从理论到数据的桥梁

> **主维度**：D4 宇宙学应用
> **次维度**：D1 概率论基础、D2 参数估计
> **关键关系**：
> - 功率谱似然 (概念) --依赖--> 似然函数 (概念)：功率谱似然依赖似然函数的一般框架
> - Boltzmann 求解器 (实验) --用于--> 参数估计 (任务)：Boltzmann 求解器用于参数估计中的理论预测
> - 协方差矩阵 (概念) --属于--> 功率谱似然 (概念)：协方差矩阵属于功率谱似然的关键组成部分
>
> **学习路径**：全景概览 → 概率论复习 → 似然与 MLE → 贝叶斯推断 → MCMC → Fisher 矩阵 → 模型选择 → **本章（宇宙学似然）** → 结果解读
>
> **前置知识**：似然函数与高斯似然矩阵形式（03 章）、贝叶斯推断与边缘化（04 章）、MCMC 采样（05 章）、Fisher 矩阵与参数退化（06 章）、宇宙学基础（CMB 域：Friedmann 方程、功率谱概念）
>
> **参考**：
> - [Planck 2018 Likelihood Paper (Planck Collaboration V)](https://arxiv.org/abs/1907.12875)
> - [R. Trotta, 2008 - Bayes in the sky](https://arxiv.org/abs/0803.4089)
> - [A. Heavens, 2009 - Statistical techniques in cosmology](https://arxiv.org/abs/0906.0664)
> - [Hamimeche & Lewis, 2008 - Likelihood analysis of CMB temperature and polarization power spectra](https://arxiv.org/abs/0801.0554)

---

## 1. 宇宙学参数估计的完整流程

**可靠程度：Level 1**

前几章我们分别学了似然函数、贝叶斯推断、MCMC、Fisher 矩阵——这些都是通用的统计工具。本章把它们组装起来，展示在宇宙学参数估计中**具体怎么用**。

参数估计的完整流程可以用一句话概括：**选定宇宙学参数 → 计算理论预测 → 与观测数据比较 → 更新参数**，不断迭代直到收敛。

### 1.1 流程详解

```
Step 1: 选择参数值 θ = (H₀, Ωm, Ωb, ns, As, τ, ...)
          │
          ▼
Step 2: Boltzmann 求解器（CAMB / CLASS）
          │  输入 θ → 数值求解线性微扰方程
          │  输出 → 理论功率谱 C_ℓ^TT(θ), C_ℓ^TE(θ), C_ℓ^EE(θ)
          ▼
Step 3: 构建似然
          │  L(θ) = P(d | θ)
          │  比较理论 C_ℓ(θ) 和观测 Ĉ_ℓ
          ▼
Step 4: MCMC 采样
          │  用 Metropolis-Hastings 决定是否接受新 θ
          │  重复 Step 1-3 数十万次
          ▼
Step 5: 后验分布 P(θ | d) → 参数约束
```

### 1.2 Boltzmann 求解器

**Boltzmann 求解器**（Boltzmann solver）是宇宙学参数估计流程中最核心的计算工具。它做的事情是：

**输入**：一组宇宙学参数 $\boldsymbol{\theta} = (H_0, \Omega_m, \Omega_b, n_s, A_s, \tau, ...)$，其中 $A_s$ 是原初功率谱的振幅（描述初始密度涨落的大小）

**输出**：理论功率谱 $C_\ell^{TT}(\boldsymbol{\theta})$, $C_\ell^{TE}(\boldsymbol{\theta})$, $C_\ell^{EE}(\boldsymbol{\theta})$ 以及物质功率谱 $P(k)$ 等

**内部过程**：从宇宙早期（$z \sim 10^9$）到今天，数值求解描述光子、重子、暗物质、中微子等各组分的**线性玻尔兹曼方程组**（耦合的微分方程）。这些方程追踪每种粒子的密度涨落和速度扰动如何在膨胀宇宙中演化。

两个标准工具：
- **CAMB**（Code for Anisotropies in the Microwave Background）：Fortran 编写，CosmoMC 默认使用
- **CLASS**（Cosmic Linear Anisotropy Solving System）：C 编写，MontePython 默认使用

你用 CosmoMC 做 kSZ 参数估计时，每次 MCMC 提议一个新参数点，背后都是 CAMB 在算对应的理论功率谱。一次 CAMB 调用大约需要几秒，整个 MCMC 运行 $\sim 10^5$–$10^6$ 步，这就是为什么跑一次完整的 MCMC 需要几小时到几天。

---

## 2. CMB 功率谱似然

**可靠程度：Level 1（基本框架）；Level 2（具体近似方案）**

### 2.1 从天图到功率谱

CMB 温度涨落 $\Delta T(\hat{n}) / T$ 可以展开为球谐函数：

$$\frac{\Delta T(\hat{n})}{T} = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat{n})$$

- $\hat{n}$：天球上的方向单位矢量
- $Y_{\ell m}(\hat{n})$：球谐函数，球面上的正交完备基函数（类比傅里叶分析中的 $e^{ikx}$，但定义在球面上）
- $a_{\ell m}$：球谐系数，描述角尺度 $\sim \pi / \ell$ 上的温度涨落
- $\ell$：多极矩（multipole），$\ell = 2$ 对应最大角尺度（$\sim 90°$），$\ell = 1000$ 对应 $\sim 0.18°$
- $m$：方位量子数，取值 $-\ell \leq m \leq \ell$，共 $2\ell + 1$ 个

观测到的功率谱是对 $m$ 求平均的结果：

$$\hat{C}_\ell = \frac{1}{2\ell + 1} \sum_{m=-\ell}^{\ell} |a_{\ell m}|^2$$

$\hat{C}_\ell$ 就是数据。理论模型预测 $C_\ell(\boldsymbol{\theta})$。参数估计的问题变成：**给定观测到的 $\hat{C}_\ell$，构建似然 $P(\hat{C}_\ell | C_\ell(\boldsymbol{\theta}))$**。

### 2.2 精确分布：Wishart 分布

如果 $a_{\ell m}$ 服从均值为零的高斯分布（标准暴涨模型的预测），那么 $\hat{C}_\ell$ 作为 $|a_{\ell m}|^2$ 的求和，服从什么分布？

对于单个分量（如纯 TT），$(2\ell + 1) \hat{C}_\ell / C_\ell$ 服从自由度为 $\nu = 2\ell + 1$ 的 **$\chi^2$ 分布**。

对于多分量（TT, TE, EE 联合），$\hat{C}_\ell$ 的采样分布是 **Wishart 分布**——这是 $\chi^2$ 分布对矩阵的推广。

> **Wishart 分布**：如果 $n$ 个独立的多元正态随机向量 $\mathbf{x}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$，则样本协方差矩阵 $\mathbf{S} = \sum_i \mathbf{x}_i \mathbf{x}_i^T$ 服从 Wishart 分布 $W(\boldsymbol{\Sigma}, n)$。在 CMB 中，"$n$" 就是 $2\ell + 1$（独立模式数），"$\boldsymbol{\Sigma}$" 就是理论功率谱矩阵。

关键点：**Wishart 分布不是高斯分布**。$\hat{C}_\ell$ 的分布是**有偏的**（右偏），特别是在小 $\ell$ 时。

### 2.3 大 ℓ 近似：高斯似然

当自由度 $\nu = 2\ell + 1$ 足够大时，由**中心极限定理**，$\hat{C}_\ell$ 的分布趋近高斯。

此时可以写成熟悉的高斯似然形式：

$$-2 \ln \mathcal{L} = \sum_\ell (2\ell + 1) \left[ \frac{\hat{C}_\ell}{C_\ell(\boldsymbol{\theta})} - \ln \frac{\hat{C}_\ell}{C_\ell(\boldsymbol{\theta})} - 1 \right]$$

或者更简化的版本（忽略 $C_\ell$ 对 $\boldsymbol{\theta}$ 的依赖在方差中的贡献）：

$$-2 \ln \mathcal{L} \approx \sum_\ell \frac{(2\ell+1)}{2} \left( \frac{\hat{C}_\ell - C_\ell(\boldsymbol{\theta})}{\sigma_\ell} \right)^2$$

其中 $\sigma_\ell^2 = \frac{2}{2\ell+1} C_\ell^2$ 是宇宙方差给出的 $\hat{C}_\ell$ 的方差。

### 2.4 小 ℓ（大角尺度）：不能用高斯近似

在小 $\ell$ 处（$\ell \lesssim 30$），$2\ell + 1$ 很小，$\hat{C}_\ell$ 的分布**明显偏离高斯**。此时必须使用：

- **完整 Wishart 似然**：精确但计算复杂
- **Offset log-normal 近似**：对 $\ln \hat{C}_\ell$ 做近似，比高斯更准确
- **Pixel-based 似然**：直接在天图像素层面构建似然（Planck 低 $\ell$ 似然采用此方法）

Planck 的处理方式正是分两段：
- **低 $\ell$**（$\ell \leq 29$）：基于像素的似然（Commander/SimAll）
- **高 $\ell$**（$30 \leq \ell \leq 2508$）：基于功率谱的高斯/pseudo-$C_\ell$ 似然（Plik）

### 2.5 宇宙方差

$$\text{Var}(\hat{C}_\ell) = \frac{2}{2\ell + 1} C_\ell^2$$

各符号含义：
- $\text{Var}(\hat{C}_\ell)$：观测功率谱 $\hat{C}_\ell$ 的方差（不确定性的平方）
- $2\ell + 1$：多极矩 $\ell$ 对应的独立模式数
- $C_\ell$：理论（真实）功率谱
- 分子中的 $2$ 来自 $\chi^2$ 分布的方差

**物理直觉**：宇宙方差的本质是**样本量有限**。对于每个 $\ell$，我们只有 $2\ell + 1$ 个独立的 $a_{\ell m}$ 来估计 $C_\ell$。$\ell$ 越小，独立样本越少，相对误差越大。

**具体数字对比**：

| $\ell$ | 独立模式数 $2\ell+1$ | 相对误差 $\Delta C_\ell / C_\ell = \sqrt{2/(2\ell+1)}$ |
|--------|---------------------|-------------------------------------------------------|
| $\ell = 2$ | 5 | $\sqrt{2/5} \approx 63\%$ |
| $\ell = 10$ | 21 | $\sqrt{2/21} \approx 31\%$ |
| $\ell = 100$ | 201 | $\sqrt{2/201} \approx 10\%$ |
| $\ell = 1000$ | 2001 | $\sqrt{2/2001} \approx 3.2\%$ |

$\ell = 2$ 时相对误差高达 63%——这就是为什么 CMB 四极矩的测量值不太可靠，也是为什么小 $\ell$ 需要特殊的似然处理方式。

> 注意：实际实验还有仪器噪声 $N_\ell$ 和不完全天空覆盖 $f_\text{sky}$ 的影响，完整公式为 $\text{Var}(\hat{C}_\ell) = \frac{2}{(2\ell+1) f_\text{sky}} (C_\ell + N_\ell)^2$，但宇宙方差项 $\frac{2}{2\ell+1} C_\ell^2$ 设定了不可消除的最小误差。

---

## 3. 协方差矩阵

**可靠程度：Level 1（概念）；Level 3（工程实践）**

### 3.1 协方差矩阵在似然中的角色

在高斯似然中，协方差矩阵 $\mathbf{C}$ 描述数据各分量之间的相关性：

$$\ln \mathcal{L} = -\frac{1}{2} (\mathbf{d} - \mathbf{m})^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}) + \text{const}$$

对于 CMB 功率谱，$\mathbf{d}$ 是各 $\ell$ 对应的观测 $\hat{C}_\ell$ 向量，$\mathbf{C}$ 的元素 $C_{ij} = \text{Cov}(\hat{C}_{\ell_i}, \hat{C}_{\ell_j})$。

如果不同 $\ell$ 的 $\hat{C}_\ell$ 完全独立（理想全天覆盖），$\mathbf{C}$ 是对角矩阵。但实际上由于天空掩膜（mask）、前景残留等效应，不同 $\ell$ 之间会产生相关，$\mathbf{C}$ 会有非对角元素。

### 3.2 解析估计 vs 模拟估计

**解析估计**：从理论推导协方差矩阵。对于高斯场、全天覆盖的理想情况，$\mathbf{C}$ 的解析表达式已知。但实际观测中的复杂效应（掩膜、前景、系统误差）很难解析处理。

**模拟估计**：运行大量蒙特卡洛模拟（mock），每次模拟生成一组虚拟数据 $\hat{C}_\ell^{(k)}$，然后用样本协方差公式估计：

$$\hat{C}_{ij} = \frac{1}{N_\text{sim} - 1} \sum_{k=1}^{N_\text{sim}} (\hat{C}_{\ell_i}^{(k)} - \bar{C}_{\ell_i})(\hat{C}_{\ell_j}^{(k)} - \bar{C}_{\ell_j})$$

- $N_\text{sim}$：模拟次数
- $\hat{C}_{\ell_i}^{(k)}$：第 $k$ 次模拟的第 $i$ 个 $\ell$ bin 的功率谱
- $\bar{C}_{\ell_i}$：对所有模拟求平均

### 3.3 Hartlap 因子

**问题**：用有限次模拟估计的样本协方差矩阵的逆 $\hat{\mathbf{C}}^{-1}$ 是**有偏的**——它系统地高估了真实逆协方差矩阵。

**Hartlap 因子**纠正这个偏差：

$$\hat{\mathbf{C}}^{-1}_\text{corrected} = \frac{N_\text{sim} - N_\text{bin} - 2}{N_\text{sim} - 1} \, \hat{\mathbf{C}}^{-1}$$

各符号含义：
- $N_\text{sim}$：模拟次数
- $N_\text{bin}$：数据向量的维度（$\ell$ bin 的数目）
- 校正因子 $\alpha = (N_\text{sim} - N_\text{bin} - 2) / (N_\text{sim} - 1)$，总是小于 1

**具体数字**：假设 $N_\text{bin} = 30$（30 个 $\ell$ bin），$N_\text{sim} = 100$：

$$\alpha = \frac{100 - 30 - 2}{100 - 1} = \frac{68}{99} \approx 0.687$$

不做校正的话，$\chi^2$ 会被高估约 $1/0.687 - 1 \approx 46\%$！

**经验法则**：需要 $N_\text{sim} \gg N_\text{bin}$，通常至少 $N_\text{sim} > 2 N_\text{bin}$，理想情况下 $N_\text{sim} > 10 N_\text{bin}$。

> 参考：[Hartlap, Simon & Schneider, 2007](https://arxiv.org/abs/0608064)

---

## 4. Nuisance 参数的处理

**可靠程度：Level 1（概念）；Level 3（具体参数列表）**

### 4.1 什么是 nuisance 参数

实际的宇宙学数据分析中，似然函数不只依赖宇宙学参数 $\boldsymbol{\theta}_\text{cosmo}$（如 $H_0$, $\Omega_m$），还依赖一系列**nuisance 参数** $\boldsymbol{\theta}_\text{nuis}$——它们描述仪器效应和天体物理前景，是参数估计中不可避免但我们不感兴趣的参数。

完整的参数向量：$\boldsymbol{\theta} = (\boldsymbol{\theta}_\text{cosmo}, \boldsymbol{\theta}_\text{nuis})$

### 4.2 常见的 nuisance 参数

**前景参数**（astrophysical foregrounds）：
- SZ（Sunyaev-Zel'dovich）效应的振幅：热 SZ 和动力学 SZ 信号
- 点源的贡献（射电源、红外源）
- 银河系尘埃发射
- CIB（宇宙红外背景）的贡献

**仪器参数**：
- 整体校准因子 $y_\text{cal}$：望远镜的绝对校准不确定性
- 束窗函数（beam）参数：望远镜角分辨率的不确定性

以 Planck 2018 为例，Plik 高 $\ell$ 似然涉及约 **20 个 nuisance 参数**，加上 6 个 ΛCDM 参数和可能的扩展参数，总共约 **25-30 个参数**。

### 4.3 消除 nuisance 参数

两种主要方法：

**边缘化**（Marginalization）：在贝叶斯框架中，对 nuisance 参数做积分，得到只关于宇宙学参数的后验分布：

$$P(\boldsymbol{\theta}_\text{cosmo} | \mathbf{d}) = \int P(\boldsymbol{\theta}_\text{cosmo}, \boldsymbol{\theta}_\text{nuis} | \mathbf{d}) \, d\boldsymbol{\theta}_\text{nuis}$$

在 MCMC 实践中，边缘化自动完成——只需在分析后验样本时忽略 nuisance 参数的维度，对它们做投影即可。

**Profiling**：频率学派的方法。对于每个固定的 $\boldsymbol{\theta}_\text{cosmo}$，找到使似然最大的 $\boldsymbol{\theta}_\text{nuis}$：

$$\mathcal{L}_\text{profile}(\boldsymbol{\theta}_\text{cosmo}) = \max_{\boldsymbol{\theta}_\text{nuis}} \mathcal{L}(\boldsymbol{\theta}_\text{cosmo}, \boldsymbol{\theta}_\text{nuis})$$

> 在高斯近似下，边缘化和 profiling 给出一样的结果。但如果后验分布明显非高斯，两者可能不同。

---

## 5. 多数据集联合约束

**可靠程度：Level 1**

### 5.1 联合似然

宇宙学参数约束的一大优势是可以**联合多个独立的数据集**。如果不同数据集之间是统计独立的，联合似然就是各似然的乘积：

$$\mathcal{L}_\text{total}(\boldsymbol{\theta}) = \mathcal{L}_\text{CMB}(\boldsymbol{\theta}) \times \mathcal{L}_\text{BAO}(\boldsymbol{\theta}) \times \mathcal{L}_\text{SN}(\boldsymbol{\theta}) \times \cdots$$

取对数后变成求和：

$$\ln \mathcal{L}_\text{total} = \ln \mathcal{L}_\text{CMB} + \ln \mathcal{L}_\text{BAO} + \ln \mathcal{L}_\text{SN} + \cdots$$

各数据集：
- $\mathcal{L}_\text{CMB}$：CMB 功率谱似然（如 Planck）
- $\mathcal{L}_\text{BAO}$：重子声学振荡似然（如 BOSS/DESI），约束尺度-红移关系
- $\mathcal{L}_\text{SN}$：Ia 型超新星似然（如 Pantheon+），约束距离-红移关系

### 5.2 独立性假设

将似然直接相乘的前提是数据集**统计独立**。这通常成立（不同实验观测不同天区或不同物理量），但也有例外——比如 CMB lensing 和大尺度结构巡天观测的是部分重叠的物质分布，它们之间存在相关性。如果忽略数据集之间的相关性，会导致**低估参数误差**。

### 5.3 互补性

不同数据集之所以值得联合，是因为它们约束了参数空间中**不同的退化方向**。

典型例子：

| 数据集 | 主要约束 | 退化方向 |
|--------|---------|---------|
| CMB（Planck） | $\Omega_b h^2$, $\Omega_c h^2$, $n_s$, $\tau$ | $\Omega_m$ 和 $H_0$ 通过 $\theta_*$ 退化 |
| BAO | $D_A(z) / r_d$, $H(z) r_d$ | 几乎正交于 CMB 的退化方向 |
| SN Ia | $D_L(z)$（光度距离） | 约束 $\Omega_m$，对 $H_0$ 不敏感（除非校准绝对亮度） |

其中：
- $\theta_*$：CMB 第一个声学峰的角尺度，被 Planck 精确测量
- $r_d$：声波视界（标准尺），BAO 用它作为标准尺
- $D_A(z)$：角直径距离
- $D_L(z)$：光度距离

CMB 对 $\Omega_m h^2$ 和 $\theta_*$ 约束非常紧，但 $\Omega_m$ 和 $H_0$ 之间存在退化（因为改变 $\Omega_m$ 和 $H_0$ 可以保持 $\theta_*$ 不变）。BAO 通过独立测量 $D_A(z)$ 打破这个退化。联合 CMB + BAO 后，$\Omega_m$ 和 $H_0$ 的等高线从一个长椭圆变成一个小圆，约束大幅提升。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $\text{Var}(\hat{C}_\ell) = \frac{2}{2\ell+1} C_\ell^2$ | 宇宙方差：$\hat{C}_\ell$ 的最小不确定性 |
| $\Delta C_\ell / C_\ell = \sqrt{2/(2\ell+1)}$ | 宇宙方差对应的相对误差 |
| $\hat{C}^{-1}_\text{corr} = \frac{N_\text{sim} - N_\text{bin} - 2}{N_\text{sim} - 1} \hat{C}^{-1}$ | Hartlap 因子：校正模拟估计的逆协方差偏差 |
| $\ln \mathcal{L}_\text{total} = \sum_i \ln \mathcal{L}_i$ | 多数据集联合似然（独立数据集取对数后相加） |
| $P(\boldsymbol{\theta}_\text{cosmo} \| \mathbf{d}) = \int P(\boldsymbol{\theta}_\text{cosmo}, \boldsymbol{\theta}_\text{nuis} \| \mathbf{d}) \, d\boldsymbol{\theta}_\text{nuis}$ | 边缘化 nuisance 参数 |

---

## 理解检测

**Q1**：解释为什么 $\ell = 2$ 的 CMB 功率谱测量比 $\ell = 1000$ 不可靠得多。用到"宇宙方差"和"独立模式数"两个概念，给出具体的相对误差数值。

你的回答：


**Q2**（计算题）：一个实验需要估计 $N_\text{bin} = 50$ 个 $\ell$ bin 的协方差矩阵。他们跑了 $N_\text{sim} = 200$ 次模拟。

(a) 计算 Hartlap 校正因子 $\alpha$。

(b) 如果不做校正，$\chi^2$ 会被高估大约百分之多少？

(c) 如果想让校正因子 $\alpha > 0.95$（即偏差小于 5%），至少需要多少次模拟？

> 提示：用 Hartlap 因子公式 $\alpha = (N_\text{sim} - N_\text{bin} - 2) / (N_\text{sim} - 1)$

你的回答：


**Q3**：为什么联合 CMB + BAO 的参数约束比单独用 CMB 好得多？具体来说，CMB 中 $\Omega_m$ 和 $H_0$ 存在什么退化，BAO 是如何打破这个退化的？

你的回答：


**Q4**：Planck 2018 的似然分成低 $\ell$（$\ell \leq 29$）和高 $\ell$（$30 \leq \ell \leq 2508$）两部分，分别用不同的方法处理。为什么要这样分？低 $\ell$ 不能用高斯近似的物理原因是什么？

你的回答：
