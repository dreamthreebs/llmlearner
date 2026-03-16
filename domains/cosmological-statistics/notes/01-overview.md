# 01 - 宇宙学统计方法：全景概览

> **主维度**：全部（D1-D4）
> **关键关系**：
> - 贝叶斯推断 (理论) --依赖--> 似然函数 (概念)：贝叶斯推断依赖似然函数构建后验
> - MCMC (方法) --用于--> 参数估计 (任务)：MCMC 用于参数估计的后验采样
> - Fisher 矩阵 (方法) --用于--> 参数估计 (任务)：Fisher 矩阵用于参数估计的预测
>
> **学习路径**：**本章（全景概览）** → 概率论复习 → 似然与 MLE → 贝叶斯推断 → MCMC → Fisher 矩阵 → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：本科概率论基本概念（会在下一章复习），宇宙学基础（参见 `domains/CMB/`）
>
> **参考**：
> - [R. Trotta, 2008 - Bayes in the sky](https://arxiv.org/abs/0803.4089)
> - [D.S. Sivia - Data Analysis: A Bayesian Tutorial](https://global.oup.com/academic/product/data-analysis-9780198568322)
> - [A. Heavens, 2009 - Statistical techniques in cosmology](https://arxiv.org/abs/0906.0664)

---

## 1. 为什么宇宙学需要统计？

**可靠程度：Level 1**

### 1.1 核心困难

宇宙学面临三个独特的统计挑战：

**只有一个宇宙**：我们不能重复实验。不像粒子物理可以对撞无数次取平均，宇宙学的很多观测量（如 CMB 全天温度图）只有一份实现。这带来了**宇宙方差**（cosmic variance）——在大角尺度上，我们能测量的独立模式数有限，统计误差有不可消除的下限。

**间接观测**：我们想知道的是宇宙学参数（$H_0$, $\Omega_m$, $\Omega_b$, $n_s$, $\sigma_8$, $\tau$），但直接测量的是完全不同的东西（CMB 温度涨落、星系红移、亮度……）。需要通过理论模型把观测量和参数联系起来。

**高维参数空间**：标准 ΛCDM 模型有 6 个自由参数，加上实验相关的 nuisance 参数（校准、前景……），实际拟合的参数空间通常是 10-30 维。在这样的高维空间中探索参数约束，需要专门的数值方法。

### 1.2 参数估计的核心流程

宇宙学参数估计（Parameter Estimation, PTE）的基本流程：

```
物理模型（如 ΛCDM）
    │  输入参数 θ = (H₀, Ωm, Ωb, ns, σ₈, τ)
    ▼
理论预测
    │  计算理论功率谱 C_ℓ(θ)
    ▼
与观测数据比较
    │  数据 d = 测量到的功率谱
    │  似然函数 L(θ) = P(d | θ)
    ▼
统计推断
    │  贝叶斯推断：后验 P(θ | d) ∝ L(θ) × π(θ)
    │  用 MCMC 采样后验分布
    ▼
参数约束
    │  最佳拟合值 + 误差棒 / 等高线图
```

每一步都涉及统计方法。本域的目标就是逐步深入理解这个流程中的每一个环节。

---

## 2. 两种统计哲学

**可靠程度：Level 1**

### 2.1 频率学派 vs 贝叶斯学派

统计学有两种基本哲学，理解它们的区别对正确解读宇宙学结果至关重要。

**频率学派**（Frequentist）：概率是**事件在大量重复实验中的出现频率**。

- 参数 $\theta$ 是固定的未知常数，不是随机变量
- $P(\theta)$ 这种写法没有意义（$\theta$ 没有概率分布）
- 给出的是**置信区间**（confidence interval）："如果重复实验无穷次，95% 的置信区间会包含真值"
- 代表方法：最大似然估计、$\chi^2$ 检验、p 值

**贝叶斯学派**（Bayesian）：概率是**对命题的信念程度**。

- 参数 $\theta$ 可以有概率分布——它反映我们对 $\theta$ 的不确定性
- 先验分布 $\pi(\theta)$ 编码了观测之前的已有知识
- 给出的是**可信区间**（credible interval）："$\theta$ 有 95% 的概率在这个区间内"
- 代表方法：贝叶斯推断、MCMC

### 2.2 宇宙学中的选择

现代宇宙学**主要使用贝叶斯方法**。原因：

1. **自然地处理 nuisance 参数**：贝叶斯框架可以通过对 nuisance 参数积分（边缘化）来消除它们的影响
2. **直接给出参数的概率分布**：后验分布 $P(\theta | d)$ 直接告诉你"$\theta$ 最可能是多少"
3. **可以融合先验信息**：比如来自其他实验的约束
4. **处理高维参数空间**：MCMC 在高维空间中比网格搜索高效得多

但频率学派方法也有用武之地——比如假设检验（"数据是否和模型一致？"）和 Fisher 矩阵预测（实验设计阶段评估预期灵敏度）。

> 参考：[Trotta, 2008 - Bayes in the sky, Section 2](https://arxiv.org/abs/0803.4089)

---

## 3. 核心概念预览

### 3.1 似然函数

**似然函数**（likelihood）$\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} | \boldsymbol{\theta})$：给定参数 $\boldsymbol{\theta}$，观测到数据 $\mathbf{d}$ 的概率。

注意区分：
- $P(\mathbf{d} | \boldsymbol{\theta})$——数据的概率（参数固定，数据是变量）→ 这是似然
- $P(\boldsymbol{\theta} | \mathbf{d})$——参数的概率（数据固定，参数是变量）→ 这是后验

似然函数是连接理论和数据的桥梁。在宇宙学中，最常见的形式是高斯似然：

$$\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta}))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta})) + \text{const}$$

其中 $\mathbf{d}$ 是数据向量，$\mathbf{m}(\boldsymbol{\theta})$ 是理论模型预测，$\mathbf{C}$ 是数据的协方差矩阵。

### 3.2 贝叶斯定理

$$P(\boldsymbol{\theta} | \mathbf{d}) = \frac{P(\mathbf{d} | \boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})}{P(\mathbf{d})}$$

| 符号 | 名称 | 含义 |
|------|------|------|
| $P(\boldsymbol{\theta} \| \mathbf{d})$ | 后验（posterior） | 看到数据后，对参数的更新信念 |
| $P(\mathbf{d} \| \boldsymbol{\theta})$ | 似然（likelihood） | 给定参数，数据出现的概率 |
| $\pi(\boldsymbol{\theta})$ | 先验（prior） | 看到数据之前对参数的信念 |
| $P(\mathbf{d})$ | 证据（evidence） | 数据在所有参数下的总概率（归一化常数） |

宇宙学参数估计的本质：**用贝叶斯定理，从似然和先验出发，计算后验分布**。

### 3.3 MCMC

后验分布 $P(\boldsymbol{\theta} | \mathbf{d})$ 在高维空间中通常无法解析计算。**马尔可夫链蒙特卡洛**（Markov Chain Monte Carlo, MCMC）是一种通过随机采样来近似后验分布的方法。

MCMC 的核心思想：构建一条在参数空间中游走的**马尔可夫链**（每一步只依赖当前位置），使得链的**稳态分布**恰好是我们想要的后验分布。走了足够多步之后，采样点的密度就近似于后验分布。

你用过的 CosmoMC 就是在做这件事——在 6+ 维的宇宙学参数空间中运行马尔可夫链，采样后验分布，最终画出参数等高线图。

### 3.4 Fisher 矩阵

**Fisher 信息矩阵** $F_{ij}$ 是似然函数在最大值处的曲率：

$$F_{ij} = -\left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j} \right\rangle$$

Fisher 矩阵告诉你：**在最佳拟合点附近，似然面在各个方向上有多"陡"**。越陡的方向，参数约束越紧。$F^{-1}$ 的对角元素给出参数误差的下限（Cramér-Rao 界）。

Fisher 矩阵是"MCMC 的快速近似版"——它假设后验是高斯分布，只需要计算似然的二阶导数，不需要跑完整的 MCMC。常用于实验设计阶段快速评估预期灵敏度。

---

## 4. 知识维度与知识地图

### 4.1 知识维度

| 维度 | 含义 | 核心概念 |
|------|------|---------|
| **D1 概率论基础** | 数学基础 | 概率分布、条件概率、贝叶斯定理、似然函数 |
| **D2 参数估计** | PTE 核心 | MLE、贝叶斯推断、先验选择、MCMC、Fisher 矩阵、收敛诊断 |
| **D3 模型选择** | 模型比较 | $\chi^2$、贝叶斯证据、Bayes factor、AIC/BIC |
| **D4 宇宙学应用** | 落地实践 | 功率谱似然、协方差矩阵、参数退化、结果解读、系统误差 |

### 4.2 知识地图

```
概率论基础（复习）
    │
    ├──→ 似然函数 + MLE
    │        │
    │        ▼
    │    贝叶斯推断（先验 → 后验）
    │        │
    │        ├──→ MCMC（后验采样）
    │        │      └──→ 收敛诊断
    │        │
    │        └──→ Fisher 矩阵（高斯近似 → 快速预测）
    │
    └──→ 模型选择（证据 → AIC/BIC → 模型比较）
              │
              ▼
         宇宙学应用
              ├── 功率谱似然（高斯 vs Wishart）
              ├── 协方差矩阵估计
              ├── 参数退化与先验影响
              └── 实际论文结果解读
```

---

## 5. 你会在论文中看到什么

学完本域后，你应该能看懂宇宙学论文中这些常见内容：

| 论文中看到的 | 对应章节 |
|-------------|---------|
| "$H_0 = 67.4 \pm 0.5$ km/s/Mpc（68% CL）" | 04（贝叶斯推断）、09（结果解读） |
| MCMC 等高线图（1σ/2σ 椭圆） | 05（MCMC）、09（结果解读） |
| "$\chi^2 / \text{dof} = 1.02$" | 07（模型选择） |
| "$\ln B_{12} = 4.3$（强烈支持模型 1）" | 07（模型选择） |
| "我们对 nuisance 参数做了边缘化" | 04（贝叶斯推断） |
| "Fisher 预测显示 CMB-S4 可以达到 $\sigma(\tau) = 0.002$" | 06（Fisher 矩阵） |
| "$H_0$ tension between Planck and SH0ES at $5\sigma$" | 09（结果解读） |

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} \| \boldsymbol{\theta})$ | 似然函数：给定参数，数据出现的概率 |
| $P(\boldsymbol{\theta} \| \mathbf{d}) \propto \mathcal{L}(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})$ | 贝叶斯定理（省略证据项）：后验 ∝ 似然 × 先验 |
| $\ln \mathcal{L} = -\frac{1}{2}(\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m}) + \text{const}$ | 高斯似然：宇宙学中最常用的似然形式 |
| $F_{ij} = -\langle \partial^2 \ln \mathcal{L} / \partial \theta_i \partial \theta_j \rangle$ | Fisher 矩阵：似然的曲率，$F^{-1}$ 给出参数误差下限 |

---

## 理解检测

**Q1**：频率学派和贝叶斯学派对"概率"的定义有什么根本区别？宇宙学主要使用哪一种？给出至少两个原因。

你的回答：


**Q2**：似然函数 $\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} | \boldsymbol{\theta})$ 和后验分布 $P(\boldsymbol{\theta} | \mathbf{d})$ 的区别是什么？"条件"的方向为什么重要？

你的回答：


**Q3**：在高斯似然 $\ln \mathcal{L} = -\frac{1}{2}(\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m})$ 中，如果数据 $\mathbf{d}$ 和模型预测 $\mathbf{m}$ 完全一致，$\ln \mathcal{L}$ 等于多少（忽略常数项）？如果 $\mathbf{d} - \mathbf{m}$ 很大，$\ln \mathcal{L}$ 会怎么变化？

> 提示：想想 $(\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m})$ 在两种情况下的值

你的回答：


**Q4**：Fisher 矩阵是似然函数在最大值处的曲率。如果某个参数方向上的 $F_{ii}$ 很大，说明似然面在这个方向上很陡还是很平？这对应参数约束紧还是松？$\sigma(\theta_i)$ 的下限是多少？

> 提示：$\sigma(\theta_i) \geq \sqrt{(F^{-1})_{ii}}$（Cramér-Rao 界）

你的回答：

