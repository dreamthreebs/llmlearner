# 05 - MCMC — 马尔可夫链蒙特卡洛

> **主维度**：D2 参数估计
> **次维度**：D4 宇宙学应用（CosmoMC/Cobaya 实现）
> **关键关系**：
> - MCMC (方法) --依赖--> 贝叶斯推断 (理论)：MCMC 依赖贝叶斯推断提供的后验分布
> - MCMC (方法) --用于--> 参数估计 (任务)：MCMC 用于参数估计的后验采样
> - MCMC (方法) --对比--> Fisher 矩阵 (方法)：两种参数约束方法的精确度-速度权衡
>
> **学习路径**：全景概览 → 概率论复习 → 似然与 MLE → 贝叶斯推断 → **本章（MCMC）** → Fisher 矩阵 → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：贝叶斯定理、后验分布、似然函数、先验、边缘化（04-bayesian-inference）；高斯似然矩阵形式（03-likelihood-mle）
>
> **参考**：
> - [R. Trotta, 2008 - Bayes in the sky, Section 5](https://arxiv.org/abs/0803.4089)
> - [D. Foreman-Mackey et al., 2013 - emcee: The MCMC Hammer](https://arxiv.org/abs/1202.3665)
> - [A. Lewis & S. Bridle, 2002 - CosmoMC](https://arxiv.org/abs/astro-ph/0205436)

---

## 1. 为什么需要 MCMC

**可靠程度：Level 1**

贝叶斯推断告诉我们：后验分布 $P(\boldsymbol{\theta} | \mathbf{d}) \propto \mathcal{L}(\boldsymbol{\theta}) \, \pi(\boldsymbol{\theta})$。但在实际宇宙学分析中，**直接计算后验分布极其困难**。

### 1.1 高维积分的困难

核心瓶颈是**归一化常数**和**边缘化积分**：

$$P(\theta_1 | \mathbf{d}) = \int P(\theta_1, \theta_2, \ldots, \theta_D | \mathbf{d}) \, d\theta_2 \cdots d\theta_D$$

- $D$ 是参数空间的维度
- 这个积分需要遍历所有 nuisance 参数（$\theta_2, \ldots, \theta_D$）的组合

**网格搜索的灾难**：如果在每个参数方向上取 $N$ 个网格点，$D$ 维空间中需要 $N^D$ 个点。

| 维度 $D$ | 网格点数 $N = 100$ | 每个点 1 秒时所需时间 |
|----------|-------------------|---------------------|
| 2 | $10^4$ | 2.8 小时 |
| 6（ΛCDM） | $10^{12}$ | 3.2 万年 |
| 20（含 nuisance） | $10^{40}$ | $> 10^{32}$ 年 |

6 维的标准 ΛCDM 已经不可行，20 维更是天文数字。这就是**维度灾难**（curse of dimensionality）——参数空间的体积随维度指数增长。

### 1.2 MCMC 的核心思想

MCMC 的解决方案：**不需要遍历整个参数空间，只需要在后验分布高的区域密集采样**。

具体来说：构建一条在参数空间中**随机游走**的链（一系列参数点 $\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \ldots$），使得链走了足够久之后，采样点的密度**正比于后验分布**。

$$\text{采样密度} \xrightarrow{\text{足够多步}} P(\boldsymbol{\theta} | \mathbf{d})$$

关键的好处：

- **不需要计算归一化常数** $P(\mathbf{d})$——因为 MCMC 只需要后验的比值 $P(\boldsymbol{\theta}') / P(\boldsymbol{\theta})$，归一化常数在比值中抵消
- **自动边缘化**——如果你有 $(\theta_1, \theta_2)$ 的联合采样，只看 $\theta_1$ 的直方图就是边缘化后的分布 $P(\theta_1 | \mathbf{d})$
- **效率远高于网格搜索**——链会自动集中在后验概率高的区域，不浪费时间在概率极低的地方

---

## 2. 马尔可夫链基础

**可靠程度：Level 1**

MCMC = **马尔可夫链**（Markov Chain） + **蒙特卡洛**（Monte Carlo 采样）。先理解马尔可夫链。

### 2.1 马尔可夫性质

一条参数空间中的链 $\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \boldsymbol{\theta}^{(3)}, \ldots$ 是**马尔可夫链**，当且仅当：

$$P(\boldsymbol{\theta}^{(t+1)} | \boldsymbol{\theta}^{(t)}, \boldsymbol{\theta}^{(t-1)}, \ldots, \boldsymbol{\theta}^{(1)}) = P(\boldsymbol{\theta}^{(t+1)} | \boldsymbol{\theta}^{(t)})$$

- $\boldsymbol{\theta}^{(t)}$：第 $t$ 步链所在的参数值
- 含义：**下一步去哪里，只取决于当前在哪里**，不取决于之前走过的路径

这就像一个喝醉的人在参数空间中走路——每一步的方向只取决于他现在站在哪，跟他之前走过的路线无关。

### 2.2 转移概率与稳态分布

**转移概率** $T(\boldsymbol{\theta}' | \boldsymbol{\theta})$：从当前位置 $\boldsymbol{\theta}$ 跳到新位置 $\boldsymbol{\theta}'$ 的概率。

如果链走了足够久（$t \to \infty$），采样点的分布会趋向一个**稳态分布**（stationary distribution）$\pi_s(\boldsymbol{\theta})$，满足：

$$\pi_s(\boldsymbol{\theta}') = \int T(\boldsymbol{\theta}' | \boldsymbol{\theta}) \, \pi_s(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$$

- $\pi_s(\boldsymbol{\theta})$：稳态分布，链达到平衡后在各位置的概率密度
- $T(\boldsymbol{\theta}' | \boldsymbol{\theta})$：转移概率，从 $\boldsymbol{\theta}$ 到 $\boldsymbol{\theta}'$ 的跳转概率

MCMC 的目标：**设计转移概率 $T$，使得稳态分布恰好是后验分布**：

$$\pi_s(\boldsymbol{\theta}) = P(\boldsymbol{\theta} | \mathbf{d})$$

### 2.3 细致平衡条件

**细致平衡**（detailed balance）是保证稳态分布的一个**充分条件**：

$$\pi_s(\boldsymbol{\theta}) \, T(\boldsymbol{\theta}' | \boldsymbol{\theta}) = \pi_s(\boldsymbol{\theta}') \, T(\boldsymbol{\theta} | \boldsymbol{\theta}')$$

- 左边：从 $\boldsymbol{\theta}$ 出发，跳到 $\boldsymbol{\theta}'$ 的"概率流"
- 右边：从 $\boldsymbol{\theta}'$ 出发，跳回 $\boldsymbol{\theta}$ 的"概率流"
- 含义：**在稳态下，任意两点之间的来回流量相等**

如果转移概率 $T$ 满足细致平衡，就能保证链的稳态分布就是我们想要的目标分布 $\pi_s$。Metropolis-Hastings 算法正是利用这一条件来设计 $T$。

---

## 3. Metropolis-Hastings 算法

**可靠程度：Level 1**

Metropolis-Hastings (MH) 是最基本的 MCMC 算法。它把转移概率 $T$ 拆成两步：**提议**（propose）+ **接受/拒绝**（accept/reject）。

### 3.1 算法步骤

```
输入：后验 P(θ|d) ∝ L(θ)π(θ)，提议分布 q(θ'|θ)，起始点 θ⁽⁰⁾
输出：采样链 {θ⁽¹⁾, θ⁽²⁾, ..., θ⁽ᴺ⁾}

for t = 0, 1, 2, ..., N-1:
    1. 提议：从 q(θ'|θ⁽ᵗ⁾) 中抽取候选点 θ'
    2. 计算接受概率：
       α = min(1, [P(θ'|d) × q(θ⁽ᵗ⁾|θ')] / [P(θ⁽ᵗ⁾|d) × q(θ'|θ⁽ᵗ⁾)])
    3. 生成均匀随机数 u ~ Uniform(0, 1)
    4. 如果 u < α：接受，θ⁽ᵗ⁺¹⁾ = θ'
       否则：拒绝，θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾（原地不动）
```

### 3.2 接受概率

接受概率的完整形式：

$$\alpha = \min\!\left(1,\; \frac{P(\boldsymbol{\theta}' | \mathbf{d}) \; q(\boldsymbol{\theta}^{(t)} | \boldsymbol{\theta}')}{P(\boldsymbol{\theta}^{(t)} | \mathbf{d}) \; q(\boldsymbol{\theta}' | \boldsymbol{\theta}^{(t)})}\right)$$

- $P(\boldsymbol{\theta}' | \mathbf{d})$：候选点的后验概率
- $P(\boldsymbol{\theta}^{(t)} | \mathbf{d})$：当前点的后验概率
- $q(\boldsymbol{\theta}' | \boldsymbol{\theta}^{(t)})$：从当前点提议到候选点的概率
- $q(\boldsymbol{\theta}^{(t)} | \boldsymbol{\theta}')$：从候选点提议回当前点的概率

因为后验 $P(\boldsymbol{\theta} | \mathbf{d}) = \mathcal{L}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) / P(\mathbf{d})$，归一化常数 $P(\mathbf{d})$ 在分子分母中抵消，所以实际计算时只需要：

$$\alpha = \min\!\left(1,\; \frac{\mathcal{L}(\boldsymbol{\theta}') \, \pi(\boldsymbol{\theta}') \; q(\boldsymbol{\theta}^{(t)} | \boldsymbol{\theta}')}{\mathcal{L}(\boldsymbol{\theta}^{(t)}) \, \pi(\boldsymbol{\theta}^{(t)}) \; q(\boldsymbol{\theta}' | \boldsymbol{\theta}^{(t)})}\right)$$

### 3.3 对称提议分布 → Metropolis 算法

如果提议分布是**对称的**，即 $q(\boldsymbol{\theta}' | \boldsymbol{\theta}) = q(\boldsymbol{\theta} | \boldsymbol{\theta}')$（比如以当前点为中心的高斯分布），那么 $q$ 在分子分母中抵消，接受概率简化为：

$$\alpha = \min\!\left(1,\; \frac{P(\boldsymbol{\theta}' | \mathbf{d})}{P(\boldsymbol{\theta}^{(t)} | \mathbf{d})}\right) = \min\!\left(1,\; \frac{\mathcal{L}(\boldsymbol{\theta}') \, \pi(\boldsymbol{\theta}')}{\mathcal{L}(\boldsymbol{\theta}^{(t)}) \, \pi(\boldsymbol{\theta}^{(t)})}\right)$$

这就是原始的 **Metropolis 算法**（1953 年提出，早于 Hastings 的推广）。直觉上：

- 如果候选点的后验 $>$ 当前点 → $\alpha = 1$，**一定接受**（上坡总是走）
- 如果候选点的后验 $<$ 当前点 → $\alpha < 1$，**以一定概率接受**（下坡有时走）

"下坡有时走"是 MCMC 与单纯优化的关键区别——它让链不会卡在局部最大值，而是在后验分布中充分探索。

### 3.4 具体数字例子：1 维高斯后验

假设后验是 $P(\theta | d) = \mathcal{N}(\theta; \mu=5, \sigma=2)$，使用对称高斯提议 $q(\theta' | \theta) = \mathcal{N}(\theta'; \theta, \sigma_q=1)$。

当前位置 $\theta^{(t)} = 4$，提议到 $\theta' = 6$：

$$\frac{P(\theta'|d)}{P(\theta^{(t)}|d)} = \frac{\exp\!\left(-\frac{(6-5)^2}{2 \times 4}\right)}{\exp\!\left(-\frac{(4-5)^2}{2 \times 4}\right)} = \frac{e^{-1/8}}{e^{-1/8}} = 1$$

因为 $\theta = 4$ 和 $\theta = 6$ 关于均值 $\mu = 5$ 对称，后验值相等，所以 $\alpha = \min(1, 1) = 1$，一定接受。

再看：当前 $\theta^{(t)} = 5$（均值处），提议到 $\theta' = 8$（离均值较远）：

$$\frac{P(\theta'|d)}{P(\theta^{(t)}|d)} = \frac{\exp\!\left(-\frac{(8-5)^2}{2 \times 4}\right)}{\exp\!\left(-\frac{(5-5)^2}{2 \times 4}\right)} = \frac{e^{-9/8}}{e^{0}} = e^{-9/8} \approx 0.325$$

$\alpha = \min(1, 0.325) = 0.325$，有 32.5% 的概率接受。这正好反映了后验在 $\theta = 8$ 处比 $\theta = 5$ 处低——链偶尔会去那里，但不会频繁去，采样密度与后验概率成正比。

---

## 4. 提议分布

**可靠程度：Level 1（原理）+ Level 3（最优接受率的具体数值）**

### 4.1 步长的权衡

提议分布的选择（特别是步长/宽度 $\sigma_q$）对 MCMC 效率至关重要。

**步长太小**（$\sigma_q \ll \sigma_\text{posterior}$）：
- 候选点总是离当前点很近，接受率很高（接近 100%）
- 但链移动缓慢，需要大量步数才能探索整个后验分布
- 像蚂蚁在走路——每步都走，但走得很慢

**步长太大**（$\sigma_q \gg \sigma_\text{posterior}$）：
- 候选点经常落在后验概率很低的区域，大部分被拒绝
- 链大部分时间原地不动，接受率很低（接近 0%）
- 像大炮打苍蝇——瞄准了但总是打不中

### 4.2 最优接受率

理论和经验表明存在最优的接受率：

| 维度 | 最优接受率 |
|------|----------|
| $D = 1$ | ~44% |
| $D \gg 1$（高维极限） | ~23% |
| 常用经验值 | 20%–50% |

- $D$：参数空间的维度

最优接受率 23%（高维）的直觉：大约 1/4 的提议被接受，链在"探索新区域"和"在高概率区域停留"之间取得平衡。

实际操作中，CosmoMC 和 Cobaya 会在 burn-in 阶段自动调整提议分布的步长，使接受率接近最优值。

---

## 5. Burn-in 与 Thinning

**可靠程度：Level 1**

### 5.1 Burn-in（预热期）

链的起始点 $\boldsymbol{\theta}^{(0)}$ 通常不在后验分布的高概率区域。链需要经过一段"预热"才能到达稳态。**Burn-in 期间的采样点不代表后验分布，必须丢弃。**

- **Burn-in**（预热期）：MCMC 链从任意起始点 $\boldsymbol{\theta}^{(0)}$ 到达稳态分布之前的过渡阶段
- 实际操作：丢弃前 $N_\text{burn}$ 个点（通常取总链长的 10%–50%）
- 判断标准：参数的链轨迹不再有明显的漂移趋势

### 5.2 Thinning（稀疏化）

由于马尔可夫性质，**相邻的采样点是相关的**（第 $t+1$ 步依赖第 $t$ 步）。直接使用所有点会低估统计误差。

- **Thinning**：每隔 $k$ 步保留一个点，丢弃中间的点
- $k$ 通常取**自相关时间** $\tau_\text{corr}$（见下节），使保留的点近似独立
- **有效样本数**（effective sample size）：$N_\text{eff} = N / \tau_\text{corr}$
  - $N$：链的总步数
  - $\tau_\text{corr}$：自相关时间，衡量相邻采样点的相关性（越大越相关）
  - $N_\text{eff}$：等效的独立样本数

> 注意：现代做法（如 GetDist）通常**不做 thinning**，而是直接用所有点并正确估计 $N_\text{eff}$。Thinning 只是为了减小存储和加速后处理。

---

## 6. 收敛诊断

**可靠程度：Level 1（方法原理）+ Level 3（阈值的具体数值）**

MCMC 的根本问题：**怎么知道链已经收敛到稳态分布？** 理论上没有完美的回答——任何有限长度的链都只能提供近似。但有几个实用的诊断工具。

### 6.1 Gelman-Rubin 统计量 $\hat{R}$

最常用的收敛诊断。核心思想：**运行多条独立的链，比较链间方差和链内方差**。

对每个参数 $\theta_i$，定义：

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

其中：

- $W$：**链内方差**（within-chain variance）——每条链内部的方差，取多条链的平均
- $B/n$：**链间方差**（between-chain variance）——各链均值之间的方差（$B$ 是未归一化的链间方差，$n$ 是每条链的长度）
- $\hat{V} = \frac{n-1}{n} W + \frac{1}{n} B$：后验方差的加权估计

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}} = \sqrt{\frac{(n-1)/n \cdot W + B/n}{W}}$$

**解读**：
- $\hat{R} \approx 1$：链间方差 $\approx$ 链内方差，说明各链在探索同一个分布 → 收敛
- $\hat{R} \gg 1$：链间方差 $\gg$ 链内方差，说明各链还在不同区域游荡 → 未收敛
- **常用阈值**：$\hat{R} < 1.1$（有些严格分析要求 $\hat{R} < 1.01$）

**具体数字例子**：假设 4 条链（$M=4$），每条 $n=1000$ 步，估计参数 $H_0$：
- 链 1 的均值 = 67.2，链 2 = 67.5，链 3 = 67.3，链 4 = 67.4
- 链间方差 $B/n \approx 0.015$，链内方差 $W \approx 0.5$
- $\hat{R} = \sqrt{(0.5 + 0.015) / 0.5} = \sqrt{1.03} \approx 1.015$ → 收敛良好

如果链 3 卡在 $H_0 = 70$ 附近不动，链间方差会大得多，$\hat{R}$ 会远大于 1.1。

### 6.2 自相关时间

**自相关函数**衡量链中相距 $\tau$ 步的样本之间的相关性：

$$\rho(\tau) = \frac{\text{Cov}(\theta^{(t)}, \theta^{(t+\tau)})}{\text{Var}(\theta)}$$

- $\rho(\tau)$：滞后 $\tau$ 步的自相关系数
- $\theta^{(t)}$、$\theta^{(t+\tau)}$：链中相隔 $\tau$ 步的两个采样点
- $\text{Cov}$：它们之间的协方差
- $\text{Var}(\theta)$：参数 $\theta$ 的方差

**自相关时间**（integrated autocorrelation time）：

$$\tau_\text{corr} = 1 + 2 \sum_{\tau=1}^{\infty} \rho(\tau)$$

- $\tau_\text{corr}$ 越大，相邻样本的相关性越强，链移动越慢
- 有效样本数 $N_\text{eff} = N / \tau_\text{corr}$
- 一般要求 $N_\text{eff} \gtrsim 1000$ 才能可靠估计后验分布

### 6.3 视觉检查

最直观的方法：**画链的轨迹图**（trace plot），横轴是步数 $t$，纵轴是参数值 $\theta^{(t)}$。

- 收敛的链：轨迹在一个稳定范围内波动，没有漂移趋势，多条链重叠
- 未收敛的链：明显的漂移、卡在某个值不动、不同链在不同区域

---

## 7. CosmoMC/Cobaya 中的实际实现

**可靠程度：Level 3**

你在用 CosmoMC 做 kSZ 参数估计时，背后运行的就是 MH 算法的改进版。

### 7.1 自适应提议分布

CosmoMC 使用**自适应 MCMC**：在 burn-in 阶段，根据已有采样自动调整提议分布的协方差矩阵。

1. **初始阶段**：使用用户提供的初始协方差矩阵（或对角矩阵）作为提议分布
2. **学习阶段**：定期从已有的链中估计参数协方差矩阵 $\hat{\mathbf{C}}$，并用 $\hat{\mathbf{C}}$ 更新提议分布
3. **稳定阶段**：提议分布不再更新，链进入正式采样

提议分布通常是多维高斯：$q(\boldsymbol{\theta}' | \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}'; \boldsymbol{\theta}, c^2 \hat{\mathbf{C}})$

- $\hat{\mathbf{C}}$：从之前的链中估计的参数协方差矩阵
- $c$：缩放因子，调整步长使接受率接近 23%

### 7.2 快慢参数分离

宇宙学参数分为两类：

- **慢参数**（如 $H_0$, $\Omega_m$）：改变后需要重新计算整个 Boltzmann 求解器（CAMB/CLASS），耗时 ~1 秒
- **快参数**（如校准系数、前景参数）：只影响似然计算，耗时 ~0.001 秒

CosmoMC 的策略：**快参数多走几步，慢参数少走几步**，因为慢参数的似然计算更昂贵。这叫做**拖拽采样**（dragging/oversampling）。

### 7.3 实际使用提示

- **收敛判断**：CosmoMC 默认输出 $\hat{R}$ 值，$R-1 < 0.02$（即 $\hat{R} < 1.02$）通常足够
- **链的数量**：通常运行 4–8 条独立链
- **后处理**：用 GetDist 读取链文件，画等高线图和 1D 边缘化分布

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $\alpha = \min\!\left(1, \frac{P(\boldsymbol{\theta}'\|\mathbf{d}) \, q(\boldsymbol{\theta}\|\boldsymbol{\theta}')}{P(\boldsymbol{\theta}\|\mathbf{d}) \, q(\boldsymbol{\theta}'\|\boldsymbol{\theta})}\right)$ | MH 接受概率：候选点后验 × 反向提议 / 当前后验 × 正向提议 |
| $\alpha = \min\!\left(1, \frac{\mathcal{L}(\boldsymbol{\theta}') \pi(\boldsymbol{\theta}')}{\mathcal{L}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta})}\right)$（对称提议） | Metropolis 接受概率：对称提议下 $q$ 抵消，只比后验之比 |
| $\pi_s(\boldsymbol{\theta}) T(\boldsymbol{\theta}'\|\boldsymbol{\theta}) = \pi_s(\boldsymbol{\theta}') T(\boldsymbol{\theta}\|\boldsymbol{\theta}')$ | 细致平衡条件：两点之间来回概率流相等 |
| $\hat{R} = \sqrt{\hat{V}/W}$ | Gelman-Rubin 收敛统计量：链间方差 / 链内方差，$< 1.1$ 表示收敛 |
| $N_\text{eff} = N / \tau_\text{corr}$ | 有效样本数 = 总步数 / 自相关时间 |
| $\rho(\tau) = \text{Cov}(\theta^{(t)}, \theta^{(t+\tau)}) / \text{Var}(\theta)$ | 自相关函数：相隔 $\tau$ 步的样本相关性 |

---

## 理解检测

**Q1**：MCMC 相比网格搜索的核心优势是什么？为什么 MCMC 不需要计算贝叶斯定理中的归一化常数 $P(\mathbf{d})$？

你的回答：


**Q2**：Metropolis-Hastings 算法中，如果候选点 $\boldsymbol{\theta}'$ 的后验概率**高于**当前点 $\boldsymbol{\theta}^{(t)}$，接受概率 $\alpha$ 是多少？如果低于当前点呢？为什么"下坡有时走"对 MCMC 是必要的？

你的回答：


**Q3（计算题）**：后验分布为 $P(\theta | d) \propto e^{-(\theta - 3)^2 / 2}$（即 $\mathcal{N}(3, 1)$）。使用对称提议分布。当前位置 $\theta^{(t)} = 2$，候选点 $\theta' = 5$。计算接受概率 $\alpha$。

> 提示：用 Metropolis 接受概率公式 $\alpha = \min(1, P(\theta'|d) / P(\theta^{(t)}|d))$，代入高斯后验

你的回答：


**Q4**：Gelman-Rubin 统计量 $\hat{R}$ 的计算需要多条链还是一条链？$\hat{R} = 1.5$ 意味着什么？CosmoMC 通常使用的收敛阈值是多少？

你的回答：


**Q5（计算题）**：一条 MCMC 链总共跑了 $N = 50000$ 步，自相关时间 $\tau_\text{corr} = 25$。有效样本数 $N_\text{eff}$ 是多少？如果要求 $N_\text{eff} \geq 1000$，这条链够长吗？

> 提示：用 $N_\text{eff} = N / \tau_\text{corr}$ 公式

你的回答：
