# 02 - 概率论复习 — 分布、条件概率、贝叶斯定理

> **主维度**：D1 概率论基础
> **次维度**：无
> **关键关系**：
> - 条件概率 (概念) --依赖--> 概率分布 (概念)：条件概率依赖概率分布的定义
> - 贝叶斯定理 (理论) --依赖--> 条件概率 (概念)：贝叶斯定理依赖条件概率
> - 似然函数 (概念) --依赖--> 概率分布 (概念)：似然函数依赖概率分布
>
> **学习路径**：全景概览 → **本章（概率论复习）** → 似然与 MLE → 贝叶斯推断 → MCMC → Fisher 矩阵 → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：本科微积分（积分、偏导）、线性代数（矩阵运算、逆矩阵、行列式）、01-overview 中的统计哲学概述
>
> **参考**：
> - [D.S. Sivia, Data Analysis: A Bayesian Tutorial, Ch.1-2](https://global.oup.com/academic/product/data-analysis-9780198568322)
> - [A. Papoulis, Probability, Random Variables, and Stochastic Processes](https://www.mheducation.com/highered/product/probability-random-variables-stochastic-processes-papoulis-pillai/M9780073515113.html)

---

本章是快速复习，目标是把后续章节会反复使用的概率论工具刷新一遍。你有本科概率论基础，所以这里不从零推导，而是**精炼要点 + 严谨定义 + 具体数值**。

---

## 1. 概率的公理化定义

**可靠程度：Level 1**

Kolmogorov 公理给出了概率的严格数学基础。

**三条公理**：

设 $\Omega$ 是样本空间（所有可能结果的集合），$A$ 是 $\Omega$ 的子集（事件），$P$ 是概率函数，则：

1. **非负性**：$P(A) \geq 0$，对所有事件 $A$
2. **归一性**：$P(\Omega) = 1$
3. **可加性**：若 $A_1, A_2, \ldots$ 两两互斥，则 $P\left(\bigcup_{i} A_i\right) = \sum_{i} P(A_i)$

这三条公理推出所有概率的计算规则（加法定理、补事件定理等），不管你是频率学派还是贝叶斯学派，数学基础是一样的。两个学派的分歧在于**如何解释** $P(A)$（见 01-overview 2.1 节）。

---

## 2. 常见概率分布及其性质

**可靠程度：Level 1**

### 2.1 连续均匀分布

$$f(x) = \frac{1}{b - a}, \quad a \leq x \leq b$$

| 符号 | 含义 |
|------|------|
| $f(x)$ | 概率密度函数（probability density function, PDF） |
| $a, b$ | 分布的下界和上界 |

性质：

- 期望：$E[X] = \frac{a + b}{2}$
- 方差：$\text{Var}(X) = \frac{(b-a)^2}{12}$

在宇宙学中，均匀分布常用作**无信息先验**——当我们对某个参数只知道它的合理范围，不知道它更可能偏向哪个值时，就给它一个均匀先验。例如：$H_0 \sim U(20, 100)$ km/s/Mpc。

### 2.2 高斯分布（正态分布）——重点

**一维高斯分布**：

$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| 符号 | 含义 |
|------|------|
| $\mu$ | 均值（分布的中心位置） |
| $\sigma$ | 标准差（分布的宽度） |
| $\sigma^2$ | 方差 |

**$n\sigma$ 区间的概率值**（必须记住）：

| 区间 | 概率 | 补充 |
|------|------|------|
| $\mu \pm 1\sigma$ | 68.27% | 约 2/3 |
| $\mu \pm 2\sigma$ | 95.45% | 约 95% |
| $\mu \pm 3\sigma$ | 99.73% | "三个九" |
| $\mu \pm 5\sigma$ | 99.99994% | 粒子物理的"发现"阈值 |

宇宙学论文中写 "$H_0 = 67.4 \pm 0.5$（68% CL）" 就意味着：假设后验近似高斯分布，$\sigma = 0.5$，68% 的可信区间是 $[66.9, 67.9]$。

**具体数字例子**：设 $\mu = 0$，$\sigma = 1$（标准正态分布）：

- $x = 0$ 时，$f(0) = \frac{1}{\sqrt{2\pi}} \approx 0.399$（峰值）
- $x = 1$ 时，$f(1) = \frac{1}{\sqrt{2\pi}} e^{-0.5} \approx 0.242$（峰值的 60.7%）
- $x = 3$ 时，$f(3) = \frac{1}{\sqrt{2\pi}} e^{-4.5} \approx 0.0044$（峰值的 1.1%）

所以高斯分布在 $3\sigma$ 处的密度只有峰值的约 1%——尾巴下降得非常快。

### 2.3 泊松分布

$$P(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

| 符号 | 含义 |
|------|------|
| $k$ | 事件发生的次数（非负整数） |
| $\lambda$ | 期望发生次数（$\lambda > 0$） |
| $P(k)$ | 恰好发生 $k$ 次的概率 |

性质：

- 期望：$E[k] = \lambda$
- 方差：$\text{Var}(k) = \lambda$（期望等于方差是泊松分布的标志性特征）
- 当 $\lambda$ 足够大时（$\lambda \gtrsim 30$），泊松分布近似为 $\mathcal{N}(\lambda, \lambda)$

宇宙学中，光子计数和星系计数常服从泊松分布。例如 CMB 某个像素收到 $\lambda = 100$ 个光子，实际计数的标准差是 $\sqrt{100} = 10$，即 10% 的泊松噪声（shot noise）。

### 2.4 多元高斯分布——重点

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\mathbf{C}|^{1/2}} \exp\!\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{C}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

| 符号 | 含义 |
|------|------|
| $\mathbf{x}$ | $n$ 维随机向量 |
| $\boldsymbol{\mu}$ | $n$ 维均值向量（分布的中心） |
| $\mathbf{C}$ | $n \times n$ 协方差矩阵 |
| $\|\mathbf{C}\|$ | $\mathbf{C}$ 的行列式（用于归一化） |
| $\mathbf{C}^{-1}$ | 精度矩阵（precision matrix），协方差矩阵的逆 |

**协方差矩阵的含义**：

$$C_{ij} = \text{Cov}(x_i, x_j) = E[(x_i - \mu_i)(x_j - \mu_j)]$$

- 对角元素 $C_{ii} = \sigma_i^2$：第 $i$ 个变量的方差
- 非对角元素 $C_{ij}$（$i \neq j$）：变量 $i$ 和 $j$ 之间的协方差
- 如果 $C_{ij} = 0$，则 $x_i$ 和 $x_j$ 不相关（对于高斯分布，不相关等价于独立）
- $\mathbf{C}$ 必须是对称正定矩阵

**具体数字例子**（2 维）：

设 $\boldsymbol{\mu} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$，$\mathbf{C} = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$。

- $\sigma_1 = \sqrt{4} = 2$，$\sigma_2 = \sqrt{2} \approx 1.41$
- 相关系数 $\rho = \frac{C_{12}}{\sigma_1 \sigma_2} = \frac{1}{2 \times 1.41} \approx 0.354$（正相关，但不强）
- $|\mathbf{C}| = 4 \times 2 - 1 \times 1 = 7$
- $\mathbf{C}^{-1} = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix}$

在宇宙学中，多元高斯分布极其常见：功率谱的测量误差、参数的联合后验分布、CMB 的像素温度涨落——都可以用多元高斯来描述或近似。

---

## 3. 条件概率与贝叶斯定理

**可靠程度：Level 1**

### 3.1 条件概率

$$P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

| 符号 | 含义 |
|------|------|
| $P(A \| B)$ | 在 $B$ 已经发生的条件下，$A$ 发生的概率 |
| $P(A \cap B)$ | $A$ 和 $B$ 同时发生的概率 |
| $P(B)$ | $B$ 发生的概率（必须大于零） |

直觉：条件概率就是**缩小了样本空间**——我们已经知道 $B$ 发生了，所以只看 $B$ 内部的情况。

### 3.2 全概率公式

如果 $B_1, B_2, \ldots, B_n$ 构成样本空间的一个划分（两两互斥且并集为 $\Omega$），则：

$$P(A) = \sum_{i=1}^{n} P(A | B_i) \, P(B_i)$$

这在贝叶斯推断中对应**边缘化**：把"条件"的所有可能情况加权求和，得到无条件概率。后续章节中你会反复看到边缘化操作。

### 3.3 贝叶斯定理的推导

从条件概率的对称性出发：

$$P(A \cap B) = P(A | B) \, P(B) = P(B | A) \, P(A)$$

两边除以 $P(B)$：

$$\boxed{P(A | B) = \frac{P(B | A) \, P(A)}{P(B)}}$$

这就是贝叶斯定理。本质上就是条件概率的"翻转"——从"已知 $A$ 求 $B$"变成"已知 $B$ 求 $A$"。

用全概率公式展开分母：

$$P(A | B) = \frac{P(B | A) \, P(A)}{\sum_{i} P(B | A_i) \, P(A_i)}$$

### 3.4 具体数字例子：检测器灵敏度

**场景**：你用一台 CMB 检测器来判断天空某区域是否有 SZ 信号（"有信号" = 事件 $S$，"无信号" = 事件 $\bar{S}$）。已知：

- 实际有信号的先验概率：$P(S) = 0.01$（很稀有，1% 的区域有信号）
- 检测器灵敏度（有信号时正确报警的概率）：$P(D | S) = 0.95$
- 检测器误报率（无信号时错误报警的概率）：$P(D | \bar{S}) = 0.05$

**问题**：检测器报警了（事件 $D$ 发生），实际有信号的概率是多少？

用贝叶斯定理：

$$P(S | D) = \frac{P(D | S) \, P(S)}{P(D)}$$

先算分母（全概率公式）：

$$P(D) = P(D|S) \, P(S) + P(D|\bar{S}) \, P(\bar{S}) = 0.95 \times 0.01 + 0.05 \times 0.99 = 0.0095 + 0.0495 = 0.059$$

所以：

$$P(S | D) = \frac{0.95 \times 0.01}{0.059} = \frac{0.0095}{0.059} \approx 0.161$$

**结果**：即使检测器报警了，实际有信号的概率只有 16.1%！这是因为先验概率太低（只有 1%），大量误报淹没了真正的信号。

**启示**：先验 $P(S)$ 在信号稀少时有巨大影响。这就是为什么宇宙学中先验选择很重要——当数据约束力有限时，先验可以显著影响后验结果（详见 04-bayesian-inference）。

---

## 4. 期望、方差、协方差

**可靠程度：Level 1**

### 4.1 基本定义

| 量 | 离散情况 | 连续情况 |
|------|---------|---------|
| 期望 $E[X]$ | $\sum_i x_i \, P(x_i)$ | $\int x \, f(x) \, dx$ |
| 方差 $\text{Var}(X)$ | $E[(X - \mu)^2]$ | $\int (x - \mu)^2 \, f(x) \, dx$ |
| 协方差 $\text{Cov}(X, Y)$ | $E[(X - \mu_X)(Y - \mu_Y)]$ | $\iint (x - \mu_X)(y - \mu_Y) \, f(x, y) \, dx \, dy$ |

| 符号 | 含义 |
|------|------|
| $E[X]$ | $X$ 的期望值（均值），记作 $\mu$ 或 $\langle X \rangle$ |
| $\text{Var}(X)$ | $X$ 的方差，记作 $\sigma^2$ |
| $\text{Cov}(X, Y)$ | $X$ 和 $Y$ 的协方差，衡量两者的线性关联程度 |
| $f(x)$ | 概率密度函数 |
| $f(x, y)$ | 联合概率密度函数 |

### 4.2 关键性质

- $\text{Var}(X) = E[X^2] - (E[X])^2$（计算方差的快捷公式）
- $\text{Cov}(X, Y) = E[XY] - E[X] \, E[Y]$
- 如果 $X, Y$ 独立，则 $\text{Cov}(X, Y) = 0$，但反过来不一定成立（不相关 $\neq$ 独立，除非是高斯分布）
- 相关系数：$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \, \sigma_Y}$，满足 $-1 \leq \rho \leq 1$

### 4.3 误差传播公式

设 $Y = g(X)$，$X$ 的误差为 $\sigma_X$，则 $Y$ 的误差近似为：

$$\sigma_Y \approx \left| \frac{dg}{dX} \right| \sigma_X$$

多变量情况：$Y = g(X_1, X_2, \ldots)$，变量之间独立时：

$$\sigma_Y^2 \approx \sum_i \left( \frac{\partial g}{\partial X_i} \right)^2 \sigma_{X_i}^2$$

如果变量之间有相关性，则需要用协方差矩阵：

$$\sigma_Y^2 \approx \sum_{i,j} \frac{\partial g}{\partial X_i} \frac{\partial g}{\partial X_j} C_{ij}$$

其中 $C_{ij} = \text{Cov}(X_i, X_j)$ 是协方差矩阵的元素。这个公式在 Fisher 矩阵（06 章）中会重新出现。

---

## 5. 大数定律与中心极限定理

**可靠程度：Level 1**

### 5.1 大数定律

**弱大数定律**：设 $X_1, X_2, \ldots, X_n$ 是独立同分布（i.i.d.）的随机变量，期望为 $\mu$，则样本均值依概率收敛到 $\mu$：

$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu, \quad n \to \infty$$

直觉：**测量次数越多，平均值越接近真值。** 这是所有"取平均来降低噪声"操作的数学基础。

### 5.2 中心极限定理（CLT）

设 $X_1, X_2, \ldots, X_n$ 是 i.i.d. 的随机变量，期望 $\mu$，方差 $\sigma^2$，则当 $n$ 足够大时，样本均值近似服从高斯分布：

$$\bar{X}_n \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$$

或等价地，标准化后的量收敛到标准正态：

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

| 符号 | 含义 |
|------|------|
| $\bar{X}_n$ | $n$ 个独立样本的均值 |
| $\sigma / \sqrt{n}$ | 均值的标准误差（standard error），随 $n$ 增大而缩小 |
| $\xrightarrow{d}$ | 依分布收敛 |

**关键结论**：

1. **不管原始分布是什么形状**（泊松、均匀、指数……），大量独立样本的均值**总是趋向高斯分布**
2. 误差按 $1/\sqrt{n}$ 缩小——要把误差减半，需要 4 倍的数据量
3. 这解释了为什么高斯分布在宇宙学中无处不在——许多观测量是大量独立贡献的叠加

**具体数字例子**：CMB 某个频率通道在一个像素上叠加了 $n = 10000$ 次独立扫描，每次扫描的噪声标准差为 $\sigma = 100\,\mu\text{K}$。根据 CLT，最终该像素的噪声标准差为：

$$\sigma_{\text{pixel}} = \frac{\sigma}{\sqrt{n}} = \frac{100}{100} = 1\,\mu\text{K}$$

噪声从 100 μK 降到了 1 μK——这就是重复观测的威力。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2 / 2\sigma^2}$ | 一维高斯分布的 PDF |
| $f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}\|\mathbf{C}\|^{1/2}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{C}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ | $n$ 维高斯分布的 PDF |
| $P(A\|B) = \frac{P(B\|A)\,P(A)}{P(B)}$ | 贝叶斯定理 |
| $P(A) = \sum_i P(A\|B_i)\,P(B_i)$ | 全概率公式 |
| $\text{Var}(X) = E[X^2] - (E[X])^2$ | 方差的快捷计算公式 |
| $\rho_{XY} = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ | 相关系数，$-1 \leq \rho \leq 1$ |
| $\sigma_{\bar{X}} = \sigma / \sqrt{n}$ | 均值的标准误差（CLT） |

---

## 理解检测

**Q1**（概念）：高斯分布的 $1\sigma$、$2\sigma$、$3\sigma$ 区间分别对应多少概率？为什么宇宙学论文中 "68% CL" 和 "$1\sigma$" 可以互换使用？在什么条件下这个等价关系不成立？

你的回答：


**Q2**（计算）：一个 2 维高斯分布的协方差矩阵为 $\mathbf{C} = \begin{pmatrix} 9 & 3 \\ 3 & 4 \end{pmatrix}$。请计算：(a) $\sigma_1$ 和 $\sigma_2$；(b) 相关系数 $\rho$；(c) $\mathbf{C}$ 的行列式 $|\mathbf{C}|$。

> 提示：$\sigma_i = \sqrt{C_{ii}}$，$\rho = C_{12}/(\sigma_1 \sigma_2)$，$|\mathbf{C}| = C_{11}C_{22} - C_{12}^2$

你的回答：


**Q3**（概念 + 计算）：在 3.4 节的检测器例子中，如果先验概率从 $P(S) = 0.01$ 变成 $P(S) = 0.10$（10% 的区域有信号），检测器报警后实际有信号的概率 $P(S|D)$ 变成多少？和原来的 16.1% 比，先验的变化对后验影响大吗？

> 提示：用贝叶斯定理 $P(S|D) = P(D|S)P(S)/P(D)$，先用全概率公式算 $P(D)$

你的回答：


**Q4**（概念）：中心极限定理说误差按 $1/\sqrt{n}$ 缩小。如果当前实验的参数测量误差为 $\sigma = 2.0$，你想把误差降到 $\sigma = 0.5$，至少需要多少倍的数据量？

> 提示：用 $\sigma_{\bar{X}} = \sigma / \sqrt{n}$ 公式

你的回答：
