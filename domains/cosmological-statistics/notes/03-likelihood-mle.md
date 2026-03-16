# 03 - 似然函数与最大似然估计

> **主维度**：D1 概率论基础 + D2 参数估计
> **次维度**：D4 宇宙学应用（高斯似然的矩阵形式）
> **关键关系**：
> - 最大似然估计 (方法) --依赖--> 似然函数 (概念)：MLE 依赖似然函数的定义
> - 似然函数 (概念) --依赖--> 概率分布 (概念)：似然函数依赖概率分布
> - 最大似然估计 (方法) --用于--> 参数估计 (任务)：MLE 用于参数估计
>
> **学习路径**：全景概览 → 概率论复习 → **本章（似然与 MLE）** → 贝叶斯推断 → MCMC → Fisher 矩阵 → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：02-probability-review（高斯分布、多元高斯分布、条件概率、贝叶斯定理）
>
> **参考**：
> - [D.S. Sivia, Data Analysis: A Bayesian Tutorial, Ch.2-3](https://global.oup.com/academic/product/data-analysis-9780198568322)
> - [R. Trotta, 2008 - Bayes in the sky, Section 3](https://arxiv.org/abs/0803.4089)
> - [A. Heavens, 2009 - Statistical techniques in cosmology, Section 2](https://arxiv.org/abs/0906.0664)

---

## 1. 似然函数的定义

**可靠程度：Level 1**

### 1.1 核心定义

**似然函数**（likelihood function）的定义：

$$\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} \,|\, \boldsymbol{\theta})$$

| 符号 | 含义 |
|------|------|
| $\mathcal{L}(\boldsymbol{\theta})$ | 似然函数，是参数 $\boldsymbol{\theta}$ 的函数 |
| $\mathbf{d}$ | 观测数据（已经确定，不是变量） |
| $\boldsymbol{\theta}$ | 模型参数（是变量，我们想找最优值） |
| $P(\mathbf{d} \,\|\, \boldsymbol{\theta})$ | 给定参数 $\boldsymbol{\theta}$ 时，观测到数据 $\mathbf{d}$ 的概率 |

### 1.2 似然 vs 概率：条件方向的区别

$P(\mathbf{d} | \boldsymbol{\theta})$ 这个表达式可以从两个方向理解：

- **当作概率**（$\boldsymbol{\theta}$ 固定，$\mathbf{d}$ 变化）：给定参数，不同数据出现的概率——这是一个关于 $\mathbf{d}$ 的概率分布，积分（或求和）为 1
- **当作似然**（$\mathbf{d}$ 固定，$\boldsymbol{\theta}$ 变化）：已有数据，不同参数值的"合理程度"——这是一个关于 $\boldsymbol{\theta}$ 的函数，**不要求积分为 1**

似然函数不是概率分布。它告诉你的是：哪组参数能更好地"解释"已有数据。似然越大，说明这组参数产生所观测数据的可能性越高。

### 1.3 具体例子：独立高斯测量

**场景**：你有 $N$ 个独立测量值 $d_1, d_2, \ldots, d_N$，每个都是对同一物理量 $\mu$（真值）的测量，测量误差为 $\sigma$（已知），噪声服从高斯分布：

$$d_i = \mu + n_i, \quad n_i \sim \mathcal{N}(0, \sigma^2)$$

| 符号 | 含义 |
|------|------|
| $d_i$ | 第 $i$ 次测量的结果 |
| $\mu$ | 要估计的参数（物理量的真值） |
| $n_i$ | 第 $i$ 次测量的噪声，服从均值为 0、方差为 $\sigma^2$ 的高斯分布 |

单次测量的概率：

$$P(d_i | \mu) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(d_i - \mu)^2}{2\sigma^2}\right)$$

因为 $N$ 次测量独立，联合概率是各次概率的乘积：

$$\mathcal{L}(\mu) = \prod_{i=1}^N P(d_i | \mu) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(d_i - \mu)^2}{2\sigma^2}\right)$$

**具体数字**：假设 $N = 3$，$\sigma = 1$，三次测量值为 $d_1 = 2.1$，$d_2 = 1.8$，$d_3 = 2.3$。那么在 $\mu = 2.0$ 处的似然值为：

$$\mathcal{L}(2.0) \propto \exp\!\left(-\frac{0.1^2 + 0.2^2 + 0.3^2}{2}\right) = \exp\!\left(-\frac{0.14}{2}\right) = e^{-0.07} \approx 0.932$$

在 $\mu = 3.0$ 处：

$$\mathcal{L}(3.0) \propto \exp\!\left(-\frac{0.9^2 + 1.2^2 + 0.7^2}{2}\right) = \exp\!\left(-\frac{2.74}{2}\right) = e^{-1.37} \approx 0.254$$

$\mu = 2.0$ 的似然比 $\mu = 3.0$ 大得多——数据更支持 $\mu \approx 2$。

---

## 2. 对数似然

**可靠程度：Level 1**

在实际计算中，几乎总是使用**对数似然**（log-likelihood）$\ln \mathcal{L}$ 而不是似然本身。原因：

1. **乘积变求和**：$\ln \prod_i P(d_i | \theta) = \sum_i \ln P(d_i | \theta)$，数值上更稳定
2. **避免下溢**：$N$ 很大时，大量小于 1 的数相乘会趋近于零，计算机无法表示；取对数后变成求和，没有这个问题
3. **求极值等价**：$\ln$ 是单调递增函数，所以 $\mathcal{L}$ 的最大值和 $\ln \mathcal{L}$ 的最大值在同一个 $\boldsymbol{\theta}$ 处

对于上面的 $N$ 个独立高斯测量：

$$\ln \mathcal{L}(\mu) = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2} \sum_{i=1}^N \frac{(d_i - \mu)^2}{\sigma^2}$$

第一项是常数（不依赖于 $\mu$），只有第二项影响 $\mu$ 的最优值。

---

## 3. 最大似然估计（MLE）

**可靠程度：Level 1**

### 3.1 定义

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \, \mathcal{L}(\boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \, \ln \mathcal{L}(\boldsymbol{\theta})$$

| 符号 | 含义 |
|------|------|
| $\hat{\theta}_{\text{MLE}}$ | 最大似然估计值（"帽子"表示估计量） |
| $\arg\max$ | 使函数取最大值的参数值 |

直觉：找那组最能"解释"数据的参数。

### 3.2 手工推导：高斯数据的 MLE

对 $\ln \mathcal{L}(\mu)$ 求导并令其为零：

$$\frac{\partial \ln \mathcal{L}}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^N (d_i - \mu) = 0$$

$$\sum_{i=1}^N d_i - N\mu = 0$$

$$\boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N d_i = \bar{d}}$$

**MLE 就是样本均值**——这是一个直觉上完全合理的结果。你有一堆测量值，对真值的最佳估计就是它们的平均值。

对上面的数字例子：$\hat{\mu}_{\text{MLE}} = (2.1 + 1.8 + 2.3)/3 = 6.2/3 \approx 2.067$。

二阶导数：

$$\frac{\partial^2 \ln \mathcal{L}}{\partial \mu^2} = -\frac{N}{\sigma^2}$$

这是负数（对数似然是向下开的抛物线），确认找到的是最大值。这个二阶导数后面在 Fisher 矩阵章节（06）会再次出现。

### 3.3 MLE 的性质

**可靠程度：Level 1（前三条），Level 2（渐近有效性的条件）**

| 性质 | 含义 | 条件 |
|------|------|------|
| **一致性** | $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_{\text{true}}$，数据量 $N \to \infty$ 时收敛到真值 | 模型正确 |
| **渐近正态性** | $N$ 足够大时，$\hat{\theta}_{\text{MLE}}$ 近似服从高斯分布 | $N$ 大 |
| **渐近有效性** | 大样本下，MLE 的方差达到 Cramér-Rao 下界 | $N$ 大 + 正则条件 |
| **不变性** | 如果 $\hat{\theta}$ 是 $\theta$ 的 MLE，则 $g(\hat{\theta})$ 是 $g(\theta)$ 的 MLE | 任何函数 $g$ |

一致性意味着"数据够多就能找到真值"。渐近正态性意味着"MLE 附近的误差近似高斯"，这是用 $\Delta\chi^2$ 做置信区间的理论基础。

---

## 4. 高斯似然的矩阵形式

**可靠程度：Level 1**

### 4.1 一般形式

对于数据向量 $\mathbf{d}$（维度 $n$），理论模型预测 $\mathbf{m}(\boldsymbol{\theta})$，如果残差 $\mathbf{d} - \mathbf{m}$ 服从多元高斯分布，则对数似然为：

$$\boxed{\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta}))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta})) - \frac{1}{2} \ln |\mathbf{C}| - \frac{n}{2} \ln(2\pi)}$$

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{d}$ | 数据向量（观测值） | $n \times 1$ |
| $\mathbf{m}(\boldsymbol{\theta})$ | 理论模型预测（依赖参数 $\boldsymbol{\theta}$） | $n \times 1$ |
| $\boldsymbol{\theta}$ | 模型参数向量 | $p \times 1$（$p$ 个参数） |
| $\mathbf{C}$ | 数据的协方差矩阵 | $n \times n$ |
| $\mathbf{C}^{-1}$ | 精度矩阵（协方差矩阵的逆） | $n \times n$ |
| $\|\mathbf{C}\|$ | 协方差矩阵的行列式 | 标量 |

当 $\mathbf{C}$ 不依赖于 $\boldsymbol{\theta}$ 时（这是最常见的情况），后两项是常数，参数估计只需要第一项：

$$\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2} (\mathbf{d} - \mathbf{m})^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}) + \text{const}$$

### 4.2 具体数字例子：2 维情况

**场景**：你测量了两个功率谱数据点 $\mathbf{d} = \begin{pmatrix} 5.2 \\ 3.1 \end{pmatrix}$，理论模型在参数 $\theta$ 下预测 $\mathbf{m}(\theta) = \begin{pmatrix} 5.0 \\ 3.0 \end{pmatrix}$，协方差矩阵为 $\mathbf{C} = \begin{pmatrix} 0.04 & 0.01 \\ 0.01 & 0.09 \end{pmatrix}$。

**Step 1**：残差向量

$$\mathbf{d} - \mathbf{m} = \begin{pmatrix} 0.2 \\ 0.1 \end{pmatrix}$$

**Step 2**：求 $\mathbf{C}^{-1}$

$$|\mathbf{C}| = 0.04 \times 0.09 - 0.01^2 = 0.0036 - 0.0001 = 0.0035$$

$$\mathbf{C}^{-1} = \frac{1}{0.0035} \begin{pmatrix} 0.09 & -0.01 \\ -0.01 & 0.04 \end{pmatrix} = \begin{pmatrix} 25.71 & -2.86 \\ -2.86 & 11.43 \end{pmatrix}$$

**Step 3**：计算二次型 $\chi^2 = (\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1} (\mathbf{d}-\mathbf{m})$

先算 $\mathbf{C}^{-1}(\mathbf{d}-\mathbf{m})$：

$$\begin{pmatrix} 25.71 & -2.86 \\ -2.86 & 11.43 \end{pmatrix} \begin{pmatrix} 0.2 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 25.71 \times 0.2 + (-2.86) \times 0.1 \\ (-2.86) \times 0.2 + 11.43 \times 0.1 \end{pmatrix} = \begin{pmatrix} 4.856 \\ 0.571 \end{pmatrix}$$

然后：

$$\chi^2 = \begin{pmatrix} 0.2 & 0.1 \end{pmatrix} \begin{pmatrix} 4.856 \\ 0.571 \end{pmatrix} = 0.2 \times 4.856 + 0.1 \times 0.571 = 0.971 + 0.057 = 1.028$$

$$\ln \mathcal{L} = -\frac{1}{2} \times 1.028 + \text{const} = -0.514 + \text{const}$$

$\chi^2 \approx 1.0$，对于 2 个数据点来说这是一个合理的拟合（$\chi^2 / \text{dof} \approx 1$，见第 5 节）。

### 4.3 特殊情况：对角协方差

当数据点之间没有相关性时，$\mathbf{C}$ 是对角矩阵：$C_{ij} = \sigma_i^2 \delta_{ij}$（$\delta_{ij}$ 是 Kronecker delta），此时：

$$\ln \mathcal{L} = -\frac{1}{2} \sum_{i=1}^n \frac{(d_i - m_i)^2}{\sigma_i^2} + \text{const}$$

这就退化成了 $n$ 个独立高斯测量的情况——每个数据点对似然的贡献独立求和。

---

## 5. $\chi^2$ 统计量

**可靠程度：Level 1**

### 5.1 定义与似然的关系

对于高斯似然，$\chi^2$ 定义为残差的加权平方和：

$$\chi^2(\boldsymbol{\theta}) = (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta}))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{m}(\boldsymbol{\theta}))$$

| 符号 | 含义 |
|------|------|
| $\chi^2$ | 卡方统计量，衡量数据与模型预测的偏离程度 |

它与对数似然的关系（当 $\mathbf{C}$ 不依赖 $\boldsymbol{\theta}$ 时）：

$$\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2}\chi^2(\boldsymbol{\theta}) + \text{const}$$

**关键关系**：最小化 $\chi^2$ 等价于最大化 $\ln \mathcal{L}$。所以 $\chi^2$ 拟合和最大似然估计在高斯似然下是同一件事。

### 5.2 自由度与拟合优度

**自由度**（degrees of freedom, dof）：

$$\text{dof} = n_{\text{data}} - n_{\text{param}}$$

| 符号 | 含义 |
|------|------|
| $n_{\text{data}}$ | 数据点的数量 |
| $n_{\text{param}}$ | 拟合参数的数量 |

**拟合优度的经验判断**：

| $\chi^2 / \text{dof}$ | 含义 |
|----------------------|------|
| $\approx 1$ | 好的拟合——数据偏差符合预期噪声水平 |
| $\gg 1$ | 差的拟合——模型与数据有系统性偏差 |
| $\ll 1$ | 过度拟合，或误差被高估了 |

---

## 6. 置信区间：$\Delta\chi^2$ 方法

**可靠程度：Level 1（基本方法），Level 2（适用条件）**

### 6.1 基本思想

在 MLE 附近，$\chi^2(\theta)$ 近似为关于 $\hat{\theta}$ 的二次函数（抛物线）。定义：

$$\Delta\chi^2(\theta) = \chi^2(\theta) - \chi^2_{\min}$$

通过设定 $\Delta\chi^2$ 的阈值来确定置信区间。

### 6.2 单参数情况

| $\Delta\chi^2$ 阈值 | 置信水平（CL） | 对应 $n\sigma$ |
|---------------------|--------------|---------------|
| 1.00 | 68.3% | $1\sigma$ |
| 4.00 | 95.4% | $2\sigma$ |
| 9.00 | 99.7% | $3\sigma$ |

**直觉**：对数似然是一个向下的抛物线。$\Delta\chi^2 = 1$ 意味着从峰顶下降 1/2 个单位的对数似然。所有满足 $\Delta\chi^2 \leq 1$ 的 $\theta$ 值构成 $1\sigma$ 置信区间。

**具体数字例子**：回到 3.2 节的高斯测量例子（$N = 3$，$\sigma = 1$，$\hat{\mu} = 2.067$）。对数似然近似为：

$$\ln \mathcal{L}(\mu) \approx \text{const} - \frac{N}{2\sigma^2}(\mu - \hat{\mu})^2 = \text{const} - \frac{3}{2}(\mu - 2.067)^2$$

所以 $\Delta\chi^2 = 2 \times \frac{3}{2}(\mu - 2.067)^2 = 3(\mu - 2.067)^2$。

令 $\Delta\chi^2 = 1$：$(\mu - 2.067)^2 = 1/3$，即 $\mu = 2.067 \pm 1/\sqrt{3} \approx 2.067 \pm 0.577$。

这与标准公式 $\sigma_{\hat{\mu}} = \sigma/\sqrt{N} = 1/\sqrt{3} \approx 0.577$ 完全一致。

### 6.3 多参数情况

| 参数个数 $p$ | $\Delta\chi^2$ for 68.3% CL | $\Delta\chi^2$ for 95.4% CL |
|-------------|---------------------------|---------------------------|
| 1 | 1.00 | 4.00 |
| 2 | 2.30 | 6.18 |
| 3 | 3.53 | 8.02 |

$p$ 个参数时，$\Delta\chi^2$ 服从 $\chi^2$ 分布（$p$ 个自由度）。宇宙学论文中常见的 2 参数等高线图，$1\sigma$ 和 $2\sigma$ 等高线对应的就是 $\Delta\chi^2 = 2.30$ 和 $6.18$。

**重要区别**：这里的"多参数"是指你**同时关心**的参数个数。如果你有 6 个参数但只关心 $H_0$，用的还是单参数的 $\Delta\chi^2 = 1$（因为其他参数被边缘化掉了，见 04 章）。

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} \,\|\, \boldsymbol{\theta})$ | 似然函数的定义 |
| $\hat{\theta}_{\text{MLE}} = \arg\max \ln \mathcal{L}(\boldsymbol{\theta})$ | 最大似然估计 |
| $\ln \mathcal{L} = -\frac{1}{2}(\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m}) + \text{const}$ | 高斯对数似然（$\mathbf{C}$ 不依赖 $\theta$ 时） |
| $\chi^2 = (\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m})$ | $\chi^2$ 统计量，最小化 $\chi^2$ ⟺ 最大化 $\ln \mathcal{L}$ |
| $\text{dof} = n_{\text{data}} - n_{\text{param}}$ | 自由度 |
| $\Delta\chi^2 = 1$（单参数 $1\sigma$） | 单参数 68.3% 置信区间的阈值 |
| $\Delta\chi^2 = 2.30$（双参数 $1\sigma$） | 双参数 68.3% 联合置信区间的阈值 |
| $\sigma_{\hat{\mu}} = \sigma / \sqrt{N}$ | $N$ 个等精度独立测量的 MLE 标准误差 |

---

## 理解检测

**Q1**（概念）：似然函数 $\mathcal{L}(\boldsymbol{\theta}) = P(\mathbf{d} | \boldsymbol{\theta})$ 为什么不是概率分布？具体来说，"对 $\boldsymbol{\theta}$ 积分不等于 1"意味着什么？如果似然不是概率分布，那我们凭什么用它来做推断？

你的回答：


**Q2**（计算）：你有 4 个独立测量值 $d_1 = 10.2$，$d_2 = 9.8$，$d_3 = 10.5$，$d_4 = 9.5$，每个测量的误差为 $\sigma = 0.5$。(a) 求 $\hat{\mu}_{\text{MLE}}$；(b) 求 MLE 的标准误差 $\sigma_{\hat{\mu}}$；(c) 写出 $1\sigma$ 置信区间。

> 提示：用 $\hat{\mu}_{\text{MLE}} = \bar{d}$ 和 $\sigma_{\hat{\mu}} = \sigma / \sqrt{N}$

你的回答：


**Q3**（计算）：在 4.2 节的 2 维数字例子中，如果数据变为 $\mathbf{d} = \begin{pmatrix} 5.0 \\ 3.0 \end{pmatrix}$（与模型预测完全一致），$\chi^2$ 等于多少？$\ln \mathcal{L}$ 是否达到最大值？

> 提示：代入 $\mathbf{d} - \mathbf{m} = \mathbf{0}$ 到 $\chi^2 = (\mathbf{d}-\mathbf{m})^T \mathbf{C}^{-1}(\mathbf{d}-\mathbf{m})$ 公式

你的回答：


**Q4**（概念）：论文中报告 "$\chi^2/\text{dof} = 1.5$，30 个数据点拟合 6 个参数"。(a) 自由度是多少？(b) 这个拟合结果好不好？(c) 可能的原因是什么？

> 提示：用 $\text{dof} = n_{\text{data}} - n_{\text{param}}$，$\chi^2/\text{dof} \approx 1$ 是好拟合

你的回答：
