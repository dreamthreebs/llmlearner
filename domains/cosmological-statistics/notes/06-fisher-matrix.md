# 06 - Fisher 矩阵 — 参数预测与退化分析

> **主维度**：D2 参数估计
> **次维度**：D4 宇宙学应用（功率谱 Fisher 矩阵）
> **关键关系**：
> - Fisher 矩阵 (方法) --依赖--> 似然函数 (概念)：Fisher 矩阵是似然的二阶导数
> - Fisher 矩阵 (方法) --用于--> 参数估计 (任务)：Fisher 矩阵用于参数约束预测
> - Fisher 矩阵 (方法) --对比--> MCMC (方法)：高斯近似 vs 精确采样
>
> **学习路径**：全景概览 → 概率论复习 → 似然与 MLE → 贝叶斯推断 → MCMC → **本章（Fisher 矩阵）** → 模型选择 → 宇宙学似然 → 结果解读
>
> **前置知识**：似然函数、对数似然、MLE（03-likelihood-mle）；后验分布、高斯似然矩阵形式（03 + 04）；MCMC 的基本原理（05-mcmc）；线性代数基础（特征值、特征向量、矩阵求逆）
>
> **参考**：
> - [M. Tegmark, A. Taylor, A. Heavens, 1997 - Karhunen-Loève eigenvalue problems in cosmology](https://arxiv.org/abs/astro-ph/9603021)
> - [A. Heavens, 2009 - Statistical techniques in cosmology](https://arxiv.org/abs/0906.0664)
> - [L. Verde, 2010 - Statistical methods in cosmology](https://arxiv.org/abs/0911.3105)
> - [R. Trotta, 2008 - Bayes in the sky, Section 4](https://arxiv.org/abs/0803.4089)

---

## 1. Fisher 信息矩阵的定义

**可靠程度：Level 1**

### 1.1 从似然的曲率出发

回顾 MLE（03 章）：最大似然估计 $\hat{\boldsymbol{\theta}}$ 是对数似然 $\ln \mathcal{L}(\boldsymbol{\theta})$ 的极大值点。在这个极值点附近，对数似然可以做 Taylor 展开：

$$\ln \mathcal{L}(\boldsymbol{\theta}) \approx \ln \mathcal{L}(\hat{\boldsymbol{\theta}}) + \frac{1}{2} \sum_{i,j} \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j}\bigg|_{\hat{\boldsymbol{\theta}}} (\theta_i - \hat{\theta}_i)(\theta_j - \hat{\theta}_j)$$

一阶项为零（因为 $\hat{\boldsymbol{\theta}}$ 是极值点）。二阶项的系数矩阵——**对数似然的 Hessian 矩阵**——就包含了参数约束的全部信息。

**Fisher 信息矩阵**的定义：

$$F_{ij} = -\left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j} \right\rangle$$

- $F_{ij}$：Fisher 矩阵的 $(i, j)$ 元素
- $\ln \mathcal{L}$：对数似然函数
- $\theta_i, \theta_j$：第 $i$ 和第 $j$ 个参数
- $\langle \cdot \rangle$：对数据的期望值（在真实参数 $\boldsymbol{\theta}_0$ 下所有可能数据实现的平均）
- 负号：因为对数似然在极值处的 Hessian 是**负定**的（极大值），加负号使 $F$ 成为正定矩阵

### 1.2 直觉：似然面的曲率

想象一个二维似然面 $\mathcal{L}(\theta_1, \theta_2)$ 的等高线图。MLE 在山顶，Fisher 矩阵描述**山顶附近的形状**：

- **$F_{ii}$ 大**：似然面在 $\theta_i$ 方向上很**陡**，偏离最佳值一点点，似然就急剧下降 → 参数 $\theta_i$ 被**紧约束**
- **$F_{ii}$ 小**：似然面在 $\theta_i$ 方向上很**平**，偏离很多似然也不怎么变 → 参数 $\theta_i$ 约束很**松**
- **$F_{ij} \neq 0$（$i \neq j$）**：似然面的等高线是**倾斜的椭圆**，$\theta_i$ 和 $\theta_j$ 之间存在**参数退化**（degeneracy）

具体例子：如果在 $\theta_1$ 方向上偏离 0.01 就让 $\ln \mathcal{L}$ 下降 50，但在 $\theta_2$ 方向上偏离 1.0 才下降 50，那么 $F_{11} \gg F_{22}$，$\theta_1$ 的约束比 $\theta_2$ 紧 100 倍。

---

## 2. Cramér-Rao 下界

**可靠程度：Level 1**

Fisher 矩阵的核心应用：**给出参数估计精度的理论下限**。

### 2.1 不等式

**Cramér-Rao 不等式**（Cramér-Rao bound, CRB）：

$$\text{Cov}(\hat{\boldsymbol{\theta}}) \geq F^{-1}$$

- $\text{Cov}(\hat{\boldsymbol{\theta}})$：任何**无偏估计量** $\hat{\boldsymbol{\theta}}$ 的协方差矩阵（无偏 = 估计量的期望值等于真值，即 $\langle \hat{\boldsymbol{\theta}} \rangle = \boldsymbol{\theta}_0$，没有系统偏差）
- $F^{-1}$：Fisher 矩阵的逆矩阵
- $\geq$ 意味着 $\text{Cov}(\hat{\boldsymbol{\theta}}) - F^{-1}$ 是半正定矩阵

**单参数情况**：

$$\sigma(\theta_i) \geq \frac{1}{\sqrt{F_{ii}}} \quad (\text{仅当该参数与其他参数无关})$$

- $\sigma(\theta_i)$：参数 $\theta_i$ 估计值的标准差
- $F_{ii}$：Fisher 矩阵的第 $i$ 个对角元素

**多参数情况**（考虑参数退化后）：

$$\sigma(\theta_i) \geq \sqrt{(F^{-1})_{ii}}$$

- $(F^{-1})_{ii}$：Fisher 矩阵**逆矩阵**的第 $i$ 个对角元素
- 注意 $(F^{-1})_{ii} \neq 1/F_{ii}$（矩阵求逆后对角元素会改变！）

### 2.2 关键区分：$1/\sqrt{F_{ii}}$ vs $\sqrt{(F^{-1})_{ii}}$

这个区分很重要，经常让人混淆：

- $1/\sqrt{F_{ii}}$：**固定其他参数**时，$\theta_i$ 的最小误差（不现实，因为其他参数也是未知的）
- $\sqrt{(F^{-1})_{ii}}$：**边缘化其他参数**后，$\theta_i$ 的最小误差（实际报告的误差）

$\sqrt{(F^{-1})_{ii}} \geq 1/\sqrt{F_{ii}}$——考虑参数退化后，误差只会**更大**，不会更小。

**具体数字例子**：假设 2 参数 Fisher 矩阵为

$$F = \begin{pmatrix} 100 & 80 \\ 80 & 100 \end{pmatrix}$$

固定另一参数时的误差：$1/\sqrt{F_{11}} = 1/\sqrt{100} = 0.1$

求逆：$F^{-1} = \frac{1}{100 \times 100 - 80 \times 80} \begin{pmatrix} 100 & -80 \\ -80 & 100 \end{pmatrix} = \frac{1}{3600} \begin{pmatrix} 100 & -80 \\ -80 & 100 \end{pmatrix}$

$(F^{-1})_{11} = 100/3600 = 0.0278$

边缘化后的误差：$\sqrt{(F^{-1})_{11}} = \sqrt{0.0278} \approx 0.167$

**误差从 0.1 增大到了 0.167**——因为两个参数高度相关（$F_{12}/\sqrt{F_{11} F_{22}} = 80/100 = 0.8$），边缘化带来额外的不确定性。

---

## 3. Fisher 矩阵与高斯近似

**可靠程度：Level 1**

### 3.1 后验的高斯近似

将对数似然在 MLE 处的 Taylor 展开代入后验公式（假设平坦先验）：

$$P(\boldsymbol{\theta} | \mathbf{d}) \propto \mathcal{L}(\boldsymbol{\theta}) \approx \mathcal{L}(\hat{\boldsymbol{\theta}}) \exp\!\left(-\frac{1}{2} \sum_{i,j} F_{ij} (\theta_i - \hat{\theta}_i)(\theta_j - \hat{\theta}_j)\right)$$

这正是一个**多维高斯分布**，均值为 $\hat{\boldsymbol{\theta}}$，协方差矩阵为 $F^{-1}$：

$$P(\boldsymbol{\theta} | \mathbf{d}) \approx \mathcal{N}(\hat{\boldsymbol{\theta}},\, F^{-1})$$

- $\hat{\boldsymbol{\theta}}$：最大似然估计值（高斯的中心）
- $F^{-1}$：Fisher 矩阵的逆（高斯的协方差矩阵）

**这就是 Fisher 矩阵方法的本质**：用 MLE 附近的二阶近似把后验近似为高斯分布。

### 3.2 什么时候近似好？什么时候不好？

**高斯近似好的条件**：
- 数据量充足——根据中心极限定理，大量数据使似然趋向高斯（**Level 1**）
- 后验是单峰的、接近对称的
- 参数之间的退化接近线性

**高斯近似差的情况**：
- **多峰后验**：后验有多个极大值，高斯只能描述一个峰
- **强非高斯性**：如参数有硬边界（$\Omega_m > 0$），或后验高度不对称（如光学深度 $\tau$ 接近 0 时）
- **香蕉形退化**：两个参数的退化关系是弯曲的，椭圆近似失败

在这些情况下，必须用 MCMC 而不是 Fisher 矩阵。

---

## 4. 参数退化与等高线椭圆

**可靠程度：Level 1**

### 4.1 Fisher 矩阵的非对角元素

$F_{ij}$（$i \neq j$）描述参数 $\theta_i$ 和 $\theta_j$ 之间的**相关性**。

**相关系数**：

$$r_{ij} = \frac{(F^{-1})_{ij}}{\sqrt{(F^{-1})_{ii} (F^{-1})_{jj}}}$$

- $r_{ij}$：参数 $\theta_i$ 和 $\theta_j$ 的相关系数
- $|r_{ij}| = 1$：完全退化——两个参数的某个线性组合完全不受约束
- $r_{ij} = 0$：两个参数独立——分别约束，互不影响

### 4.2 等高线椭圆

2 参数的 Fisher 矩阵 $F^{-1}$ 决定了后验等高线的**形状、方向和大小**：

- **椭圆的大小**：由 $F^{-1}$ 的**特征值**决定——特征值越大，椭圆在该方向越长，约束越松
- **椭圆的方向**：由 $F^{-1}$ 的**特征向量**决定——特征向量是椭圆的主轴方向
- **椭圆面积** $\propto \sqrt{\det(F^{-1})} = 1/\sqrt{\det(F)}$

**1σ 等高线**的方程：

$$\Delta \boldsymbol{\theta}^T F \, \Delta \boldsymbol{\theta} = \Delta \chi^2$$

- $\Delta \boldsymbol{\theta} = \boldsymbol{\theta} - \hat{\boldsymbol{\theta}}$：参数偏离最佳值的向量
- $\Delta \chi^2 = 2.30$（68% CL，2 参数）或 $6.17$（95% CL，2 参数）

### 4.3 具体例子：2 参数等高线

取 Fisher 矩阵

$$F = \begin{pmatrix} 25 & 15 \\ 15 & 16 \end{pmatrix}$$

**Step 1**：求逆得协方差矩阵

$\det(F) = 25 \times 16 - 15 \times 15 = 400 - 225 = 175$

$$F^{-1} = \frac{1}{175} \begin{pmatrix} 16 & -15 \\ -15 & 25 \end{pmatrix} = \begin{pmatrix} 0.0914 & -0.0857 \\ -0.0857 & 0.1429 \end{pmatrix}$$

**Step 2**：读出参数误差

- $\sigma(\theta_1) = \sqrt{(F^{-1})_{11}} = \sqrt{0.0914} \approx 0.302$
- $\sigma(\theta_2) = \sqrt{(F^{-1})_{22}} = \sqrt{0.1429} \approx 0.378$

**Step 3**：读出相关系数

$$r_{12} = \frac{-0.0857}{\sqrt{0.0914 \times 0.1429}} = \frac{-0.0857}{0.1143} \approx -0.75$$

$r_{12} = -0.75$：$\theta_1$ 和 $\theta_2$ 有较强的**负相关**——增大 $\theta_1$ 可以通过减小 $\theta_2$ 来补偿，等高线椭圆的长轴沿着 $\theta_1 + \theta_2 \approx \text{const}$ 的方向。

**Step 4**：椭圆方向（特征值和特征向量）

$F^{-1}$ 的特征值 $\lambda_{1,2}$ 满足 $\det(F^{-1} - \lambda I) = 0$：

$(0.0914 - \lambda)(0.1429 - \lambda) - (-0.0857)^2 = 0$

$\lambda^2 - 0.2343\lambda + 0.005716 = 0$

$\lambda_1 \approx 0.208, \quad \lambda_2 \approx 0.027$

椭圆的**长半轴** $\propto \sqrt{\lambda_1} \approx 0.456$，**短半轴** $\propto \sqrt{\lambda_2} \approx 0.164$。长轴方向对应最松的约束方向（退化方向）。

---

## 5. 高斯似然下的 Fisher 矩阵

**可靠程度：Level 1**

在宇宙学中，最常见的场景是对**功率谱** $C_\ell$ 做参数约束。如果假设每个 $C_\ell$ 的测量误差是独立的高斯分布，Fisher 矩阵有一个简洁的形式。

### 5.1 一般公式

对于具有高斯似然的观测量 $\mathbf{d}$，理论预测 $\mathbf{m}(\boldsymbol{\theta})$，固定协方差矩阵 $\mathbf{C}$：

$$F_{ij} = \sum_{\alpha, \beta} \frac{\partial m_\alpha}{\partial \theta_i} \left(\mathbf{C}^{-1}\right)_{\alpha\beta} \frac{\partial m_\beta}{\partial \theta_j}$$

- $m_\alpha$：理论模型对第 $\alpha$ 个数据点的预测
- $\theta_i, \theta_j$：第 $i, j$ 个参数
- $(\mathbf{C}^{-1})_{\alpha\beta}$：数据协方差矩阵逆的 $(\alpha, \beta)$ 元素
- $\partial m_\alpha / \partial \theta_i$：理论预测对参数 $\theta_i$ 的偏导数（灵敏度）

### 5.2 功率谱的特殊情况

对于 CMB 功率谱 $C_\ell$，如果不同 $\ell$ 之间独立（对角协方差），Fisher 矩阵简化为：

$$F_{ij} = \sum_\ell \frac{1}{\sigma_{C_\ell}^2} \frac{\partial C_\ell}{\partial \theta_i} \frac{\partial C_\ell}{\partial \theta_j}$$

- $C_\ell$：角功率谱在多极矩 $\ell$ 处的值
- $\sigma_{C_\ell}$：$C_\ell$ 的测量误差（包括宇宙方差 + 噪声）
- $\partial C_\ell / \partial \theta_i$：$C_\ell$ 对参数 $\theta_i$ 的偏导数——描述"改变 $\theta_i$ 会如何影响功率谱"

每一项的物理含义：

- $\partial C_\ell / \partial \theta_i$：功率谱对参数 $\theta_i$ 的**灵敏度**——如果 $C_\ell$ 对 $\theta_i$ 不敏感（导数小），那 $\theta_i$ 就约束不了
- $1/\sigma_{C_\ell}^2$：第 $\ell$ 个数据点的**权重**——测量越精确（$\sigma$ 越小），权重越大，对 Fisher 矩阵贡献越大
- 求和 $\sum_\ell$：累加所有 $\ell$ 的贡献——更多数据点提供更紧的约束

**宇宙方差下的** $\sigma_{C_\ell}$：对于全天覆盖的 CMB 实验，$\sigma_{C_\ell} = C_\ell \sqrt{2/(2\ell+1)}$。这给出了 $C_\ell$ 测量精度的**不可消除下限**——宇宙只有一个，在多极矩 $\ell$ 上只有 $2\ell + 1$ 个独立模式可以测量。

---

## 6. Fisher 矩阵 vs MCMC

**可靠程度：Level 1（方法论）+ Level 3（实际使用建议）**

| 特性 | Fisher 矩阵 | MCMC |
|------|------------|------|
| **速度** | 快（分钟级）| 慢（小时到天） |
| **近似假设** | 后验是高斯 | 无（精确采样） |
| **计算需求** | 二阶导数 + 矩阵求逆 | 大量似然计算 |
| **适用阶段** | 实验设计 / 预测 | 实际数据分析 |
| **处理非高斯** | 不能 | 能 |
| **处理多峰** | 不能 | 能（取决于采样效率） |
| **参数退化** | 线性退化 | 任意非线性退化 |

### 两者何时一致？

当后验**接近高斯**时，Fisher 矩阵和 MCMC 给出几乎相同的结果。这通常发生在：
- 数据量充足（信噪比高）
- 参数没有接近边界
- 退化关系接近线性

### 典型使用场景

**Fisher 矩阵**：
- "CMB-S4 能把 $\sigma(\tau)$ 约束到多少？"——实验还没开始，用 Fisher 矩阵**预测**预期灵敏度
- "增加 $\ell > 3000$ 的数据能改善多少？"——快速对比不同实验设计

**MCMC**：
- Planck 数据发布了，用 MCMC 从实际数据中提取参数约束
- 后验可能有非高斯结构或强退化（如 $\Omega_m$-$H_0$ 退化）

---

### 公式速查卡

| 公式 | 含义 |
|------|------|
| $F_{ij} = -\langle \partial^2 \ln \mathcal{L} / \partial \theta_i \partial \theta_j \rangle$ | Fisher 矩阵定义：对数似然在 MLE 处的曲率 |
| $\sigma(\theta_i) \geq 1/\sqrt{F_{ii}}$（固定其他参数） | 单参数 Cramér-Rao 界 |
| $\sigma(\theta_i) \geq \sqrt{(F^{-1})_{ii}}$（边缘化其他参数） | 多参数 Cramér-Rao 界（实际使用） |
| $P(\boldsymbol{\theta}\|\mathbf{d}) \approx \mathcal{N}(\hat{\boldsymbol{\theta}}, F^{-1})$ | 后验的高斯近似 |
| $F_{ij} = \sum_\ell \frac{1}{\sigma_{C_\ell}^2} \frac{\partial C_\ell}{\partial \theta_i} \frac{\partial C_\ell}{\partial \theta_j}$ | 功率谱的 Fisher 矩阵（独立 $\ell$） |
| $r_{ij} = (F^{-1})_{ij} / \sqrt{(F^{-1})_{ii}(F^{-1})_{jj}}$ | 参数相关系数 |
| $\Delta \boldsymbol{\theta}^T F \, \Delta \boldsymbol{\theta} = \Delta\chi^2$ | 等高线椭圆方程 |

---

## 理解检测

**Q1**：Fisher 矩阵描述的是似然面的什么几何性质？如果 $F_{ii}$ 很大，意味着似然面在 $\theta_i$ 方向是陡还是平？对应参数约束紧还是松？

你的回答：


**Q2**：$1/\sqrt{F_{ii}}$ 和 $\sqrt{(F^{-1})_{ii}}$ 分别表示什么？为什么后者总是 $\geq$ 前者？在参数高度相关时，差别会大还是小？

你的回答：


**Q3（计算题）**：已知 2 参数 Fisher 矩阵为

$$F = \begin{pmatrix} 400 & 0 \\ 0 & 100 \end{pmatrix}$$

(a) 求 $\sigma(\theta_1)$ 和 $\sigma(\theta_2)$。
(b) 两个参数的相关系数 $r_{12}$ 是多少？等高线椭圆的长轴沿什么方向？

> 提示：用 $\sigma(\theta_i) = \sqrt{(F^{-1})_{ii}}$ 公式。对角矩阵的逆还是对角矩阵

你的回答：


**Q4（计算题）**：CMB 功率谱 $C_\ell$ 在某个 $\ell$ 处的值为 $C_\ell = 1000 \, \mu\text{K}^2$，假设只有宇宙方差，$\sigma_{C_\ell} = C_\ell \sqrt{2/(2\ell+1)}$。当 $\ell = 2$ 和 $\ell = 1000$ 时，$\sigma_{C_\ell}/C_\ell$ 分别是多少？哪个 $\ell$ 的测量相对精度更高？

> 提示：$\sigma_{C_\ell}/C_\ell = \sqrt{2/(2\ell+1)}$，代入 $\ell = 2$ 和 $\ell = 1000$ 计算

你的回答：


**Q5**：Fisher 矩阵和 MCMC 各自适合什么场景？给出一个必须用 MCMC 而不能用 Fisher 矩阵的具体例子，并解释为什么。

你的回答：
