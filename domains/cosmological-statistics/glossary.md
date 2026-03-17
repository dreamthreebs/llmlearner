# 宇宙学统计方法 术语表

| 缩写/术语 | 全称 | 简述 |
|-----------|------|------|
| AIC | Akaike Information Criterion | 赤池信息准则，平衡拟合优度和模型复杂度的模型选择准则 |
| Bayes factor | Bayes factor / 贝叶斯因子 | 两个模型的贝叶斯证据之比，用于模型比较 |
| BIC | Bayesian Information Criterion | 贝叶斯信息准则，比 AIC 对参数数量惩罚更重 |
| CL | Confidence Level | 置信水平，如 68% CL、95% CL |
| Cobaya | Cobaya | 宇宙学贝叶斯推断框架（CosmoMC 的 Python 继承者） |
| CosmoMC | Cosmological Monte Carlo | 宇宙学 MCMC 标准工具 |
| Cramér-Rao bound | Cramér-Rao bound | 参数估计方差的理论下限，等于 Fisher 矩阵逆的对角元素 |
| Credible interval | Credible interval / 可信区间 | 贝叶斯学派的区间估计："参数有 X% 的概率在此区间内" |
| emcee | emcee | Python 的 Affine-invariant MCMC 库 |
| Evidence | Bayesian Evidence / 贝叶斯证据 | 数据在某模型下的总概率，用于模型选择 |
| Fisher matrix | Fisher Information Matrix | 似然函数在最大值处的曲率矩阵，用于快速参数预测 |
| GetDist | GetDist | 后验分布可视化工具（等高线图） |
| Likelihood | Likelihood / 似然函数 | $P(\mathbf{d} \| \boldsymbol{\theta})$，给定参数时数据出现的概率 |
| Marginalization | Marginalization / 边缘化 | 对不感兴趣的参数积分，得到感兴趣参数的分布 |
| MCMC | Markov Chain Monte Carlo | 马尔可夫链蒙特卡洛，通过随机采样近似后验分布 |
| MLE | Maximum Likelihood Estimation | 最大似然估计，选择使似然最大的参数值 |
| Nuisance parameter | Nuisance parameter / 干扰参数 | 模型中需要但不感兴趣的参数（如校准常数） |
| Posterior | Posterior / 后验分布 | $P(\boldsymbol{\theta} \| \mathbf{d})$，看到数据后参数的概率分布 |
| Prior | Prior / 先验分布 | $\pi(\boldsymbol{\theta})$，观测前对参数的信念 |
| PTE | Probability To Exceed | 拟合优度指标：模型正确时得到更差拟合的概率，即 $\chi^2$ 分布的上尾概率 |
| BAO | Baryon Acoustic Oscillations | 重子声学振荡，标准尺方法测量宇宙距离-红移关系 |
| Boltzmann solver | Boltzmann solver / 玻尔兹曼求解器 | 数值求解线性微扰方程，输入宇宙学参数输出理论功率谱（如 CAMB/CLASS） |
| CAMB | Code for Anisotropies in the Microwave Background | Fortran 编写的 Boltzmann 求解器，CosmoMC 默认使用 |
| CIB | Cosmic Infrared Background | 宇宙红外背景，CMB 分析中的前景污染源 |
| CLASS | Cosmic Linear Anisotropy Solving System | C 编写的 Boltzmann 求解器，MontePython 默认使用 |
| Contour plot | Contour plot / 等高线图 | 二维联合后验分布的等概率密度线，内圈 68%、外圈 95% |
| Cosmic variance | Cosmic variance / 宇宙方差 | 因独立模式数有限导致的 $\hat{C}_\ell$ 最小不确定性 |
| $f_\text{sky}$ | Sky fraction | 天空覆盖率，不完全天空覆盖增大功率谱方差 |
| Hartlap factor | Hartlap correction factor | 校正用有限模拟估计的逆协方差矩阵的系统偏差 |
| Offset log-normal | Offset log-normal likelihood | 小 $\ell$ 下功率谱似然的近似方案，比高斯更准确 |
| Profiling | Profile likelihood | 频率学派消除 nuisance 参数的方法，对 nuisance 参数取极大值 |
| SH0ES | Supernova $H_0$ for the Equation of State | 用造父变星 + Ia 超新星距离阶梯测量 $H_0$ 的项目 |
| SN Ia | Type Ia Supernova | Ia 型超新星，标准烛光方法测量光度距离 |
| SZ | Sunyaev-Zel'dovich effect | 苏尼亚耶夫-泽尔多维奇效应，CMB 光子被热电子散射 |
| Tension | Tension / 张力 | 两个独立实验对同一参数的不一致，用 $N\sigma$ 量化 |
| Triangle plot | Triangle plot / 三角图 | 对角线为一维后验、下三角为二维等高线的标准可视化 |
| Volume effect | Volume effect / 体积效应 | 高维空间边缘化导致后验峰值偏离最大似然点的现象 |
| Wishart distribution | Wishart distribution | 多元正态协方差矩阵的采样分布，CMB $\hat{C}_\ell$ 的精确分布 |
| CLT | Central Limit Theorem / 中心极限定理 | 大量独立样本的均值趋向高斯分布，误差按 $1/\sqrt{n}$ 缩小 |
| Confidence interval | Confidence interval / 置信区间 | 频率学派的区间估计："重复实验无穷次，X% 的区间包含真值" |
| Covariance matrix | Covariance matrix / 协方差矩阵 | $C_{ij} = \text{Cov}(x_i, x_j)$，描述多个变量之间的关联程度 |
| dof | Degrees of freedom / 自由度 | $n_{\text{data}} - n_{\text{param}}$，用于评估拟合优度 |
| Equal-tail interval | Equal-tail interval / 等尾区间 | 后验分布上下尾概率相等的可信区间 |
| Flat prior | Flat prior / 均匀先验 | 在给定范围内所有值概率相等的先验分布 |
| Gaussian distribution | Gaussian distribution / 高斯分布 | $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$，最常用的概率分布 |
| HPD | Highest Posterior Density interval | 最高后验密度区间，包含后验密度最高的区域 |
| Jeffreys prior | Jeffreys prior | $\pi(\theta) \propto \sqrt{|F(\theta)|}$，参数化不变的无信息先验 |
| Log-likelihood | Log-likelihood / 对数似然 | $\ln \mathcal{L}$，似然的对数形式，计算更稳定 |
| MAP | Maximum A Posteriori | 最大后验估计，$\hat{\theta}_{\text{MAP}} = \arg\max [\ln \mathcal{L} + \ln \pi]$ |
| Multivariate Gaussian | Multivariate Gaussian / 多元高斯分布 | 多维高斯分布，由均值向量和协方差矩阵完全描述 |
| PDF | Probability Density Function | 概率密度函数 |
| Poisson distribution | Poisson distribution / 泊松分布 | 计数数据的分布，期望等于方差，常用于光子计数 |
| Precision matrix | Precision matrix / 精度矩阵 | 协方差矩阵的逆 $\mathbf{C}^{-1}$ |
| Profile likelihood | Profile likelihood | 对 nuisance 参数取最大值（而非积分）得到的似然 |
| Burn-in | Burn-in / 预热期 | MCMC 链到达稳态分布之前的过渡阶段，采样点需丢弃 |
| $\chi^2$ | Chi-squared | 卡方统计量，衡量数据与模型的偏离程度 |
| $\chi^2_\text{red}$ | Reduced chi-squared / 约化卡方 | $\chi^2_\text{min} / \nu$，$\approx 1$ 表示好的拟合 |
| Degeneracy | Parameter degeneracy / 参数退化 | 多个参数的某种组合不受数据约束，等高线椭圆倾斜拉伸 |
| Detailed balance | Detailed balance / 细致平衡 | MCMC 的充分条件：任意两点之间的来回概率流相等 |
| $\Delta\chi^2$ | Delta chi-squared | 相对最小 $\chi^2$ 的差值，用于确定置信/可信区间 |
| $\hat{R}$ | Gelman-Rubin statistic | MCMC 收敛诊断：链间方差与链内方差之比，$< 1.1$ 表示收敛 |
| Jeffreys scale | Jeffreys scale / Jeffreys 判据 | 用 $\|\ln B_{12}\|$ 量化贝叶斯因子强度的经验尺度 |
| LRT | Likelihood Ratio Test | 似然比检验，频率学派比较嵌套模型的方法 |
| Markov chain | Markov chain / 马尔可夫链 | 下一步只依赖当前位置的随机游走序列 |
| Metropolis-Hastings | Metropolis-Hastings algorithm | 最基本的 MCMC 算法，通过提议+接受/拒绝构建马尔可夫链 |
| MultiNest | MultiNest | 嵌套采样算法的实现，用于计算贝叶斯证据 |
| $N_\text{eff}$ | Effective sample size / 有效样本数 | $N / \tau_\text{corr}$，等效的独立采样点数 |
| Nested models | Nested models / 嵌套模型 | 一个模型是另一个模型的特殊情况 |
| Nested sampling | Nested sampling / 嵌套采样 | 专为计算贝叶斯证据设计的采样算法 |
| Occam's Razor | Occam's Razor / 奥卡姆剃刀 | 同等拟合下优先选择更简单的模型 |
| PolyChord | PolyChord | 高维嵌套采样算法 |
| Proposal distribution | Proposal distribution / 提议分布 | MCMC 中用于生成候选点的分布 |
| Stationary distribution | Stationary distribution / 稳态分布 | 马尔可夫链走足够久后趋向的平衡分布 |
| $\tau_\text{corr}$ | Autocorrelation time / 自相关时间 | 衡量 MCMC 链中相邻采样点的相关性 |
| Thinning | Thinning / 稀疏化 | 每隔 $k$ 步保留一个 MCMC 采样点以降低自相关 |
| Wilks' theorem | Wilks' theorem / Wilks 定理 | 嵌套模型的似然比统计量渐近服从 $\chi^2_{\Delta k}$ 分布 |
