# 量化投资术语表

| 缩写/术语 | 全称/原语 | 简述 |
|-----------|----------|------|
| ADF | Augmented Dickey-Fuller test | 平稳性检验，判断时间序列是否有单位根 |
| alpha ($\alpha$) | 阿尔法 | 因子模型无法解释的超额收益；量化策略追求的目标 |
| alternative data | 另类数据 | 传统金融数据之外的数据源（卫星图像、NLP情感等） |
| AR | AutoRegressive model / 自回归模型 | 当前值取决于过去值：$y_t = c + \phi y_{t-1} + \epsilon_t$ |
| ARCH | AutoRegressive Conditional Heteroskedasticity | 条件异方差模型，Engle (1982) |
| ARMA | AutoRegressive Moving Average | AR + MA 的组合模型 |
| backtest | 回测 | 用历史数据模拟策略的表现 |
| Brownian motion | 布朗运动 / Wiener process | 连续时间随机过程，随机游走的连续版本 |
| Calmar ratio | 卡尔玛比率 | 年化收益率 / 最大回撤 |
| cointegration | 协整 | 两个非平稳序列的线性组合是平稳的 |
| CTA | Commodity Trading Advisor | 管理期货策略，常用趋势跟踪 |
| CVaR | Conditional VaR / Expected Shortfall | 超过 VaR 部分的条件期望损失 |
| Donchian Channel | 唐奇安通道 | 价格突破 N 日最高/最低价时交易 |
| EMH | Efficient Market Hypothesis / 有效市场假说 | 价格已反映所有信息 |
| Engle-Granger | 恩格尔-格兰杰两步法 | 协整检验方法：OLS回归→检验残差平稳性 |
| fat tail | 厚尾 | 极端事件发生的概率比正态分布预测的高得多 |
| GARCH | Generalized ARCH | $\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ |
| GBM | Geometric Brownian Motion / 几何布朗运动 | $dS = \mu S dt + \sigma S dW$，股价常用模型 |
| Kelly criterion | 凯利准则 | 最优下注比例：$f^* = (p(b+1)-1)/b$ |
| Level 2 data | 盘口深度数据 | 买卖双方各价位的挂单数量 |
| limit order | 限价单 | 指定价格的交易委托 |
| look-ahead bias | 前视偏差 | 回测中使用了当时不可能获得的未来信息 |
| MA (model) | Moving Average model / 移动平均模型 | $y_t = c + \epsilon_t + \theta\epsilon_{t-1}$（不是均线！） |
| MA (indicator) | Moving Average / 移动平均线/均线 | 过去 N 天收盘价的平均值 |
| market impact | 市场冲击 | 大单交易本身推动价格的成本 |
| market maker | 做市商 | 同时报出买价和卖价的交易商 |
| market order | 市价单 | 以当前最优价格立即成交的委托 |
| MDD | Maximum Drawdown / 最大回撤 | 从峰值到谷底的最大跌幅 |
| mean reversion | 均值回复 | 价格偏离均值后倾向回归 |
| momentum | 动量 | 过去涨的继续涨，跌的继续跌 |
| OHLCV | Open/High/Low/Close/Volume | 开盘/最高/最低/收盘价+成交量 |
| OOS | Out-of-Sample / 样本外 | 未参与策略开发的数据期间 |
| OU process | Ornstein-Uhlenbeck process | 均值回复随机过程 |
| overfitting | 过拟合 | 策略拟合了历史噪声而非真实信号 |
| pairs trading | 配对交易 | 找两只协整股票，价差偏离时交易 |
| PCA | Principal Component Analysis / 主成分分析 | 降维方法，提取统计因子 |
| Purged K-Fold | 清洗K折交叉验证 | 时间序列专用交叉验证，防止数据泄露 |
| random walk | 随机游走 | $P_t = P_{t-1} + \epsilon_t$，价格变化不可预测 |
| risk parity | 风险平价 | 按风险贡献等比例配置资产 |
| Sharpe ratio | 夏普比率 | $(E[r]-r_f)/\sigma$，每单位风险的超额收益 |
| slippage | 滑点 | 预期成交价与实际成交价的差异 |
| Sortino ratio | 索提诺比率 | 用下行波动率替换总波动率的夏普比率 |
| spread | 价差 | 两个相关资产的价格差或价格比 |
| stat arb | Statistical Arbitrage / 统计套利 | 利用统计关系获利，非无风险 |
| stationarity | 平稳性 | 统计性质不随时间变化 |
| stop-loss | 止损 | 亏损达到阈值时强制平仓 |
| survivorship bias | 幸存者偏差 | 只考察存活资产导致的偏差 |
| take-profit | 止盈 | 盈利达到阈值时主动平仓 |
| tick data | 逐笔成交数据 | 每笔交易的价格和数量 |
| TSMOM | Time-Series Momentum | 单一资产自己过去的趋势动量 |
| TWAP | Time-Weighted Average Price | 将大单拆分为等时间间隔小单执行 |
| VaR | Value at Risk / 在险价值 | 给定置信度下的最大可能亏损 |
| volatility clustering | 波动率聚类 | 大波动后跟大波动，小波动后跟小波动 |
| VWAP | Volume-Weighted Average Price | 按成交量加权的平均价格，执行基准 |
| Walk-Forward | 滚动前推分析 | 滚动窗口训练+样本外测试 |
| XSMOM | Cross-Sectional Momentum | 截面动量，不同资产之间排名 |
| z-score | 标准分数 | $(x-\mu)/\sigma$，衡量偏离均值的程度 |
