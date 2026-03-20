# 多因子模型术语表

| 缩写/术语 | 全称/原语 | 简述 |
|-----------|----------|------|
| alpha ($\alpha$) | 阿尔法 | 因子模型无法解释的超额收益，代表真正的投资技能 |
| alternative data | 另类数据 | 传统财务数据之外的数据源（卫星图像、NLP情感等） |
| anomaly | 异象 | CAPM 无法解释的系统性收益规律 |
| APT | Arbitrage Pricing Theory / 套利定价理论 | Ross (1976)，基于无套利条件的多因子定价理论 |
| backtest | 回测 | 用历史数据检验投资策略的表现 |
| beta ($\beta$) | 贝塔 | 资产对某个因子的暴露/敏感度 |
| BM ratio | Book-to-Market ratio / 账面市值比 | 公司账面价值与市值的比率，价值因子的核心指标 |
| CAPM | Capital Asset Pricing Model / 资本资产定价模型 | Sharpe (1964)，单因子定价模型 |
| CMA | Conservative Minus Aggressive | FF5投资因子——保守投资减激进投资 |
| covariance | 协方差 | 两个变量共同变动的程度 |
| cross-section | 截面 | 同一时间点上不同资产之间的比较 |
| data snooping | 数据窥探 | 对同一数据集反复检验导致的虚假发现 |
| diversification | 分散化 | 通过持有多种资产降低组合风险 |
| drawdown | 回撤 | 从峰值到谷底的价值下降幅度 |
| efficient frontier | 有效前沿 | 给定风险下收益率最高（或给定收益率下风险最低）的组合集合 |
| ESG | Environmental, Social, Governance | 环境、社会和治理因素 |
| excess return | 超额收益 | 收益率减去无风险利率 |
| exposure | 暴露 | 资产对某个因子的敏感程度（同 beta） |
| factor | 因子 | 系统性解释大量资产收益率差异的变量 |
| factor crowding | 因子拥挤 | 过多投资者追逐同一因子导致溢价下降 |
| factor premium | 因子溢价 | 承担某种因子风险所获得的额外收益 |
| factor zoo | 因子动物园 | 学术界发现的数百个因子（大部分可能是假的） |
| Fama-French | 法玛-弗伦奇 | 三因子/五因子模型的提出者 |
| Fama-MacBeth | 法玛-麦克贝斯 | 两步回归法——先时序回归估beta，再截面回归检验因子溢价 |
| FF3 | Fama-French 三因子模型 | 市场+SMB+HML |
| FF5 | Fama-French 五因子模型 | 市场+SMB+HML+RMW+CMA |
| GP/A | Gross Profitability / Assets | 毛利润率，质量因子指标（Novy-Marx 2013） |
| HML | High Minus Low | 高账面市值比减低账面市值比——价值因子 |
| idiosyncratic risk | 特异性风险 | 只影响个别资产的风险，可通过分散化消除 |
| log return | 对数收益率 | $\ln(P_t/P_{t-1})$，具有时间可加性 |
| long-short portfolio | 多空组合 | 同时做多（买入）一组和做空（卖出）另一组资产 |
| look-ahead bias | 前视偏差 | 在回测中使用了当时不可能获得的未来信息 |
| market portfolio | 市场组合 | 所有可投资资产的加权组合（实操中用指数代替） |
| Markowitz | 马科维茨 | Harry Markowitz，均值-方差投资组合理论创始人 |
| MDD | Maximum Drawdown / 最大回撤 | 历史上从峰值到谷底的最大下降幅度 |
| momentum | 动量 | 过去表现好的资产继续表现好的趋势 |
| P/E ratio | Price-to-Earnings ratio / 市盈率 | 股价与每股收益的比率 |
| p-hacking | p 值操纵 | 通过反复尝试不同规格使结果达到统计显著 |
| $r_f$ | 无风险利率 | 不承担任何风险的收益率（通常用短期国债利率） |
| $r_m$ | 市场收益率 | 市场组合的收益率 |
| reversal | 反转 | 过去表现好的资产未来表现差（与动量相反） |
| risk parity | 风险平价 | 按风险贡献（而非资金量）等比例配置资产 |
| risk premium | 风险溢价 | 承担风险获得的额外收益 |
| RMW | Robust Minus Weak | FF5盈利因子——高盈利减低盈利 |
| ROE | Return on Equity / 净资产收益率 | 净利润除以股东权益 |
| Sharpe ratio | 夏普比率 | 每单位风险获得的超额收益：$(E[r]-r_f)/\sigma$ |
| short selling | 卖空/做空 | 借入资产卖出，等价格下跌后买回归还 |
| simple return | 简单收益率 | $(P_t - P_{t-1})/P_{t-1}$ |
| SML | Security Market Line / 证券市场线 | CAPM下预期收益率与beta的线性关系 |
| SMB | Small Minus Big | 小市值减大市值——规模因子 |
| Smart Beta | 智能贝塔 | 因子投资的产品化——通过规则化的方式获取因子溢价 |
| survivorship bias | 幸存者偏差 | 只考察存活资产而忽略退市资产导致的偏差 |
| systematic risk | 系统性风险 | 影响所有资产的风险（市场风险），不可分散 |
| time series regression | 时间序列回归 | 对单一资产在时间维度上做回归 |
| turnover | 换手率 | 交易量与总股本的比率 |
| UMD | Up Minus Down | 过去赢家减过去输家——动量因子 |
| value | 价值 | "便宜的"股票（高BM、低PE）的系统性溢价 |
| VaR | Value at Risk / 在险价值 | 给定置信水平下的最大可能损失 |
| volatility | 波动率 | 收益率的标准差，最常用的风险度量 |
