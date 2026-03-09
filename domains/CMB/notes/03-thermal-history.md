# 03 - 宇宙热历史

> **主维度**：D1 物理原理
> **关键关系**：
> - 尺度因子 a(t) (概念) --用于--> 热历史 (概念)：尺度因子用于描述热历史的温度演化 T ∝ 1/a
> - 复合与退耦 (概念) --属于--> 热历史 (概念)：复合与退耦属于热历史
> - 热历史 (概念) --依赖--> 膨胀宇宙 (理论)：热历史依赖膨胀框架才能理解
>
> 学习路径：Step 2（理解宇宙从极热到冷却的演化过程）
> 前置知识：02 膨胀宇宙（尺度因子、Friedmann 方程、辐射/物质密度的演化）
> 参考：[Wikipedia - Chronology of the universe](https://en.wikipedia.org/wiki/Chronology_of_the_universe) · [Kolb & Turner《The Early Universe》Ch.3](https://www.amazon.com/Early-Universe-Frontiers-Physics/dp/0201626748) · [Dodelson《Modern Cosmology》Ch.2-3](https://www.amazon.com/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

## 核心问题

宇宙从大爆炸到今天经历了什么？不同粒子什么时候产生、什么时候不再互相作用？这个演化过程怎么决定了我们今天看到的宇宙成分？

## 核心思想：温度决定一切

上一章我们知道宇宙在膨胀，辐射密度 $\rho_{\text{rad}} \propto a^{-4}$。对于热辐射（黑体辐射），能量密度和温度的关系你在热统里学过：

$$\rho_{\text{rad}} \propto T^4$$

结合 $\rho_{\text{rad}} \propto a^{-4}$，立刻得到：

$$T \propto a^{-1} = 1 + z$$

**宇宙的温度和尺度因子成反比。膨胀越多，温度越低。** 今天 $T_0 = 2.725\,\text{K}$，在红移 $z$ 时温度是 $T = T_0 (1+z)$。

这意味着往回看（$a \to 0$），温度 $T \to \infty$。宇宙早期的温度极高，高到足以产生各种粒子-反粒子对。

> 参考：[Wikipedia - Cosmic microwave background § Temperature](https://en.wikipedia.org/wiki/Cosmic_microwave_background#Features)

## 热平衡与退耦：一个通用机制

早期宇宙中，各种粒子通过相互作用（碰撞、散射、湮灭-产生）保持**热平衡**——你在热统里学过，平衡态意味着详细平衡成立，所有粒子共享同一温度。

但保持热平衡有一个条件：**反应速率 $\Gamma$ 必须比宇宙膨胀速率 $H$ 快。**

$$\Gamma \gg H \quad \Rightarrow \quad \text{粒子保持热平衡}$$
$$\Gamma \ll H \quad \Rightarrow \quad \text{粒子退耦（decoupling），不再相互作用}$$

物理图像：粒子 A 和粒子 B 通过某种反应保持联系。如果反应足够频繁（$\Gamma \gg H$），它们一直保持平衡。但宇宙在膨胀，粒子密度在下降，反应速率 $\Gamma$（通常正比于密度和截面）也在下降。当 $\Gamma$ 降到和 $H$ 差不多时，粒子之间来不及再反应，就"各走各路"了——这就是**退耦（decoupling）**。

> **类比**：一个派对上大家在聊天（热平衡）。如果房间不断变大（膨胀），人越来越稀疏，到某个时刻大家互相喊话已经听不见了，就各自冻结在当时的状态（退耦）。

这个 $\Gamma \sim H$ 的判据在整个宇宙热历史中反复出现——中微子退耦、核合成、复合，全都是同一个机制。

> 参考：[Wikipedia - Decoupling (cosmology)](https://en.wikipedia.org/wiki/Decoupling_(cosmology)) · [Dodelson Ch.2 §2.3](https://www.amazon.com/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

## 宇宙的时间线

从大爆炸到今天，按时间和温度列出关键事件：

| 时间 | 温度 | 红移 $z$ | 事件 | 和 CMB 的关系 |
|------|------|---------|------|-------------|
| $\sim 10^{-36}$ s | $\sim 10^{28}$ K | — | **暴胀（Inflation）** | 产生原初扰动——CMB 涨落的种子（Level 2-3） |
| $\sim 10^{-6}$ s | $\sim 10^{12}$ K | — | **夸克-强子转变** | 夸克被束缚成质子和中子 |
| $\sim 1$ s | $\sim 10^{10}$ K | $\sim 10^{10}$ | **中微子退耦** | 中微子不再和其他物质互相作用，成为"宇宙中微子背景"（CνB） |
| $\sim 1$ s | $\sim 10^{10}$ K | $\sim 10^{10}$ | **电子-正电子湮灭** | $e^+e^-$ 对湮灭成光子，加热了光子气体（所以中微子温度比光子低） |
| 1-3 min | $\sim 10^{9}$ K | $\sim 10^{9}$ | **大爆炸核合成（BBN）** | 质子+中子合成 He、D、Li 等轻元素。BBN 预测的 He 丰度和观测吻合，是大爆炸理论的独立验证 |
| $\sim 6万$ 年 | $\sim 9000$ K | $\sim 3400$ | **物质-辐射等量** | $\rho_{\text{matter}} = \rho_{\text{rad}}$，宇宙从辐射主导进入物质主导。影响 CMB 功率谱的峰高 |
| **$\sim 38万$ 年** | **$\sim 3000$ K** | **$\sim 1100$** | **复合与退耦（→ CMB 释放）** | **光子退耦，自由飞行至今。这就是 CMB。** |
| $\sim 1-10亿$ 年 | — | $\sim 6-20$ | **再电离（Reionization）** | 第一批恒星/星系的紫外辐射重新电离氢。CMB 光子在穿越时被部分散射，影响大尺度偏振 |
| $\sim 138亿$ 年 | 2.725 K | 0 | **今天** | 我们观测 CMB |

> 参考：[Wikipedia - Chronology of the universe](https://en.wikipedia.org/wiki/Chronology_of_the_universe) · [图解版时间线](https://en.wikipedia.org/wiki/Chronology_of_the_universe#/media/File:History_of_the_Universe.svg)

## 几个重要事件详解

### 中微子退耦（$T \sim 1\,\text{MeV}$，$t \sim 1$ s）

中微子通过弱相互作用与电子、正电子保持热平衡。弱相互作用截面 $\sigma \propto G_F^2 T^2$（$G_F$ 是 Fermi 常数），反应速率 $\Gamma \propto G_F^2 T^5$。而 Hubble 参数在辐射主导时代 $H \propto T^2$。

$\Gamma/H \propto T^3$，所以温度下降时 $\Gamma$ 比 $H$ 掉得更快。当 $T$ 降到 $\sim 1\,\text{MeV}$（$\sim 10^{10}\,\text{K}$），$\Gamma \sim H$，中微子退耦。

退耦后，中微子温度独立演化。之后 $e^+e^-$ 湮灭只加热了光子而没加热中微子，所以今天中微子温度 $T_\nu = (4/11)^{1/3} T_\gamma \approx 1.95\,\text{K}$。

> 参考：[Wikipedia - Cosmic neutrino background](https://en.wikipedia.org/wiki/Cosmic_neutrino_background) · [Dodelson Ch.2 §2.4](https://www.amazon.com/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

### 大爆炸核合成（BBN，$T \sim 0.1\,\text{MeV}$，$t \sim$ 几分钟）

温度降到 $\sim 0.1\,\text{MeV}$ 时，质子和中子可以结合成氘（D），然后迅速反应成 $^4\text{He}$。

BBN 的预测只依赖一个自由参数：重子-光子数密度比 $\eta = n_b/n_\gamma \approx 6 \times 10^{-10}$。这个 $\eta$ 决定了：
- $^4\text{He}$ 的质量丰度 $\sim 24\%$
- D 的丰度 $\sim 2.5 \times 10^{-5}$

这些预测和天文观测精确吻合。而且——**CMB 功率谱独立测出的重子密度 $\Omega_b h^2$ 和 BBN 要求的 $\eta$ 完全一致**。这是两个完全不同的物理过程（核反应 vs 光子散射），在完全不同的时代（3分钟 vs 38万年），给出了同一个答案。这是大爆炸理论最有力的验证之一。

> 参考：[Wikipedia - Big Bang nucleosynthesis](https://en.wikipedia.org/wiki/Big_Bang_nucleosynthesis) · [BBN 综述 - Fields et al.](https://arxiv.org/abs/1505.01076)

### 物质-辐射等量（$z \sim 3400$）

在 02 中我们知道 $\rho_{\text{rad}} \propto a^{-4}$，$\rho_{\text{matter}} \propto a^{-3}$。两者相等的时刻 $a_{\text{eq}}$ 满足：

$$\rho_{\text{rad}}(a_{\text{eq}}) = \rho_{\text{matter}}(a_{\text{eq}})$$

这发生在 $z_{\text{eq}} \approx 3400$（CMB 释放之前不久）。

这个时刻对 CMB 很重要：在物质-辐射等量前后，引力势 $\Phi$ 的演化行为不同（辐射主导时势会衰减，物质主导时势不变），这直接影响声波振荡的驱动方式和功率谱峰的高度。

> 参考：[Wayne Hu - Radiation driving](https://background.uchicago.edu/~whu/intermediate/driving2.html)

## 和 CMB 的联系

热历史中和 CMB 直接相关的几个关键点：

1. **复合/退耦**（下一章详讲）：CMB 的产生时刻
2. **物质-辐射等量**：影响功率谱峰的高度比
3. **BBN**：独立验证了 CMB 测出的重子密度
4. **暴胀**：产生了 CMB 涨落的种子（原初扰动）
5. **再电离**：影响 CMB 的大尺度偏振信号

---

## 理解检测

请直接在下面写回答，保存文件后告诉我。

**Q1**：为什么宇宙越早温度越高？用 02 中学到的 $\rho_{\text{rad}} \propto a^{-4}$ 和 $\rho_{\text{rad}} \propto T^4$ 推导一下 $T$ 和 $a$ 的关系。

你的回答：



**Q2**：粒子退耦的条件是什么？用你自己的话解释一下 $\Gamma \sim H$ 这个判据的物理含义——为什么是和膨胀速率比，而不是和别的什么量比？

你的回答：



**Q3**：BBN 发生在大爆炸后 ~3 分钟，CMB 释放在 ~38 万年后。但两者独立测出了同一个重子密度。为什么这件事这么重要？（提示：想想如果大爆炸理论是错的，这两个测量有没有理由一致。）

你的回答：



