# 02 - 膨胀宇宙

> **主维度**：D1 物理原理
> **关键关系**：
> - Friedmann 方程 (理论) --推广了--> 牛顿能量守恒 (理论)：Friedmann 推广了牛顿能量守恒到膨胀宇宙
> - Friedmann 方程 (理论) --用于--> 尺度因子 a(t) (概念)：Friedmann 方程用于求解尺度因子 a(t)
> - 尺度因子 a(t) (概念) --用于--> 热历史 (概念)：尺度因子用于描述宇宙热历史的演化
>
> 学习路径：Step 1（最基础的框架，后面所有概念都依赖它）
> 前置知识：牛顿力学（能量守恒）
> 参考：[Wikipedia - Friedmann equations](https://en.wikipedia.org/wiki/Friedmann_equations) · [Dodelson《Modern Cosmology》Ch.1](https://www.amazon.com/Modern-Cosmology-Scott-Dodelson/dp/0128159480) · [Wikipedia - FLRW metric](https://en.wikipedia.org/wiki/Friedmann%E2%80%93Lema%C3%AEtre%E2%80%93Robertson%E2%80%93Walker_metric)

## 核心问题

宇宙在膨胀——这是观测事实（[哈勃定律](https://en.wikipedia.org/wiki/Hubble%27s_law)：越远的星系退行越快）。怎么用一个方程描述膨胀有多快？

## 关键概念：尺度因子 $a(t)$

宇宙中任意两个"跟着膨胀走"的点之间的物理距离：

$$d(t) = a(t) \cdot x$$

- $x$：**共动坐标**（固定不变，可以理解为物体在膨胀网格上的"编号"）
- $a(t)$：**尺度因子**（随时间增长，描述网格整体的缩放）

**类比**：在一块正在膨胀的面团上画两个点，两点在面团上的"格子编号"没变（共动坐标不变），但实际距离随面团膨胀而增大。$a(t)$ 就是面团的缩放比例。

**约定**：今天 $a(t_0) = 1$。宇宙早期 $a$ 很小，比如 CMB 释放时 $a \approx 1/1100$，意味着那时候宇宙的线性尺度只有现在的 $1/1100$。

**红移 $z$**：观测上常用红移代替 $a$，定义为 $1 + z = 1/a$。CMB 的红移 $z \approx 1100$。

> 参考：[Wikipedia - Scale factor](https://en.wikipedia.org/wiki/Scale_factor_(cosmology)) · [Wikipedia - Redshift](https://en.wikipedia.org/wiki/Redshift)

## Friedmann 方程的推导（牛顿版本）

严格来说这需要广义相对论，但牛顿力学的能量守恒就可以给出正确的形式（这不是巧合，可以证明在均匀球对称情况下牛顿推导给出和 GR 完全一样的结果）。

取宇宙中一个半径为 $r$ 的均匀球，球壳上有一个质量为 $m$ 的粒子：

- 动能：$\frac{1}{2}m\dot{r}^2$
- 引力势能：$-\frac{G M m}{r}$，其中 $M = \frac{4}{3}\pi r^3 \rho$ 是球内总质量
- （球外均匀分布的物质对球壳上粒子没有净引力——这是牛顿壳定理，你在力学课上学过）

能量守恒：

$$\frac{1}{2}m\dot{r}^2 - \frac{4\pi G}{3}\rho \cdot m r^2 = E$$

代入 $r = a(t) \cdot x$（$x$ 固定），所以 $\dot{r} = \dot{a} \cdot x$：

$$\frac{1}{2}m \dot{a}^2 x^2 - \frac{4\pi G}{3}\rho \cdot m a^2 x^2 = E$$

两边除以 $\frac{1}{2}m a^2 x^2$：

$$\left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G}{3}\rho + \frac{2E}{m a^2 x^2}$$

定义 **Hubble 参数** $H \equiv \dot{a}/a$，把右边第二项记为 $-k/a^2$（$k$ 和空间曲率对应，在 GR 中有严格含义），就得到了 **Friedmann 方程**：

$$\boxed{H^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2}}$$

> 参考：[Friedmann 方程的牛顿推导](https://en.wikipedia.org/wiki/Friedmann_equations#Newtonian_derivation) · [Barbara Ryden《Introduction to Cosmology》Ch.4](https://www.amazon.com/Introduction-Cosmology-Barbara-Ryden/dp/1107154839)（这本书对初学者更友好）

## 这个方程在说什么

一句话：**宇宙膨胀的速率（$H$），由里面装了多少东西（$\rho$）和空间的几何（$k$）决定。**

| 情况 | 含义 |
|------|------|
| $\rho$ 大 → $H$ 大 | 物质越多，膨胀越快（引力势能越深，初始速度也越大） |
| $k > 0$（正曲率） | 空间像球面，膨胀会减速，可能回缩（类似"投球没达到逃逸速度"） |
| $k = 0$（平坦） | 恰好在逃逸边缘 |
| $k < 0$（负曲率） | 空间像马鞍面，永远膨胀 |

**观测结论**：CMB 第一声学峰的位置（$\ell \approx 220$）精确告诉我们 $k \approx 0$，宇宙是平坦的。（这个后面学功率谱时会回来解释。）

> 参考：[Wikipedia - Shape of the universe](https://en.wikipedia.org/wiki/Shape_of_the_universe)

## 不同物质的密度怎么随膨胀变化

这个很关键——决定了宇宙在不同时期由谁主导。

**普通物质（重子、暗物质）**：粒子数守恒，体积 $\propto a^3$：

$$\rho_{\text{matter}} \propto a^{-3}$$

**辐射（光子、中微子）**：粒子数也守恒，体积也 $\propto a^3$，但多一个效应——**光子波长随膨胀拉伸**（[宇宙学红移](https://en.wikipedia.org/wiki/Cosmological_redshift)），单个光子能量 $E = h\nu \propto 1/\lambda \propto a^{-1}$。所以总能量密度多降一个因子：

$$\rho_{\text{radiation}} \propto a^{-4}$$

**宇宙学常数（暗能量）**：密度是空间本身的性质，膨胀不改变它：

$$\rho_{\Lambda} = \text{const}$$

三种成分的衰减速度不同，所以宇宙的历史自然分成三个时代：

```
早期  ──→  辐射主导（ρ_rad ∝ a⁻⁴ 降最快，说明更早的时候它最大）
         ↓ a ~ 3400 时 ρ_rad = ρ_matter（物质-辐射等量时刻）
中期  ──→  物质主导（ρ_mat ∝ a⁻³）
         ↓ a ~ 0.7 时 ρ_matter = ρ_Λ
现在  ──→  暗能量主导（ρ_Λ = const，最终胜出）
```

> 参考：[Wikipedia - Friedmann equations § Density parameter](https://en.wikipedia.org/wiki/Friedmann_equations#Density_parameter) · [Wayne Hu - Expansion of the Universe](https://background.uchicago.edu/~whu/intermediate/expansion2.html)

## 和 CMB 的联系

- **红移**：CMB 光子从 $z \approx 1100$ 传播到今天，波长拉伸了 1100 倍，温度从 ~3000K 降到 2.725K
- **辐射主导时代**：CMB 释放前的宇宙处于辐射主导时代，$\rho_{\text{rad}}$ 决定了膨胀速率
- **平坦宇宙**：$k \approx 0$ 这个结论直接来自 CMB 功率谱第一峰的位置

---

## 理解检测

请直接在下面写你的回答，保存文件后告诉我。

**Q1**：假设宇宙从现在（$a=1$）膨胀到 $a = 2$（尺度因子翻倍）。物质密度变为原来的多少？辐射密度呢？为什么辐射比物质多降了一个因子？

你的回答：



**Q2**：Friedmann 方程里，如果宇宙是平坦的（$k=0$），$H$ 完全由什么决定？如果宇宙里只有物质（$\rho \propto a^{-3}$），随着宇宙膨胀，$H$ 是增大还是减小？物理上怎么理解？

你的回答：



**Q3**：为什么宇宙早期是辐射主导而不是物质主导？（提示：想想两者的密度分别怎么随 $a$ 变化，然后往 $a \to 0$ 的方向想。）

你的回答：



