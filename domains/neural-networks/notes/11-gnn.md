# 11 - GNN：图神经网络

> **主维度**：D1 基础架构
> **关键关系**：
> - GCN (架构) --推广--> CNN (架构)：GCN 把 CNN 的卷积从网格推广到图结构
> - GCN (架构) --实例--> GNN (架构)：GCN 是 GNN 的实例
> - GAT (架构) --对比--> GCN (架构)：GAT 与 GCN 在消息传递方式上形成对比
>
> **学习路径**：01-overview → 深入 MLP（02） → CNN（03） → Transformer（参见 domains/LLM/notes/04-05） → **本章**
>
> **前置知识**：MLP 与反向传播、CNN 的卷积思想（03 已覆盖）、注意力机制（Transformer 中的 self-attention，参见 LLM/04-05）、线性代数（矩阵乘法、特征值直觉即可）
>
> **参考**：
> - [Kipf & Welling 2017 - Semi-Supervised Classification with GCN](https://arxiv.org/abs/1609.02907)
> - [Veličković et al. 2018 - Graph Attention Networks](https://arxiv.org/abs/1710.10903)
> - [Distill - A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/)
> - [Wikipedia - Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network)
> - [d2l.ai](https://d2l.ai/)
> - [Stanford CS224W - Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)

---

## 核心问题

到目前为止我们学过的所有网络架构，都有一个隐含假设：**数据有规则的结构**。

- **MLP**：输入是一个固定长度的向量（"表格数据"）
- **CNN**：输入是网格结构（图像是二维网格，像素排列整齐）
- **RNN/Transformer**：输入是序列（一维有序排列）

但现实世界中大量数据天然是**图**（graph）的形式——节点之间通过边连接，没有固定的网格或顺序：

- **社交网络**：用户是节点，好友关系是边
- **分子结构**：原子是节点，化学键是边
- **交通网络**：路口是节点，道路是边
- **引用网络**：论文是节点，引用关系是边
- **蛋白质结构**：氨基酸残基是节点，空间邻近关系是边

你没法把一个社交网络"拉平"成一个固定长度的向量送进 MLP（节点数不固定、没有自然的排列顺序），也没法把分子结构排列成网格送进 CNN（原子的连接方式是不规则的）。

**图神经网络**（Graph Neural Network，GNN）就是为处理这种不规则的图结构数据而设计的网络架构。

> 可靠程度：**Level 1**（教科书共识）

---

## 1. 图的基本定义

在正式学 GNN 之前，需要先明确"图"的数学定义。如果你没学过图论，这里给出最小化的介绍。

### 1.1 图的组成

一个**图** $G = (V, E)$ 由两部分组成：

- **节点集** $V = \{v_1, v_2, \dots, v_n\}$：图中的"对象"，共 $n$ 个
- **边集** $E \subseteq V \times V$：节点之间的"关系"。边 $(v_i, v_j) \in E$ 表示节点 $v_i$ 和 $v_j$ 之间有连接

**无向图**：边没有方向，$(v_i, v_j)$ 和 $(v_j, v_i)$ 是同一条边。例如：好友关系是双向的。

**有向图**：边有方向，$(v_i, v_j)$ 表示从 $v_i$ 指向 $v_j$。例如：论文引用关系有方向（A 引用 B 不等于 B 引用 A）。

### 1.2 邻接矩阵

图的结构可以用一个 $n \times n$ 的**邻接矩阵** $A$ 来表示：

$$A_{ij} = \begin{cases} 1, & \text{如果 } (v_i, v_j) \in E \\ 0, & \text{否则} \end{cases}$$

对于无向图，$A$ 是对称的（$A_{ij} = A_{ji}$）。

一个简单的例子：三个节点，1-2 相连，2-3 相连，1-3 不相连：

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

### 1.3 节点特征矩阵

除了图的连接结构，每个节点通常还携带**特征**（属性）。比如在社交网络中，每个用户有年龄、兴趣等属性；在分子图中，每个原子有原子序数、电荷等属性。

所有节点的特征排成一个矩阵 $X \in \mathbb{R}^{n \times f}$：

$$X = \begin{bmatrix} — x_1^T — \\ — x_2^T — \\ \vdots \\ — x_n^T — \end{bmatrix}$$

其中 $x_i \in \mathbb{R}^f$ 是节点 $v_i$ 的特征向量，$f$ 是特征维度。

### 1.4 节点的邻居

节点 $v$ 的**邻居**（neighborhood）是所有与 $v$ 直接相连的节点：

$$\mathcal{N}(v) = \{u \in V : (v, u) \in E\}$$

节点的**度**（degree）是它的邻居个数：$\text{deg}(v) = |\mathcal{N}(v)|$。所有节点的度构成**度矩阵** $D$，它是一个对角矩阵：

$$D_{ii} = \text{deg}(v_i) = \sum_{j} A_{ij}$$

> 参考：[Wikipedia - Graph theory](https://en.wikipedia.org/wiki/Graph_theory) · [Wikipedia - Adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)

---

## 2. 消息传递机制（Message Passing）

### 2.1 核心思想

GNN 的核心思想其实非常自然：**每个节点从它的邻居收集信息，然后更新自己的表示。**

类比：想象一个社交网络。每个人（节点）向自己的朋友（邻居）打听消息，综合所有朋友的信息后更新自己的"知识状态"。经过几轮这样的信息传播，每个人就不仅知道朋友的信息，还知道朋友的朋友的信息——信息在图上扩散开了。

这个过程叫做**消息传递**（message passing），它是几乎所有 GNN 的统一框架。

### 2.2 一般公式

第 $k$ 轮消息传递的更新规则：

$$h_v^{(k+1)} = \text{UPDATE}\left(h_v^{(k)}, \; \text{AGGREGATE}\left(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)\right)$$

逐项解释：

- $h_v^{(k)} \in \mathbb{R}^{d_k}$：节点 $v$ 在第 $k$ 轮的**表示向量**（也叫隐藏状态）。初始时 $h_v^{(0)} = x_v$（就是原始节点特征）。
- $\mathcal{N}(v)$：节点 $v$ 的邻居集合。
- $\text{AGGREGATE}$：**聚合函数**——把所有邻居的表示汇总成一个向量。常见选择：求和（sum）、均值（mean）、最大值（max）。
- $\text{UPDATE}$：**更新函数**——结合节点自身的旧表示和聚合后的邻居信息，生成新的表示。通常包含一个可学习的线性变换加非线性激活。

**和 CNN 的类比**：CNN 的卷积核在规则的网格上滑动，聚合一个固定大小窗口内的像素。GNN 的消息传递在不规则的图上操作，聚合每个节点的邻居——邻居的数量和排列因节点而异。GNN 可以看作**把 CNN 的卷积思想推广到了非欧几里得域**。

### 2.3 多层堆叠的效果

一层消息传递让每个节点获得 1-hop 邻居的信息。两层让每个节点获得 2-hop 邻居的信息（邻居的邻居）。$K$ 层让每个节点的表示融合了 $K$-hop 邻域的信息。

这和 CNN 中**感受野**（receptive field）的概念很像：CNN 中更深的层"看到"更大的图像区域，GNN 中更深的层"感知到"更远的图邻域。

> 可靠程度：**Level 1**（消息传递框架是 GNN 的统一范式，见 [Gilmer et al. 2017 - MPNN](https://arxiv.org/abs/1704.01212)）

---

## 3. GCN（Graph Convolutional Network）

**GCN**（图卷积网络，Kipf & Welling 2017）是最经典的 GNN 模型之一。它给出了一个简洁优雅的消息传递规则。

### 3.1 GCN 的层公式

GCN 一层的矩阵形式：

$$H^{(k+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)}\right)$$

公式中每个符号的含义：

- $H^{(k)} \in \mathbb{R}^{n \times d_k}$：第 $k$ 层所有节点的表示矩阵，第 $i$ 行是节点 $v_i$ 的表示 $h_i^{(k)}$。初始时 $H^{(0)} = X$（节点特征矩阵）。
- $\tilde{A} = A + I$：**加了自环的邻接矩阵**。$I$ 是单位矩阵，加上它意味着每个节点把自己也算作"邻居"——聚合邻居信息时也考虑自身。这很重要：没有自环的话，节点会丢失自身的信息。
- $\tilde{D}$：$\tilde{A}$ 对应的度矩阵，$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$。
- $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$：**对称归一化的邻接矩阵**。它的作用是在聚合邻居信息时，根据节点的度进行归一化，防止度高的节点"淹没"度低的节点。
- $W^{(k)} \in \mathbb{R}^{d_k \times d_{k+1}}$：第 $k$ 层的**可学习权重矩阵**，是 GCN 的参数。它对聚合后的表示做线性变换（类似 MLP 中的权重矩阵）。
- $\sigma$：非线性激活函数（通常是 ReLU）。

### 3.2 单个节点的视角

矩阵公式看起来密，但对于单个节点 $v_i$，GCN 做的事情很直觉：

$$h_i^{(k+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \cdot \tilde{d}_j}} \; h_j^{(k)} \; W^{(k)}\right)$$

其中 $\tilde{d}_i = \tilde{D}_{ii}$ 是加自环后节点 $i$ 的度。

用文字说：**把自己和所有邻居的表示加权求和（权重和两端节点的度有关），乘以一个可学习矩阵，过一个激活函数。** 这就是 GCN 一层做的全部事情。

### 3.3 为什么要归一化

如果不做归一化（直接用 $AH^{(k)}W^{(k)}$），会出什么问题？

矩阵乘法 $AH^{(k)}$ 的效果是：对每个节点，把所有邻居的表示**求和**。但不同节点的度差别可能很大（社交网络中有人有 10 个朋友，有人有 10000 个）。度高的节点聚合后的向量模长会非常大，导致数值不稳定。

$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 的归一化使得聚合变成**加权均值**（而非求和），消除了度带来的尺度差异。这和 CNN 中卷积核的归一化有类似的作用。

> 可靠程度：**Level 1**（GCN 是经典模型，见 [Kipf & Welling 2017](https://arxiv.org/abs/1609.02907)）

---

## 4. GAT（Graph Attention Network）

### 4.1 GCN 的局限

GCN 在聚合邻居时，每个邻居的权重完全由图的结构决定（度归一化），和邻居的特征无关。但直觉上，**不同邻居的重要性应该不同**——你最亲密的朋友提供的信息比泛泛之交更重要。

### 4.2 GAT 的改进：注意力聚合

**GAT**（Graph Attention Network，Veličković et al. 2018）把 Transformer 中的**注意力机制**引入了 GNN：不再用固定权重聚合邻居，而是用注意力分数动态决定每个邻居的重要性。

对于节点 $v_i$，GAT 的计算过程：

**第一步**：计算节点 $i$ 和邻居 $j$ 之间的**注意力系数**（attention coefficient）：

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [W h_i \| W h_j]\right)$$

其中：
- $W \in \mathbb{R}^{d' \times d}$ 是可学习的线性变换矩阵
- $\|$ 表示向量拼接（concatenation）
- $\mathbf{a} \in \mathbb{R}^{2d'}$ 是一个可学习的注意力向量
- $\text{LeakyReLU}$ 是带泄漏的 ReLU 激活

直觉：$e_{ij}$ 衡量的是"节点 $j$ 对节点 $i$ 有多重要"。这个重要性不是预设的，而是从数据中学出来的。

**第二步**：用 softmax 归一化注意力系数（只在邻居上做 softmax）：

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

**第三步**：用注意力权重聚合邻居表示：

$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \; W h_j\right)$$

### 4.3 和 Transformer 的联系

如果你学过 Transformer（参见 domains/LLM/notes/），GAT 的注意力机制会非常眼熟：

- Transformer 的 self-attention：每个 token 关注序列中的所有其他 token，注意力权重由 query-key 点积决定
- GAT 的 attention：每个节点关注图中的所有邻居（不是所有节点！），注意力权重由可学习的打分函数决定

关键区别：**Transformer 中每个 token 可以关注序列中的任意位置（全连接注意力），而 GAT 只关注图中的邻居节点（稀疏注意力）**。事实上，如果你把 Transformer 看成在一个"所有 token 两两相连"的完全图上做 GNN——那 Transformer 就是一个特殊的 GAT。

实际上，后来的研究也验证了这个联系：Graph Transformer 把两者融合，在图的邻域结构上使用 Transformer 风格的注意力。

> 可靠程度：**Level 1**（GAT 是经典模型，见 [Veličković et al. 2018](https://arxiv.org/abs/1710.10903)）

---

## 5. GNN 的三种任务

GNN 可以在不同粒度上做预测：

### 5.1 节点级任务（Node-level）

预测每个节点的属性。例如：在引用网络中，预测每篇论文的学科类别。

做法：用 GNN 计算每个节点的表示 $h_v$，然后接一个分类器 $\hat{y}_v = \text{softmax}(W h_v)$。

### 5.2 边级任务（Edge-level）

预测两个节点之间是否存在边，或边的属性。例如：社交网络中预测两个人是否会成为朋友（链接预测）。

做法：把两个端点的表示拼接或点积，送进分类器。

### 5.3 图级任务（Graph-level）

预测整个图的属性。例如：预测一个分子是否有毒性。

做法：用 GNN 计算所有节点的表示，然后用一个**读出函数**（readout / pooling）把它们汇总成一个图级表示：

$$h_G = \text{READOUT}(\{h_v : v \in V\})$$

常见的 readout：所有节点表示的均值、求和、或可学习的注意力池化。

---

## 6. GNN 的应用

### 6.1 分子属性预测

分子天然是图：原子是节点，化学键是边。GNN 可以预测分子的溶解度、毒性、药物活性等。这在**药物发现**（drug discovery）中有重大应用价值，因为传统方法需要实验测量，而 GNN 可以快速筛选候选分子。

### 6.2 社交网络分析

预测用户兴趣（节点分类）、推荐好友（链接预测）、检测社区结构（图聚类）。

### 6.3 推荐系统

把用户和物品建模为二部图（bipartite graph），用户-物品交互是边。GNN 通过消息传递学习用户和物品的表示，用于推荐。Pinterest 就在生产环境中使用了基于 GNN 的推荐系统（[PinSage](https://arxiv.org/abs/1806.01973)）。

### 6.4 交通流预测

交通网络中，路口是节点，道路是边，车流量是时变的节点特征。GNN 结合时间序列模型（如 RNN）可以预测未来的交通状况。

### 6.5 AlphaFold 中的蛋白质结构预测

DeepMind 的 AlphaFold2（2020）在蛋白质结构预测上取得了革命性突破。它的核心架构之一就是在氨基酸残基构成的图上做消息传递——每个残基是节点，空间邻近的残基之间有边。通过多轮消息传递，网络学会了预测蛋白质的三维折叠结构。

> 可靠程度：**Level 1-2**（分子预测和蛋白质结构是成熟应用；推荐系统和交通预测中 GNN 是否显著优于传统方法仍有争议）
>
> 参考：[AlphaFold2 (Jumper et al. 2021)](https://www.nature.com/articles/s41586-021-03819-2) · [PinSage (Ying et al. 2018)](https://arxiv.org/abs/1806.01973)

---

## 7. GNN 的局限

### 7.1 过平滑问题（Over-smoothing）

这是 GNN 最重要的局限之一。

每一层 GNN 都在聚合邻居信息。直觉上，层数越多，每个节点"感知到"的图区域越大——$K$ 层 GNN 聚合了 $K$-hop 邻域。但当层数太多时，一个问题出现了：**所有节点的表示趋于相同。**

为什么？因为经过足够多轮消息传递，每个节点都聚合了整个图的信息。如果图是连通的，所有节点最终会收敛到同一个表示——就像在一杯水中滴入墨水，搅拌足够久后整杯水颜色均匀。

这叫做**过平滑**（over-smoothing）。它意味着：

- 不能简单地通过加层来提升 GNN 性能（和 CNN/Transformer 不同）
- 大多数 GNN 只用 2-4 层，远浅于 CNN（几十层到上百层）或 Transformer
- 如何让 GNN 变深是一个活跃的研究方向

### 7.2 表达能力的理论上限（WL test）

GNN 的表达能力有一个理论上限，和图论中的 **Weisfeiler-Leman (WL) 测试**有关。

WL 测试是一个经典的图同构测试算法：它通过迭代更新节点的"颜色"（标签）来判断两个图是否同构（结构相同）。基本的 1-WL 测试和消息传递 GNN 的操作几乎一模一样——收集邻居信息，更新自身标签。

Xu et al. (2019) 证明了一个重要结论：**标准的消息传递 GNN 的区分能力不超过 1-WL 测试。** 也就是说，如果 1-WL 测试无法区分两个图，那么任何消息传递 GNN 也无法区分它们。

存在一些 1-WL 无法区分的非同构图（它们结构不同，但 WL 测试认为它们相同）。这意味着 GNN 对这些图也无法区分——这是消息传递范式的根本局限。

后续研究（如 higher-order GNN、subgraph GNN）试图突破这个限制，但通常以更高的计算成本为代价。

> 可靠程度：**Level 1**（WL 测试与 GNN 表达能力的关系已有严格证明，见 [Xu et al. 2019 - How Powerful are GNNs?](https://arxiv.org/abs/1810.00826)）

### 7.3 可扩展性

大规模图（如社交网络有数十亿节点）上的 GNN 训练面临内存和计算瓶颈。因为消息传递需要访问邻居节点的特征，随着层数增加，需要访问的节点数指数增长（"邻居爆炸"问题）。

解决方案包括：图采样（GraphSAGE 的邻居采样）、子图训练（Cluster-GCN）、分布式训练等。

> 参考：[GraphSAGE (Hamilton et al. 2017)](https://arxiv.org/abs/1706.02216) · [Cluster-GCN (Chiang et al. 2019)](https://arxiv.org/abs/1905.07953)

---

## 理解检测

**Q1**：GCN 的层公式中用了 $\tilde{A} = A + I$（给每个节点加自环）。如果不加这个自环（直接用 $A$ 而不是 $\tilde{A}$），在消息传递过程中会发生什么？节点会丢失什么信息？提示：想想聚合公式 $AH$ 的第 $i$ 行是什么。

你的回答：


**Q2**：Transformer 的 self-attention 可以看成在一个"完全图"（所有节点两两相连）上做 GNN。如果我们把 GAT 也放在一个完全图上运行——即每个节点都关注所有其他节点——它和 Transformer 有什么区别？（提示：比较两者的注意力计算方式。）

你的回答：


**Q3**：过平滑（over-smoothing）是 GNN 的核心限制：层数太多时所有节点的表示趋同。但 CNN 和 Transformer 堆几十甚至上百层却没有这个问题。请从"信息聚合范围"的角度解释：CNN/Transformer 和 GNN 在深层时的行为为什么不同？（提示：CNN 即使很深，每层的卷积核大小是固定的，总的感受野增长是线性的；而 GNN 呢？）

你的回答：

