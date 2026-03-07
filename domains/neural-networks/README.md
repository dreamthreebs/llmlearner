# 神经网络（Neural Networks）

> 创建日期：2026-03-07

## 背景与起点

- **已有知识**：本科物理（线性代数、微积分、概率论），已学过 LLM/02 的神经网络基础（神经元、MLP、损失函数、反向传播、梯度下降）
- **从哪开始**：深入 MLP 的数学细节，然后展开各种架构
- **目的**：理解各种网络架构的设计思想、训练的数学理论、具备动手实现的能力

## 领域概览

神经网络是一族参数化的函数，通过多层非线性变换将输入映射到输出。它的力量来自两件事：（1）结构上足够灵活（万能近似），（2）有高效的训练算法（反向传播 + 梯度下降）。

不同的问题需要不同的网络结构：图像用 CNN（利用空间局部性），序列用 RNN/Transformer（利用时序结构），图数据用 GNN。理解"为什么这个结构适合这个问题"是学习各种架构的核心线索。

同时，神经网络为什么在理论上应该不工作（参数远多于数据，经典统计理论预测必然过拟合）却在实践中表现极好——这是深度学习理论要回答的核心问题。

## 知识地图

```
LLM/02 基础（神经元、MLP、损失函数、反向传播、梯度下降）
    ↓
深入 MLP（初始化、BatchNorm、残差连接、优化器细节）
    ↓
  ┌────────────── 架构分支 ──────────────┐
  │                                      │
  ├── CNN（卷积网络）— 图像/空间数据     │
  │     └── LeNet → AlexNet → ResNet     │
  ├── RNN / LSTM — 序列数据              │
  │     └── 梯度消失 → 门控机制          │
  ├── Transformer — 注意力机制           │
  │     └── (已在 LLM/04-05 中覆盖)      │
  ├── GNN — 图结构数据                   │
  ├── AutoEncoder / VAE — 生成模型       │
  └── GAN — 对抗生成                     │
  └──────────────────────────────────────┘
    ↓
  ┌────────────── 训练数学 ──────────────┐
  │                                      │
  ├── 优化理论（损失面、SGD 收敛性）     │
  ├── 正则化理论（为什么过参数化不过拟合）│
  ├── 泛化理论（PAC、VC 维、Rademacher） │
  └── 初始化与归一化                     │
  └──────────────────────────────────────┘
    ↓
  ┌────────────── 实现 ────────────────┐
  │                                    │
  ├── PyTorch 基础                     │
  ├── 手写 MLP / CNN / Transformer     │
  └── 训练技巧（调参、调试）           │
  └────────────────────────────────────┘
    ↓
  ┌────────────── 前沿 ────────────────┐
  │                                    │
  ├── Neural Scaling Laws              │
  ├── Double Descent / 泛化谜题        │
  ├── Neural Tangent Kernel（NTK）     │
  ├── 可解释性（Mechanistic Interp.）  │
  └── 架构搜索（NAS）                  │
  └────────────────────────────────────┘
```

## 学习路径

1. **深入 MLP** — 初始化策略、BatchNorm、残差连接、优化器（Adam 细节）→ `02-deep-mlp.md`
2. **CNN** — 卷积、池化、感受野，从 LeNet 到 ResNet 的设计演进 → `03-cnn.md`
3. **RNN 与 LSTM** — 序列建模、梯度消失问题、门控机制 → `04-rnn-lstm.md`
4. **训练的数学** — 损失面结构、SGD 收敛性、泛化理论 → `05-training-math.md`
5. **正则化与泛化** — 为什么过参数化反而好、double descent → `06-generalization.md`
6. **PyTorch 实战** — 从零手写 MLP、CNN、简单 Transformer → `07-pytorch-practice.md`
7. **前沿理论** — NTK、scaling laws、可解释性 → `08-frontiers.md`
8. **AutoEncoder 与 VAE** — 无监督特征学习、生成模型基础 → `09-autoencoder-vae.md`
9. **GAN** — 对抗生成、minimax 博弈、从 DCGAN 到 StyleGAN → `10-gan.md`
10. **GNN** — 图结构数据的消息传递、GCN、GAT → `11-gnn.md`

## 推荐资源

### 教材
1. Goodfellow et al.,《Deep Learning》(花书) — 最标准的深度学习教材，[免费在线](https://www.deeplearningbook.org/)
2. Zhang et al.,《Dive into Deep Learning》— 理论+代码，[免费在线](https://d2l.ai/)

### 视频
1. 3Blue1Brown 神经网络系列 — 直觉建立
2. Andrej Karpathy 的 YouTube 教程 — 从零实现

### 论文
1. "Deep Residual Learning" (He et al., 2015) — ResNet，影响力巨大
2. "Batch Normalization" (Ioffe & Szegedy, 2015) — 训练稳定性的关键技术
