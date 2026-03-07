# LLM（大语言模型）

> 创建日期：2026-03-07

## 背景与起点

- **已有知识**：本科物理背景，有数学基础（线性代数、微积分、概率论），听说过 Transformer 但不了解，机器学习/深度学习从零开始
- **从哪开始**：神经网络基础
- **目的**：理解 LLM 的原理——怎么训练出来的、为什么有效

## 领域概览

LLM（Large Language Model）的核心思想极其简单：**给定前面的文字，预测下一个词。** 就这一件事，做到极致。

一个 LLM 本质上是一个巨大的函数 $f$：输入一段文本，输出每个可能下一个词的概率分布。训练就是用海量文本调整函数的参数，让它在预测下一个词时尽可能准确。令人惊讶的是，当模型足够大、数据足够多时，这个"预测下一个词"的任务迫使模型学会了语法、语义、推理、甚至世界知识。

关键架构是 Transformer（2017），核心机制是 Attention（注意力机制）——让模型在处理每个词时可以"看到"序列中所有其他词，并选择性地关注最相关的部分。

## 知识地图

```
线性代数 + 微积分 + 概率论
    ↓
神经网络基础（前馈网络、反向传播、梯度下降）
    ↓
词嵌入（Word Embedding）—— 怎么把文字变成数字
    ↓
序列建模的挑战 —— RNN 的思路和局限
    ↓
Attention 机制 —— 核心创新
    ↓
Transformer 架构 —— 现代 LLM 的骨架
    ├── Self-Attention
    ├── 多头注意力（Multi-Head Attention）
    ├── 位置编码（Positional Encoding）
    ├── 前馈网络（FFN）
    └── 层归一化（Layer Normalization）
    ↓
预训练（Pretraining）—— 用海量文本训练"下一个词预测"
    ↓
微调与对齐（Fine-tuning & Alignment）
    ├── SFT（监督微调）
    ├── RLHF / DPO（人类偏好对齐）
    └── 安全性与可控性
    ↓
  ┌─────────── 前沿方向 ───────────────┐
  │                                    │
  ├── Scaling Laws（规模定律）         │
  ├── 推理能力（Reasoning / CoT）      │
  ├── 多模态（Vision-Language）        │
  ├── 高效架构（MoE, SSM, Mamba）      │
  ├── 对齐与安全（Alignment & Safety） │
  └── 长上下文与记忆                   │
  └────────────────────────────────────┘
```

## 学习路径

1. **神经网络基础** — 不理解神经网络，后面全是黑箱
2. **词嵌入** — 文字怎么变成向量，为什么语义相近的词向量也相近
3. **Attention 机制** — LLM 的灵魂，必须彻底理解
4. **Transformer 架构** — 把 Attention 组装成完整模型
5. **预训练** — 怎么用"预测下一个词"训练出通用能力
6. **微调与对齐** — 怎么把一个"续写机器"变成有用的助手

## 核心机制

见 notes/ 中的逐章讲解。

## 推荐资源

### 教材/教程
1. 3Blue1Brown 的神经网络视频系列 — 直觉建立极佳
2. Andrej Karpathy 的 "Let's build GPT from scratch" — 从零手写一个小 GPT

### 论文
1. "Attention Is All You Need" (Vaswani et al., 2017) — Transformer 原始论文
2. "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020) — Scaling 的里程碑
