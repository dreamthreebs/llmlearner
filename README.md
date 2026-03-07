# LLM Learner — 研究者学习系统

用 Cursor Agent 结构化学习新领域。

## 使用

直接告诉 Agent："我想学习 [领域名称]"

Agent 会：先问你知道什么 → 给出个性化的知识地图和路径 → 讲解核心机制 → 边讲边测

或者手动初始化：`./init-domain.sh <领域名>`

## 目录结构

```
domains/<领域名>/
├── README.md          # 概览 + 知识地图 + 学习路径
├── notes/             # 学习笔记
├── papers/            # 论文 PDF
├── references.bib     # 参考文献
└── questions.md       # 问题追踪
```
