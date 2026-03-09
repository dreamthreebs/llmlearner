# LLM Learner — 研究者学习系统

用 Cursor Agent 结构化学习新领域。

## 开始使用

1. 在 Cursor 中打开本项目
2. 告诉 Agent："我想学习 [领域名称]"
3. Agent 会自动创建领域目录、知识地图、学习笔记

## 环境配置

详见 [README](https://github.com/dreamthreebs/llmlearner/tree/template#快速开始)。

## 目录结构

```
domains/<领域名>/
├── README.md              # 概览 + 知识图 SVG
├── knowledge-graph.dot    # Graphviz 知识图源文件
├── glossary.md            # 术语表
├── notes/                 # 学习笔记
├── images/                # 生成的 SVG 等
└── questions.md           # 问题追踪
```
