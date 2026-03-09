# LLM Learner — 研究者学习系统

用 Cursor Agent 结构化学习新领域。

## 在线阅读

**GitHub Pages**：<https://dreamthreebs.github.io/llmlearner/>

每次 push 到 `main` 分支后自动部署。

## 本地预览

```bash
./serve.sh
```

浏览器打开 <http://127.0.0.1:8800>（需要 conda `test` 环境 + `mkdocs-material`）。

## 使用

直接告诉 Agent："我想学习 [领域名称]"

Agent 会：先问你知道什么 → 给出个性化的知识地图和路径 → 讲解核心机制 → 边讲边测

或者手动初始化：`./init-domain.sh <领域名>`

## 目录结构

```
domains/<领域名>/
├── README.md              # 概览 + 知识图 SVG
├── knowledge-graph.dot    # Graphviz 知识图源文件
├── glossary.md            # 术语表
├── notes/                 # 学习笔记
├── images/                # 生成的 SVG 等
├── papers/                # 论文 PDF
├── references.bib         # 参考文献
└── questions.md           # 问题追踪
```
