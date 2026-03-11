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
）
conda create -n llmlearner python=3.12 -y
conda activate llmlearner
# 或者用 venv:
# python -m venv .venv && source .venv/bin/activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 安装 Graphviz（如果没装过）
# macOS
brew install graphviz
# Ubuntu/Debian
# sudo apt install graphviz

# 5. 生成知识图 SVG
bash build-graphs.sh

# 6. 启动本地预览
mkdocs serve -a 127.0.0.1:8800
```

浏览器打开 <http://127.0.0.1:8800> 即可。

---

## 使用

直接告诉 Cursor Agent："我想学习 [领域名称]"

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
