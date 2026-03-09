# LLM Learner — 研究者学习系统

用 Cursor Agent 结构化学习新领域。

> **想自己用？** 请克隆 [template 分支](https://github.com/dreamthreebs/llmlearner/tree/template)（干净模板，没有学习内容）：
> ```bash
> git clone -b template https://github.com/dreamthreebs/llmlearner.git
> ```

## 在线阅读

<https://dreamthreebs.github.io/llmlearner/>

push 到 `main` 后 GitHub Actions 自动部署。

---

## 环境配置

### 依赖

- Python >= 3.10
- [Graphviz](https://graphviz.org/)（用于生成知识图 SVG）

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/dreamthreebs/llmlearner.git
cd llmlearner

# 2. 创建虚拟环境（conda 或 venv 均可）
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
