# LLM Learner — 研究者学习系统

用 Cursor Agent 结构化学习新领域。

## 使用方式

在 Cursor 中打开本项目，告诉 Agent："我想学习 [领域名称]"

### 三种交互模式

| 关键词 | 模式 | Agent 做什么 |
|--------|------|-------------|
| `想学` `新领域` | 新领域 | 问背景 → 领域建模 → 写 overview → 画知识图 → 开始讲解 |
| `深入` `展开` `没懂` | 深入 | 在对话中解释概念，用户说"加到文章里"再写入 notes |
| `批改` `看看我回答` | 批改 | 读取回答 → 逐题点评 → 结果写入 questions.md → 清空 notes 中的回答 |

## 环境配置

详见 [README](https://github.com/dreamthreebs/llmlearner#环境配置)。
