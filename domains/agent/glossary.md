# AI Agent 术语表

| 缩写/术语 | 全称 | 简述 |
|-----------|------|------|
| Agent | AI Agent / 智能体 | 以 LLM 为核心，具备规划、记忆、工具调用能力的自主系统 |
| ANN | Approximate Nearest Neighbor | 近似最近邻搜索，在大规模向量中快速找到相似向量 |
| AutoGen | AutoGen | 微软开源的 Multi-Agent 框架 |
| AutoGPT | AutoGPT | 早期的长期自主 Agent 尝试 |
| Benchmark | Benchmark | 标准化的任务集，用于评估 Agent 能力 |
| Chunking | Chunking / 分块 | 把长文档切成短块以便 embedding 和检索 |
| CoT | Chain-of-Thought | 链式思维，让 LLM 逐步写出推理过程以提升准确率 |
| Constrained Decoding | Constrained Decoding / 受限解码 | 强制 LLM 只输出符合特定格式的 token |
| Few-shot | Few-shot Prompting | 在 prompt 中给出少量输入-输出示例来引导 LLM |
| Function Calling | Function Calling | LLM 生成结构化的函数调用请求，由外部系统执行 |
| HNSW | Hierarchical Navigable Small World | 一种高效的 ANN 算法 |
| Human-in-the-Loop | Human-in-the-Loop | 人机协作，在关键节点引入人类判断 |
| In-context Learning | In-context Learning / 上下文学习 | LLM 从 prompt 中的示例推断任务模式，不更新参数 |
| JSON Schema | JSON Schema | 描述 JSON 数据结构的标准，用于定义工具参数格式 |
| LangChain | LangChain | Python Agent 框架，封装 LLM 调用、工具集成、链式调用 |
| LangGraph | LangGraph | LangChain 团队的图结构 Agent 框架，支持有状态的多步工作流 |
| LangSmith | LangSmith | LangChain 的配套调试工具，提供可视化 trace 查看 |
| LATS | Language Agent Tree Search | 结合树搜索和 ReAct 的 Agent 架构 |
| LLM-as-Judge | LLM-as-Judge | 用另一个 LLM 评估 Agent 输出质量的方法 |
| MCP | Model Context Protocol | Anthropic 提出的统一工具与 Agent 通信的开放标准 |
| MCTS | Monte Carlo Tree Search | 蒙特卡洛树搜索，通过模拟估计每个决策的价值 |
| Multi-Agent | Multi-Agent System | 多个 Agent 分工协作完成复杂任务 |
| Plan-and-Execute | Plan-and-Execute | 先生成完整计划再逐步执行的 Agent 架构 |
| Prompt Injection | Prompt Injection / 提示注入 | 通过恶意输入操纵 LLM 行为的攻击方式 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，先检索相关文档再让 LLM 生成回答 |
| ReAct | Reasoning + Acting | 推理与行动交替循环的 Agent 框架 |
| Reflexion | Reflexion | 在 ReAct 基础上加入自我反思和经验总结的改进 |
| Self-Consistency | Self-Consistency | 生成多条推理链条，用多数投票选答案 |
| SWE-bench | SWE-bench | 基于 GitHub 真实 issue 的代码修复 Benchmark |
| System Prompt | System Prompt / 系统提示 | 设定 LLM 角色和行为规则的指令 |
| Tool Use | Tool Use / 工具调用 | Agent 调用外部工具（搜索、代码执行、API）的能力 |
| Toolformer | Toolformer | 训练时让模型学会在文本中自然插入 API 调用的方法 |
| ToT | Tree of Thoughts | 树状思维，把推理从链条扩展为树状搜索 |
| Trace | Trace / 执行轨迹 | Agent 完整的执行记录（每步输入、输出、工具调用、耗时） |
| WebArena | WebArena | 在真实网站镜像中执行任务的 Agent Benchmark |
| Zero-shot | Zero-shot Prompting | 无示例，直接给任务让 LLM 回答 |
| Zero-shot CoT | Zero-shot Chain-of-Thought | 加 "Let's think step by step" 触发推理链条 |
