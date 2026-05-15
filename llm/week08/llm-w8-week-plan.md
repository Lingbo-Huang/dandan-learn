---
layout: default
title: "Week 8 周规划 · 大模型应用工程"
render_with_liquid: false
---

# 大模型线 Week 8 周规划总览

**主题：大模型应用工程——Prompt、Function Calling、结构化输出、安全**  
**周期：Day 1 - Day 6**

---

## 本周目标

掌握大模型应用层工程实践，能够构建生产级 LLM 应用：

- 掌握 Prompt 工程的核心技巧（CoT、Few-shot、Self-consistency）
- 理解 Function Calling 原理，构建 LLM Agent
- 掌握结构化输出（JSON mode、Pydantic、Outlines）
- 了解 LLM 安全问题（Prompt Injection、越狱、内容安全）
- 能够设计生产级 LLM 应用架构

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | Prompt 工程 | Zero-shot/Few-shot/CoT/Self-consistency |
| D2 | Function Calling | Tool use、JSON Schema、OpenAI API |
| D3 | 结构化输出 | JSON mode、Pydantic、Outlines、约束解码 |
| D4 | LLM Agent | ReAct、Planning、Memory、Multi-agent |
| D5 | LLM 安全 | Prompt Injection、越狱、内容审核 |
| D6 | Capstone | 构建完整 LLM 应用 |

---

## 面试高频题

1. CoT 为什么能提升模型的推理能力？
2. Function Calling 的实现原理？
3. 如何保证 LLM 输出格式正确？
4. Prompt Injection 攻击是什么？如何防御？
5. ReAct Agent 和 Plan-and-Execute 的区别？
6. LLM 应用的延迟瓶颈在哪里？如何优化？

---

## 参考资料

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [Outlines 库](https://github.com/outlines-dev/outlines)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Instructor 库](https://github.com/jxnl/instructor)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
