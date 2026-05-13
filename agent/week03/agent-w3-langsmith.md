---
layout: default
title: "D6 · LangSmith：调试与评估"
---

# D6 · LangSmith：Agent 的监控基础设施

> **Agent Week 3**  
> 没有监控的 Agent 是黑盒。LangSmith 让你看清每一步发生了什么。

---

## 一、为什么需要 LangSmith？

| 问题 | LangSmith 的解决 |
|------|----------------|
| Agent 输出不对，不知道哪一步错了 | 完整追踪每一步的输入输出 |
| 每次 LLM 调用花了多少 Token？ | Token 使用量统计 |
| 模型在测试集上表现如何？ | 批量评估（Evaluation） |
| 同一个问题两次结果不一样？ | 对比不同 Run 的结果 |

---

## 二、配置 LangSmith

```python
import os

# 环境变量配置（或放在 .env 文件）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"   # https://smith.langchain.com 获取
os.environ["LANGCHAIN_PROJECT"] = "dandan-agent-w3"  # 项目名称

# 配置后，所有 LangChain 调用自动上传追踪
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
chain = (
    ChatPromptTemplate.from_template("解释：{topic}") 
    | llm 
    | StrOutputParser()
)

# 这次调用会自动被追踪
result = chain.invoke({"topic": "注意力机制"})
```

---

## 三、添加自定义追踪

```python
from langsmith import traceable

@traceable(name="my_rag_pipeline", run_type="chain")
def rag_pipeline(question: str) -> str:
    """带追踪的 RAG 管道"""
    # 检索
    docs = retriever.invoke(question)
    
    # 生成
    answer = rag_chain.invoke(question)
    
    return answer

# 添加元数据
from langsmith import Client

client = Client()

@traceable(
    name="custom_agent_run",
    metadata={"version": "1.0", "user_id": "user_123"},
    tags=["production", "rag"]
)
def my_agent_call(input: str) -> str:
    return agent_executor.invoke({"input": input})["output"]
```

---

## 四、批量评估（Evaluation）

```python
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

client = Client()

# 准备测试数据集
dataset_name = "AI基础问答测试集"

# 如果数据集不存在，创建它
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    
    examples = [
        {
            "inputs": {"question": "什么是梯度下降？"},
            "outputs": {"answer": "梯度下降是一种迭代优化算法，通过沿损失函数梯度的反方向更新参数来最小化损失函数。"}
        },
        {
            "inputs": {"question": "ReLU 激活函数有什么优点？"},
            "outputs": {"answer": "ReLU 计算简单，不存在梯度消失问题，在正区间梯度恒为1，有助于深层网络训练。"}
        },
    ]
    
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id
    )

# 定义待评估的函数
def predict(inputs: dict) -> dict:
    question = inputs["question"]
    answer = chain.invoke({"question": question})
    return {"answer": answer}

# 运行评估
results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[
        LangChainStringEvaluator("qa"),           # 回答是否正确
        LangChainStringEvaluator("conciseness"),   # 是否简洁
    ],
    experiment_prefix="gpt-4o-mini-baseline"
)

print(f"评估完成！在 LangSmith 查看详情。")
```

---

## 五、A/B 测试不同模型

```python
# 对比 GPT-4o-mini vs GPT-4o 的效果
def predict_mini(inputs):
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    return {"answer": chain.invoke(inputs)}

def predict_4o(inputs):
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm | StrOutputParser()
    return {"answer": chain.invoke(inputs)}

# 分别评估
results_mini = evaluate(predict_mini, data=dataset_name, experiment_prefix="gpt-4o-mini")
results_4o = evaluate(predict_4o, data=dataset_name, experiment_prefix="gpt-4o")

# 在 LangSmith UI 中可以直接对比两次实验的结果
```

---

## 六、实时监控 Dashboard

LangSmith 提供的关键指标：

```
监控面板 → https://smith.langchain.com

核心指标：
├── Latency（延迟）：P50/P90/P99 响应时间
├── Token Usage：每次调用的输入/输出 Token 数
├── Error Rate：失败率
├── Cost：估算的 API 费用
└── Feedback：用户反馈评分
```

---

## 今天的关键认识

1. **LangSmith = 追踪 + 评估 + 监控**：Agent 开发的必备基础设施
2. **追踪**：每一步的输入输出全部记录，方便定位问题
3. **Evaluation**：批量测试，量化模型质量
4. **A/B 测试**：数据驱动地对比不同方案

---

## 明天预告

D7：**综合实战**——用本周学的所有技术，搭建一个完整的智能客服 Agent。
