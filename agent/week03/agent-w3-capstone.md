---
layout: default
title: "D7 · 综合实战：智能客服 Agent"
---

# D7 · 综合实战：智能客服 Agent

> **Agent Week 3 收官**  
> 把本周学的 LangChain 全家桶整合起来，搭建一个真实可用的客服 Agent。

---

## 本周回顾

| 天 | 核心技术 |
|----|---------|
| D1 | LangChain 架构：Model I/O / Chains / Agents |
| D2 | LCEL：管道 / 并行 / 流式 / 条件路由 |
| D3 | 记忆系统：短期历史 / 持久化 / 向量记忆 |
| D4 | RAG：文档分割 / 嵌入 / 检索 / 生成 |
| D5 | Tool：工具定义 / ReAct Agent |
| D6 | LangSmith：追踪 / 评估 / 监控 |

---

## 系统设计

```
用户输入
    ↓
[意图分类]
    ├── FAQ 问题 → [RAG 检索知识库] → 回答
    ├── 订单查询 → [order_tool] → 查询订单系统
    ├── 退款申请 → [refund_tool] → 处理退款
    └── 投诉/复杂问题 → [转人工] → 通知人工客服
    ↓
[对话历史管理]  ← 保证多轮对话连贯
    ↓
最终回复（带来源引用）
```

---

## 完整实现

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langsmith import traceable
import json

# ============================================================
# Step 1: 构建知识库（FAQ）
# ============================================================

faqs = [
    Document(page_content="退款政策：购买后7天内可无理由退款，超过7天需联系客服审核。退款到账时间为3-5个工作日。", metadata={"category": "退款"}),
    Document(page_content="配送时间：北京、上海、广州等一线城市次日达，其他城市2-3天。偏远地区5-7天。", metadata={"category": "配送"}),
    Document(page_content="会员权益：黄金会员享受9折优惠，铂金会员享受8折优惠，钻石会员享受7折优惠并享受专属客服。", metadata={"category": "会员"}),
    Document(page_content="售后保障：所有商品提供1年质保，电子产品提供2年质保。质保期内免费维修或换新。", metadata={"category": "售后"}),
    Document(page_content="积分规则：每消费1元积1分，100积分可兑换1元优惠券。积分有效期为1年。", metadata={"category": "积分"}),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(faqs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ============================================================
# Step 2: 定义工具
# ============================================================

# 模拟订单数据库
ORDERS = {
    "ORD001": {"status": "已发货", "product": "iPhone 15", "tracking": "SF1234567"},
    "ORD002": {"status": "待发货", "product": "MacBook Pro", "eta": "明日发出"},
    "ORD003": {"status": "已完成", "product": "AirPods Pro", "refund_eligible": True},
}

@tool
def query_order(order_id: str) -> str:
    """查询订单状态。输入订单号（如 ORD001），返回订单详情。"""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"未找到订单号 {order_id}，请确认订单号是否正确。"
    
    status = order["status"]
    product = order["product"]
    details = []
    
    if "tracking" in order:
        details.append(f"快递单号：{order['tracking']}")
    if "eta" in order:
        details.append(f"预计：{order['eta']}")
    
    return f"订单 {order_id.upper()}：{product}，状态：{status}。{' '.join(details)}"

@tool
def submit_refund(order_id: str, reason: str) -> str:
    """提交退款申请。需要提供订单号和退款原因。"""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"未找到订单 {order_id}。"
    if not order.get("refund_eligible", False):
        return f"订单 {order_id} 当前状态（{order['status']}）不支持直接退款，请联系人工客服处理。"
    
    # 实际场景中调用退款 API
    return f"退款申请已提交！订单 {order_id.upper()}（{order['product']}），原因：{reason}。预计3-5个工作日退回。申请编号：REF{hash(order_id) % 10000:04d}"

@tool
def escalate_to_human(reason: str) -> str:
    """将对话转接给人工客服。当问题复杂、用户情绪激动或需要特殊处理时使用。"""
    # 实际场景中发送通知给客服团队
    return f"已为您转接人工客服，预计等待时间：3-5分钟。转接原因：{reason}。感谢您的耐心等待。"

@tool  
def search_faq(query: str) -> str:
    """在知识库中搜索常见问题的答案。用于回答退款政策、配送时间、会员权益等通用问题。"""
    docs = retriever.invoke(query)
    if not docs:
        return "暂无相关FAQ，建议转接人工客服。"
    return "\n".join(f"- {doc.page_content}" for doc in docs)

# ============================================================
# Step 3: 构建 Agent
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [query_order, submit_refund, escalate_to_human, search_faq]

system_prompt = """你是「丹丹商城」的智能客服助手，名字叫「小丹」。

你的职责：
1. 友好、专业地回答用户问题
2. 查询订单状态（需要用户提供订单号）
3. 处理退款申请
4. 回答商城政策类问题（使用搜索工具）
5. 复杂问题转接人工

注意事项：
- 始终保持礼貌和耐心
- 不要编造信息，不确定的信息使用工具查询
- 用户提到订单时，主动询问订单号
- 用户情绪激动时，先安抚，再解决问题

开场白：「您好！我是丹丹商城智能客服小丹，很高兴为您服务！请问有什么可以帮助您的？」"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)

# ============================================================
# Step 4: 带历史的对话循环
# ============================================================

@traceable(name="customer_service_session")
def run_customer_service():
    history = []
    print("小丹：您好！我是丹丹商城智能客服小丹，很高兴为您服务！请问有什么可以帮助您的？")
    print("（输入 'quit' 退出）\n")
    
    while True:
        user_input = input("您：").strip()
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("小丹：感谢您的联系，祝您生活愉快！再见！")
            break
        
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": history
        })
        
        answer = response["output"]
        print(f"小丹：{answer}\n")
        
        # 更新历史
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))

# 运行（交互式对话）
if __name__ == "__main__":
    run_customer_service()
```

---

## 测试脚本

```python
# 非交互式测试（验证各功能）
test_cases = [
    "你好",
    "你们的退款政策是什么？",
    "我的订单 ORD001 发货了吗？",
    "我想退款，订单号是 ORD003",
    "我非常不满意！你们的服务太差了！",
]

history = []
for user_input in test_cases:
    print(f"用户：{user_input}")
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": history
    })
    answer = response["output"]
    print(f"小丹：{answer}\n")
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
```

---

## Week 3 完成！

🎉 **Agent Week 3 全部完成！** 

- ✅ LangChain 全栈：Prompt / Chain / Agent / Memory / RAG / Tool
- ✅ LCEL 管道：流式、并行、条件路由
- ✅ RAG 完整流程：分割→嵌入→检索→生成
- ✅ 工具系统：自定义工具 + ReAct Agent
- ✅ 工程实战：可运行的智能客服

**Week 4 预告**：OpenAI Assistants API 深度实战 + Function Calling 进阶
