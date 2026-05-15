---
layout: default
title: "W8 Capstone · 生产级 Agent 系统"
---

# Capstone：构建生产级 Agent 系统

> **Week 8 · Capstone** | 难度：⭐⭐⭐⭐⭐

---

## 项目：企业智能助手系统

整合 Week 8 全部技术，构建一个真实可用的企业知识库问答 Agent，具备：
- 可观测性（日志 + 指标）
- 容错机制（重试 + 熔断 + 降级）
- 成本优化（缓存 + 模型分级）
- 完整 API 服务（FastAPI + 异步）

## 系统架构

```
用户请求
   │
   ▼
┌──────────────┐
│  FastAPI     │  ← 请求验证、限流、路由
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│                  Agent 核心层                     │
│                                                  │
│  ┌─────────────┐    ┌────────────────────────┐  │
│  │ 任务分类器   │    │    ReAct Agent          │  │
│  │ （模型路由） │ →  │  gpt-4o / gpt-4o-mini  │  │
│  └─────────────┘    └──────────┬─────────────┘  │
│                                │                 │
│              ┌─────────────────┤                 │
│              ▼                 ▼                 │
│         ┌─────────┐      ┌─────────┐            │
│         │ 知识检索 │      │  工具集  │            │
│         │ (RAG)   │      │ 搜索/计算│            │
│         └─────────┘      └─────────┘            │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│         基础设施层                │
│  Redis（缓存+会话）               │
│  SQLite/向量DB（知识存储）         │
│  结构化日志 + Prometheus 指标     │
└──────────────────────────────────┘
```

## 完整实现

```python
# enterprise_agent.py
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# ═══════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    
    def _log(self, level: str, msg: str, **kw):
        rec = {"ts": datetime.utcnow().isoformat()+"Z", "level": level,
               "msg": msg, **kw}
        getattr(self.logger, level.lower())(json.dumps(rec, ensure_ascii=False))
    
    def info(self, msg, **kw): self._log("INFO", msg, **kw)
    def warning(self, msg, **kw): self._log("WARNING", msg, **kw)
    def error(self, msg, **kw): self._log("ERROR", msg, **kw)

log = StructuredLogger("enterprise-agent")

# ═══════════════════════════════════════
# 成本追踪
# ═══════════════════════════════════════

class CostTracker:
    MODEL_PRICING = {
        "gpt-4o":      {"input": 0.000003, "output": 0.000015},
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.call_count = 0
    
    def record(self, model: str, input_tokens: int, output_tokens: int):
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["gpt-4o-mini"])
        cost = (input_tokens * pricing["input"] + 
                output_tokens * pricing["output"])
        self.total_cost += cost
        self.call_count += 1
        return cost

cost_tracker = CostTracker()

# ═══════════════════════════════════════
# 简单内存缓存（生产用 Redis 替代）
# ═══════════════════════════════════════

class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self._store: Dict[str, tuple] = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[str]:
        k = self._key(text)
        if k in self._store:
            val, ts = self._store[k]
            if time.time() - ts < self.ttl:
                self.hits += 1
                return val
        self.misses += 1
        return None
    
    def set(self, text: str, value: str):
        self._store[self._key(text)] = (value, time.time())

cache = SimpleCache(ttl=3600)

# ═══════════════════════════════════════
# 熔断器
# ═══════════════════════════════════════

class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.opened_at = None
        self.open = False
    
    def call(self, func, *args, **kwargs):
        # 检查是否可以半开
        if self.open:
            if time.time() - self.opened_at > self.timeout:
                self.open = False
                self.failures = 0
                log.info("熔断器半开，尝试恢复")
            else:
                raise Exception("服务熔断中，请稍后重试")
        
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open = True
                self.opened_at = time.time()
                log.warning("熔断器开路", failures=self.failures)
            raise

breaker = CircuitBreaker(threshold=5, timeout=60)

# ═══════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════

@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math,
            "abs": abs, "round": round, "sum": sum, "min": min, "max": max})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库（模拟）"""
    # 实际实现：向量检索 → Qdrant/Chroma
    mock_kb = {
        "部署": "系统支持 Docker 和 K8s 部署。镜像：company/service:latest，端口 8080。",
        "api": "REST API 文档：https://docs.internal.company.com/api",
        "监控": "使用 Prometheus + Grafana 监控。告警阈值：CPU>80%, 错误率>5%",
        "成本": "月均 LLM 成本约 $200，通过缓存优化后降至 $45",
    }
    
    for key, value in mock_kb.items():
        if key.lower() in query.lower():
            return f"知识库结果：{value}"
    
    return "知识库中未找到相关信息，建议联系相关团队。"

# ═══════════════════════════════════════
# Agent 核心
# ═══════════════════════════════════════

class EnterpriseAgent:
    SYSTEM_PROMPT = """你是企业智能助手，帮助员工解答问题和完成任务。

能力：
- 搜索内部知识库
- 数学计算
- 获取当前时间

原则：
1. 优先查询知识库
2. 不确定时明确说明
3. 保密不对外透露内部信息
4. 回答简洁、专业"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.tools = [calculate, get_time, search_knowledge_base]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=self.tools,
            verbose=False, max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    async def run(self, task: str, session_id: str) -> Dict[str, Any]:
        # 查缓存
        cached = cache.get(f"{self.model}:{task}")
        if cached:
            log.info("缓存命中", session_id=session_id)
            return {"output": cached, "from_cache": True, "steps": 0}
        
        start = time.time()
        
        try:
            def _execute():
                return self.executor.invoke({"input": task})
            
            result = await asyncio.to_thread(
                lambda: breaker.call(_execute)
            )
            
            duration_ms = (time.time() - start) * 1000
            output = result["output"]
            steps = len(result.get("intermediate_steps", []))
            
            # 写缓存
            cache.set(f"{self.model}:{task}", output)
            
            log.info("任务完成",
                     session_id=session_id,
                     steps=steps,
                     duration_ms=round(duration_ms, 1),
                     cache_hit_rate=f"{cache.hits/(cache.hits+cache.misses+1):.1%}")
            
            return {
                "output": output,
                "from_cache": False,
                "steps": steps,
                "duration_ms": duration_ms
            }
        
        except Exception as e:
            log.error("任务失败", session_id=session_id, error=str(e))
            raise

# ═══════════════════════════════════════
# FastAPI 服务
# ═══════════════════════════════════════

app = FastAPI(title="Enterprise Agent API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

agent = EnterpriseAgent(model="gpt-4o-mini")

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    session_id: str
    from_cache: bool
    steps: int
    duration_ms: float

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cache_hit_rate": f"{cache.hits/(cache.hits+cache.misses+1):.1%}",
        "circuit_breaker_open": breaker.open,
        "total_cost_usd": round(cost_tracker.total_cost, 4)
    }

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    try:
        result = await agent.run(request.question, session_id)
        return AskResponse(
            answer=result["output"],
            session_id=session_id,
            from_cache=result["from_cache"],
            steps=result["steps"],
            duration_ms=result.get("duration_ms", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════
# 本地测试
# ═══════════════════════════════════════

async def test():
    print("=== 企业 Agent 系统测试 ===\n")
    
    test_cases = [
        "我们系统怎么部署？",
        "今天几号？",
        "计算 1234 * 5678",
        "我们系统怎么部署？",  # 测试缓存
    ]
    
    for q in test_cases:
        print(f"Q: {q}")
        result = await agent.run(q, "test")
        cache_mark = "🎯 缓存" if result["from_cache"] else f"⚙️ {result['steps']}步"
        print(f"A: {result['output'][:150]}")
        print(f"   [{cache_mark} | {result.get('duration_ms', 0):.0f}ms]\n")
    
    print(f"缓存命中率：{cache.hits/(cache.hits+cache.misses+1):.1%}")

if __name__ == "__main__":
    asyncio.run(test())
    # 启动服务：uvicorn enterprise_agent:app --reload
```

## 本系列完整技术图谱

```
Week 4: 规划与推理
  CoT → ToT → ReAct进阶 → 自我反思 → 任务分解 → 规划引擎

Week 5: 多Agent系统
  通信协议 → 角色分工 → 协作框架 → AutoGen → LangGraph → 监控

Week 6: 记忆系统
  短期记忆 → 向量DB → 长期记忆 → 情景语义记忆 → 记忆模式

Week 7: 工具与环境
  工具设计 → 浏览器 → 代码沙箱 → API集成 → 安全隔离 → 工具链编排

Week 8: 生产级系统 ← 本周
  可观测性 → 评估体系 → 容错设计 → 成本控制 → 部署扩展 → 面试精讲
```

---

**恭喜完成 Agent 线 Week 4-8 全部学习！你现在具备了构建生产级 Agent 系统的完整技能树。**

*W8 Capstone · 生产级 Agent 系统 | Agent + Claw 系列*
