---
layout: default
title: "W8D1 · Agent 可观测性"
---

# Agent 可观测性：生产系统的眼睛

> **Week 8 · Day 1** | 难度：⭐⭐⭐⭐

---

## 可观测性的三大支柱

```
┌─────────────────────────────────────────────────────┐
│                 可观测性三支柱                        │
│                                                     │
│  Logs（日志）    Traces（追踪）   Metrics（指标）      │
│  "发生了什么"    "为什么发生"     "发生了多少"          │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 结构化   │  │ 分布式   │  │ 延迟/错误率/      │  │
│  │ 日志     │  │ 链路追踪 │  │ Token用量/成本   │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## LangSmith：LangChain 官方可观测平台

```python
import os
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled

# 设置环境变量
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# 使用追踪（自动记录所有 LLM 调用）
llm = ChatOpenAI(model="gpt-4o")

with tracing_v2_enabled():
    response = llm.invoke("解释什么是可观测性")
    # 这次调用会自动记录到 LangSmith

# 手动创建 run
from langsmith import traceable

@traceable(name="my_agent_task", tags=["production"])
def run_agent_task(task: str) -> str:
    """被追踪的 Agent 任务"""
    result = llm.invoke(task)
    return result.content

result = run_agent_task("分析最新AI趋势")

# 查询运行数据
client = Client()
runs = list(client.list_runs(project_name="my-agent-project", limit=10))
for run in runs:
    print(f"Run: {run.name}, Status: {run.status}, "
          f"Tokens: {run.total_tokens}, "
          f"Latency: {run.end_time - run.start_time if run.end_time else 'N/A'}")
```

## 自定义结构化日志

```python
import logging
import json
import uuid
from datetime import datetime
from typing import Any, Optional
from contextvars import ContextVar

# 上下文变量存储追踪 ID
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')

class AgentLogger:
    """Agent 专用结构化日志器"""
    
    def __init__(self, service_name: str, log_file: str = None):
        self.service_name = service_name
        
        logger = logging.getLogger(f"agent.{service_name}")
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(message)s')
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件输出
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.logger = logger
    
    def _base_record(self, level: str, message: str, **extra) -> dict:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service_name,
            "trace_id": trace_id_var.get() or str(uuid.uuid4())[:8],
            "message": message,
            **extra
        }
    
    def info(self, message: str, **kwargs):
        record = self._base_record("INFO", message, **kwargs)
        self.logger.info(json.dumps(record, ensure_ascii=False))
    
    def warning(self, message: str, **kwargs):
        record = self._base_record("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(record, ensure_ascii=False))
    
    def error(self, message: str, exc: Exception = None, **kwargs):
        if exc:
            kwargs["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        record = self._base_record("ERROR", message, **kwargs)
        self.logger.error(json.dumps(record, ensure_ascii=False))
    
    def llm_call(self, model: str, tokens: int, latency_ms: float, 
                 cost_usd: float = None, **kwargs):
        """记录 LLM 调用"""
        record = self._base_record(
            "INFO", "llm_call",
            model=model,
            tokens=tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6) if cost_usd else None,
            **kwargs
        )
        self.logger.info(json.dumps(record, ensure_ascii=False))
    
    def tool_call(self, tool_name: str, success: bool, 
                 latency_ms: float, **kwargs):
        """记录工具调用"""
        record = self._base_record(
            "INFO", "tool_call",
            tool_name=tool_name,
            success=success,
            latency_ms=round(latency_ms, 2),
            **kwargs
        )
        self.logger.info(json.dumps(record, ensure_ascii=False))

# 使用示例
logger = AgentLogger("research-agent", "/tmp/agent.log")

# 设置追踪 ID
trace_id_var.set(str(uuid.uuid4())[:8])

logger.info("Agent 启动", task="研究量子计算趋势")
logger.llm_call("gpt-4o", tokens=1500, latency_ms=1200.5, cost_usd=0.015)
logger.tool_call("web_search", success=True, latency_ms=850.2)
```

## OpenTelemetry 集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化 TracerProvider
provider = TracerProvider()

# 导出到控制台（开发阶段）
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# 导出到 OTLP 收集器（生产阶段，如 Jaeger/Tempo）
# otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
# provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent.tracer")

class InstrumentedAgent:
    """集成 OpenTelemetry 的 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
    
    def run_task(self, task: str) -> str:
        """带追踪的任务执行"""
        with tracer.start_as_current_span("agent.run_task") as span:
            span.set_attribute("agent.task", task[:200])
            span.set_attribute("agent.model", "gpt-4o")
            
            try:
                # 子 span：LLM 调用
                with tracer.start_as_current_span("llm.invoke") as llm_span:
                    import time
                    start = time.time()
                    response = self.llm.invoke(task)
                    duration = (time.time() - start) * 1000
                    
                    llm_span.set_attribute("llm.latency_ms", duration)
                    llm_span.set_attribute("llm.output_length", len(response.content))
                
                span.set_attribute("agent.status", "success")
                return response.content
            
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

agent = InstrumentedAgent()
result = agent.run_task("分析大模型的最新进展")
```

## 监控告警规则

```python
from dataclasses import dataclass
from typing import Callable, List
import time
from collections import deque

@dataclass
class AlertRule:
    name: str
    check: Callable
    threshold: float
    window_size: int   # 滑动窗口大小
    cooldown: int = 300  # 告警冷却时间（秒）
    
    def __post_init__(self):
        self._last_alert_time = 0
    
    def should_alert(self) -> bool:
        return time.time() - self._last_alert_time > self.cooldown
    
    def mark_alerted(self):
        self._last_alert_time = time.time()

class AgentAlertSystem:
    """Agent 告警系统"""
    
    def __init__(self):
        self.metrics = {
            "error_rate": deque(maxlen=100),
            "latency_ms": deque(maxlen=100),
            "cost_usd": deque(maxlen=100),
            "token_count": deque(maxlen=100),
        }
        
        # 告警规则
        self.rules = [
            AlertRule(
                name="高错误率",
                check=lambda m: sum(m["error_rate"]) / max(len(m["error_rate"]), 1),
                threshold=0.1,  # 10% 错误率
                window_size=50
            ),
            AlertRule(
                name="高延迟",
                check=lambda m: sum(m["latency_ms"]) / max(len(m["latency_ms"]), 1),
                threshold=5000,  # 5秒
                window_size=20
            ),
            AlertRule(
                name="高成本",
                check=lambda m: sum(list(m["cost_usd"])[-10:]),
                threshold=1.0,  # 最近10次调用成本超$1
                window_size=10
            ),
        ]
    
    def record(self, success: bool, latency_ms: float, 
               cost_usd: float = 0, tokens: int = 0):
        """记录一次 Agent 调用"""
        self.metrics["error_rate"].append(0 if success else 1)
        self.metrics["latency_ms"].append(latency_ms)
        self.metrics["cost_usd"].append(cost_usd)
        self.metrics["token_count"].append(tokens)
        
        # 检查告警
        self._check_alerts()
    
    def _check_alerts(self):
        for rule in self.rules:
            if not rule.should_alert():
                continue
            
            value = rule.check(self.metrics)
            if value > rule.threshold:
                self._send_alert(rule, value)
                rule.mark_alerted()
    
    def _send_alert(self, rule: AlertRule, value: float):
        """发送告警（实际应接 PagerDuty/Slack）"""
        print(f"🚨 告警：{rule.name} | 当前值：{value:.2f} | 阈值：{rule.threshold}")

# 使用
alert_system = AgentAlertSystem()
alert_system.record(success=True, latency_ms=1200, cost_usd=0.012, tokens=1500)
alert_system.record(success=False, latency_ms=8000, cost_usd=0)  # 高延迟+失败
```

## 踩坑经验

### 坑1：日志量太大，存储爆炸

**解法**：
1. 只记录 INFO 及以上级别到持久化存储
2. DEBUG 日志只在开发环境启用
3. 设置日志轮转（RotatingFileHandler）

### 坑2：追踪 ID 在异步场景下丢失

**问题**：asyncio 任务切换时，contextvars 的 trace_id 正确继承，但线程池会丢失。  
**解法**：使用 `asyncio.create_task` 时显式传递 trace_id；线程池用 `contextvars.copy_context()` 。

---

*W8D1 · Agent 可观测性 | Agent + Claw 系列*
