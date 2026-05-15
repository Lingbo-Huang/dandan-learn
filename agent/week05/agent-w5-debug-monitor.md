---
layout: default
title: "W5D6 · 多 Agent 调试与监控"
---

# 多 Agent 调试与监控：生产系统的必备能力

> **Week 5 · Day 6** | 难度：⭐⭐⭐⭐

---

## 为什么多 Agent 系统特别难调试？

单 Agent 出问题好定位，多 Agent 就复杂得多：
- **非确定性**：同样输入可能走不同路径
- **涌现行为**：单个 Agent 正常，组合起来出问题
- **消息追踪**：消息在 Agent 间流转，很难追踪
- **级联失败**：一个 Agent 失败，级联影响其他 Agent

## 全链路追踪系统

```python
import uuid
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class TraceSpan:
    """单个追踪 span"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: Optional[str] = None
    trace_id: str = ""
    name: str = ""
    agent_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"  # running/success/error
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    token_usage: Dict = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def complete(self, output: Any = None, error: str = None):
        self.end_time = time.time()
        self.output_data = output
        if error:
            self.status = "error"
            self.error = error
        else:
            self.status = "success"

class MultiAgentTracer:
    """多 Agent 全链路追踪器"""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
    
    def start_trace(self, name: str) -> str:
        """开始一条新的追踪链"""
        trace_id = str(uuid.uuid4())[:12]
        self.traces[trace_id] = []
        return trace_id
    
    @contextmanager
    def span(self, name: str, agent_id: str, trace_id: str, 
             parent_span_id: str = None, input_data: Any = None):
        """创建追踪 span 的上下文管理器"""
        span = TraceSpan(
            name=name,
            agent_id=agent_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            input_data=input_data
        )
        self.traces[trace_id].append(span)
        self.active_spans[span.span_id] = span
        
        try:
            yield span
            span.complete(output=span.output_data)
        except Exception as e:
            span.complete(error=str(e))
            raise
        finally:
            self.active_spans.pop(span.span_id, None)
    
    def get_trace_summary(self, trace_id: str) -> dict:
        """获取追踪摘要"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return {}
        
        total_duration = sum(s.duration_ms or 0 for s in spans)
        error_spans = [s for s in spans if s.status == "error"]
        
        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "success_rate": (len(spans) - len(error_spans)) / len(spans),
            "errors": [{"span": s.name, "error": s.error} for s in error_spans],
            "spans": [
                {
                    "name": s.name,
                    "agent": s.agent_id,
                    "duration_ms": s.duration_ms,
                    "status": s.status
                }
                for s in spans
            ]
        }
    
    def visualize_trace(self, trace_id: str) -> str:
        """ASCII 可视化追踪树"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return "无追踪数据"
        
        lines = [f"追踪 ID: {trace_id}", ""]
        
        for span in spans:
            indent = "  " if span.parent_span_id else ""
            status_icon = "✅" if span.status == "success" else "❌" if span.status == "error" else "🔄"
            duration = f"{span.duration_ms:.0f}ms" if span.duration_ms else "进行中"
            
            lines.append(f"{indent}{status_icon} [{span.agent_id}] {span.name} ({duration})")
            if span.error:
                lines.append(f"{indent}   ⚠️ 错误：{span.error[:100]}")
        
        return "\n".join(lines)

# 使用示例
tracer = MultiAgentTracer()

def traced_agent_call(agent_id: str, task: str, tracer: MultiAgentTracer, 
                      trace_id: str, parent_span_id: str = None):
    """带追踪的 Agent 调用"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    with tracer.span(f"agent_call_{agent_id}", agent_id, trace_id, parent_span_id, task) as span:
        response = llm.invoke(task)
        span.output_data = response.content
        span.token_usage = response.response_metadata.get("token_usage", {})
        return response.content

# 模拟多 Agent 调用链
trace_id = tracer.start_trace("research_pipeline")

research_result = traced_agent_call("researcher", "研究Python异步编程", tracer, trace_id)
analysis_result = traced_agent_call("analyst", f"分析：{research_result[:200]}", tracer, trace_id)

print(tracer.visualize_trace(trace_id))
summary = tracer.get_trace_summary(trace_id)
print(f"\n成功率：{summary['success_rate']:.0%}")
print(f"总耗时：{summary['total_duration_ms']:.0f}ms")
```

## 实时监控 Dashboard

```python
import asyncio
from collections import defaultdict, deque

class AgentMonitor:
    """实时 Agent 监控"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # 滑动窗口存储最近的调用记录
        self.call_history: deque = deque(maxlen=window_size)
        self.agent_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "total_latency_ms": 0,
            "total_tokens": 0,
        })
    
    def record_call(self, agent_id: str, latency_ms: float, 
                   success: bool, tokens: int = 0):
        """记录一次 Agent 调用"""
        self.call_history.append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "latency_ms": latency_ms,
            "success": success,
            "tokens": tokens
        })
        
        metrics = self.agent_metrics[agent_id]
        metrics["total_calls"] += 1
        metrics["total_latency_ms"] += latency_ms
        metrics["total_tokens"] += tokens
        if success:
            metrics["success_calls"] += 1
        else:
            metrics["error_calls"] += 1
    
    def get_dashboard(self) -> str:
        """生成监控 Dashboard"""
        lines = ["=" * 60, "Multi-Agent 实时监控", "=" * 60, ""]
        
        for agent_id, metrics in self.agent_metrics.items():
            total = metrics["total_calls"]
            if total == 0:
                continue
            
            success_rate = metrics["success_calls"] / total
            avg_latency = metrics["total_latency_ms"] / total
            
            lines.append(f"Agent: {agent_id}")
            lines.append(f"  调用次数：{total}")
            lines.append(f"  成功率：{success_rate:.1%}")
            lines.append(f"  平均延迟：{avg_latency:.0f}ms")
            lines.append(f"  总 token：{metrics['total_tokens']:,}")
            lines.append("")
        
        # 最近错误
        recent_errors = [c for c in self.call_history if not c["success"]][-5:]
        if recent_errors:
            lines.append("最近错误：")
            for err in recent_errors:
                lines.append(f"  [{err['agent_id']}] {datetime.fromtimestamp(err['timestamp']).strftime('%H:%M:%S')}")
        
        return "\n".join(lines)
    
    def get_alerts(self, 
                  error_rate_threshold: float = 0.2,
                  latency_threshold_ms: float = 5000) -> List[str]:
        """检查是否触发告警"""
        alerts = []
        
        for agent_id, metrics in self.agent_metrics.items():
            total = metrics["total_calls"]
            if total < 5:  # 样本太少，不告警
                continue
            
            error_rate = metrics["error_calls"] / total
            avg_latency = metrics["total_latency_ms"] / total
            
            if error_rate > error_rate_threshold:
                alerts.append(
                    f"⚠️ 高错误率告警：{agent_id} 错误率 {error_rate:.1%}"
                )
            
            if avg_latency > latency_threshold_ms:
                alerts.append(
                    f"⚠️ 高延迟告警：{agent_id} 平均延迟 {avg_latency:.0f}ms"
                )
        
        return alerts

# 使用监控
monitor = AgentMonitor()

# 模拟一些调用记录
import random
for i in range(20):
    agent = random.choice(["researcher", "analyst", "writer"])
    latency = random.uniform(500, 3000)
    success = random.random() > 0.1  # 90% 成功率
    tokens = random.randint(100, 1000)
    monitor.record_call(agent, latency, success, tokens)

print(monitor.get_dashboard())

alerts = monitor.get_alerts()
if alerts:
    print("告警：")
    for alert in alerts:
        print(f"  {alert}")
```

## 调试工具：Agent 通信回放

```python
class AgentDebugger:
    """多 Agent 系统调试工具"""
    
    def __init__(self, tracer: MultiAgentTracer):
        self.tracer = tracer
    
    def replay_trace(self, trace_id: str, inject_mock: Dict[str, str] = None):
        """回放追踪，可注入 mock 输出用于调试"""
        spans = self.tracer.traces.get(trace_id, [])
        print(f"回放追踪 {trace_id}（{len(spans)} 个 span）")
        
        for span in spans:
            print(f"\n[{span.agent_id}] {span.name}")
            print(f"  输入：{str(span.input_data)[:100]}")
            
            if inject_mock and span.agent_id in inject_mock:
                output = inject_mock[span.agent_id]
                print(f"  输出（注入Mock）：{output[:100]}")
            else:
                print(f"  输出：{str(span.output_data)[:100]}")
            
            print(f"  状态：{span.status}, 耗时：{span.duration_ms:.0f}ms")
    
    def find_bottleneck(self, trace_id: str) -> Optional[TraceSpan]:
        """找到最慢的 span"""
        spans = self.tracer.traces.get(trace_id, [])
        if not spans:
            return None
        
        return max(spans, key=lambda s: s.duration_ms or 0)
    
    def export_trace(self, trace_id: str, filepath: str):
        """导出追踪数据用于分析"""
        spans = self.tracer.traces.get(trace_id, [])
        data = [
            {
                "span_id": s.span_id,
                "name": s.name,
                "agent_id": s.agent_id,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "error": s.error
            }
            for s in spans
        ]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"追踪数据已导出到 {filepath}")
```

## 踩坑经验

### 坑1：日志太多淹没关键信息

**解法**：分级记录（DEBUG/INFO/WARN/ERROR），生产环境只开 INFO，调试时开 DEBUG。

### 坑2：监控数据量爆炸

**问题**：每次 token 使用都记录，数据库很快爆满。  
**解法**：采样记录（如只记录10%的调用），聚合统计用滚动窗口。

### 坑3：追踪 ID 在 Agent 间丢失

**问题**：Agent A 调用 Agent B 时，没有传递 trace_id，导致无法关联同一条追踪链。  
**解法**：用上下文变量（contextvars）或在 AgentMessage 中强制携带 trace_id。

---

*W5D6 · 多 Agent 调试与监控 | Agent + Claw 系列*
