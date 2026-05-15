---
layout: default
title: "W8D4 · 成本控制与优化"
---

# 成本控制：让 Agent 既强大又经济

> **Week 8 · Day 4** | 难度：⭐⭐⭐⭐

---

## 成本结构分析

```
Agent 成本来源（典型分布）：
┌────────────────────────────────────────────────┐
│  LLM API 调用         ████████████████  70-85% │
│  向量数据库            ███              5-10%  │
│  搜索/外部 API         ██               3-8%   │
│  存储/计算             █                2-5%   │
└────────────────────────────────────────────────┘

LLM 成本 = Input Tokens × 输入单价 + Output Tokens × 输出单价
         = 约$0.003/1K 到 $0.015/1K（gpt-4o，2024年）
```

## 成本追踪

```python
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from typing import Dict
import json

class CostTracker:
    """成本追踪器"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.session_costs: Dict[str, float] = {}
    
    def track(self, func):
        """装饰器：追踪函数的 LLM 成本"""
        def wrapper(*args, **kwargs):
            with get_openai_callback() as cb:
                result = func(*args, **kwargs)
                
                self.total_tokens += cb.total_tokens
                self.total_cost_usd += cb.total_cost
                self.call_count += 1
                
                print(f"本次调用：{cb.total_tokens} tokens, ${cb.total_cost:.6f}")
            
            return result
        return wrapper
    
    def report(self) -> str:
        avg_cost = self.total_cost_usd / max(self.call_count, 1)
        return (f"总调用：{self.call_count}次 | "
                f"总 Token：{self.total_tokens:,} | "
                f"总成本：${self.total_cost_usd:.4f} | "
                f"平均成本：${avg_cost:.6f}/次")

tracker = CostTracker()

@tracker.track
def run_expensive_agent(task: str) -> str:
    llm = ChatOpenAI(model="gpt-4o")
    return llm.invoke(task).content

run_expensive_agent("分析大模型在企业中的应用场景")
print(tracker.report())
```

## 优化策略1：模型分级

```python
from langchain_openai import ChatOpenAI

# 不同任务用不同模型
MODELS = {
    "simple": "gpt-4o-mini",    # $0.00015/1K tokens
    "standard": "gpt-4o",       # $0.003/1K tokens  
    "complex": "gpt-4o",        # $0.003/1K tokens
}

# 成本比：gpt-4o vs gpt-4o-mini ≈ 20:1
# 简单任务用 mini 可节省约90%成本

def classify_task_complexity(task: str) -> str:
    """分类任务复杂度，选择合适模型"""
    # 规则层快速分类
    if len(task) < 50 and not any(kw in task for kw in ["分析", "比较", "规划", "设计"]):
        return "simple"
    elif any(kw in task for kw in ["详细分析", "全面评估", "系统设计", "多角度"]):
        return "complex"
    else:
        return "standard"

def adaptive_llm_call(task: str) -> str:
    """根据任务复杂度自动选择模型"""
    complexity = classify_task_complexity(task)
    model = MODELS[complexity]
    
    llm = ChatOpenAI(model=model, temperature=0)
    print(f"任务复杂度：{complexity}，使用模型：{model}")
    return llm.invoke(task).content

# 测试
adaptive_llm_call("今天几号？")  # → simple → gpt-4o-mini
adaptive_llm_call("详细分析LangGraph在生产环境中的性能优化策略")  # → complex → gpt-4o
```

## 优化策略2：Prompt 压缩

```python
from langchain_openai import ChatOpenAI

class PromptCompressor:
    """Prompt 压缩：减少输入 Token"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def compress_context(self, context: str, max_tokens: int = 1000) -> str:
        """压缩上下文，保留关键信息"""
        # 简单估算：1 token ≈ 1.5 中文字
        estimated_tokens = len(context) / 1.5
        
        if estimated_tokens <= max_tokens:
            return context  # 不需要压缩
        
        compression_ratio = max_tokens / estimated_tokens
        target_chars = int(len(context) * compression_ratio)
        
        prompt = f"""将以下内容压缩到约 {target_chars} 字，保留所有关键信息：

{context}

压缩后内容："""
        
        return self.llm.invoke(prompt).content
    
    def remove_redundancy(self, text: str) -> str:
        """去除重复和冗余信息"""
        prompt = f"""去除以下文本中的重复和冗余内容，同时保持信息完整性：

{text}

精简后的版本："""
        return self.llm.invoke(prompt).content

compressor = PromptCompressor()
long_context = "这是很长的上下文..." * 100
compressed = compressor.compress_context(long_context, max_tokens=500)
print(f"压缩前：{len(long_context)} 字 → 压缩后：{len(compressed)} 字")
```

## 优化策略3：缓存层

```python
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta

class LLMResponseCache:
    """LLM 响应缓存：相同请求不重复调用"""
    
    def __init__(self, db_path: str = "/tmp/llm_cache.db", ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl = timedelta(hours=ttl_hours)
        self._init_db()
        self.hit_count = 0
        self.miss_count = 0
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT,
                created_at TEXT,
                model TEXT,
                token_count INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def _make_key(self, prompt: str, model: str, temperature: float) -> str:
        content = json.dumps({"prompt": prompt, "model": model, "temperature": temperature})
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float = 0) -> str:
        """获取缓存的响应"""
        key = self._make_key(prompt, model, temperature)
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute(
            "SELECT response, created_at FROM cache WHERE key=?", (key,)
        ).fetchone()
        
        conn.close()
        
        if row:
            created_at = datetime.fromisoformat(row[1])
            if datetime.now() - created_at < self.ttl:
                self.hit_count += 1
                return row[0]
        
        self.miss_count += 1
        return None
    
    def set(self, prompt: str, model: str, response: str, 
            temperature: float = 0, token_count: int = 0):
        """缓存响应"""
        key = self._make_key(prompt, model, temperature)
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO cache (key, response, created_at, model, token_count)
            VALUES (?, ?, ?, ?, ?)
        """, (key, response, datetime.now().isoformat(), model, token_count))
        
        conn.commit()
        conn.close()
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0
    
    def stats(self) -> str:
        return (f"缓存命中：{self.hit_count} | "
                f"未命中：{self.miss_count} | "
                f"命中率：{self.hit_rate:.1%}")

# 带缓存的 LLM 调用
cache = LLMResponseCache(ttl_hours=24)

def cached_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    # 先查缓存
    cached = cache.get(prompt, model)
    if cached:
        print("🎯 缓存命中")
        return cached
    
    # 调用 LLM
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(prompt)
    result = response.content
    
    # 存入缓存
    cache.set(prompt, model, result)
    return result

# 第一次调用
result1 = cached_llm_call("Python 和 Go 的主要区别是什么？")
# 第二次调用同一问题 → 缓存命中
result2 = cached_llm_call("Python 和 Go 的主要区别是什么？")
print(cache.stats())
```

## 优化策略4：批量处理

```python
import asyncio
from typing import List

class BatchLLMProcessor:
    """批量 LLM 处理：减少 overhead，提升吞吐"""
    
    def __init__(self, model: str = "gpt-4o-mini", batch_size: int = 10):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.batch_size = batch_size
    
    async def process_many(self, prompts: List[str]) -> List[str]:
        """批量处理多个 prompt"""
        results = []
        
        # 分批处理
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            print(f"处理批次 {i//self.batch_size + 1}: {len(batch)} 个请求")
            
            # 并发处理同一批次
            tasks = [self.llm.ainvoke(p) for p in batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for resp in responses:
                if isinstance(resp, Exception):
                    results.append(f"错误：{resp}")
                else:
                    results.append(resp.content)
        
        return results

processor = BatchLLMProcessor()

async def demo():
    tasks = [
        "用一句话解释量子纠缠",
        "Python 的 GIL 是什么",
        "什么是服务网格",
        "解释 CAP 定理",
        "什么是边缘计算",
    ]
    
    with get_openai_callback() as cb:
        results = await processor.process_many(tasks)
        print(f"批量处理完成，总 token：{cb.total_tokens}，成本：${cb.total_cost:.6f}")
    
    for task, result in zip(tasks, results):
        print(f"Q: {task}\nA: {result[:100]}\n")

asyncio.run(demo())
```

## 成本优化效果汇总

```
优化策略              预期节省    适用场景
──────────────────────────────────────────────
模型分级              40-70%     有简单任务的混合场景
Prompt 压缩           20-40%     上下文很长的场景
响应缓存              30-80%     有重复查询的场景
批量处理              10-30%     大量小请求场景
输出长度控制          10-30%     输出普遍过长的场景
```

## 踩坑经验

### 坑1：缓存导致用户获取过时信息

**问题**：新闻、天气等实时信息被缓存，用户获取到过时数据。  
**解法**：区分内容类型，实时信息设 TTL=0（不缓存），稳定信息可缓存更长时间。

### 坑2：过度优化导致质量下降

**问题**：为节省成本把所有任务都路由到 gpt-4o-mini，复杂分析任务质量大幅下降。  
**解法**：先建立质量基线（用 gpt-4o 跑全部评估），然后逐步下放任务到小模型，持续监控质量指标。

---

*W8D4 · 成本控制与优化 | Agent + Claw 系列*
