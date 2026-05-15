---
layout: default
title: "W8D3 · 容错与可靠性"
---

# 容错设计：让 Agent 在故障中优雅降级

> **Week 8 · Day 3** | 难度：⭐⭐⭐⭐

---

## 故障类型与应对策略

```
故障类型         发生概率   影响      应对策略
──────────────────────────────────────────────
LLM API 超时      中        高        重试+超时
LLM 返回乱码      低        高        格式验证+重试
工具调用失败      中        中        降级+跳过
Context 溢出      低        高        截断+压缩
速率限制(429)     高        中        指数退避
Token 超限        中        高        分块处理
网络中断          低        高        本地缓存
```

## 重试机制

```python
import asyncio
import time
import random
from typing import TypeVar, Callable, Optional
from functools import wraps

T = TypeVar('T')

class RetryConfig:
    """重试配置"""
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)  # 添加抖动
        return delay

def with_retry(config: RetryConfig = None, 
               exceptions: tuple = (Exception,)):
    """重试装饰器"""
    config = config or RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        print(f"第{attempt+1}次失败（{type(e).__name__}），"
                              f"{delay:.1f}秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"达到最大重试次数（{config.max_retries}次）")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        print(f"重试 {attempt+1}/{config.max_retries}（{delay:.1f}s）")
                        time.sleep(delay)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用
llm_retry_config = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0
)

@with_retry(llm_retry_config, exceptions=(Exception,))
async def call_llm_with_retry(prompt: str) -> str:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")
    response = await llm.ainvoke(prompt)
    return response.content
```

## 熔断器模式

```python
from enum import Enum
from datetime import datetime, timedelta
import threading

class CircuitState(Enum):
    CLOSED = "closed"       # 正常状态
    OPEN = "open"           # 熔断状态（拒绝请求）
    HALF_OPEN = "half_open" # 试探状态（允许少量请求）

class CircuitBreaker:
    """熔断器：防止级联故障"""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == CircuitState.OPEN:
                    # 检查是否可以转为半开
                    if (datetime.now() - self.last_failure_time).seconds >= self.timeout_seconds:
                        self.state = CircuitState.HALF_OPEN
                        self.success_count = 0
                    else:
                        raise Exception(f"熔断器开路，服务不可用（{self.timeout_seconds}秒后重试）")
            
            try:
                result = func(*args, **kwargs)
                
                with self._lock:
                    if self.state == CircuitState.HALF_OPEN:
                        self.success_count += 1
                        if self.success_count >= self.success_threshold:
                            self.state = CircuitState.CLOSED
                            self.failure_count = 0
                            print("熔断器恢复正常")
                    elif self.state == CircuitState.CLOSED:
                        self.failure_count = 0
                
                return result
            
            except Exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = datetime.now()
                    
                    if (self.state == CircuitState.CLOSED and 
                        self.failure_count >= self.failure_threshold):
                        self.state = CircuitState.OPEN
                        print(f"熔断器触发！连续失败 {self.failure_count} 次")
                    elif self.state == CircuitState.HALF_OPEN:
                        self.state = CircuitState.OPEN
                        print("半开状态测试失败，重新熔断")
                
                raise

        return wrapper

# 使用熔断器
circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=30
)

@circuit_breaker
def call_external_api(query: str) -> str:
    """调用外部 API（带熔断保护）"""
    import requests
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
    response.raise_for_status()
    return response.json()
```

## 降级策略

```python
from typing import List, Any
from langchain_openai import ChatOpenAI

class FallbackChain:
    """降级链：主方案失败时自动切换备选方案"""
    
    def __init__(self, strategies: List[Callable]):
        self.strategies = strategies
    
    def run(self, input_data: Any) -> Any:
        """按顺序尝试每个策略"""
        last_error = None
        
        for i, strategy in enumerate(self.strategies):
            try:
                print(f"尝试方案 {i+1}/{len(self.strategies)}")
                result = strategy(input_data)
                
                if i > 0:
                    print(f"降级到方案 {i+1} 成功")
                
                return result
            
            except Exception as e:
                last_error = e
                print(f"方案 {i+1} 失败：{type(e).__name__}")
        
        raise Exception(f"所有方案均失败，最后一个错误：{last_error}")

# 示例：LLM 模型降级
def strategy_gpt4o(query: str) -> str:
    """主方案：GPT-4o"""
    return ChatOpenAI(model="gpt-4o").invoke(query).content

def strategy_gpt4o_mini(query: str) -> str:
    """备选方案1：GPT-4o-mini（更便宜）"""
    return ChatOpenAI(model="gpt-4o-mini").invoke(query).content

def strategy_simple_response(query: str) -> str:
    """备选方案2：简单模板回复"""
    return f"抱歉，当前服务繁忙，无法处理您的请求：{query}"

fallback = FallbackChain([
    strategy_gpt4o,
    strategy_gpt4o_mini,
    strategy_simple_response
])

result = fallback.run("分析大模型的发展趋势")
```

## 幂等性设计

```python
import hashlib
import json
from typing import Optional

class IdempotentCache:
    """幂等性缓存：相同输入不重复执行"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self.ttl = ttl_seconds
    
    def _cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        content = json.dumps({"func": func_name, "args": args, "kwargs": kwargs}, 
                            sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_or_execute(self, func: Callable, *args, **kwargs):
        key = self._cache_key(func.__name__, args, kwargs)
        
        # 检查缓存
        if key in self._cache:
            cached_at, result = self._cache[key]
            if time.time() - cached_at < self.ttl:
                print(f"缓存命中：{func.__name__}")
                return result
        
        # 执行函数
        result = func(*args, **kwargs)
        self._cache[key] = (time.time(), result)
        return result

idempotent_cache = IdempotentCache(ttl_seconds=3600)

def expensive_agent_call(query: str) -> str:
    """模拟昂贵的 Agent 调用"""
    print("实际执行 Agent...")
    time.sleep(1)  # 模拟耗时
    return f"对 '{query}' 的分析结果"

# 相同输入只执行一次
result1 = idempotent_cache.get_or_execute(expensive_agent_call, "Python 异步编程")
result2 = idempotent_cache.get_or_execute(expensive_agent_call, "Python 异步编程")  # 缓存命中
print(f"结果一致：{result1 == result2}")
```

## 踩坑经验

### 坑1：重试风暴

**问题**：大量并发请求同时失败，重试时间一致，造成同步的"重试风暴"再次打垮服务。  
**解法**：重试时间加随机抖动（jitter），避免同步重试。

### 坑2：熔断器阈值设置不当

**问题**：阈值太低（3次失败就熔断），短暂网络波动就触发熔断；阈值太高，故障放大。  
**解法**：结合错误率（而非绝对数量）判断，至少有一定请求量才触发。

---

*W8D3 · 容错与可靠性 | Agent + Claw 系列*
