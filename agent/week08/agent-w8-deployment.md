---
layout: default
title: "W8D5 · Agent 部署与扩展"
---

# Agent 部署：从本地跑通到生产可用

> **Week 8 · Day 5** | 难度：⭐⭐⭐⭐

---

## 部署架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 生产部署架构                         │
│                                                             │
│  用户请求 → API Gateway → 负载均衡                           │
│                              │                              │
│              ┌───────────────┼───────────────┐             │
│              ▼               ▼               ▼             │
│         Agent Pod 1    Agent Pod 2    Agent Pod N           │
│              │                                              │
│         ┌────┴─────────────────────────────────┐           │
│         │            共享服务层                  │           │
│         │  Redis（缓存+会话）  向量DB  工具服务   │           │
│         └────────────────────────────────────┘            │
│                          │                                  │
│                    监控/日志/追踪                             │
└─────────────────────────────────────────────────────────────┘
```

## Docker 容器化

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 安全：非 root 用户
RUN useradd --create-home appuser
WORKDIR /app

# 安装依赖（分层缓存优化）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY --chown=appuser:appuser . .
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

```python
# main.py - FastAPI Agent 服务
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid
import time
from datetime import datetime

app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务存储（生产环境用 Redis）
tasks = {}

class AgentRequest(BaseModel):
    task: str
    session_id: Optional[str] = None
    max_steps: int = 20
    timeout_seconds: int = 120

class AgentResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    steps: int = 0
    duration_ms: float = 0

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent_sync(request: AgentRequest):
    """同步执行 Agent 任务（适合短任务）"""
    task_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # 带超时的 Agent 执行
        result = await asyncio.wait_for(
            execute_agent_task(request.task, request.max_steps),
            timeout=request.timeout_seconds
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            task_id=task_id,
            status="completed",
            result=result,
            duration_ms=duration_ms
        )
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"任务超时（>{request.timeout_seconds}秒）")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/submit")
async def submit_agent_task(request: AgentRequest, background_tasks: BackgroundTasks):
    """异步提交 Agent 任务（适合长任务）"""
    task_id = str(uuid.uuid4())[:8]
    
    tasks[task_id] = {
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # 在后台执行
    background_tasks.add_task(run_background_task, task_id, request)
    
    return {"task_id": task_id, "status": "submitted"}

@app.get("/agent/status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return tasks[task_id]

async def run_background_task(task_id: str, request: AgentRequest):
    """后台任务执行"""
    try:
        result = await execute_agent_task(request.task, request.max_steps)
        tasks[task_id].update({"status": "completed", "result": result})
    except Exception as e:
        tasks[task_id].update({"status": "failed", "error": str(e)})

async def execute_agent_task(task: str, max_steps: int) -> str:
    """实际的 Agent 执行逻辑"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = await llm.ainvoke(task)
    return response.content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent
        image: your-registry/agent-service:latest
        ports:
        - containerPort: 8080
        
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "2"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
# 水平自动扩展
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 配置管理

```python
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class AgentSettings(BaseSettings):
    """Agent 服务配置（从环境变量读取）"""
    
    # LLM 配置
    openai_api_key: str
    default_model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    
    # 缓存配置
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 3600
    
    # 限流配置
    rate_limit_per_minute: int = 60
    max_concurrent_tasks: int = 100
    
    # 安全配置
    api_key_header: str = "X-API-Key"
    allowed_origins: list = ["*"]
    
    # 监控配置
    log_level: str = "INFO"
    enable_tracing: bool = True
    langsmith_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> AgentSettings:
    return AgentSettings()

settings = get_settings()
```

## 踩坑经验

### 坑1：无状态部署 + 有状态 Session

**问题**：Agent 支持多轮对话，但水平扩展后不同请求可能落到不同 Pod，会话状态丢失。  
**解法**：会话状态统一存 Redis，Pod 本身完全无状态。

```python
import redis
import json
from typing import List

redis_client = redis.from_url("redis://redis:6379")

def save_session(session_id: str, messages: List[dict], ttl: int = 3600):
    redis_client.setex(
        f"session:{session_id}",
        ttl,
        json.dumps(messages, ensure_ascii=False)
    )

def load_session(session_id: str) -> List[dict]:
    data = redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else []
```

### 坑2：K8s OOMKilled

**问题**：Agent 处理大型文档时内存突增，容器被 OOM Kill。  
**解法**：
1. 设置合理的 memory limit（如请求的 4-8 倍）
2. 大文档分块处理，避免全文加载
3. 监控内存使用，提前预警

---

*W8D5 · Agent 部署与扩展 | Agent + Claw 系列*
