---
layout: post
title: "Python 工程栈实战"
track: "🤖 大模型"
---

# Python 工程栈实战

> 大模型应用开发的基础工程能力，所有岗位通用。重点：asyncio异步、FastAPI服务、Docker部署。

---

## 1. asyncio 异步编程

大模型 API 调用本质是网络 IO，天然适合异步。用 asyncio 可以同时处理数十个请求，吞吐量是同步代码的 10-50 倍。

### 核心概念

```python
import asyncio
import httpx

# async def 定义协程
async def call_llm_api(prompt: str, client: httpx.AsyncClient) -> str:
    """单次调用大模型API"""
    response = await client.post(   # await = 等待但不阻塞
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30.0
    )
    return response.json()["choices"][0]["message"]["content"]

async def batch_process(prompts: list[str]) -> list[str]:
    """并发处理多个请求 — 核心模式"""
    async with httpx.AsyncClient() as client:
        # gather = 同时启动所有任务
        tasks = [call_llm_api(p, client) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# 运行入口
if __name__ == "__main__":
    prompts = ["解释Transformer", "什么是RAG", "解释LoRA"]
    results = asyncio.run(batch_process(prompts))
    for p, r in zip(prompts, results):
        print(f"Q: {p}\nA: {r}\n")
```

### 流式输出（Streaming）

生产环境必须支持流式，避免用户等待：

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_response(prompt: str):
    """流式输出，逐token打印"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # 关键参数
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()  # 换行

asyncio.run(stream_response("详细解释注意力机制"))
```

### 并发限制（防止打爆API限额）

```python
import asyncio

async def process_with_semaphore(prompts: list[str], max_concurrent: int = 5):
    """限制并发数，避免超出API速率限制"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_call(prompt: str) -> str:
        async with semaphore:  # 最多5个并发
            return await call_llm_api(prompt)
    
    tasks = [bounded_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

---

## 2. Pydantic 数据验证

大模型输出不可靠，必须用 Pydantic 强制校验格式：

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10000)
    
class RAGConfig(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    @validator("chunk_overlap")
    def overlap_less_than_chunk(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap必须小于chunk_size")
        return v

class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
# 使用示例
config = RAGConfig(chunk_size=800, chunk_overlap=100, top_k=5)
print(config.model_dump())
# {'chunk_size': 800, 'chunk_overlap': 100, 'top_k': 5, ...}
```

---

## 3. FastAPI 大模型服务

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import uvicorn

app = FastAPI(title="LLM Service", version="1.0.0")

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048

@app.post("/v1/chat")
async def chat(req: ChatRequest) -> LLMResponse:
    """普通对话接口"""
    try:
        result = await llm_client.chat(req.messages, req.model, req.temperature)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话接口 (SSE)"""
    async def event_stream() -> AsyncGenerator[str, None]:
        async for chunk in llm_client.stream_chat(req.messages):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}

# 启动: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 4. Docker 容器化

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 先复制依赖文件（利用Docker层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 生产环境用 gunicorn + uvicorn worker
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml（完整服务栈）

```yaml
version: "3.9"

services:
  llm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/llmdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
      - chroma
    restart: unless-stopped
    
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: llmdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
      
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chromadata:/chroma/chroma

volumes:
  pgdata:
  redisdata:
  chromadata:
```

```bash
# 常用操作
docker-compose up -d              # 后台启动所有服务
docker-compose logs -f llm-api    # 实时查看API日志
docker-compose down               # 停止所有服务
docker-compose exec llm-api bash  # 进入容器调试
docker stats                      # 查看资源使用
```

---

## 5. 环境变量管理（生产规范）

```python
# config.py — 所有配置集中管理
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o"
    
    # 数据库
    database_url: str
    redis_url: str = "redis://localhost:6379"
    
    # RAG
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # 服务
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"  # 本地开发读.env文件
        case_sensitive = False

@lru_cache()  # 单例模式，只初始化一次
def get_settings() -> Settings:
    return Settings()

# 使用
settings = get_settings()
print(settings.openai_api_key)
```

```bash
# .env 文件（不要提交到git！）
OPENAI_API_KEY=sk-xxx
DATABASE_URL=postgresql://user:pass@localhost/llmdb
REDIS_URL=redis://localhost:6379
```

---

## 6. 面试高频问题

**Q: 什么情况用 async，什么情况用多进程？**

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| LLM API调用 | asyncio | IO密集型，协程切换开销小 |
| 向量检索 | asyncio | 数据库IO |
| 模型推理（PyTorch） | 多进程/多线程 | CPU/GPU密集型，绕过GIL |
| 数据预处理 | 多进程（multiprocessing） | CPU密集 |

**Q: FastAPI 怎么做接口限流？**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/chat")
@limiter.limit("10/minute")  # 每分钟最多10次
async def chat(request: Request, req: ChatRequest):
    ...
```

**Q: 如何保证 API 服务的高可用？**
- Docker Compose：`restart: unless-stopped`
- K8s：配置 `replicas: 3` + `HorizontalPodAutoscaler`
- 健康检查：`/health` 接口 + K8s liveness/readiness probe
- 熔断降级：`tenacity` 库做重试，超时返回默认响应

---

[← 岗位能力地图](./skill-map) | [→ RAG全链路工程化](./rag-engineering)
