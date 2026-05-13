---
layout: post
title: "LLMOps 全链路"
track: "🤖 大模型"
---

# LLMOps 全链路

> 从原型到生产的分水岭技能。能做Demo的人很多，能做生产级可运维系统的人稀缺。

---

## LLMOps 全链路图

```
数据工程
  ↓
实验管理（W&B/MLflow）
  ↓
模型训练/微调
  ↓
模型评测
  ↓
容器化打包（Docker）
  ↓
服务化部署（K8s + vLLM）
  ↓
监控告警（Prometheus + Grafana）
  ↓
持续迭代（数据飞轮）
```

---

## 1. 实验管理（W&B）

```python
import wandb
from transformers import TrainingArguments

# 初始化实验
wandb.init(
    project="llm-finetuning",
    name="qwen-lora-v1",
    config={
        "model": "Qwen2.5-7B-Instruct",
        "lora_r": 16,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 3,
        "dataset": "customer-service-v2"
    }
)

# 在训练中记录指标
def on_log(args, state, control, logs=None, **kwargs):
    if logs:
        wandb.log({
            "train/loss": logs.get("loss"),
            "train/learning_rate": logs.get("learning_rate"),
            "train/epoch": logs.get("epoch"),
            "eval/loss": logs.get("eval_loss"),
        })

# TrainingArguments集成W&B
training_args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",      # 自动记录到W&B
    run_name="qwen-lora-v1",
    # ...
)

# 记录模型（模型版本管理）
artifact = wandb.Artifact(
    name="qwen-lora-v1",
    type="model",
    description="QLoRA微调的客服模型，r=16"
)
artifact.add_dir("./qwen-lora-adapter")
wandb.log_artifact(artifact)

wandb.finish()
```

---

## 2. 模型评测流水线

```python
import json
from openai import OpenAI
from datasets import load_dataset

class LLMEvaluator:
    def __init__(self, model_endpoint: str):
        self.client = OpenAI(base_url=model_endpoint)
    
    def evaluate_task(self, dataset_path: str) -> dict:
        """在测试集上评测模型"""
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        
        results = []
        for sample in dataset:
            response = self.client.chat.completions.create(
                model="finetuned-model",
                messages=[{"role": "user", "content": sample["input"]}],
                max_tokens=512,
                temperature=0
            )
            predicted = response.choices[0].message.content
            results.append({
                "input": sample["input"],
                "expected": sample["expected_output"],
                "predicted": predicted
            })
        
        # 计算指标
        metrics = {
            "exact_match": self._exact_match(results),
            "bleu": self._bleu_score(results),
            "llm_judge_score": self._llm_judge(results),  # 用GPT-4评分
        }
        return metrics
    
    def _llm_judge(self, results: list[dict]) -> float:
        """用GPT-4作为评判者（LLM-as-Judge）"""
        scores = []
        for r in results[:20]:  # 抽样评估
            judge_prompt = f"""评估以下回答的质量（1-5分）：
问题：{r['input']}
参考答案：{r['expected']}
模型回答：{r['predicted']}

评分标准：5=完全正确且全面，3=基本正确但不完整，1=错误或无关
只回复数字分数："""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10
            )
            try:
                score = float(response.choices[0].message.content.strip())
                scores.append(min(max(score, 1), 5))
            except:
                pass
        
        return sum(scores) / len(scores) if scores else 0
```

---

## 3. K8s 生产部署

### 部署配置（deployment.yaml）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
  namespace: production
spec:
  replicas: 3                          # 3个副本，高可用
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api
        image: registry.company.com/llm-api:v1.2.0
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "8Gi"
            nvidia.com/gpu: "1"        # 需要1块GPU
          limits:
            memory: "16Gi"
            nvidia.com/gpu: "1"
        livenessProbe:                 # 存活检查
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:                # 就绪检查（流量切入前）
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70        # CPU超70%时扩容
```

---

## 4. 监控告警（Prometheus + Grafana）

```python
# metrics.py — 在FastAPI中暴露Prometheus指标
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response
import time

app = FastAPI()

# 定义指标
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "status"]              # 标签：按模型和状态分类
)

REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TOKEN_COUNT = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["model", "type"]               # type: prompt/completion
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently active requests"
)

HALLUCINATION_SCORE = Histogram(
    "llm_hallucination_score",
    "Hallucination detection scores",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """自动记录所有请求的指标"""
    if request.url.path == "/metrics":
        return await call_next(request)
    
    ACTIVE_REQUESTS.inc()
    start = time.time()
    
    try:
        response = await call_next(request)
        status = "success" if response.status_code < 400 else "error"
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start
        model = request.headers.get("X-Model", "unknown")
        
        REQUEST_COUNT.labels(model=model, status=status).inc()
        REQUEST_LATENCY.labels(model=model).observe(duration)
        ACTIVE_REQUESTS.dec()
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus拉取指标的端点"""
    return Response(generate_latest(), media_type="text/plain")
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['llm-api-service:80']
    scrape_interval: 15s

# 告警规则
groups:
  - name: llm_alerts
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.99, llm_request_duration_seconds_bucket) > 10
      for: 5m
      annotations:
        summary: "P99延迟超过10秒"
    
    - alert: HighErrorRate
      expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
      for: 2m
      annotations:
        summary: "错误率超过5%"
    
    - alert: HighHallucinationScore
      expr: histogram_quantile(0.5, llm_hallucination_score_bucket) < 0.6
      for: 10m
      annotations:
        summary: "幻觉分数中位数低于0.6，模型质量下降"
```

---

## 5. CI/CD 流水线（GitHub Actions）

```yaml
# .github/workflows/deploy.yml
name: LLM Service CI/CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: 运行测试
      run: |
        pip install -r requirements.txt
        pytest tests/ -v --cov=app

  evaluate:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: 模型评测
      run: |
        python scripts/evaluate.py \
          --model-endpoint ${{ secrets.STAGING_ENDPOINT }} \
          --dataset tests/eval_dataset.json \
          --min-score 3.5         # 评分低于3.5拒绝部署

  deploy:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
    - name: 构建镜像
      run: |
        docker build -t registry.company.com/llm-api:${{ github.sha }} .
        docker push registry.company.com/llm-api:${{ github.sha }}
    
    - name: 滚动更新（零停机）
      run: |
        kubectl set image deployment/llm-api \
          llm-api=registry.company.com/llm-api:${{ github.sha }} \
          --record
        kubectl rollout status deployment/llm-api --timeout=5m
    
    - name: 烟雾测试
      run: |
        curl -f http://llm-api-service/health
        python scripts/smoke_test.py
    
    - name: 回滚（失败时）
      if: failure()
      run: kubectl rollout undo deployment/llm-api
```

---

## 6. 数据飞轮（持续迭代）

```
用户使用 → 收集反馈（点赞/踩/修正）
    ↓
数据清洗与质量筛选
    ↓
增量微调（新数据+原数据）
    ↓
评测（A/B测试）
    ↓
上线（蓝绿部署/金丝雀发布）
    ↓
监控（效果是否提升）
    ↓ 循环
```

```python
# 收集用户反馈，构建训练数据
class FeedbackCollector:
    async def record_feedback(
        self,
        session_id: str,
        message_id: str,
        rating: int,        # 1-5分
        correction: str = None  # 用户提供的正确答案
    ):
        feedback = {
            "session_id": session_id,
            "message_id": message_id,
            "rating": rating,
            "correction": correction,
            "timestamp": datetime.now().isoformat()
        }
        await db.insert("user_feedback", feedback)
        
        # 低分 + 有修正 = 高质量训练数据
        if rating <= 2 and correction:
            original = await db.get_message(message_id)
            await db.insert("training_candidates", {
                "input": original["user_input"],
                "bad_output": original["response"],
                "good_output": correction,
                "source": "user_correction"
            })
```

---

## 面试高频问题

**Q: 蓝绿部署和金丝雀发布有什么区别？**
> 蓝绿部署：维护两套完整环境，切流量时100%切换，回滚快但成本高；金丝雀发布：逐步将少量流量（5%→20%→100%）切到新版本，风险更低，适合大规模服务。

**Q: 如何做大模型的A/B测试？**
> 将流量按用户ID哈希分流，A组用旧模型，B组用新模型，统计TTFT、用户满意度、任务完成率等指标，统计显著后全量切换。

**Q: LLMOps和传统MLOps的核心区别？**
> LLMOps新增：①幻觉检测和监控②Prompt版本管理③长上下文的KV Cache管理④LLM-as-Judge评估方式⑤更复杂的数据飞轮（人类反馈+AI反馈）。

---

[← 推理加速与量化](./inference-optimization) | [→ 安全合规与幻觉治理](./safety-compliance)
