---
layout: default
title: "D5 · RAG 评估"
render_with_liquid: false
---

# D5 · RAG 系统评估

> **没有评估，就没有改进。** RAG 系统的评估需要覆盖检索质量和生成质量两个维度。

---

## 一、RAG 评估框架

```
RAG 评估指标：
┌──────────────────────────────────────────┐
│           检索质量                        │
│   ├─ 召回率（Recall@k）                   │
│   ├─ 精确率（Precision@k）                │
│   └─ NDCG（归一化折损累计增益）            │
├──────────────────────────────────────────┤
│           生成质量                        │
│   ├─ 忠实度（Faithfulness）               │
│   ├─ 答案相关性（Answer Relevance）        │
│   ├─ 上下文相关性（Context Relevance）     │
│   └─ 幻觉率（Hallucination Rate）         │
└──────────────────────────────────────────┘
```

---

## 二、检索评估指标

```python
from typing import List, Set

def recall_at_k(
    retrieved: List[int],   # 检索到的文档 ID 列表（按相关性排序）
    relevant: Set[int],     # 真正相关的文档 ID 集合
    k: int,
) -> float:
    """Recall@k = |检索到的相关文档| / |所有相关文档|"""
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / max(len(relevant), 1)

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@k = |检索到的相关文档| / k"""
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / k

def ndcg_at_k(
    retrieved: List[int],
    relevant_scores: dict,  # {doc_id: relevance_score}，0-3 分
    k: int,
) -> float:
    """
    NDCG@k（归一化折损累计增益）
    考虑相关文档的排名位置，排名越靠前贡献越大
    """
    import math
    
    def dcg(ranked_list):
        dcg_val = 0.0
        for i, doc_id in enumerate(ranked_list[:k]):
            rel = relevant_scores.get(doc_id, 0)
            dcg_val += rel / math.log2(i + 2)  # log2(rank+1)
        return dcg_val
    
    actual_dcg = dcg(retrieved)
    # 理想 DCG：相关文档按相关性从高到低排列
    ideal_list = sorted(relevant_scores.keys(), key=lambda d: -relevant_scores[d])
    ideal_dcg = dcg(ideal_list)
    
    return actual_dcg / max(ideal_dcg, 1e-9)

def mrr(retrieved_list: List[List[int]], relevant_list: List[Set[int]]) -> float:
    """
    MRR（Mean Reciprocal Rank）
    第一个相关文档出现的位置倒数的平均值
    """
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# 评估示例
retrieved = [3, 1, 7, 2, 8, 5, 4, 6, 9, 0]
relevant = {1, 3, 5}

print("检索评估指标:")
for k in [1, 3, 5, 10]:
    r = recall_at_k(retrieved, relevant, k)
    p = precision_at_k(retrieved, relevant, k)
    print(f"  @k={k}: Recall={r:.2f}, Precision={p:.2f}")
```

---

## 三、RAGAS：端到端 RAG 评估

```python
"""
RAGAS (Evaluation of Retrieval Augmented Generation) 是目前最主流的 RAG 评估框架，
使用 LLM 自动评估以下指标：

1. Faithfulness（忠实度）：答案是否完全基于检索到的上下文？
   - 无参考答案
   - 评估：LLM 判断答案中的每个声明是否来自上下文
   
2. Answer Relevance（答案相关性）：答案是否切题？
   - 无参考答案
   - 评估：LLM 生成问题，看生成的问题是否与原问题相似
   
3. Context Precision（上下文精确率）：检索到的上下文是否相关？
   - 需要参考答案（ground truth）
   - 评估：检索到的 k 个上下文中，有多少是真正相关的
   
4. Context Recall（上下文召回率）：参考答案中的信息是否在上下文中？
   - 需要参考答案
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def run_ragas_evaluation(rag_outputs: list[dict]) -> dict:
    """
    运行 RAGAS 评估
    
    rag_outputs 格式：
    [
        {
            "question": "...",
            "answer": "...",  # 模型生成的答案
            "contexts": ["上下文1", "上下文2"],
            "ground_truth": "...",  # 参考答案（可选）
        }
    ]
    """
    dataset = Dataset.from_list(rag_outputs)
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=None,  # 使用默认 LLM（OpenAI GPT-4o-mini）
    )
    
    return result


# 幻觉检测（自定义实现）
class HallucinationDetector:
    """基于 NLI 的幻觉检测"""
    
    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small"):
        from transformers import pipeline
        self.nli = pipeline("zero-shot-classification", model=nli_model)
    
    def check_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> dict:
        """
        检测答案是否忠实于上下文
        使用 NLI：context 是否蕴含 answer 中的每个声明
        """
        # 简化：将答案按句子分割
        sentences = [s.strip() for s in answer.split('。') if s.strip()]
        
        results = []
        for sent in sentences:
            if len(sent) < 10:
                continue
            
            # 检查上下文是否蕴含这个句子
            output = self.nli(
                sequences=sent,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="根据以下文档：" + context[:200],
            )
            
            label = output['labels'][0]
            score = output['scores'][0]
            
            results.append({
                'claim': sent,
                'label': label,
                'score': score,
                'is_hallucination': label == 'contradiction' and score > 0.7
            })
        
        hallucination_rate = sum(r['is_hallucination'] for r in results) / max(len(results), 1)
        return {
            'faithfulness_score': 1 - hallucination_rate,
            'claim_analysis': results,
        }
```

---

## 四、系统化评估流程

```python
class RAGEvaluator:
    """RAG 系统完整评估器"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
    
    async def evaluate_on_dataset(
        self,
        test_cases: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        在测试数据集上评估 RAG 系统
        
        test_cases 格式：[{"question": "...", "ground_truth": "..."}]
        """
        results = []
        
        for case in test_cases:
            question = case['question']
            ground_truth = case.get('ground_truth', '')
            
            # 运行 RAG
            output = await self.rag.answer(question)
            
            result = {
                'question': question,
                'answer': output['answer'],
                'contexts': output['sources'],
                'ground_truth': ground_truth,
            }
            results.append(result)
            
            if verbose:
                print(f"Q: {question[:50]}")
                print(f"A: {output['answer'][:80]}...")
                print()
        
        # RAGAS 评估
        metrics = run_ragas_evaluation(results)
        
        return {
            'num_cases': len(test_cases),
            'faithfulness': metrics.get('faithfulness', 0),
            'answer_relevancy': metrics.get('answer_relevancy', 0),
            'context_precision': metrics.get('context_precision', 0),
            'context_recall': metrics.get('context_recall', 0),
            'raw_results': results,
        }
    
    def compare_configs(
        self,
        configs: dict,
        test_cases: list[dict],
    ) -> dict:
        """
        对比不同 RAG 配置的效果
        
        configs: {"config_name": rag_system, ...}
        """
        comparison = {}
        for name, rag in configs.items():
            print(f"\n评估配置: {name}")
            self.rag = rag
            metrics = asyncio.run(self.evaluate_on_dataset(test_cases))
            comparison[name] = metrics
        
        # 打印对比表
        print("\n配置对比:")
        print(f"{'配置':<20} {'忠实度':<12} {'答案相关':<12} {'上下文精':<12} {'上下文召'}")
        print("-" * 70)
        for name, m in comparison.items():
            print(f"{name:<20} {m['faithfulness']:<12.3f} "
                  f"{m['answer_relevancy']:<12.3f} "
                  f"{m['context_precision']:<12.3f} "
                  f"{m['context_recall']:.3f}")
        
        return comparison


# 基准测试题库
rag_benchmark_questions = [
    {
        "question": "LoRA 的低秩分解是什么原理？",
        "ground_truth": "LoRA 将权重更新 ΔW 分解为两个低秩矩阵 B 和 A 的乘积，其中秩 r 远小于原始矩阵维度。"
    },
    {
        "question": "Chinchilla Scaling Law 的核心结论是什么？",
        "ground_truth": "最优训练 token 数约等于 20 倍的模型参数量，即 token 数和参数数同等重要。"
    },
]
```

---

## 五、面试题精讲

**Q: RAG 系统的幻觉来自哪里？如何减少？**

A: 幻觉来源：
1. **检索阶段**：召回了不相关文档（上下文噪声）
2. **生成阶段**：LLM 在上下文不足时"创造"信息
3. **指令遵循**：LLM 没有严格遵守"只使用上下文信息"的指令

减少方法：
- 提升检索质量（Reranker、混合检索）
- 在 Prompt 中明确指令（"仅基于提供的文档回答"）
- 添加答案验证步骤（Faithfulness 检测）
- 当置信度低时，输出"信息不足"而非猜测

**Q: 如何构建 RAG 的测试集？**

A:
1. **人工标注**：专家撰写问题和参考答案（黄金标准但昂贵）
2. **LLM 生成**：从文档中自动生成问题（RAGAS 的 testset_generation）
3. **业务日志**：收集用户真实问题（需要隐私处理）
4. **多样性保证**：覆盖简单事实题、多文档综合题、推理题等不同类型

---

## 小结

```
RAG 评估体系：
  检索：Recall@k, NDCG, MRR
  生成：Faithfulness, Answer Relevancy
  
  工具：RAGAS（自动化评估首选）

优化方向（按优先级）：
  1. 检索召回率不够 → 改进 Chunking/混合检索
  2. 忠实度低 → 改进 Prompt/Reranker
  3. 答案相关性低 → 改进生成 Prompt
  4. 上下文精确率低 → 改进 Reranker 或减少 k
```
