---
layout: default
title: "D5 · Medusa / Lookahead：并行解码进阶"
render_with_liquid: false
---

# D5 · Medusa / Lookahead：并行解码进阶

## Medusa：多头并行预测

[Medusa](https://arxiv.org/abs/2401.10774)（Cai et al., 2024）不使用独立的 Draft 模型，而是在原始 LLM 的基础上添加多个**预测头（Medusa Heads）**，并行预测未来多个 token。

### 架构

```
原始 LLM:
[Token 1] → [Transformer Layers] → [LM Head] → P(token_t+1)

Medusa:
[Token 1] → [Transformer Layers] → [LM Head]   → P(token_t+1)    # 原始 head
                                  → [Medusa Head 1] → P(token_t+2)  # 新增
                                  → [Medusa Head 2] → P(token_t+3)  # 新增
                                  → [Medusa Head 3] → P(token_t+4)  # 新增
                                  → [Medusa Head 4] → P(token_t+5)  # 新增
```

每个 Medusa Head 是一个简单的 MLP：
```python
import torch
import torch.nn as nn

class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_size, vocab_size, bias=False))
        self.net = nn.Sequential(*layers)
    
    def forward(self, hidden_states):
        return self.net(hidden_states)


class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads: int = 4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        vocab_size  = base_model.config.vocab_size
        
        # 4 个额外预测头，分别预测 t+2, t+3, t+4, t+5
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size)
            for _ in range(num_heads)
        ])
        
        # 只训练 Medusa Heads，冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids):
        # 基础模型的隐藏状态
        outputs = self.base_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # 原始 LM Head（不变）
        lm_logits = outputs.logits
        
        # Medusa Heads
        medusa_logits = [head(hidden_states) for head in self.medusa_heads]
        # medusa_logits[i]: [batch, seq, vocab] - 预测 t+i+2
        
        return lm_logits, medusa_logits
```

### Medusa Tree Attention

Medusa 的关键创新：用**树形结构**来组织候选序列，避免枚举所有 $K^H$ 种组合（K=top-K选择, H=heads数）。

```python
from typing import List, Tuple
import torch

def build_medusa_tree(
    medusa_logits: List[torch.Tensor],
    top_k: int = 10
) -> Tuple[torch.Tensor, List]:
    """
    构建 Medusa 候选 token 树
    
    例如 H=4 heads, top_k=3:
    root(当前 token)
    ├── A1 (top-1 from head 0)
    │   ├── A1A1 (top-1 from head 1)
    │   │   ├── A1A1A1 (top-1 from head 2)
    │   │   └── A1A1A2 (top-2 from head 2)
    │   └── A1A2 (top-2 from head 1)
    ├── A2 (top-2 from head 0)
    │   └── A2A1 (top-1 from head 1)
    └── A3 (top-3 from head 0)
    
    不是所有 3^4=81 种，而是选择高概率路径，约 20-30 个候选
    """
    num_heads = len(medusa_logits)
    
    # 获取每个 head 的 top-k token 和概率
    candidates = []
    probs = []
    for head_logits in medusa_logits:
        head_probs = torch.softmax(head_logits[:, -1, :], dim=-1)  # [batch, vocab]
        top_probs, top_tokens = head_probs.topk(top_k, dim=-1)  # [batch, top_k]
        candidates.append(top_tokens[0])  # 假设 batch=1
        probs.append(top_probs[0])
    
    # 构建候选序列（简化：笛卡尔积的剪枝版本）
    # 实际 Medusa 使用预定义的树结构（论文附录 A）
    tree_candidates = []
    
    # 第一层：head 0 的 top-3
    for i in range(min(3, top_k)):
        path = [candidates[0][i].item()]
        path_prob = probs[0][i].item()
        tree_candidates.append(path)
        
        # 第二层：head 1 的 top-2（条件概率乘积 > 阈值）
        if num_heads > 1:
            for j in range(min(2, top_k)):
                joint_prob = path_prob * probs[1][j].item()
                if joint_prob > 0.01:  # 概率阈值
                    tree_candidates.append(path + [candidates[1][j].item()])
                    
                    # 第三层...
                    if num_heads > 2:
                        for k_idx in range(min(2, top_k)):
                            joint_prob2 = joint_prob * probs[2][k_idx].item()
                            if joint_prob2 > 0.001:
                                tree_candidates.append(
                                    path + [candidates[1][j].item(), candidates[2][k_idx].item()]
                                )
    
    return torch.tensor(tree_candidates), candidates
```

### Tree Attention Mask

树形候选序列需要特殊的 Attention Mask：

```
候选序列树：
  0: [A]
  1: [A, B]
  2: [A, B, C]
  3: [A, B, D]
  4: [A, E]
  5: [A, E, F]

Attention Mask（1=可看到，0=不可看到）：
      A  B  C  D  E  F
  A: [1, 0, 0, 0, 0, 0]
  B: [1, 1, 0, 0, 0, 0]  # B 可以看到 A
  C: [1, 1, 1, 0, 0, 0]  # C 可以看到 A, B
  D: [1, 1, 0, 1, 0, 0]  # D 可以看到 A, B（但看不到 C！）
  E: [1, 0, 0, 0, 1, 0]  # E 只能看到 A
  F: [1, 0, 0, 0, 1, 1]  # F 可以看到 A, E
```

```python
def build_tree_attention_mask(tree_paths: List[List[int]], seq_len: int) -> torch.Tensor:
    """构建树形 Attention Mask"""
    n = sum(len(path) for path in tree_paths)
    mask = torch.zeros(n, n, dtype=torch.bool)
    
    idx = 0
    for path in tree_paths:
        for j, token in enumerate(path):
            # token 可以看到路径上的所有祖先（包括自己）
            for ancestor_idx in range(j + 1):
                # 找到祖先在 flat 数组中的位置
                # ... 实际实现较复杂，需要维护位置映射
                pass
    
    return mask
```

### Medusa 性能数据

A100 80GB，Vicuna-7B + 4 Medusa Heads：

| 指标 | 标准解码 | Medusa |
|------|---------|--------|
| 接受率（每个 head）| N/A | ~75% |
| 平均每步生成 tokens | 1.0 | 2.8 |
| 端到端加速 | 1× | 2.3× |
| 额外参数 | 0 | ~100M (< 1.5%) |

## Lookahead 解码

[Lookahead Decoding](https://arxiv.org/abs/2402.02057) 不需要 Draft 模型，也不需要训练，用 n-gram 模式来预测未来 token。

### 核心思想

```
维护一个 "Lookahead Window"（前瞻窗口）：
- W 个并行序列（每个序列提前预测 N 步）
- 从这些预测中提取 N-gram 作为候选

例如 W=3, N=5:
并行运行:
  Path 0: [x_t] → [a, b, c, d, e]  # 预测 5 步
  Path 1: [x_t] → [p, q, r, s, t]  # 预测 5 步
  Path 2: [x_t] → [m, n, o, u, v]  # 预测 5 步

提取 N-gram:
  从 Path 0: (a,b,c), (b,c,d), (c,d,e), ...
  从 Path 1: (p,q,r), (q,r,s), ...
  ...

用这些 N-gram 作为候选，并行验证（Verify 步骤）
```

```python
import collections
from typing import Dict, List

class LookaheadDecoder:
    def __init__(self, model, tokenizer, window_size=3, lookahead_steps=5, 
                 ngram_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.W = window_size
        self.N = lookahead_steps
        self.n = ngram_size
        
        # N-gram 缓存（相当于从之前的解码中积累的"先验知识"）
        self.ngram_cache: Dict[tuple, List[int]] = collections.defaultdict(list)
    
    def lookup_ngram(self, context: List[int]) -> List[int]:
        """查找 N-gram 缓存"""
        key = tuple(context[-self.n+1:])
        return self.ngram_cache.get(key, [])
    
    def decode(self, input_ids: torch.Tensor, max_new_tokens: int) -> List[int]:
        """Lookahead 解码"""
        generated = []
        current_ids = input_ids.tolist()[0]
        
        while len(generated) < max_new_tokens:
            # Step 1: Lookahead 步骤（探索性）
            # 从当前位置并行生成 W 个"探索序列"
            lookahead_candidates = []
            for w in range(self.W):
                # 每个探索序列独立生成 N 步
                candidate = self._lookahead_sample(current_ids, steps=self.N)
                lookahead_candidates.append(candidate)
                
                # 更新 N-gram 缓存
                for i in range(len(candidate) - self.n + 1):
                    key = tuple(candidate[i:i+self.n-1])
                    val = candidate[i+self.n-1]
                    self.ngram_cache[key].append(val)
            
            # Step 2: Verify 步骤（从缓存中找候选）
            # 查找最长匹配的 N-gram
            context = current_ids[-self.n+1:]
            cache_candidate = self.lookup_ngram(context)
            
            if cache_candidate:
                # 找到 N-gram 候选，并行验证
                verify_sequence = context + cache_candidate[:self.N]
                # ... 并行验证并接受/拒绝
                accepted = self._verify_and_accept(current_ids, verify_sequence)
                generated.extend(accepted)
                current_ids.extend(accepted)
            else:
                # 没有缓存命中，回退到标准解码
                next_token = self._greedy_sample(current_ids)
                generated.append(next_token)
                current_ids.append(next_token)
        
        return generated
    
    def _lookahead_sample(self, context: List[int], steps: int) -> List[int]:
        """贪心生成 steps 步"""
        ids = torch.tensor([context], device='cuda')
        result = []
        for _ in range(steps):
            with torch.no_grad():
                logits = self.model(ids).logits[0, -1, :]
            next_tok = logits.argmax().item()
            result.append(next_tok)
            ids = torch.cat([ids, torch.tensor([[next_tok]], device='cuda')], dim=1)
        return result
    
    def _greedy_sample(self, context: List[int]) -> int:
        ids = torch.tensor([context], device='cuda')
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1, :]
        return logits.argmax().item()
    
    def _verify_and_accept(self, prefix: List[int], candidates: List[int]) -> List[int]:
        """并行验证候选序列，返回接受的 tokens"""
        full_seq = torch.tensor([prefix + candidates], device='cuda')
        with torch.no_grad():
            logits = self.model(full_seq).logits[0, len(prefix)-1:, :]
        
        accepted = []
        for i, candidate_tok in enumerate(candidates):
            predicted_tok = logits[i].argmax().item()
            if predicted_tok == candidate_tok:
                accepted.append(candidate_tok)
            else:
                accepted.append(predicted_tok)  # 用 Target 的预测替换
                break  # 遇到不匹配则停止
        
        return accepted
```

## 各方法对比

| 方法 | 需要训练 | 需要 Draft 模型 | 加速比 | 精度损失 |
|------|---------|----------------|-------|---------|
| 标准解码 | - | - | 1× | 0% |
| 投机解码 | 仅 Draft | 需要 | 2-3× | 0% |
| Medusa | 需要 (Heads) | 不需要 | 2-3× | 0% |
| Lookahead | 不需要 | 不需要 | 1.5-2.5× | 0% |
| Eagle | 需要 (Draft) | 需要 | 3-4× | 0% |

## 面试题

**Q: Medusa 和标准投机解码的核心区别是什么？各有什么优劣？**

A: Medusa 在原始 LLM 上添加额外的预测头（MedusaHead），无需独立的 Draft 模型；而投机解码使用一个独立的小模型作为 Drafter。Medusa 的优势：无需维护两个模型，部署简单，KV Cache 共享（Draft 和 Verify 共用基础模型的 KV Cache）；缺点：需要微调添加 Medusa Head，且预测准确率受限（毕竟只是浅层 MLP）。投机解码的优势：更高的接受率（专门训练的 Draft 模型分布更接近 Target）；缺点：需要额外显存和双模型管理复杂性。一般来说：有配套小模型用投机解码，否则用 Medusa。
