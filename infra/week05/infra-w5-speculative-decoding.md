---
layout: default
title: "D4 · 投机解码：用小模型加速大模型"
render_with_liquid: false
---

# D4 · 投机解码：用小模型加速大模型

## 核心思想

LLM 的 Decode 阶段是严重 Memory-Bound 的——每步只生成 1 个 token，但需要读取全部模型权重。

**投机解码（Speculative Decoding）**的洞察：
- 小模型生成速度快（权重少 → IO 少）
- 大模型验证速度快（并行验证多个 token）
- 大部分时候小模型猜对了 → 等价于大模型"加速"了

**算法概述**：
1. **Draft（起草）**：用小 Draft 模型快速生成 K 个 candidate tokens
2. **Verify（验证）**：用大 Target 模型**并行**验证这 K 个 token（1次前向传播）
3. **Accept/Reject**：接受符合 Target 分布的 token，从第一个拒绝处重采样

## 正确性保证

关键：投机解码必须保证输出分布**完全等价**于 Target 模型！

设：
- $p(x)$：Target 模型的分布
- $q(x)$：Draft 模型的分布

**Accept-Reject 采样**：

对于 Draft 模型提出的 token $x$：
$$\text{Accept 概率} = \min\left(1, \frac{p(x)}{q(x)}\right)$$

若拒绝，从校正分布中重采样：
$$p'(x) = \text{normalize}\left(\max(0, p(x) - q(x))\right)$$

**定理**（Chen et al., 2022）：上述过程产生的输出分布精确等于 $p(x)$。

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple

def speculative_decode_step(
    target_logits: torch.Tensor,    # [K+1, vocab_size] - Target 模型在 K+1 个位置的 logits
    draft_probs: torch.Tensor,      # [K, vocab_size]   - Draft 模型的概率分布
    draft_tokens: torch.Tensor,     # [K]               - Draft 模型生成的 token
    temperature: float = 1.0,
) -> Tuple[List[int], int]:
    """
    投机解码的接受/拒绝步骤
    返回：(接受的 token 列表, 接受的数量)
    """
    K = draft_tokens.shape[0]
    target_probs = F.softmax(target_logits / temperature, dim=-1)  # [K+1, vocab_size]
    
    accepted_tokens = []
    
    for i in range(K):
        x = draft_tokens[i].item()
        
        # 计算接受概率
        p_x = target_probs[i, x].item()
        q_x = draft_probs[i, x].item()
        
        accept_prob = min(1.0, p_x / (q_x + 1e-10))
        
        # 随机接受或拒绝
        if torch.rand(1).item() < accept_prob:
            accepted_tokens.append(x)
        else:
            # 拒绝：从校正分布中采样
            correction = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            correction = correction / correction.sum()
            
            # 如果校正分布全为 0，使用 Target 分布
            if correction.sum() > 0:
                new_token = torch.multinomial(correction, num_samples=1).item()
            else:
                new_token = torch.multinomial(target_probs[i], num_samples=1).item()
            
            accepted_tokens.append(new_token)
            return accepted_tokens, len(accepted_tokens) - 1  # 最后一个是修正 token
    
    # 全部接受：从 Target 的最后一个位置采样一个新 token
    bonus_token = torch.multinomial(target_probs[K], num_samples=1).item()
    accepted_tokens.append(bonus_token)
    
    return accepted_tokens, K  # 接受了所有 K 个 Draft token + 1 个额外 token


class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, K: int = 4):
        """
        target_model: 大模型（慢但准确）
        draft_model:  小模型（快但不够准确）
        K: 每步 Draft 的 token 数
        """
        self.target = target_model
        self.draft  = draft_model
        self.K = K
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        投机解码完整流程
        """
        generated = []
        current_ids = input_ids
        
        total_draft_tokens = 0
        total_accepted_tokens = 0
        
        while len(generated) < max_new_tokens:
            # Step 1: Draft 模型自回归生成 K 个 token
            draft_tokens = []
            draft_probs_list = []
            draft_ids = current_ids
            
            for k in range(self.K):
                with torch.no_grad():
                    draft_out = self.draft(draft_ids)
                    draft_logits = draft_out.logits[:, -1, :]
                    draft_prob   = F.softmax(draft_logits, dim=-1)
                    
                    # 贪心采样（或温度采样）
                    next_token = torch.argmax(draft_logits, dim=-1)
                    
                    draft_tokens.append(next_token.item())
                    draft_probs_list.append(draft_prob[0])
                    
                    draft_ids = torch.cat([draft_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            draft_tokens_tensor = torch.tensor(draft_tokens)
            draft_probs_tensor  = torch.stack(draft_probs_list)  # [K, vocab_size]
            
            # Step 2: Target 模型一次性验证所有 K 个 Draft token
            # 关键：Target 只做 1 次前向传播！
            verify_ids = torch.cat([
                current_ids, 
                draft_tokens_tensor.unsqueeze(0).unsqueeze(0).squeeze(-1)
            ], dim=1)
            
            with torch.no_grad():
                target_out = self.target(verify_ids)
                target_logits = target_out.logits[0, -(self.K+1):, :]  # [K+1, vocab]
            
            # Step 3: Accept-Reject
            accepted, num_accepted = speculative_decode_step(
                target_logits, draft_probs_tensor, draft_tokens_tensor
            )
            
            total_draft_tokens    += self.K
            total_accepted_tokens += num_accepted
            
            # 更新上下文
            for tok in accepted:
                generated.append(tok)
                current_ids = torch.cat([
                    current_ids, 
                    torch.tensor([[tok]])
                ], dim=1)
                
                if tok == 2:  # EOS token
                    break
            
            if len(generated) >= max_new_tokens:
                break
        
        # 统计接受率
        acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        print(f"Acceptance rate: {acceptance_rate:.2%}")
        print(f"Effective speedup: ~{1 + acceptance_rate * (self.K - 1):.2f}×")
        
        return torch.tensor(generated)
```

## 加速比推导

设：
- $K$：每步 Draft 的 token 数
- $\alpha$：Draft 模型的接受率（每个 token 被接受的概率）
- $t_d$：Draft 模型每步的延迟
- $t_v$：Target 模型做一次 K+1 token 验证的延迟（≈ Target 模型单步延迟）

**期望每次"大模型调用"接受的 token 数**：

$$\mathbb{E}[\text{accepted}] = \sum_{k=0}^{K-1} \alpha^k \cdot (k+1) \cdot (1-\alpha) + K\alpha^K \cdot (K+1)$$

对于高接受率（$\alpha \approx 1$），近似为 $K+1$。

**实际加速比**（忽略 Draft 开销，$K=4, \alpha=0.8$）：

$$\text{Speedup} \approx \frac{1-\alpha^{K+1}}{(1-\alpha)(t_v + K \cdot t_d)}$$

实测数据（LLaMA-2 70B + LLaMA-2 7B Draft, K=5）：

| 场景 | 接受率 | 加速比 |
|------|-------|-------|
| 代码生成 | 85% | 2.5× |
| 创意写作 | 65% | 1.8× |
| 数学推理 | 70% | 1.9× |
| 通用对话 | 75% | 2.1× |

## 实战：使用 HuggingFace 投机解码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

target_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
draft_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",  # 53× 小的 Draft 模型
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 标准解码
start = time.time()
with torch.no_grad():
    standard_output = target_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )
standard_time = time.time() - start

# 投机解码
start = time.time()
with torch.no_grad():
    spec_output = target_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        # HuggingFace 原生投机解码支持
        assistant_model=draft_model,
        num_assistant_tokens=5,  # K=5
    )
spec_time = time.time() - start

print(f"Standard: {standard_time:.2f}s")
print(f"Speculative: {spec_time:.2f}s")
print(f"Speedup: {standard_time/spec_time:.2f}×")
```

## Self-Speculative Decoding

不需要单独的 Draft 模型！用目标模型本身的早退（early exit）来起草：

```python
"""
Self-Speculative Decoding 思路：
- 在 LLM 的第 L 层（总共 N 层）处早退，得到近似 logits
- 用这些近似 logits 起草 K 个 token
- 完整 N 层模型并行验证

优点：不需要额外的 Draft 模型
缺点：实现复杂，加速比略低于独立 Draft 模型
"""

class EarlyExitLLM(torch.nn.Module):
    def __init__(self, model, early_exit_layer: int):
        super().__init__()
        self.model = model
        self.early_exit_layer = early_exit_layer
        # 在 early_exit_layer 处添加一个 LM head
        self.draft_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size)
    
    def forward_draft(self, input_ids):
        """只运行到 early_exit_layer"""
        hidden_states = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            if i == self.early_exit_layer:
                break
            hidden_states = layer(hidden_states)[0]
        return self.draft_head(hidden_states)
    
    def forward_full(self, input_ids):
        """完整前向传播"""
        return self.model(input_ids).logits
```

## 面试题

**Q: 投机解码如何保证输出分布不变？**

A: 通过 Acceptance-Rejection Sampling。对于 Draft 模型提出的每个 token x，以 min(1, p(x)/q(x)) 的概率接受（p 是 Target 分布，q 是 Draft 分布）。如果拒绝，从校正分布 normalize(max(0, p-q)) 中重新采样。可以数学证明，这个过程产生的最终分布精确等于 Target 模型的分布 p(x)，因此投机解码不损失任何精度，只是提升速度。当 Draft 和 Target 分布接近时（高接受率），可以获得接近 K+1× 的加速。
