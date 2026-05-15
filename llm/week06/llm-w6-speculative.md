---
layout: default
title: "D4 · 投机采样"
render_with_liquid: false
---

# D4 · 投机采样（Speculative Sampling）

> **核心思想**：用小模型（Draft Model）快速生成多个候选 token，再用大模型一次性验证，接受正确的，拒绝错误的。等效加速自回归生成。

---

## 一、为什么能加速？

### 1.1 自回归推理的瓶颈

大模型生成每个 token 都是独立的一次前向传播，且受**内存带宽**限制（Decode 阶段，计算量很小但要不断加载权重）。

如果能让大模型一次处理**多个 token**（验证），就能提高计算利用率。

### 1.2 投机采样流程

```
Step 1: 小模型（Draft）串行生成 k=5 个 token
  Input: [x1, x2, x3, x4, x5, x6, x7]
  Draft: [d1, d2, d3, d4, d5]  ← 小模型猜的

Step 2: 大模型（Target）并行验证 k+1 个位置
  Input: [x1...x7, d1, d2, d3, d4, d5]  ← 一次前向，处理 k+1 个新 token
  Output logits at each position: [p1, p2, p3, p4, p5, p6]

Step 3: 对每个 draft token 进行接受/拒绝
  - 如果大模型也选 d_i：接受
  - 如果大模型不选 d_i：以概率拒绝，从大模型分布重采样

Step 4: 假设前 3 个被接受，第 4 个被拒绝：
  接受 d1, d2, d3，从 p4 重采样一个新 token，丢弃 d4, d5
  本次循环：生成了 4 个 token，但只用了 1 次大模型前向
```

---

## 二、接受/拒绝机制

```python
import torch
import torch.nn.functional as F

def speculative_sampling_step(
    target_probs: torch.Tensor,    # 大模型在 draft position 的概率 [k, vocab]
    draft_probs: torch.Tensor,     # 小模型的概率 [k, vocab]
    draft_tokens: torch.Tensor,    # 小模型选的 token [k]
) -> tuple[list[int], int]:
    """
    投机采样的接受/拒绝算法
    
    Returns:
        accepted_tokens: 接受的 token 列表
        first_rejected: 第一个被拒绝的位置（-1 表示全接受）
    """
    accepted = []
    
    for i in range(len(draft_tokens)):
        d_i = draft_tokens[i].item()
        
        # 目标概率和草稿概率
        p_target = target_probs[i, d_i].item()
        p_draft = draft_probs[i, d_i].item()
        
        # 接受概率 = min(1, p_target / p_draft)
        # 直觉：如果大模型比小模型"更喜欢"这个 token，就接受
        accept_prob = min(1.0, p_target / (p_draft + 1e-9))
        
        if torch.rand(1).item() < accept_prob:
            accepted.append(d_i)
        else:
            # 拒绝：从调整后的分布重采样
            # p_adjusted ∝ max(0, p_target - p_draft)
            adjusted_probs = F.relu(target_probs[i] - draft_probs[i])
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            new_token = torch.multinomial(adjusted_probs, 1).item()
            accepted.append(new_token)
            return accepted, i  # 返回第一个被拒绝的位置
    
    return accepted, -1  # -1 表示全部接受


def speculative_generate(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    num_speculative: int = 5,    # 每次投机 k 个 token
    temperature: float = 1.0,
) -> torch.Tensor:
    """完整的投机采样生成"""
    
    generated = input_ids.clone()
    total_accepted = 0
    total_drafted = 0
    
    while len(generated[0]) < len(input_ids[0]) + max_new_tokens:
        # Step 1: 小模型投机生成 k 个 token
        draft_tokens = []
        draft_probs_list = []
        
        context = generated
        for _ in range(num_speculative):
            with torch.no_grad():
                draft_logits = draft_model(context).logits[:, -1, :] / temperature
                draft_prob = F.softmax(draft_logits, dim=-1)
                next_token = torch.multinomial(draft_prob[0], 1)
                
                draft_tokens.append(next_token.item())
                draft_probs_list.append(draft_prob[0])
                context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
        
        draft_tokens_tensor = torch.tensor(draft_tokens)
        draft_probs = torch.stack(draft_probs_list)
        
        # Step 2: 大模型一次性验证
        # 将 draft tokens 拼接到 generated 后，一次前向
        verification_input = torch.cat([
            generated,
            draft_tokens_tensor.unsqueeze(0).to(generated.device)
        ], dim=1)
        
        with torch.no_grad():
            target_logits = target_model(verification_input).logits / temperature
            # 取 draft token 位置的概率
            target_probs = F.softmax(
                target_logits[0, -num_speculative-1:-1, :], dim=-1
            )  # [k, vocab]
        
        # Step 3: 接受/拒绝
        accepted, first_rejected = speculative_sampling_step(
            target_probs, draft_probs, draft_tokens_tensor
        )
        
        # 拼接接受的 token
        accepted_tensor = torch.tensor(accepted, device=generated.device).unsqueeze(0)
        generated = torch.cat([generated, accepted_tensor], dim=1)
        
        total_accepted += len(accepted)
        total_drafted += num_speculative
        
        # 全接受时，还需要从大模型额外采样一个 token（bonus token）
        if first_rejected == -1:
            with torch.no_grad():
                bonus_logits = target_model(generated).logits[:, -1, :] / temperature
                bonus_prob = F.softmax(bonus_logits, dim=-1)
                bonus_token = torch.multinomial(bonus_prob[0], 1).unsqueeze(0)
            generated = torch.cat([generated, bonus_token], dim=1)
            total_accepted += 1
    
    acceptance_rate = total_accepted / max(total_drafted, 1)
    print(f"接受率: {acceptance_rate:.2%}, 平均每次 decode 接受 {acceptance_rate * num_speculative:.1f} 个 token")
    
    return generated
```

---

## 三、加速比分析

```python
def analyze_speedup(
    num_speculative: int = 5,
    acceptance_rate: float = 0.8,
    target_model_time: float = 100,  # ms
    draft_model_time: float = 10,    # ms（小模型约 10x 快）
) -> dict:
    """
    理论加速比分析
    
    期望接受 token 数：sum_{i=0}^{k} alpha^i = (1 - alpha^{k+1}) / (1 - alpha)
    （几何级数，alpha 是接受率）
    """
    import math
    
    # 期望每次 decode 产生的 token 数
    alpha = acceptance_rate
    k = num_speculative
    expected_tokens = sum(alpha**i for i in range(k + 1))
    
    # 时间：k 次小模型 + 1 次大模型（并行验证）
    time_per_cycle = k * draft_model_time + target_model_time
    
    # 对比：不用投机采样，需要 expected_tokens 次大模型
    time_standard = expected_tokens * target_model_time
    
    speedup = time_standard / time_per_cycle
    
    return {
        '期望接受 token 数': round(expected_tokens, 2),
        '每轮时间 (ms)': time_per_cycle,
        '无投机等效时间 (ms)': round(time_standard, 1),
        '理论加速比': round(speedup, 2),
        '接受率': f'{acceptance_rate:.0%}',
        '投机长度 k': k,
    }

# 不同场景分析
print("投机采样加速比分析:")
scenarios = [
    (5, 0.5, "接受率低（50%）"),
    (5, 0.7, "接受率中（70%）"),
    (5, 0.9, "接受率高（90%）"),
    (3, 0.7, "k=3, 接受率70%"),
    (10, 0.7, "k=10, 接受率70%"),
]

for k, alpha, desc in scenarios:
    result = analyze_speedup(k, alpha)
    print(f"\n{desc}:")
    for key, val in result.items():
        print(f"  {key}: {val}")
```

---

## 四、实际应用

```python
"""
投机采样的实际使用

1. vLLM 内置支持（推荐）
   vllm serve --speculative-model ngram --num-speculative-tokens 5

2. 常见 Draft Model 搭配
   - Target: Llama-3-70B → Draft: Llama-3-8B（同系列更好）
   - Target: Qwen2.5-72B → Draft: Qwen2.5-7B
   - Target: GPT-4 → Draft: GPT-3.5（API 场景）
   
3. N-gram Draft（不需要单独 Draft Model）
   - 从已生成序列中找重复的 n-gram 作为 draft
   - 适合文档续写、代码生成等重复性高的场景

4. 适用场景
   ✅ 代码生成（重复模式多，接受率高）
   ✅ 文档摘要（常见短语，接受率高）
   ❌ 创意写作（输出随机，接受率低）
   ❌ 多语言场景（draft 和 target 词表不匹配）
"""

# vLLM 投机采样配置
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",  # Draft 模型
    num_speculative_tokens=5,                               # k=5
    speculative_draft_tensor_parallel_size=1,
    tensor_parallel_size=4,  # 大模型 4 卡
)
```

---

## 五、面试题精讲

**Q: 投机采样为什么不改变输出分布？**

A: 这是投机采样最优雅的地方——它是**无损加速**。数学上可以证明，经过接受/拒绝机制后，最终接受的 token 序列与直接用大模型采样的分布完全相同（精确等价）。拒绝时从调整分布 $\max(0, p_t - p_d)$ 重采样，保证了这一点。

**Q: Draft 模型和 Target 模型需要什么条件？**

A: 
- Draft 应该是 Target 的"同系列小版本"（分词器必须相同！）
- 或者 n-gram 模型（不需要神经网络）
- 接受率是关键：一般要求 alpha > 0.6 才有加速效果

**Q: 投机采样适合并行化吗？**

A: 对单个序列，小模型生成 k 个 token 是串行的；但多序列可以批处理。更现代的方法如 Medusa（在同一模型上添加多个预测头）可以完全并行生成 draft token，进一步减少时间开销。

---

## 小结

```
投机采样流程：
1. Draft 模型串行生成 k 个 token
2. Target 模型并行验证 k+1 个位置
3. 接受率 alpha 决定期望收益
4. 理论加速比：~2-3x（alpha≈0.7, k=5）

适用场景：
✅ 代码生成（接受率高，效果最好）
❌ 自由创作（接受率低，效果差）
```
