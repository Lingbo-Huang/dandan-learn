---
layout: default
title: "D4 · Scaling Law"
render_with_liquid: false
---

# D4 · Scaling Law

> **一句话总结**：给定计算预算，Scaling Law 告诉你如何在模型大小和数据量之间做最优分配。这是设计大模型最重要的指导原则。

---

## 一、Kaplan Scaling Law（2020）

OpenAI 2020 年发现：模型性能（loss）与模型大小 N、数据量 D、计算量 C 呈幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

- $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$
- Loss 随参数量的幂律递减
- **关键发现**：参数量比数据量更重要（指数更大）

---

## 二、Chinchilla Scaling Law（2022，最重要）

DeepMind 的 Chinchilla 论文推翻了 Kaplan 的结论：

**最优训练 token 数 ≈ 20 × 参数量**

$$N^* = \frac{C}{6D^*}, \quad D^* = \frac{C}{6N^*}$$

| 模型 | 参数量 | 计算最优 Token 数 | 实际使用 |
|------|--------|-----------------|---------|
| GPT-3 | 175B | 3.5T | 300B（训练不足）|
| LLaMA-1 | 65B | 1.3T | 1.4T（接近最优）|
| LLaMA-2 | 70B | 1.4T | 2T（过量训练）|
| LLaMA-3 | 8B | 160B | 15T（大幅过量）|

> **为什么要"过量训练"？** Chinchilla 最优是对**训练成本**最优，但实际部署关注**推理成本**。用更小的模型 + 更多数据训练，推理更便宜。

---

## 三、FLOPs 计算

```python
def estimate_training_flops(
    n_params: int,    # 参数量（不含 embedding）
    n_tokens: int,    # 训练 token 数
) -> dict:
    """
    估算训练所需 FLOPs
    
    规则：每个 token 前向 ≈ 2N FLOPs（N 为非 embedding 参数）
         反向传播 ≈ 2 × 前向 = 4N FLOPs
         总计 ≈ 6N × D
    
    来源：Kaplan et al. 2020
    """
    forward_flops = 2 * n_params * n_tokens
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    return {
        'forward_FLOPs': forward_flops,
        'backward_FLOPs': backward_flops,
        'total_FLOPs': total_flops,
        'total_PFLOPs': total_flops / 1e15,
    }

def estimate_model_params(
    d_model: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
    seq_len: int = None,  # 不影响参数量
) -> dict:
    """估算 GPT 各层参数量"""
    
    # Attention 层：Q, K, V 投影 + Output 投影
    attn_params = 4 * d_model * d_model  # 每层
    
    # FFN 层：两个线性层（4x 扩展）
    ffn_params = 2 * d_model * (4 * d_model)  # 每层
    
    # LayerNorm：2 * d_model（gamma + beta）
    ln_params = 2 * d_model  # 每层有2个 LN
    
    total_per_layer = attn_params + ffn_params + ln_params * 2
    transformer_params = total_per_layer * n_layers
    
    # Embedding 层（通常不计入 N）
    embedding_params = vocab_size * d_model
    
    total_non_embedding = transformer_params
    total_with_embedding = transformer_params + embedding_params
    
    return {
        'd_model': d_model,
        'n_layers': n_layers,
        'attn_per_layer': attn_params,
        'ffn_per_layer': ffn_params,
        'transformer_total': transformer_params,
        'embedding': embedding_params,
        'total_non_emb_B': total_non_embedding / 1e9,
        'total_B': total_with_embedding / 1e9,
    }

# 验证：GPT-3 175B
gpt3_params = estimate_model_params(
    d_model=12288, n_layers=96, n_heads=96, vocab_size=50257
)
print("GPT-3 参数估算:")
for k, v in gpt3_params.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}")

# GPT-3 训练 FLOPs
flops = estimate_training_flops(
    n_params=int(175e9),
    n_tokens=int(300e9)
)
print(f"\nGPT-3 训练 FLOPs: {flops['total_PFLOPs']:.0f} PFLOPs")
print(f"（A100 一张 80GB，峰值 312 TFLOPs/s，需约 {flops['total_FLOPs'] / (312e12 * 0.3 * 3600 * 24):.0f} 天，利用率30%）")
```

---

## 四、神经 Scaling Law 的数学推导

```python
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Chinchilla 损失函数
    L(N, D) = E + A/N^alpha + B/D^beta
    
    E：不可减少的损失（自然语言的固有熵）
    A/N^alpha：参数量不足引起的损失
    B/D^beta：数据量不足引起的损失
    """
    return E + A / (N ** alpha) + B / (D ** beta)

def optimal_allocation(C: float) -> tuple:
    """
    给定计算预算 C（FLOPs），求最优 (N*, D*)
    
    由 dL/dN = dL/dD = 0 且 C = 6ND 得：
    N* = G * (C/6)^(beta/(alpha+beta))
    D* = G^(-1) * (C/6)^(alpha/(alpha+beta))
    """
    alpha, beta = 0.34, 0.28
    A, B = 406.4, 410.7
    
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    
    N_star = G * (C / 6) ** (beta / (alpha + beta))
    D_star = (C / 6) ** (alpha / (alpha + beta)) / G
    
    return N_star, D_star

# 不同计算预算下的最优分配
compute_budgets = {
    '1e21 FLOPs (7B 级别)': 1e21,
    '1e22 FLOPs (13B 级别)': 1e22,
    '1e23 FLOPs (70B 级别)': 1e23,
    '1e24 FLOPs (175B 级别)': 1e24,
}

print("Chinchilla 最优分配:")
print(f"{'预算':<30} {'N* (参数)':<20} {'D* (Tokens)':<20} {'N:D 比'}")
print("-" * 80)
for desc, C in compute_budgets.items():
    N, D = optimal_allocation(C)
    ratio = N / D
    print(f"{desc:<30} {N/1e9:.1f}B{'':<15} {D/1e9:.1f}B{'':<15} 1:{D/N:.0f}")
```

---

## 五、Scaling Law 的实践指导

```python
class ScalingLawAdvisor:
    """基于 Scaling Law 的实践建议"""
    
    @staticmethod
    def gpu_days_to_flops(a100_count: int, days: int, efficiency: float = 0.4) -> float:
        """
        计算可用 FLOPs
        A100 峰值: 312 TFLOPs/s (BF16)
        实际效率通常 30-50%
        """
        a100_tflops = 312e12  # FLOPs/s
        seconds = days * 24 * 3600
        return a100_count * a100_tflops * efficiency * seconds
    
    @staticmethod
    def recommend_config(
        a100_count: int, 
        training_days: int,
        efficiency: float = 0.4
    ) -> dict:
        """给出训练建议"""
        total_flops = ScalingLawAdvisor.gpu_days_to_flops(
            a100_count, training_days, efficiency
        )
        
        N_star, D_star = optimal_allocation(total_flops)
        
        return {
            'compute_budget_PFLOPs': total_flops / 1e15,
            'optimal_params_B': N_star / 1e9,
            'optimal_tokens_B': D_star / 1e9,
            'params_to_tokens_ratio': f'1:{D_star/N_star:.0f}',
            'expected_loss': chinchilla_loss(N_star, D_star),
            'note': f"若要 cheaper inference，可缩小模型到 {N_star/4/1e9:.1f}B，增大 tokens 到 {D_star*3/1e9:.0f}B"
        }

# 实际场景：你有 64 张 A100，训练 30 天
advisor = ScalingLawAdvisor()
config = advisor.recommend_config(a100_count=64, training_days=30)
print("推荐训练配置:")
for k, v in config.items():
    print(f"  {k}: {v}")
```

---

## 六、面试题精讲

**Q: Kaplan 和 Chinchilla Scaling Law 的核心区别？**

A: Kaplan 认为参数比数据更重要（推荐"大模型少数据"），导致 GPT-3 数据严重不足。Chinchilla 证明两者应该等量增长（20 tokens/param），GPT-3 在 Chinchilla 最优下应该训练 3.5T tokens，实际只用了 300B。

**Q: 为什么 LLaMA 用"过量训练"策略？**

A: 因为训练成本 vs 推理成本。LLaMA 的目标是**推理高效**：用 7B/13B 的小模型，训练 2T tokens（远超最优），得到的模型比 Chinchilla 最优的大模型推理快很多，且性能接近。这在工业部署中更实用。

**Q: 如何快速估算一个模型的 FLOPs？**

A: 
- **参数量估算**：对于 Transformer，主要参数来自 Attention（$4d^2$/层）和 FFN（$8d^2$/层），共约 $12d^2 L$（L 为层数）
- **训练 FLOPs**：$\approx 6ND$，N 为非 embedding 参数，D 为训练 token 数
- **例子**：7B 模型训练 1T tokens ≈ 6 × 7B × 1T = 4.2 × 10²² FLOPs

---

## 小结

```
Kaplan (2020): L ∝ N^{-0.076}  → 参数比数据重要
Chinchilla (2022): N* ≈ D*/20   → 两者同等重要

实践原则：
1. 计算预算确定后，N 和 D 各占一半权重
2. 推理部署场景下，可以"过量训练"小模型
3. 记住：FLOPs ≈ 6ND，参数 ≈ 12d²L
```
