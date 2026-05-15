---
layout: default
title: "D4 · MoE 架构：专家路由与负载均衡"
render_with_liquid: false
---

# D4 · MoE 架构：专家路由与负载均衡

## 什么是 Mixture of Experts？

**MoE**（Mixture of Experts）将单个 FFN 替换为 N 个"专家" FFN，每个 token 只激活 K 个专家。

```
标准 Transformer FFN：
  token → [FFN] → output

MoE Transformer FFN：
  token → [Router] → 选择 Top-K 专家
         ├→ Expert 1 ┐
         ├→ Expert 3 ├→ 加权求和 → output
         └...        ┘
```

**关键数字**（以 Mixtral 8×7B 为例）：
- 总专家数：8 个
- 每 token 激活专家数：K=2
- 每个专家大小：7B 参数
- 实际激活参数：~12B（2/8 × 56B + 其他层）
- 效果：接近 70B 稠密模型

## Top-K 路由机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    Top-K 专家路由器
    """
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由网络：简单的线性层
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        """
        x: [batch * seq_len, hidden_dim]（通常先 reshape 为 2D）
        返回：
          - expert_weights: [batch * seq_len, top_k] - 专家权重（softmax 后）
          - expert_indices: [batch * seq_len, top_k] - 选择的专家 ID
        """
        # 计算路由 logits
        router_logits = self.gate(x)  # [tokens, num_experts]
        
        # Top-K 选择
        expert_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Softmax 归一化（只对选中的 top-k 归一化）
        expert_weights = F.softmax(expert_weights, dim=-1)  # [tokens, top_k]
        
        return expert_weights, expert_indices


class SparseExpertLayer(nn.Module):
    """
    稀疏 MoE 层（Switch Transformer 风格）
    """
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器
        self.router = TopKRouter(hidden_dim, num_experts, top_k)
        
        # 专家 FFN（每个专家独立）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim, bias=False),
                nn.SiLU(),
                nn.Linear(ffn_dim, hidden_dim, bias=False)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, hidden = x.shape
        
        # Flatten batch 和 seq 维度
        x_flat = x.view(-1, hidden)  # [tokens, hidden]
        num_tokens = x_flat.shape[0]
        
        # 路由
        expert_weights, expert_indices = self.router(x_flat)
        # expert_weights:  [tokens, top_k]
        # expert_indices:  [tokens, top_k]
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 对每个专家，收集分配给它的 token，批量计算
        for expert_id in range(self.num_experts):
            # 找到哪些 token 选择了这个专家
            # expert_indices == expert_id: [tokens, top_k] 的布尔掩码
            expert_mask = (expert_indices == expert_id)  # [tokens, top_k]
            
            if not expert_mask.any():
                continue
            
            # 获取 token 索引和对应的权重
            token_indices, k_indices = expert_mask.nonzero(as_tuple=True)
            weights = expert_weights[token_indices, k_indices]  # [num_selected]
            
            # 批量处理这些 token
            selected_tokens = x_flat[token_indices]  # [num_selected, hidden]
            expert_output = self.experts[expert_id](selected_tokens)  # [num_selected, hidden]
            
            # 加权累加回输出
            output.scatter_add_(
                0,
                token_indices.unsqueeze(1).expand(-1, hidden),
                expert_output * weights.unsqueeze(1)
            )
        
        return output.view(batch, seq_len, hidden)
```

## 负载均衡问题

**训练崩溃场景**：如果没有负载均衡，路由器会退化——越热门的专家越被选择，越冷门的专家越不被训练，最终所有 token 都路由到少数专家（"专家塌缩"）。

### 辅助负载均衡损失

```python
def load_balance_loss(
    router_logits: torch.Tensor,  # [tokens, num_experts] - 未 softmax 的 logits
    expert_indices: torch.Tensor, # [tokens, top_k] - 选择的专家
    num_experts: int,
    top_k: int,
    alpha: float = 0.01           # 平衡损失的权重
) -> torch.Tensor:
    """
    Switch Transformer 风格的负载均衡辅助损失
    
    L_balance = alpha * N * sum_i(f_i * P_i)
    
    f_i: 路由到专家 i 的 token 比例（离散，不可微）
    P_i: 专家 i 的平均路由概率（连续，可微）
    N:   专家数量
    """
    num_tokens = router_logits.shape[0]
    
    # P_i: 每个专家的平均路由概率（可微）
    router_probs = F.softmax(router_logits, dim=-1)  # [tokens, num_experts]
    P = router_probs.mean(dim=0)  # [num_experts]
    
    # f_i: 每个专家实际处理的 token 比例（使用 one-hot 近似，不可微）
    # expert_indices: [tokens, top_k] → one-hot: [tokens, num_experts]
    one_hot = torch.zeros(num_tokens, num_experts, device=router_logits.device)
    for k in range(top_k):
        one_hot.scatter_(1, expert_indices[:, k:k+1], 1.0 / top_k)
    f = one_hot.mean(dim=0)  # [num_experts]
    
    # 辅助损失
    balance_loss = num_experts * (f * P).sum()
    
    return alpha * balance_loss


def training_step(model, batch, alpha=0.01):
    """带负载均衡损失的训练步骤"""
    outputs, router_logits_list, expert_indices_list = model(batch)
    
    # 任务损失
    task_loss = F.cross_entropy(outputs, batch.labels)
    
    # 每层的负载均衡损失
    aux_loss = 0
    for router_logits, expert_indices in zip(router_logits_list, expert_indices_list):
        aux_loss += load_balance_loss(router_logits, expert_indices, 
                                       model.num_experts, model.top_k, alpha)
    aux_loss /= len(router_logits_list)
    
    total_loss = task_loss + aux_loss
    
    return total_loss, task_loss.item(), aux_loss.item()
```

## Expert Choice 路由（替代方案）

Switch/Top-K 路由让每个 **token** 选择专家；Expert Choice 让每个**专家**选择 token：

```python
class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice 路由：每个专家选择 top-k token
    自动保证负载均衡！（每个专家固定选 k 个 token）
    """
    def __init__(self, hidden_dim: int, num_experts: int, expert_capacity: int):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = expert_capacity  # 每个专家的 capacity
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        """
        x: [tokens, hidden_dim]
        """
        num_tokens = x.shape[0]
        
        # 每个 token 对每个专家的路由概率
        logits = self.gate(x)                         # [tokens, num_experts]
        probs = F.softmax(logits, dim=0)              # softmax over tokens（不是 experts！）
        
        # 每个专家选择概率最高的 capacity 个 token
        # topk over tokens
        expert_weights, token_indices = probs.T.topk(self.capacity, dim=-1)
        # expert_weights: [num_experts, capacity]
        # token_indices:  [num_experts, capacity]
        
        return expert_weights, token_indices

# Expert Choice 的好处：
# 1. 自动负载均衡（每个专家固定处理 capacity 个 token）
# 2. 不需要辅助损失
# 缺点：
# 1. token 可能被多个专家选中（也可能没有专家选中）
# 2. 计算模式不同，实现较复杂
```

## Mixtral 8×7B 架构

```python
"""
Mixtral 8×7B 的 MoE 配置
"""
config = {
    "hidden_size": 4096,
    "intermediate_size": 14336,  # 专家 FFN 大小
    "num_experts_per_tok": 2,    # K=2
    "num_local_experts": 8,      # 8 个专家
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,    # GQA
}

# 参数量估算：
# 每个专家: 2 × (4096 × 14336) × 2(SiLU gate) = ~400M params/expert
# 8 个专家: ~3.2B × 32 层 = ~100B（全部专家参数）
# 但 K=2：每次激活 2/8 × 3.2B = 800M × 32 层 ≈ 25B
# 加上 Attention 等：实际激活 ~12B

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    # MoE 推理：每次只激活 K=2 个专家，速度接近 12B 稠密模型
)
```

## 面试题

**Q: MoE 的负载均衡为什么重要？如何实现？**

A: 负载均衡在 MoE 训练中至关重要，原因是路由器容易陷入局部最优——某些专家因为初始权重略好而被频繁选中，被选中的专家因为接收更多训练数据而变得更强，形成正反馈循环，最终大多数 token 路由到少数几个专家（"专家塌缩"），其他专家几乎不参与训练，MoE 退化为稠密模型。解决方案主要有两种：①辅助损失（Switch Transformer 风格）：在总损失中加入 L_balance = α × N × Σ(f_i × P_i)，其中 f_i 是每个专家的实际使用频率，P_i 是路由器给专家的平均概率，鼓励均匀分配；②Expert Choice 路由：每个专家主动选择固定数量的 token，从设计上保证负载均衡，无需辅助损失。
