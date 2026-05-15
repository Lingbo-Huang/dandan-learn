---
layout: default
title: "D4 · DPO：直接偏好优化"
render_with_liquid: false
---

# D4 · DPO（Direct Preference Optimization）

> **DPO 的洞见**：RLHF 中存在一个隐式的闭合解——可以直接从偏好数据学习策略，完全绕过奖励模型和 PPO 的复杂流程。

---

## 一、DPO 的推导

### 1.1 从 RLHF 目标推导

RLHF 的目标是找到最大化以下目标的策略 $\pi_\theta$：

$$\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta \cdot \text{KL}[\pi_\theta || \pi_{ref}]$$

这个问题有**解析解**：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

反解出奖励函数 $r$：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 偏好模型（$Z(x)$ 在 $y_w$ 和 $y_l$ 的差中抵消）：

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**这就是 DPO 的损失函数**！只需要策略模型和参照模型的对数概率之差。

### 1.2 DPO vs RLHF

| | RLHF (PPO) | DPO |
|--|-----------|-----|
| 需要 RM | ✅ 需要单独训练 | ❌ 不需要 |
| 需要 RL | ✅ PPO | ❌ 监督学习 |
| 训练稳定性 | ❌ 不稳定 | ✅ 稳定 |
| 显存占用 | ❌ 3个模型 | ✅ 2个模型 |
| 效果 | 上限更高 | 接近，更简单 |

---

## 二、DPO 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class DPOBatch:
    """DPO 训练 batch"""
    prompt_input_ids: torch.Tensor         # [B, seq_len]
    chosen_input_ids: torch.Tensor         # [B, seq_len]（prompt + chosen response）
    rejected_input_ids: torch.Tensor       # [B, seq_len]（prompt + rejected response）
    chosen_labels: torch.Tensor            # [B, seq_len]（只在 response 位置有值）
    rejected_labels: torch.Tensor          # [B, seq_len]

def get_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    计算模型对序列的对数概率（只在 response 位置）
    
    Returns:
        log_probs: [batch_size] - 每个样本的平均对数概率
    """
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids=input_ids).logits  # [B, T, V]
    
    # 移位：logits[t] 预测 labels[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # 计算每个 token 的对数概率
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 只取 label 位置的对数概率（label=-100 的位置是 padding/prompt）
    token_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask 掉 -100 位置
    mask = shift_labels != -100
    token_log_probs = token_log_probs * mask
    
    # 平均（归一化序列长度）
    return token_log_probs.sum(dim=-1) / mask.sum(dim=-1).float()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,     # 策略模型对 chosen 的对数概率
    policy_rejected_logps: torch.Tensor,   # 策略模型对 rejected 的对数概率
    ref_chosen_logps: torch.Tensor,        # 参照模型对 chosen 的对数概率
    ref_rejected_logps: torch.Tensor,      # 参照模型对 rejected 的对数概率
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO 损失函数
    
    Returns:
        loss: 标量
        chosen_rewards: chosen 回复的隐式奖励
        rejected_rewards: rejected 回复的隐式奖励
    """
    # 隐式奖励（相对于参照模型的对数概率比）
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    # DPO 损失
    reward_diff = chosen_rewards - rejected_rewards
    
    if label_smoothing > 0:
        # Label smoothing DPO（更稳定）
        loss = (
            -F.logsigmoid(reward_diff) * (1 - label_smoothing) +
            -F.logsigmoid(-reward_diff) * label_smoothing
        ).mean()
    else:
        loss = -F.logsigmoid(reward_diff).mean()
    
    return loss, chosen_rewards.detach(), rejected_rewards.detach()


class DPOTrainer:
    """DPO 训练器"""
    
    def __init__(self, policy_model, ref_model, beta=0.1, learning_rate=1e-6):
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta
        
        # 冻结参照模型
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
    
    def training_step(self, batch: DPOBatch) -> dict:
        self.policy.train()
        
        # 策略模型的对数概率
        policy_chosen_logps = get_log_probs(
            self.policy, batch.chosen_input_ids, batch.chosen_labels
        )
        policy_rejected_logps = get_log_probs(
            self.policy, batch.rejected_input_ids, batch.rejected_labels
        )
        
        # 参照模型的对数概率（不计算梯度）
        with torch.no_grad():
            ref_chosen_logps = get_log_probs(
                self.ref, batch.chosen_input_ids, batch.chosen_labels
            )
            ref_rejected_logps = get_log_probs(
                self.ref, batch.rejected_input_ids, batch.rejected_labels
            )
        
        # 计算 DPO 损失
        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # 计算准确率（chosen 奖励 > rejected 奖励的比例）
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_rewards_mean': chosen_rewards.mean().item(),
            'rejected_rewards_mean': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
        }
```

---

## 三、使用 TRL 的 DPO

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def run_dpo():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 偏好数据集（格式：prompt, chosen, rejected）
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    dpo_config = DPOConfig(
        beta=0.1,                     # KL 系数
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,           # 比 SFT 更小的 lr
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        output_dir="./dpo_output",
        bf16=True,
        loss_type="sigmoid",          # 标准 DPO
        # loss_type="ipo"             # IPO 变体（更稳定）
        # loss_type="robust"          # 带 label smoothing
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
```

---

## 四、DPO 变体

```python
"""
DPO 变体对比：

1. 标准 DPO (Rafailov et al., 2023)
   - Loss: -log σ(β(log π_w/π_ref_w - log π_l/π_ref_l))
   
2. IPO (Azar et al., 2023)
   - 防止过拟合偏好数据
   - Loss: (log π_w/π_ref_w - log π_l/π_ref_l - 1/(2β))²
   
3. KTO (Ethayarajh et al., 2024)
   - 不需要配对数据，只需 (prompt, response, label) 三元组
   - 基于 Kahneman-Tversky 效用理论
   
4. SimPO (Meng et al., 2024)
   - 不需要参照模型
   - Loss: -log σ(β(logprob_w/len_w - logprob_l/len_l) - γ)
"""

def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    SimPO：不需要参照模型的 DPO 变体
    
    关键：用序列平均对数概率代替相对于参照模型的对数概率比
    """
    # 归一化为每 token 平均对数概率
    chosen_logps_norm = policy_chosen_logps / chosen_lengths
    rejected_logps_norm = policy_rejected_logps / rejected_lengths
    
    loss = -F.logsigmoid(beta * (chosen_logps_norm - rejected_logps_norm) - gamma).mean()
    return loss
```

---

## 五、面试题精讲

**Q: DPO 和 RLHF 的数学等价性？**

A: DPO 通过数学推导证明：在 Bradley-Terry 偏好模型下，RLHF 的最优策略有闭合解，可以直接通过最大化偏好数据的似然来学习，无需显式训练奖励模型和运行 PPO。两者在理论上目标相同，但 DPO 实践更简单。

**Q: 什么时候选 RLHF，什么时候选 DPO？**

A:
- **选 RLHF**：有在线标注能力（模型生成 → 人标注 → 继续训练），需要极致对齐效果（GPT-4 级别），允许复杂训练基础设施
- **选 DPO**：有现成的偏好数据集，资源有限，追求简单稳定，多数开源模型微调场景

**Q: DPO 训练监控哪些指标？**

A:
- `accuracy`：chosen 的隐式奖励 > rejected 的比例，应该随训练上升到 70-90%
- `reward_margin`：两者差值，越大越好（但过大可能过拟合）
- `chosen_rewards` 和 `rejected_rewards` 的趋势（前者应上升，后者应下降）

---

## 小结

```
DPO 公式：
L = -E[log σ(β(log π_θ(yw|x)/π_ref(yw|x) - log π_θ(yl|x)/π_ref(yl|x)))]

优点：稳定、简单、无需 RM 和 PPO
适用：有偏好数据对，资源有限的场景

变体选择：
- 标准 DPO：默认选择
- IPO：数据偏好噪声大时
- KTO：没有配对数据时
- SimPO：不想维护参照模型时
```
