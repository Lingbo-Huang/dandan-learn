---
layout: default
title: "D3 · RLHF"
render_with_liquid: false
---

# D3 · RLHF（从人类反馈中强化学习）

> **RLHF 的本质**：用人类的偏好判断训练一个奖励模型，再用 RL 让语言模型的输出最大化这个奖励——同时不要跑偏太远（KL 约束）。

---

## 一、RLHF 三阶段流程

```
阶段 1: SFT（监督微调）
  → 让模型学会基本的指令跟随

阶段 2: RM（奖励模型训练）
  → 用人类偏好数据训练"打分模型"
  → 给定 (prompt, response)，输出质量分数

阶段 3: PPO（强化学习微调）
  → 用 PPO 最大化 RM 打出的分数
  → KL 散度约束，防止模型跑偏
```

---

## 二、奖励模型（Reward Model）

### 2.1 Bradley-Terry 偏好模型

给定两个回复 $y_w$（更优）和 $y_l$（较差），人类偏好 $y_w > y_l$ 的概率：

$$P(y_w > y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $r(\cdot)$ 是奖励模型，$\sigma$ 是 sigmoid 函数。

训练损失（最大化正确偏好的对数似然）：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}[\log \sigma(r(x, y_w) - r(x, y_l))]$$

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel(nn.Module):
    """奖励模型：在语言模型基础上加一个线性 head"""
    
    def __init__(self, base_model_name: str):
        super().__init__()
        # 使用序列分类模型（最后的 CLS 位置输出标量）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,  # 输出一个标量分数
        )
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        reward = outputs.logits.squeeze(-1)  # [batch_size]
        return reward

def compute_rm_loss(
    reward_model: RewardModel,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> torch.Tensor:
    """
    奖励模型的 pairwise ranking loss
    
    chosen: 人类偏好的回复
    rejected: 人类不偏好的回复
    """
    r_chosen = reward_model(chosen_ids, chosen_mask)    # [B]
    r_rejected = reward_model(rejected_ids, rejected_mask)  # [B]
    
    # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
    loss = -nn.functional.logsigmoid(r_chosen - r_rejected).mean()
    
    # 准确率（chosen 分 > rejected 分的比例）
    accuracy = (r_chosen > r_rejected).float().mean()
    
    return loss, accuracy.item()


# 训练示例
def train_reward_model():
    """奖励模型训练循环"""
    from torch.optim import AdamW
    
    model = RewardModel("Qwen/Qwen2.5-1.5B")
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # 偏好数据格式（来自 Anthropic HH-RLHF 或 UltraFeedback）
    # {
    #   "prompt": "...",
    #   "chosen": "好的回答...",
    #   "rejected": "差的回答..."
    # }
    
    for step, batch in enumerate(dataloader):
        loss, acc = compute_rm_loss(
            model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 50 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, acc={acc:.3f}")
```

---

## 三、PPO 训练（强化学习阶段）

### 3.1 目标函数

$$\mathcal{L}_{PPO} = \mathbb{E}[r(x, y)] - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_{ref}(y|x)]$$

- 第一项：最大化奖励（越高越好）
- 第二项：KL 散度惩罚（防止偏离 SFT 模型太远）
- $\beta$：KL 系数（通常 0.01-0.1）

### 3.2 关键公式

```python
def compute_kl_penalty(
    log_probs_policy: torch.Tensor,   # RL 策略的对数概率
    log_probs_ref: torch.Tensor,       # SFT 参照模型的对数概率
    beta: float = 0.1
) -> torch.Tensor:
    """
    计算 KL 散度惩罚
    KL(π_θ || π_ref) = E[log π_θ - log π_ref]
    """
    kl = log_probs_policy - log_probs_ref
    return beta * kl

def compute_ppo_loss(
    log_probs: torch.Tensor,       # 当前策略的对数概率
    log_probs_old: torch.Tensor,   # 旧策略的对数概率（PPO clip 用）
    advantages: torch.Tensor,       # 优势函数
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    PPO-Clip 目标函数
    """
    # 重要性采样比率
    ratio = torch.exp(log_probs - log_probs_old)
    
    # Clip
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    # 取较小值（PPO 的保守更新）
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


class PPOTrainer:
    """RLHF PPO 训练器（简化版）"""
    
    def __init__(self, policy_model, ref_model, reward_model, beta=0.1):
        self.policy = policy_model   # 要训练的模型（从 SFT 初始化）
        self.ref = ref_model         # 冻结的参照模型（SFT 模型）
        self.rm = reward_model       # 奖励模型（冻结）
        self.beta = beta
    
    def generate_and_score(self, prompts):
        """生成回复并打分"""
        # 生成
        responses = self.policy.generate(prompts, max_new_tokens=256)
        
        # 计算奖励
        rewards = self.rm(prompts, responses)
        
        # 计算 KL 惩罚
        log_probs_policy = self.policy.log_probs(responses)
        log_probs_ref = self.ref.log_probs(responses)
        kl = self.beta * (log_probs_policy - log_probs_ref)
        
        # 总奖励
        total_reward = rewards - kl
        return responses, total_reward
    
    def ppo_step(self, prompts):
        """一步 PPO 更新"""
        # 1. 生成并打分
        responses, rewards = self.generate_and_score(prompts)
        
        # 2. 计算优势（这里简化为奖励本身，实际用 GAE）
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 3. 多轮 mini-batch 更新（PPO 核心）
        for epoch in range(4):  # PPO epoch
            log_probs = self.policy.log_probs(responses)
            loss = compute_ppo_loss(log_probs, log_probs.detach(), advantages)
            loss.backward()
            # optimizer.step() ...
```

---

## 四、使用 TRL 库进行 RLHF

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# PPO 配置
ppo_config = PPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    learning_rate=1.4e-5,
    log_with="tensorboard",
    mini_batch_size=1,
    batch_size=16,
    gradient_accumulation_steps=1,
    kl_penalty="kl",         # KL 惩罚方式
    init_kl_coef=0.2,        # 初始 KL 系数
    adap_kl_ctrl=True,       # 自适应 KL 控制
    target_kl=6.0,           # 目标 KL（超过就增大 beta）
)

# 带 Value Head 的策略模型（PPO 需要 Critic）
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# 训练循环
for batch in dataloader:
    query_tensors = batch["input_ids"]
    
    # 生成回复
    response_tensors = ppo_trainer.generate(
        query_tensors, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    
    # 计算奖励（来自 RM 或规则）
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]
    
    # PPO 步骤
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
```

---

## 五、RLHF 的挑战

| 挑战 | 描述 | 解决方案 |
|------|------|---------|
| 奖励 Hacking | 模型找到 RM 的漏洞刷分 | KL 约束、迭代更新 RM |
| 训练不稳定 | PPO 收敛困难 | 自适应 KL、梯度裁剪 |
| 数据收集昂贵 | 人工标注偏好数据 | AI 辅助标注 |
| 三模型显存 | Policy + Ref + RM | 量化 + offload |

---

## 六、面试题精讲

**Q: KL 散度惩罚的作用是什么？**

A: 防止"奖励 Hacking"——如果没有 KL 约束，模型会找到奖励模型的漏洞，生成高分但质量很差的文本（如不断重复得高分的短语）。KL 约束确保优化后的模型不会跑偏太远，保留 SFT 阶段学到的语言能力。

**Q: RLHF 中的 Reference Model 是什么？**

A: Reference Model 是 SFT 阶段训练好的模型（参数冻结），用于计算 KL 散度。目标函数是：最大化 $r(x,y) - \beta \cdot KL(\pi_\theta || \pi_{ref})$。

**Q: 为什么 PPO 在 RLHF 中比其他 RL 算法更受欢迎？**

A: PPO 的 Clip 机制天然限制了策略更新幅度（类似 KL 约束），训练稳定。相比 TRPO（需要计算 Hessian，计算昂贵），PPO 实现简单高效。对于语言模型这种高维动作空间，PPO 的稳定性尤为重要。

---

## 小结

```
RLHF 流程：
1. SFT：让模型会说话
2. RM：教会模型什么是"好的回答"（人类偏好打分）
3. PPO：最大化 RM 分数，KL 约束防止跑偏

关键公式：
- RM Loss: -E[log σ(r_chosen - r_rejected)]
- PPO Target: E[r(x,y)] - β·KL(π_θ || π_ref)
```
