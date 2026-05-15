---
layout: default
title: "D3 · GPT 预训练目标"
render_with_liquid: false
---

# D3 · GPT 预训练目标

> **一句话总结**：GPT 的预训练就是最大化序列中每个 token 的条件概率——给定前文，预测下一个词。这个看似简单的目标，催生了涌现能力。

---

## 一、语言建模目标

### 1.1 自回归语言模型（CLM）

GPT 使用**因果语言建模（Causal Language Modeling）**：

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

对序列 $[x_1, x_2, ..., x_T]$，模型学习：
- $P(x_2 | x_1)$
- $P(x_3 | x_1, x_2)$
- ...
- $P(x_T | x_1, ..., x_{T-1})$

### 1.2 对比 MLM（BERT 的方式）

| | CLM (GPT) | MLM (BERT) |
|--|-----------|-----------|
| 目标 | 预测下一个 token | 预测被 [MASK] 的 token |
| 注意力 | 因果（只看左侧） | 双向（左右都看） |
| 生成能力 | 天然支持 | 不适合生成 |
| 理解能力 | 强 | 更强（但需 fine-tune） |

---

## 二、交叉熵损失与困惑度

### 2.1 损失函数推导

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算 CLM 损失
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - 模型输出的 logit
        labels: [batch_size, seq_len] - 目标 token ID
    
    Returns:
        标量损失值
    """
    # 关键：输入是 x[:-1]，标签是 x[1:]（移位操作）
    # logits[:, :-1] 预测 labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # 展平后计算交叉熵
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100  # -100 表示不参与 loss 计算（padding 位置）
    )
    return loss

# 示例
batch_size, seq_len, vocab_size = 2, 10, 50257
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

loss = causal_lm_loss(logits, labels)
print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")
```

### 2.2 困惑度（Perplexity）

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t|x_{<t})\right)$$

困惑度的直觉：PPL=100 意味着模型每次预测相当于在 100 个候选中随机猜。

```python
def compute_perplexity(model, dataloader, device='cuda'):
    """计算模型在验证集上的困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # 对于 CLM，labels = input_ids
            )
            
            loss = outputs.loss
            # 只计算非 padding 位置的 token 数
            num_tokens = attention_mask[:, 1:].sum()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()
```

---

## 三、mini-GPT 完整实现

```python
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    """因果自注意力（带 KV Cache 支持）"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 因果掩码（下三角矩阵）
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV 投影
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # 多头 reshape
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # [B, n_heads, T, d_head]
        
        # Scaled Dot-Product Attention
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # [B, n_heads, T, T]
        
        # 因果掩码
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        out = attn @ v  # [B, n_heads, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class GPTBlock(nn.Module):
    """GPT Transformer 块"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-LN
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """mini-GPT：标准 Decoder-only 架构"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（token_emb 与 lm_head 共享权重）
        self.lm_head.weight = self.token_emb.weight
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        assert T <= self.max_seq_len
        
        pos = torch.arange(T, device=input_ids.device)
        
        x = self.drop(
            self.token_emb(input_ids) + self.pos_emb(pos)
        )
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """自回归文本生成"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 截断到 max_seq_len
                context = input_ids[:, -self.max_seq_len:]
                logits = self(context)
                
                # 取最后一个位置的 logit
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k 采样
                if top_k > 0:
                    values, _ = torch.topk(next_logits, top_k)
                    threshold = values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float('-inf')
                    )
                
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# 参数量统计
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }

# 实例化不同规模
configs = {
    'mini': dict(d_model=256, n_layers=4, n_heads=4),
    'small': dict(d_model=512, n_layers=8, n_heads=8),
    'medium': dict(d_model=1024, n_layers=24, n_heads=16),
}

for name, cfg in configs.items():
    model = MiniGPT(**cfg)
    params = count_parameters(model)
    print(f"{name}: {params['total_M']:.1f}M parameters")
```

---

## 四、预训练训练循环

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_gpt(
    model: MiniGPT,
    train_dataloader,
    num_epochs: int = 3,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    grad_clip: float = 1.0,
    device: str = 'cuda'
):
    """GPT 预训练主循环"""
    model.to(device)
    model.train()
    
    # AdamW：带权重衰减，bias 和 LayerNorm 不衰减
    decay_params = [p for n, p in model.named_parameters() 
                   if p.dim() >= 2]  # 矩阵权重
    no_decay_params = [p for n, p in model.named_parameters() 
                      if p.dim() < 2]  # bias, LayerNorm
    
    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    
    total_steps = num_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
    
    step = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            
            # Warmup（线性预热）
            if step < warmup_steps:
                lr_scale = step / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr * lr_scale
            
            # 前向传播
            logits = model(input_ids)
            loss = causal_lm_loss(logits, input_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            step += 1
            
            if step % 100 == 0:
                ppl = torch.exp(loss).item()
                print(f"Step {step}: loss={loss.item():.4f}, ppl={ppl:.2f}, "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
```

---

## 五、面试题精讲

**Q: 为什么 GPT 用 Pre-LayerNorm 而不是 Post-LayerNorm？**

A: 原始 Transformer 用 Post-LN（LayerNorm 在残差之后）。Pre-LN（LayerNorm 在残差之前）在大模型中更稳定：梯度通过残差路径直接传播，不经过 LayerNorm 归一化，训练更稳定，可以不用 warmup 也能收敛。

**Q: Causal Mask 的实现细节？**

A: 下三角矩阵（对角线及以下为 1，其余为 -inf）。通常用 `torch.tril` 生成，然后 `masked_fill(mask == 0, float('-inf'))` 在 softmax 之前应用。注意要在 softmax **之前**填充 -inf，否则 softmax 后不是 0。

**Q: 权重绑定（weight tying）是什么？有什么作用？**

A: 将 token embedding 矩阵和 LM head 线性层共享同一组参数。理论基础是两者处理的是同一个语义空间（输入 token → 向量 和 向量 → 输出 token 应该用相同的表示）。好处：减少参数量（vocab_size × d_model 这么大的矩阵只需一份），通常还能提升性能。

---

## 小结

| 概念 | 关键点 |
|------|--------|
| CLM 目标 | 最大化 $P(x_t\|x_{<t})$ |
| 损失函数 | 交叉熵，移位一位 |
| 困惑度 | exp(avg_loss)，越低越好 |
| 因果 Mask | 下三角矩阵，防止看到未来 |
| 权重绑定 | Embedding ↔ LM Head 共享 |
