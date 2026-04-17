# D7：综合实战——从零搭建小型 Transformer

> **Week 2 · Day 7** | 大模型学习路线

---

## 一、本篇目标

综合 D1-D6 的所有知识，从零搭建一个**完整可训练的小型 GPT**：

- 完整的 Transformer Decoder 架构
- 训练字符级语言模型（Character-Level Language Model）
- 在莎士比亚文本上训练
- 从零推理生成文本

模型规模参考（训练时间约 5-10 分钟，CPU 可运行）：
- 参数量：约 300K
- 上下文长度：128
- 层数：4，头数：4，维度：128

---

## 二、完整架构概览

```
输入 Tokens (B, T)
    ↓ Token Embedding
    ↓ Position Embedding
    ↓ Dropout
    ↓ [TransformerBlock × N]
        ↓ LayerNorm → Multi-Head Self-Attention (Causal) → Residual
        ↓ LayerNorm → Feed-Forward Network → Residual
    ↓ LayerNorm
    ↓ Linear Projection → Logits (B, T, vocab_size)
    ↓ Cross-Entropy Loss
```

---

## 三、完整代码实现

```python
"""
从零搭建小型 GPT（字符级语言模型）
综合实现：Embedding + Attention + FFN + 训练 + 推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple


# ==================== 配置 ====================

@dataclass
class GPTConfig:
    # 模型超参数
    vocab_size: int = 65        # 字符级词表大小（莎士比亚文本）
    block_size: int = 128       # 最大序列长度（上下文窗口）
    n_layer: int = 4            # Transformer 层数
    n_head: int = 4             # 注意力头数
    n_embd: int = 128           # 嵌入维度
    dropout: float = 0.1        # Dropout 概率
    bias: bool = True           # 是否使用偏置
    
    @property
    def d_k(self):
        return self.n_embd // self.n_head


# ==================== 组件实现 ====================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力（GPT 风格）
    每个 token 只能关注自己和之前的 token
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Q/K/V 合并投影（一次线性变换得到所有三个）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 注册因果掩码（不是参数，不参与梯度更新）
        # 上三角矩阵：位置 (i,j) 为 True 表示 j > i（未来位置）
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool), 
                diagonal=1
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels(n_embd)
        
        # 一次投影得到 Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*n_embd)
        Q, K, V = qkv.split(self.n_embd, dim=2)  # 各 (B, T, n_embd)
        
        # 拆分多头
        # (B, T, n_embd) -> (B, n_head, T, d_k)
        Q = Q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, n_head, T, T)
        
        # 应用因果掩码（只看过去）
        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),  # (1, 1, T, T)
            float('-inf')
        )
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权聚合
        out = torch.matmul(attn_weights, V)  # (B, n_head, T, d_k)
        
        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        
        # 输出投影 + Dropout
        out = self.resid_dropout(self.c_proj(out))
        
        return out


class FeedForward(nn.Module):
    """
    前馈网络（FFN）
    
    结构：Linear(d_model → 4d_model) → GELU → Linear(4d_model → d_model)
    
    为什么是 4x？原论文选择，实践中效果好
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),  # 比 ReLU 更平滑，现代 LLM 标配
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer 解码器块
    
    结构（Pre-LN，GPT-2 风格）：
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    注：原论文是 Post-LN（先 Attention 再 LayerNorm），
    但 Pre-LN 训练更稳定，现代模型大多采用
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + 残差连接
        x = x + self.attn(self.ln_1(x))   # 注意力子层
        x = x + self.ffn(self.ln_2(x))    # FFN 子层
        return x


class GPT(nn.Module):
    """
    小型 GPT 模型
    
    字符级语言模型，给定前 T 个字符，预测第 T+1 个字符
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            # Token Embedding：将离散 token ID 映射到连续向量
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding：为每个位置提供位置信息
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            # Transformer 块堆叠
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            # 最终 LayerNorm
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        
        # 语言模型头：将 n_embd 映射到词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定：lm_head 与 wte 共享权重（减少参数，提升性能）
        self.transformer['wte'].weight = self.lm_head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 对残差连接的投影层使用特殊初始化（GPT-2 策略）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        print(f"GPT 参数量: {self.get_num_params()/1e6:.2f}M")
    
    def _init_weights(self, module: nn.Module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """计算参数量（默认不含 embedding）"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params
    
    def forward(
        self, 
        idx: torch.Tensor,          # (B, T) token indices
        targets: Optional[torch.Tensor] = None  # (B, T) target indices
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: (B, T) 输入 token 索引
            targets: (B, T) 目标 token 索引（训练时使用）
        
        Returns:
            logits: (B, T, vocab_size)
            loss: 交叉熵损失（如果提供 targets）
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"序列长度 {T} 超过最大上下文长度 {self.config.block_size}"
        
        device = idx.device
        
        # Token Embedding + Position Embedding
        tok_emb = self.transformer['wte'](idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.transformer['wpe'](pos)  # (T, n_embd)，会广播到 (B, T, n_embd)
        
        x = self.transformer['drop'](tok_emb + pos_emb)
        
        # 通过 N 层 Transformer
        for block in self.transformer['h']:
            x = block(x)
        
        # 最终 LayerNorm
        x = self.transformer['ln_f'](x)
        
        if targets is not None:
            # 训练：计算所有位置的 loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # 将 (B, T, vocab_size) 展平为 (B*T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # 可用 -1 标记不计算 loss 的位置
            )
        else:
            # 推理：只计算最后一个位置的 logits
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor,          # (B, T) 起始 tokens
        max_new_tokens: int = 100,  # 生成的最大 token 数
        temperature: float = 1.0,   # 温度（越低越确定，越高越随机）
        top_k: Optional[int] = None,  # Top-K 采样（限制候选词数量）
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        温度采样：logits / temperature → softmax → sample
        Top-K 采样：只保留概率最高的 K 个词
        """
        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]
            
            # 前向传播（推理模式）
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)，取最后一个位置
            
            # 温度缩放
            logits = logits / temperature
            
            # Top-K 截断
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将 Top-K 之外的 logits 设为 -inf
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax → 概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 追加到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ==================== 数据处理 ====================

def prepare_shakespeare_data(data_dir: str = "./data"):
    """下载并处理莎士比亚文本数据"""
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "shakespeare.txt")
    
    if not os.path.exists(data_path):
        print("下载莎士比亚文本...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, data_path)
            print("下载完成")
        except Exception:
            # 如果下载失败，使用示例文本
            print("下载失败，使用示例文本")
            sample = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come""" * 100  # 重复以获得足够数据
            with open(data_path, 'w') as f:
                f.write(sample)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 构建字符级词表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 编码全文
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    # 切分训练/验证集（90/10）
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"文本长度: {len(text):,} 字符")
    print(f"词表大小: {vocab_size}")
    print(f"训练集: {len(train_data):,} tokens")
    print(f"验证集: {len(val_data):,} tokens")
    
    return train_data, val_data, char_to_idx, idx_to_char, vocab_size


def get_batch(
    data: torch.Tensor, 
    block_size: int, 
    batch_size: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样一个 batch
    
    Returns:
        x: (batch_size, block_size) 输入序列
        y: (batch_size, block_size) 目标序列（x 向右偏移 1）
    """
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ==================== 训练循环 ====================

def train_gpt(
    config: GPTConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    idx_to_char: dict,
    # 训练超参数
    batch_size: int = 32,
    max_iters: int = 3000,
    eval_interval: int = 300,
    eval_iters: int = 50,
    learning_rate: float = 3e-4,
    device: str = 'cpu',
) -> GPT:
    """完整训练流程"""
    
    model = GPT(config).to(device)
    
    # 优化器（AdamW 是 LLM 标配）
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.95),  # GPT 常用 beta 值
        weight_decay=0.1,   # 权重衰减（正则化）
    )
    
    # 学习率调度：cosine warmup
    def get_lr(it: int) -> float:
        warmup_iters = 100
        lr_decay_iters = max_iters
        min_lr = learning_rate / 10
        
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        
        # Cosine 退火
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    @torch.no_grad()
    def estimate_loss() -> dict:
        """评估训练集和验证集的平均 loss"""
        out = {}
        model.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(data, config.block_size, batch_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out
    
    print("\n开始训练...")
    print(f"设备: {device} | 批大小: {batch_size} | 最大迭代: {max_iters}")
    print("-" * 60)
    
    t0 = time.time()
    
    for it in range(max_iters + 1):
        # 更新学习率
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 定期评估
        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"步骤 {it:4d}/{max_iters} | "
                  f"训练 loss: {losses['train']:.4f} | "
                  f"验证 loss: {losses['val']:.4f} | "
                  f"lr: {lr:.2e} | "
                  f"时间: {elapsed:.1f}s")
            
            # 生成样本（展示训练效果）
            if it > 0 and it % (eval_interval * 2) == 0:
                model.eval()
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=80, temperature=0.8, top_k=40)
                text = ''.join([idx_to_char[i] for i in generated[0].tolist()])
                print(f"  生成样本: {text[:100]!r}")
                model.train()
        
        if it == max_iters:
            break
        
        # 前向传播
        x, y = get_batch(train_data, config.block_size, batch_size, device)
        logits, loss = model(x, y)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)  # set_to_none 更省内存
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    print(f"\n训练完成！总时间: {time.time()-t0:.1f}s")
    return model


# ==================== 主程序 ====================

def main():
    # 配置
    config = GPTConfig(
        vocab_size=65,     # 莎士比亚字符级词表
        block_size=128,    # 上下文长度
        n_layer=4,         # 层数
        n_head=4,          # 注意力头数
        n_embd=128,        # 嵌入维度
        dropout=0.1,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 准备数据
    train_data, val_data, char_to_idx, idx_to_char, vocab_size = \
        prepare_shakespeare_data()
    config.vocab_size = vocab_size
    
    # 训练
    model = train_gpt(
        config=config,
        train_data=train_data,
        val_data=val_data,
        idx_to_char=idx_to_char,
        batch_size=32,
        max_iters=3000,
        eval_interval=300,
        learning_rate=3e-4,
        device=device,
    )
    
    # 推理：生成文本
    print("\n" + "="*60)
    print("生成文本示例")
    print("="*60)
    
    model.eval()
    
    # 方式 1：从空白开始生成
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    print("\n[ 温度=1.0，随机性高 ]")
    generated = model.generate(context, max_new_tokens=200, temperature=1.0, top_k=40)
    print(''.join([idx_to_char[i] for i in generated[0].tolist()]))
    
    print("\n[ 温度=0.5，更确定 ]")
    generated = model.generate(context, max_new_tokens=200, temperature=0.5, top_k=40)
    print(''.join([idx_to_char[i] for i in generated[0].tolist()]))
    
    # 方式 2：给定起始文本生成
    starter = "To be, or not to be"
    starter_tokens = torch.tensor(
        [[char_to_idx[c] for c in starter if c in char_to_idx]], 
        dtype=torch.long, device=device
    )
    print(f"\n[ 起始文本: {starter!r} ]")
    generated = model.generate(starter_tokens, max_new_tokens=200, temperature=0.8, top_k=40)
    print(''.join([idx_to_char[i] for i in generated[0].tolist()]))
    
    return model


# ==================== 单元测试 ====================

def quick_test():
    """快速测试模型的基本功能（无需真实数据）"""
    print("运行快速测试...")
    
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    
    # 测试前向传播
    B, T = 4, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(idx, targets)
    print(f"✅ 前向传播: logits={logits.shape}, loss={loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print(f"✅ 反向传播成功")
    
    # 测试生成
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=20, temperature=1.0)
    print(f"✅ 生成: {generated.shape} (生成了 {generated.size(1)-1} 个 token)")
    
    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数统计:")
    print(f"  Token Embedding: {config.vocab_size * config.n_embd:,}")
    print(f"  Position Embedding: {config.block_size * config.n_embd:,}")
    print(f"  Attention (per layer): {4 * config.n_embd**2:,}")
    print(f"  FFN (per layer): {8 * config.n_embd**2:,}")
    print(f"  总参数量: {total:,}")
    
    print("\n🎉 快速测试全部通过！")


if __name__ == "__main__":
    # 快速测试（始终运行）
    quick_test()
    
    # 完整训练（需要时间）
    # main()
```

---

## 四、关键设计决策解析

### 4.1 为什么用 Pre-LN 而不是 Post-LN？

| | Post-LN（原论文） | Pre-LN（现代） |
|--|--------|--------|
| 结构 | $x = \text{LN}(x + \text{sublayer}(x))$ | $x = x + \text{sublayer}(\text{LN}(x))$ |
| 稳定性 | 训练初期不稳定，需要 warmup | 训练更稳定 |
| 性能 | 理论上更好 | 实践中相当或更好 |

### 4.2 学习率 Warmup 的原因

$$\text{lr}(t) = \begin{cases} \frac{t}{\text{warmup}} \cdot \text{lr}_{\max} & t < \text{warmup} \\ \text{cosine decay} & t \geq \text{warmup} \end{cases}$$

训练初期参数随机，梯度方向不可靠，大学习率会导致发散。Warmup 让模型先在小 lr 下"预热"。

### 4.3 权重共享：lm_head 和 wte

$$\text{logits} = h \cdot W_{emb}^\top$$

语言模型头（预测下一个 token）和 embedding 层（将 token ID 映射到向量）共享权重：
- 减少参数量（省 vocab_size × n_embd 个参数）
- 直觉上合理：相同的向量空间既用于编码也用于解码

### 4.4 梯度裁剪

$$\text{if } \|\nabla\|_2 > \text{max\_norm}: \quad \nabla \leftarrow \frac{\text{max\_norm}}{\|\nabla\|_2} \nabla$$

防止梯度爆炸（特别是深层 Transformer 训练初期）。

---

## 五、训练曲线解读

| 阶段 | 现象 | 含义 |
|------|------|------|
| 初期（0-200 步） | Loss 快速下降 | 模型学习基本字符频率 |
| 中期（200-1000 步） | 稳定下降 | 学习短程模式（词语） |
| 后期（1000-3000 步） | 缓慢下降 | 学习长程结构（语法、句式） |
| 收敛后 | val loss > train loss | 正常过拟合，适当 Dropout |

**典型 Loss 值**：
- 随机猜测：$\ln(65) \approx 4.17$
- 训练 3000 步后：约 1.5-1.8
- 理论下限（完美记忆训练集）：接近 0

---

## 六、Week 2 总结

本周从 Attention 机制的直觉出发，层层深入：

| 篇章 | 核心概念 | 关键公式/代码 |
|------|---------|-------------|
| D1 | Q/K/V 隐喻，自注意力 | 图书馆检索类比 |
| D2 | Scaled Dot-Product 完整推导 | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| D3 | 多头注意力，GQA | Split → Attend → Merge |
| D4 | $O(n^2)$ 瓶颈，KV Cache | 复杂度分析 |
| D5 | Sparse/Linear/Flash 变体 | FlashAttention 分块算法 |
| D6 | 生产级代码实现 | 完整测试套件 |
| D7 | 小型 GPT 实战 | 训练字符级语言模型 |

**下一步**：Week 3 将进入位置编码（RoPE、ALiBi）、高效训练技巧（混合精度、梯度检查点）和现代 LLM 架构细节（LLaMA、Mistral 等）。

---

*参考实现：nanoGPT（Andrej Karpathy）；GPT-2 论文（Radford et al., 2019）；Transformer 原论文（Vaswani et al., 2017）*
