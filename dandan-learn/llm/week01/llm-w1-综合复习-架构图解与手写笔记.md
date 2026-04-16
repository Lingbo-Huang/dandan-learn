# Day 7：动手实现最小 Transformer + 本周总结

---

## 学习目标

- 将本周所学整合，从零搭建一个可运行的最小 Transformer
- 在字符级语言模型任务上验证模型能学习
- 梳理本周知识体系，找出还不清楚的盲点
- 阅读原论文关键章节

---

## 一、任务设定：字符级语言模型

使用 **Decoder-only** 架构（GPT 风格），在一段简单文本上训练字符级语言模型，验证 Transformer 能学到规律。

**任务**：给定前 N 个字符，预测下一个字符。
**数据集**：一小段莎士比亚文本（或任意中文文本）。

---

## 二、完整实现代码

```python
# minimal_transformer.py
# 字符级 GPT：Decoder-only Transformer 最小实现
# 参考：Andrej Karpathy "Let's build GPT"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 16
BLOCK_SIZE = 64       # 上下文长度（每次处理的字符数）
MAX_ITERS = 3000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 128          # d_model
N_HEAD = 4            # 注意力头数
N_LAYER = 4           # Transformer 层数
DROPOUT = 0.1

print(f"使用设备: {DEVICE}")

# ============================================================
# 数据准备
# ============================================================
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause.
""" * 20  # 重复以增加数据量

chars = sorted(set(text))
vocab_size = len(chars)
print(f"词表大小: {vocab_size}")
print(f"文本长度: {len(text)}")

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ============================================================
# 模型组件
# ============================================================

class CausalSelfAttention(nn.Module):
    """单个多头因果自注意力"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.d_k = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Q/K/V 合并投影
        self.c_proj = nn.Linear(n_embd, n_embd)       # 输出投影
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 因果 mask（注册为 buffer，不是参数）
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
    
    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd
        
        # 计算 Q/K/V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 分头
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, h, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 拼接多头
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """FFN"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block（Pre-LN）"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差 + Pre-LN
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """最小 GPT"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, n_embd)  # 可学习位置编码
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        print(f"参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(T, device=DEVICE)
        pos_emb = self.pos_emb(pos)    # (T, n_embd)
        
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]         # 截断到 block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]               # 只取最后一个 token 的预测
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ============================================================
# 训练
# ============================================================

model = MiniGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        loss_sum = 0
        for _ in range(50):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            loss_sum += loss.item()
        losses[split] = loss_sum / 50
    model.train()
    return losses

print("\n开始训练...")
for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================================================
# 生成文本
# ============================================================
print("\n生成示例（200 个字符）:")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))
```

---

## 三、本周知识梳理

### 3.1 Transformer 一图流

```
输入 tokens
    ↓ Token Embedding + Positional Encoding
    ↓
[Encoder Block] × N                [Decoder Block] × N
  ├─ Multi-Head Self-Attention        ├─ Masked Multi-Head Self-Attn
  ├─ Add & Norm                       ├─ Add & Norm
  ├─ Feed-Forward Network             ├─ Cross-Attention (↔ Encoder)
  └─ Add & Norm                       ├─ Add & Norm
                                      ├─ Feed-Forward Network
                                      └─ Add & Norm
    ↓                                       ↓
Encoder Output                   Linear + Softmax
                                       ↓
                                  输出 token 概率
```

### 3.2 本周关键公式

| 公式 | 说明 |
|------|------|
| $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ | Scaled Dot-Product Attention |
| $\text{MultiHead} = \text{Concat}(\text{head}_i)W^O$ | 多头注意力 |
| $\text{FFN}(x) = \text{GELU}(xW_1+b_1)W_2+b_2$ | 前馈网络 |
| $PE_{pos,2i} = \sin(pos/10000^{2i/d})$ | 正弦位置编码 |
| $x' = x + \text{Sublayer}(\text{LN}(x))$ | Pre-LN 残差 |

### 3.3 三种架构记忆口诀

- **BERT（Encoder）** = 双向理解，完形填空
- **GPT（Decoder）** = 单向生成，自回归预测
- **T5（Enc-Dec）** = 序列到序列，完整映射

---

## 四、原论文精读任务

今日阅读《Attention Is All You Need》以下章节：

- **Section 3.1** Model Architecture（整体结构）
- **Section 3.2** Attention（Attention 机制）
- **Section 3.3** Position-wise Feed-Forward Networks（FFN）
- **Section 3.4** Embeddings and Softmax
- **Section 3.5** Positional Encoding

阅读时记录：
1. 与你本周理解不同的地方
2. 论文中提到但本课程未覆盖的内容
3. 你还不理解的地方（留到 Week 2 解决）

---

## 五、动手练习

### 练习 1：运行代码并记录 Loss 曲线

运行上方完整代码，记录每 500 步的 train/val loss，判断：
- 模型是否在学习（loss 是否下降）？
- 是否过拟合（val loss 是否远高于 train loss）？

### 练习 2：超参数实验

修改以下超参数，观察影响：
- `N_LAYER = 2` vs `N_LAYER = 6`
- `N_HEAD = 1` vs `N_HEAD = 8`
- `BLOCK_SIZE = 32` vs `BLOCK_SIZE = 128`

### 练习 3：本周自测

不看笔记，回答以下问题：

1. Transformer 中 Attention 的计算公式是什么？为什么要除以 √d_k？
2. Multi-Head Attention 相比 Single-Head 有什么优势？
3. Encoder-only 和 Decoder-only 的 Attention Mask 有什么区别？
4. Pre-LayerNorm 和 Post-LayerNorm 的区别是什么？哪个更常用？
5. 为什么现代大模型（GPT、LLaMA）都选择 Decoder-only 架构？

---

## 六、Week 1 总结与 Week 2 预告

### 本周收获

| Day | 主题 | 核心收获 |
|-----|------|---------|
| D1 | 环境 + 路线 | uv 环境 + 大模型学习地图 |
| D2 | 整体架构 | Encoder/Decoder 结构 + 数据流 |
| D3 | Attention | Q/K/V + Scaled Dot-Product |
| D4 | Multi-Head + PE | 多头并行 + 位置编码 |
| D5 | FFN + 归一化 | FFN 知识存储 + Pre-LN |
| D6 | 架构变体 | BERT/GPT/T5 对比 |
| D7 | 动手实现 | 从零写出可运行 Transformer |

### Week 2 预告（建议）

- **主题**：深入预训练与 Tokenization
- D1-D2：BPE / WordPiece / SentencePiece Tokenizer
- D3-D4：BERT 预训练细节（MLM + NSP）
- D5-D6：GPT 系列进化史（GPT-1 → GPT-3）
- D7：HuggingFace Trainer 微调实战

> 🎉 **恭喜完成 Week 1！** 你已经掌握了 Transformer 的核心架构。Week 2 开始深入预训练世界。
