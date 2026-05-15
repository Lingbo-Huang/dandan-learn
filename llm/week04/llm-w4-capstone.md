---
layout: default
title: "D6 · Capstone：手写 mini-GPT 预训练"
render_with_liquid: false
---

# D6 · Capstone：手写 mini-GPT 预训练

> **目标**：从零实现一个完整的 mini-GPT，包含 BPE Tokenizer 训练、数据加载、RoPE 位置编码，在莎士比亚文本上预训练，验证 loss 下降和文本生成。

---

## 项目结构

```
mini_gpt/
├── tokenizer.py     # BPE Tokenizer
├── dataset.py       # 数据加载
├── model.py         # GPT 模型（带 RoPE）
├── train.py         # 训练主循环
└── generate.py      # 文本生成
```

---

## 一、数据准备

```python
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request

def download_shakespeare():
    """下载莎士比亚文本"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")
    with open("shakespeare.txt", encoding='utf-8') as f:
        return f.read()

class TextDataset(Dataset):
    """GPT 预训练数据集"""
    
    def __init__(self, text: str, tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        # Tokenize 全文
        self.tokens = tokenizer.encode(text)
        print(f"文本长度: {len(text):,} 字符")
        print(f"Token 数量: {len(self.tokens):,}")
        print(f"压缩比: {len(text)/len(self.tokens):.2f} chars/token")
    
    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx: int) -> dict:
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': labels}
```

---

## 二、完整模型（带 RoPE）

```python
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        freqs = precompute_freqs_cis(self.d_head, max_seq_len)
        self.register_buffer('freqs_cis', freqs)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)  # 现代 LLM 用 RMSNorm
        self.attn = Attention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.SiLU(),  # SwiGLU 的简化版
            nn.Linear(4 * d_model, d_model, bias=False),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, max_seq_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, max_seq_len) for _ in range(n_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight  # 权重绑定
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"参数量: {n_params/1e6:.1f}M")
    
    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -256:])  # 截断到 max_seq_len
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
```

---

## 三、训练主循环

```python
# train.py
import torch
from torch.optim import AdamW
import time

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 数据
    text = download_shakespeare()
    
    # 使用字符级 tokenizer（简化，实际用 BPE）
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    
    class CharTokenizer:
        def encode(self, s): return [stoi[c] for c in s]
        def decode(self, ids): return ''.join(itos[i] for i in ids)
    
    tokenizer = CharTokenizer()
    dataset = TextDataset(text, tokenizer, seq_len=256)
    
    # 训练/验证集分割
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. 模型
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=256
    ).to(device)
    
    # 3. 优化器
    optimizer = AdamW(
        model.parameters(), 
        lr=3e-4, 
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 4. 训练循环
    num_epochs = 10
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Warmup
            if step < 1000:
                lr = 3e-4 * step / 1000
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            
            _, loss = model(input_ids, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            
            if step % 200 == 0:
                ppl = torch.exp(loss).item()
                print(f"Step {step:5d} | loss {loss.item():.4f} | ppl {ppl:.1f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}")
        
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                _, loss = model(input_ids, labels)
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        elapsed = time.time() - t0
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.1f} | "
              f"时间 {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mini_gpt_best.pt')
            print(f"  ✅ 保存最佳模型 (val_loss={val_loss:.4f})")
        
        # 生成示例
        prompt = "ROMEO:"
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        generated = model.generate(prompt_ids, max_new_tokens=100)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"\n生成示例:\n{generated_text}\n{'='*50}")
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")

if __name__ == '__main__':
    train()
```

---

## 四、预期训练曲线

```
Epoch 1:  val_loss ≈ 2.8,  val_ppl ≈ 16  （刚开始学，生成基本是乱的）
Epoch 3:  val_loss ≈ 1.8,  val_ppl ≈ 6   （开始学到英文单词结构）
Epoch 5:  val_loss ≈ 1.5,  val_ppl ≈ 4.5 （学会了基本对话格式）
Epoch 10: val_loss ≈ 1.2,  val_ppl ≈ 3.3 （能生成莎翁风格文本）
```

---

## 五、本周回顾

| 主题 | 核心知识点 |
|------|-----------|
| 数据处理 | 去重、质量过滤、配比（代码占比关键）|
| Tokenizer | BPE 训练过程、字节级 BPE、SentencePiece |
| GPT 目标 | CLM 损失、因果 Mask、权重绑定 |
| Scaling Law | Chinchilla 最优、6ND FLOPs、推理效率 trade-off |
| 位置编码 | RoPE（旋转复数）、ALiBi（线性偏置）|

---

## 六、面试综合题

**Q: 从头训练一个 7B 模型，你会怎么规划？**

A:
1. **数据**：收集 140B-2T tokens，按代码15%、书籍20%、网页50%、其他配比；去重后质量过滤
2. **Tokenizer**：训练 32K-128K BPE，中文/代码覆盖率高
3. **架构**：7B 参数，d_model=4096，32层，32头，GQA（降低KV Cache），RoPE，RMSNorm，SwiGLU
4. **训练**：AdamW（β₁=0.9, β₂=0.95, wd=0.1），cosine LR，warmup，梯度裁剪
5. **基础设施**：64-128张A100，Megatron-LM 流水线并行，Flash Attention 2
6. **评估**：每1B tokens 评估 MMLU/HellaSwag/HumanEval
