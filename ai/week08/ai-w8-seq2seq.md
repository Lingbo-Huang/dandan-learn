---
layout: default
title: "Seq2Seq 模型"
render_with_liquid: false
---

# Seq2Seq 模型

## 直觉：编码 + 解码

翻译"猫喝牛奶" → "Cat drinks milk"。

无法逐字翻译（词序不同），需要先**理解整句**，再**生成目标语**。

Seq2Seq = Encoder（压缩源序列 → 上下文向量）+ Decoder（从上下文解码目标序列）。

## 架构

### Encoder

$$h_t^{\text{enc}} = \text{LSTM}(x_t, h_{t-1}^{\text{enc}})$$

上下文向量（Context Vector）：$c = h_T^{\text{enc}}$（最后一步隐藏状态）

### Decoder

条件语言模型：$P(y_1, \ldots, y_M \mid x_1, \ldots, x_N)$

$$h_t^{\text{dec}} = \text{LSTM}([y_{t-1}, c], h_{t-1}^{\text{dec}})$$

$$P(y_t \mid y_{<t}, c) = \text{softmax}(W h_t^{\text{dec}})$$

### 训练：Teacher Forcing

训练时 Decoder 输入用**真实的上一步目标词**（而非自身预测），加速训练，避免错误累积。

推理时使用自回归生成（每步用自己的预测）。

## 信息瓶颈问题

所有信息被压缩到单一向量 $c$，长序列信息损失严重。

**Attention 是解决方案**（下一节详述）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ===== 简单 Seq2Seq（数字序列反转任务）=====
# 任务：给定序列 [1,3,5,2,4]，输出 [4,2,5,3,1]

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (h, c) = self.rnn(embedded)
        return outputs, h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h, c):
        # x: (batch,) 单个时步
        embedded = self.dropout(self.embedding(x.unsqueeze(1)))  # (B, 1, E)
        output, (h, c) = self.rnn(embedded, (h, c))
        logits = self.fc(output.squeeze(1))  # (B, vocab_size)
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        _, h, c = self.encoder(src)
        
        # Decoder 第一步输入：<SOS> token（这里用 0 代替）
        dec_input = trg[:, 0]
        
        for t in range(1, trg_len):
            logits, h, c = self.decoder(dec_input, h, c)
            outputs[:, t, :] = logits
            
            # Teacher forcing
            use_tf = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_tf else logits.argmax(1)
        
        return outputs

# ===== 数据生成（序列反转）=====
def generate_data(n_samples=2000, max_len=10, vocab_size=15):
    data = []
    SOS, EOS = 1, 2
    for _ in range(n_samples):
        seq_len = random.randint(3, max_len)
        seq = [random.randint(3, vocab_size-1) for _ in range(seq_len)]
        src = [SOS] + seq + [EOS]
        trg = [SOS] + seq[::-1] + [EOS]
        data.append((src, trg))
    return data

VOCAB_SIZE = 15
SOS, EOS, PAD = 1, 2, 0

data = generate_data(2000, max_len=8, vocab_size=VOCAB_SIZE)

def pad_sequence(seqs, pad_val=0):
    max_len = max(len(s) for s in seqs)
    return [[s + [pad_val] * (max_len - len(s)) for s in seqs]]

# 简单批处理
def make_batch(batch_data):
    srcs = [d[0] for d in batch_data]
    trgs = [d[1] for d in batch_data]
    max_src = max(len(s) for s in srcs)
    max_trg = max(len(t) for t in trgs)
    srcs_pad = [s + [PAD]*(max_src-len(s)) for s in srcs]
    trgs_pad = [t + [PAD]*(max_trg-len(t)) for t in trgs]
    return (torch.LongTensor(srcs_pad), torch.LongTensor(trgs_pad))

device = torch.device('cpu')
random.seed(42)
torch.manual_seed(42)

train_data = data[:1600]
test_data = data[1600:]
batch_size = 64

encoder = Encoder(VOCAB_SIZE, embed_size=32, hidden_size=64)
decoder = Decoder(VOCAB_SIZE, embed_size=32, hidden_size=64)
model = Seq2Seq(encoder, decoder, device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

losses = []
for epoch in range(20):
    model.train()
    random.shuffle(train_data)
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        src, trg = make_batch(batch)
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        # output: (B, T, V), trg: (B, T)
        output_flat = output[:, 1:, :].reshape(-1, VOCAB_SIZE)
        trg_flat = trg[:, 1:].reshape(-1)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    
    if epoch % 5 == 0:
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# 推理示例
model.eval()
sample_src, sample_trg = make_batch(test_data[:3])
with torch.no_grad():
    output = model(sample_src, sample_trg, teacher_forcing_ratio=0.0)
    preds = output.argmax(-1)

for i in range(min(3, len(test_data))):
    src_tokens = [t for t in sample_src[i].tolist() if t > 0]
    trg_tokens = [t for t in sample_trg[i].tolist() if t > 0]
    pred_tokens = [t for t in preds[i].tolist() if t > 0]
    print(f"Input: {src_tokens}")
    print(f"Target: {trg_tokens}")
    print(f"Pred:   {pred_tokens}")
    print()
```

## Beam Search（推理质量提升）

```python
def beam_search(model, src, beam_size=3, max_len=20):
    """Beam Search 解码"""
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0)  # (1, T)
        enc_out, h, c = model.encoder(src)
        
        # 初始化 beam：(累计log概率, 序列, (h, c))
        beams = [(0.0, [SOS], h, c)]
        
        for _ in range(max_len):
            all_candidates = []
            for score, seq, h, c in beams:
                if seq[-1] == EOS:
                    all_candidates.append((score, seq, h, c))
                    continue
                
                inp = torch.LongTensor([seq[-1]])
                logits, new_h, new_c = model.decoder(inp, h, c)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                
                top_scores, top_indices = log_probs.topk(beam_size)
                for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
                    all_candidates.append((score + s, seq + [idx], new_h, new_c))
            
            # 保留 top beam_size
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]
        
        best = beams[0]
        return best[1]  # 最高分序列

# 推理
sample_src_single = sample_src[0]
result = beam_search(model, sample_src_single, beam_size=3)
print("Beam Search 结果:", result)
```

## 面试要点

**Q: Teacher Forcing 的优缺点？**

A: 优点：加速收敛（不受错误积累影响）；缺点：训练-推理不一致（Exposure Bias），训练时总看真实词，推理时用自己预测，分布偏移可能导致错误级联。解决：Scheduled Sampling（逐步降低 teacher forcing 比例）。

**Q: Beam Search 为什么比 Greedy 好？**

A: Greedy 每步取最高概率词，可能陷入次优序列（局部最优）。Beam Search 维护 $k$ 个候选序列，遍历更大的搜索空间，通常得到更高质量结果。但 $k$ 过大计算代价高，且有时更长的生成过于保守。
