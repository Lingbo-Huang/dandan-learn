---
layout: default
title: "Week 8 综合实战：LSTM 情感分析与文本生成"
render_with_liquid: false
---

# Week 8 综合实战：LSTM 情感分析与文本生成

## 任务1：双向 LSTM 情感分析

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import re

# ===== 简单分词与词汇表 =====
class SimpleTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        
        for word, count in counter.most_common(self.max_vocab - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词汇表大小: {len(self.word2idx)}")
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text, max_len=100):
        tokens = self._tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        return ids

# 模拟训练数据（实际用 IMDB/SST-2）
positive_templates = [
    "this movie is absolutely fantastic and wonderful",
    "great performance by all the actors in this film",
    "i loved every minute of this brilliant masterpiece",
    "amazing storyline with incredible character development",
    "highly recommend this excellent and entertaining movie",
]
negative_templates = [
    "this film is terrible and extremely boring waste of time",
    "horrible acting and very poor script writing",
    "one of the worst movies i have ever seen in my life",
    "complete disappointment and a total disaster of a film",
    "avoid this awful and dreadful movie at all costs",
]

# 数据增强：随机组合
np.random.seed(42)
texts, labels = [], []
for _ in range(2000):
    if np.random.rand() > 0.5:
        texts.append(np.random.choice(positive_templates) + " " + 
                     np.random.choice(positive_templates[:3]))
        labels.append(1)
    else:
        texts.append(np.random.choice(negative_templates) + " " +
                     np.random.choice(negative_templates[:3]))
        labels.append(0)

tokenizer = SimpleTokenizer(max_vocab=500)
tokenizer.build_vocab(texts)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=50):
        self.data = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text, max_len)
            self.data.append((torch.LongTensor(ids), label))
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    lengths = []
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
        lengths.append(len(t))
    return padded, torch.FloatTensor(labels), torch.LongTensor(lengths)

# 划分数据集
n_train = int(0.8 * len(texts))
train_ds = SentimentDataset(texts[:n_train], labels[:n_train], tokenizer)
test_ds = SentimentDataset(texts[n_train:], labels[n_train:], tokenizer)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

# ===== 模型定义 =====
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers,
                             batch_first=True, bidirectional=True, dropout=dropout)
        
        # Self-Attention 池化
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        
        if lengths is not None:
            # Pack padded sequence（提升效率和准确性）
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(emb)
        
        # Attention 池化
        attn_scores = self.attention(out)  # (B, T, 1)
        
        # Mask padding
        if lengths is not None:
            mask = torch.arange(out.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores[mask.unsqueeze(-1)] = float('-inf')
        
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (out * attn_weights).sum(dim=1)
        
        return self.classifier(context).squeeze(), attn_weights.squeeze(-1)

device = torch.device('cpu')
vocab_size = len(tokenizer.word2idx)
model = BiLSTMSentiment(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=20
)

history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(20):
    model.train()
    tr_loss, tr_correct, tr_total = 0, 0, 0
    
    for X_b, y_b, lens in train_loader:
        optimizer.zero_grad()
        out, _ = model(X_b, lens)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        tr_loss += loss.item() * len(y_b)
        tr_correct += ((out.sigmoid() > 0.5) == y_b.bool()).sum().item()
        tr_total += len(y_b)
    
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X_b, y_b, lens in test_loader:
            out, _ = model(X_b, lens)
            val_correct += ((out.sigmoid() > 0.5) == y_b.bool()).sum().item()
            val_total += len(y_b)
    
    history['train_loss'].append(tr_loss / tr_total)
    history['train_acc'].append(tr_correct / tr_total)
    history['val_acc'].append(val_correct / val_total)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}: Train={history['train_acc'][-1]:.4f} "
              f"Val={history['val_acc'][-1]:.4f}")

# ===== 任务2：字符级文本生成 =====
print("\n===== 字符级文本生成 =====")

CORPUS = """
machine learning is a fascinating field of artificial intelligence
deep learning uses neural networks with many layers
attention mechanism is the key innovation behind transformers
language models can generate coherent and meaningful text
natural language processing enables computers to understand human language
the transformer architecture revolutionized natural language processing
"""

chars = sorted(set(CORPUS))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
n_chars = len(chars)

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embed_size=32, hidden_size=128, n_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_chars, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers,
                             batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, n_chars)
    
    def forward(self, x, h=None):
        emb = self.embedding(x)
        if h is None:
            out, (h, c) = self.lstm(emb)
        else:
            out, (h, c) = self.lstm(emb, h)
        return self.fc(out), (h, c)
    
    def generate(self, seed_text, max_len=100, temperature=0.8):
        self.eval()
        chars_gen = list(seed_text)
        x = torch.LongTensor([[c2i.get(c, 0) for c in seed_text]])
        
        with torch.no_grad():
            logits, hidden = self(x)
            
            for _ in range(max_len):
                # 温度采样
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = i2c[next_char_idx]
                chars_gen.append(next_char)
                
                x = torch.LongTensor([[next_char_idx]])
                logits, hidden = self(x, hidden)
        
        return ''.join(chars_gen)

# 准备序列数据
seq_len = 40
corpus_idx = [c2i[c] for c in CORPUS]
sequences = []
for i in range(0, len(corpus_idx) - seq_len - 1, 3):
    sequences.append((corpus_idx[i:i+seq_len], corpus_idx[i+1:i+seq_len+1]))

char_model = CharLSTM(n_chars)
char_optimizer = torch.optim.Adam(char_model.parameters(), lr=1e-3)
char_criterion = nn.CrossEntropyLoss()

gen_losses = []
for epoch in range(50):
    char_model.train()
    np.random.shuffle(sequences)
    epoch_loss = 0
    
    for src, trg in sequences:
        x = torch.LongTensor([src])
        y = torch.LongTensor(trg)
        
        char_optimizer.zero_grad()
        logits, _ = char_model(x)
        loss = char_criterion(logits.squeeze(0), y)
        loss.backward()
        nn.utils.clip_grad_norm_(char_model.parameters(), 5.0)
        char_optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:
        gen_losses.append(epoch_loss / len(sequences))
        print(f"Epoch {epoch}: Loss={gen_losses[-1]:.4f}")
        # 生成示例
        generated = char_model.generate("machine learning", max_len=80, temperature=0.7)
        print(f"  生成: {generated}")

# ===== 可视化 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 情感分析：训练曲线
axes[0].plot(history['train_acc'], label='训练')
axes[0].plot(history['val_acc'], label='验证')
axes[0].set_title('双向LSTM情感分析准确率')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# 文本生成：Loss曲线
axes[1].plot(range(0, 50, 10), gen_losses, 'o-')
axes[1].set_title('字符级LSTM生成 Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True)

# 注意力权重示例
model.eval()
sample_text = "this movie is absolutely fantastic and wonderful"
sample_ids = tokenizer.encode(sample_text)
sample_tensor = torch.LongTensor([sample_ids])
sample_len = torch.LongTensor([len(sample_ids)])

with torch.no_grad():
    _, attn_w = model(sample_tensor, sample_len)

words = sample_text.split()[:len(sample_ids)]
axes[2].bar(range(len(words)), attn_w[0, :len(words)].numpy())
axes[2].set_xticks(range(len(words)))
axes[2].set_xticklabels(words, rotation=45, ha='right', fontsize=8)
axes[2].set_title('情感分析注意力权重')
axes[2].set_ylabel('权重')
axes[2].grid(True)

plt.suptitle("Week 8 · LSTM 情感分析 + 文本生成", fontsize=14, fontweight='bold')
plt.tight_layout()

print("""
===== Week 8 总结 =====
✅ RNN：循环结构处理序列，但梯度消失限制长程依赖
✅ LSTM：四门控制记忆，细胞状态提供梯度高速公路
✅ GRU：两门简化版，参数少速度快
✅ Seq2Seq：Encoder-Decoder，Teacher Forcing 加速训练
✅ Attention：动态上下文，任意位置直接交互
✅ Self-Attention：Transformer 的核心，QKV 框架
✅ 下一步：Transformer = Self-Attention + FFN + 位置编码 → GPT/BERT
""")
```
