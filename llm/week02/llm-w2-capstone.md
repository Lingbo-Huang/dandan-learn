# Week 2 综合项目：从零训练一个 Tiny Transformer

> 系列：大模型线 Week 2 | 文件：llm-w2-capstone.md

---

## 学习目标

- 将 Attention 代码集成到完整 Transformer 层
- 在玩具数据集（字符级语言模型）上训练模型
- 可视化训练过程中 Attention 权重的变化
- 验证因果掩码在自回归生成中的效果
- 复现 Andrej Karpathy 风格的 nanoGPT 简化版

---

## 项目结构

```
capstone/
├── model.py          # Transformer 模型（使用上文的 MHA 实现）
├── train.py          # 训练循环
├── generate.py       # 文本生成
├── visualize.py      # Attention 可视化
└── data/
    └── tiny_shakespeare.txt  # 数据（自动下载）
```

---

## 1. 模型定义（model.py）

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0"]
# ///
"""
Tiny Transformer (字符级语言模型)

参考: Andrej Karpathy nanoGPT
      https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================
# 配置
# ============================================================

@dataclass
class TinyGPTConfig:
    vocab_size: int = 65        # 字符数（莎士比亚数据集）
    block_size: int = 128       # 最大序列长度（上下文窗口）
    d_model: int = 128          # 模型维度
    num_heads: int = 4          # 注意力头数
    num_layers: int = 4         # Transformer 层数
    dropout: float = 0.1
    # 派生参数
    d_ff: int = 512             # FFN 中间层维度（通常 4×d_model）


# ============================================================
# Scaled Dot-Product Attention（内联）
# ============================================================

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights_dropped = self.dropout(weights)

        output = torch.matmul(weights_dropped, V)
        return output, weights


# ============================================================
# Multi-Head Causal Self-Attention（自回归版）
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    带因果掩码的 Multi-Head Self-Attention。
    每个位置只能看到当前及之前的位置（不看未来）。
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads

        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn = ScaledDotProductAttention(dropout=config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 注册因果掩码（不是参数，只是 buffer）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape  # batch, seq_len, d_model

        # 一次投影得到 Q, K, V
        qkv = self.W_qkv(x)  # (B, T, 3*d_model)
        Q, K, V = qkv.split(self.d_model, dim=2)  # each: (B, T, d_model)

        # 拆分多头: (B, T, d_model) → (B, h, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 因果掩码
        mask = self.causal_mask[:T, :T]  # (T, T)

        # Attention
        context, weights = self.attn(Q, K, V, mask=mask)

        # 合并头: (B, h, T, d_k) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.W_o(context))

        return output, weights


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block（Layer Norm 前置）"""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN: LayerNorm → Attention → Residual
        attn_out, weights = self.attn(self.ln1(x))
        x = x + attn_out
        # Pre-LN: LayerNorm → FFN → Residual
        x = x + self.ffn(self.ln2(x))
        return x, weights


# ============================================================
# Tiny GPT 模型
# ============================================================

class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享：token embedding 与 lm_head 共享权重（GPT-2 风格）
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()
        print(f"TinyGPT 参数量: {self.num_params():,}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,  # (B, T) 输入 token ids
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list]:
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, device=idx.device)
        x = self.dropout(
            self.token_embedding(idx) + self.position_embedding(pos)
        )

        all_attention_weights = []
        for block in self.blocks:
            x, weights = block(x)
            all_attention_weights.append(weights)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, all_attention_weights

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归文本生成"""
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)

            # 取最后一个 token 的 logits
            logits = logits[:, -1, :] / temperature

            # Top-k 采样（可选）
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx


if __name__ == "__main__":
    config = TinyGPTConfig()
    model = TinyGPT(config)

    # 前向传播测试
    x = torch.randint(0, config.vocab_size, (2, 64))
    y = torch.randint(0, config.vocab_size, (2, 64))

    logits, loss, attn_weights = model(x, y)
    print(f"Logits: {logits.shape}")           # (2, 64, 65)
    print(f"Loss: {loss.item():.4f}")
    print(f"Attention layers: {len(attn_weights)}")  # 4
    print(f"Attention shape: {attn_weights[0].shape}")  # (2, 4, 64, 64)

    # 生成测试
    prompt = torch.zeros((1, 1), dtype=torch.long)  # 空 prompt
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")  # (1, 21)
```

---

## 2. 训练脚本（train.py）

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0", "requests>=2.28"]
# ///
"""字符级语言模型训练脚本"""

import os
import requests
import time
import torch
import sys
sys.path.insert(0, ".")
from model import TinyGPT, TinyGPTConfig


# ============================================================
# 数据准备
# ============================================================

def get_data(data_dir: str = "data") -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """下载并预处理 Tiny Shakespeare 数据集"""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(path):
        print("下载 Tiny Shakespeare 数据集...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url, timeout=30)
        with open(path, "w") as f:
            f.write(r.text)
        print(f"✅ 下载完成，{len(r.text):,} 字符")

    with open(path) as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size}, 数据量: {len(text):,} 字符")

    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, stoi, itos


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样一个 batch"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 训练
# ============================================================

def train():
    # 超参数
    config = TinyGPTConfig(
        block_size=128,
        d_model=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    )
    batch_size = 32
    max_iters = 3000
    eval_interval = 300
    eval_iters = 50
    learning_rate = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用设备: {device}")

    # 数据
    train_data, val_data, stoi, itos = get_data()
    config.vocab_size = len(stoi)

    # 模型
    model = TinyGPT(config).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=learning_rate * 0.1
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split, data in [("train", train_data), ("val", val_data)]:
            L = []
            for _ in range(eval_iters):
                xb, yb = get_batch(data, config.block_size, batch_size, device)
                _, loss, _ = model(xb, yb)
                L.append(loss.item())
            losses[split] = sum(L) / len(L)
        model.train()
        return losses

    # 训练循环
    print("\n开始训练...")
    train_losses, val_losses = [], []
    t0 = time.time()

    for step in range(max_iters + 1):
        # 评估
        if step % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"Step {step:4d}/{max_iters} | "
                  f"train loss: {losses['train']:.4f} | "
                  f"val loss: {losses['val']:.4f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                  f"时间: {elapsed:.1f}s")
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        if step == max_iters:
            break

        # 训练步骤
        xb, yb = get_batch(train_data, config.block_size, batch_size, device)
        _, loss, _ = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # 保存模型
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, "tiny_gpt.pt")
    print("\n✅ 模型已保存到 tiny_gpt.pt")

    # 生成示例
    print("\n=== 生成示例 ===")
    model.eval()
    prompt_str = "ROMEO:"
    prompt_ids = torch.tensor([[stoi[c] for c in prompt_str]], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40)
    result = "".join([itos[i.item()] for i in generated[0]])
    print(result)

    return model, stoi, itos, train_losses, val_losses


if __name__ == "__main__":
    train()
```

---

## 3. 可视化（visualize.py）

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""训练曲线 + Attention 权重可视化"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, ".")
from model import TinyGPT, TinyGPTConfig


def plot_training_curves(train_losses: list, val_losses: list, eval_interval: int = 300):
    """绘制训练/验证损失曲线"""
    steps = [i * eval_interval for i in range(len(train_losses))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_losses, label="Train Loss", color="#e74c3c", linewidth=2)
    ax.plot(steps, val_losses, label="Val Loss", color="#3498db", linewidth=2)
    ax.set_xlabel("训练步数", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("TinyGPT 训练曲线", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.annotate(f"最终 Val Loss: {val_losses[-1]:.3f}",
                xy=(steps[-1], val_losses[-1]),
                xytext=(-100, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->'), fontsize=11)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 训练曲线已保存 training_curves.png")


def visualize_attention_on_text(
    model: TinyGPT,
    text: str,
    stoi: dict,
    itos: dict,
    layer: int = 0,
    head: int = 0,
    device: str = "cpu",
):
    """在给定文本上可视化 attention 权重"""
    model.eval()
    ids = [stoi[c] for c in text if c in stoi]
    idx = torch.tensor([ids], device=device)
    tokens = [itos[i] for i in ids]

    with torch.no_grad():
        _, _, attn_weights = model(idx)

    # attn_weights: list of (B, H, T, T)
    weights = attn_weights[layer][0, head].cpu().numpy()  # (T, T)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.7), max(6, len(tokens) * 0.5)))
    im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels([repr(t) for t in tokens], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([repr(t) for t in tokens], fontsize=9)
    ax.set_title(f"Layer {layer + 1}, Head {head + 1} Attention\n输入: {repr(text[:30])}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Key（被关注位置）", fontsize=11)
    ax.set_ylabel("Query（当前位置）", fontsize=11)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"attn_layer{layer}_head{head}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Attention 热力图已保存 attn_layer{layer}_head{head}.png")


if __name__ == "__main__":
    # 加载训练好的模型
    checkpoint = torch.load("tiny_gpt.pt", map_location="cpu")
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 绘制训练曲线
    plot_training_curves(
        checkpoint["train_losses"],
        checkpoint["val_losses"]
    )

    # 可视化 Attention
    sample_text = "To be, or not to be, that is the question"
    for layer in range(min(2, config.num_layers)):
        for head in range(min(2, config.num_heads)):
            visualize_attention_on_text(
                model, sample_text, stoi, itos,
                layer=layer, head=head
            )
```

---

## 4. 运行完整项目

```bash
# 安装依赖
cd capstone
uv init
uv add torch requests matplotlib numpy

# 运行训练（约 5-10 分钟，CPU）
uv run python train.py

# 可视化结果
uv run python visualize.py
```

---

## 5. 预期结果

| 指标 | 预期值 |
|------|--------|
| 初始 Val Loss | ~4.2（随机猜测 ~4.17 = ln(65)） |
| 训练后 Val Loss | ~1.5 - 1.8 |
| 生成质量 | 莎士比亚风格的英文文本，语法基本正确 |
| 训练时间 | CPU ~10min，GPU ~2min |

### 预期生成示例（非固定）

```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon...
```

---

## 6. 探索与作业

1. **修改超参数**：
   - 增加 `num_heads` 到 8，观察训练曲线变化
   - 将 `block_size` 从 128 增加到 256，观察内存变化

2. **可视化实验**：
   - 在不同训练阶段（step 0, 500, 3000）保存 Attention 权重
   - 观察训练过程中 Attention 权重分布如何变化

3. **消融实验**：
   - 去掉因果掩码，观察生成质量变化（理解为什么需要 Masking）
   - 去掉缩放因子 $\sqrt{d_k}$，观察训练曲线（理解为什么需要 Scaling）

4. **实现 GQA 版本**：
   - 将 `CausalSelfAttention` 替换为 `GroupedQueryAttention`
   - 比较参数量和训练速度

---

## 小结

| 模块 | 对应知识点 |
|------|-----------|
| `CausalSelfAttention` | Scaled Dot-Product + 因果掩码 + Multi-Head |
| `TransformerBlock` | 残差连接 + Pre-LN + FFN |
| `TinyGPT` | 位置编码 + 权重共享 + 自回归生成 |
| `train.py` | AdamW + 梯度裁剪 + 余弦调度 |
| `visualize.py` | Attention 热力图 + 训练曲线 |

**本周结束检验**：

✅ 能从零写出 `ScaledDotProductAttention`（无需看笔记）
✅ 能解释 $\sqrt{d_k}$ 的来源
✅ 能解释 GQA 相比 MHA 节省 KV Cache 的原因
✅ TinyGPT 能在莎士比亚数据上收敛（val loss < 1.8）
✅ Attention 热力图有可观察到的结构（对角线、局部模式等）

**下一步（Week 3 预告）**：位置编码深度分析——从绝对位置编码到 RoPE，以及 Transformer 如何感知序列顺序。
