---
layout: default
title: "Week 6 综合实战：PyTorch MLP 全流程"
render_with_liquid: false
---

# Week 6 综合实战：PyTorch MLP 全流程

## 目标

用 PyTorch 从头训练一个完整的 MLP，覆盖数据加载、模型设计、训练循环、验证、保存加载全流程。

## 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===== 1. 数据准备 =====
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def to_tensor(*arrays):
    return [torch.FloatTensor(a) for a in arrays]

X_tr_t, X_val_t, X_te_t = to_tensor(X_train, X_val, X_test)
y_tr_t = torch.FloatTensor(y_train)
y_val_t = torch.FloatTensor(y_val)
y_te_t = torch.FloatTensor(y_test)

# DataLoader
train_dataset = TensorDataset(X_tr_t, y_tr_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# ===== 2. 模型定义 =====
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x).squeeze()

model = MLP(
    input_dim=30,
    hidden_dims=[128, 64, 32],
    dropout_rate=0.3
)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数: {total_params:,}")

# ===== 3. 训练配置 =====
criterion = nn.BCEWithLogitsLoss()  # 内置 Sigmoid，数值更稳定
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader), epochs=100
)

# ===== 4. 训练循环 =====
def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, y_b in loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * len(y_b)
        preds = (out.sigmoid() > 0.5)
        correct += (preds == y_b.bool()).sum().item()
        total += len(y_b)
    
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            out = model(X_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * len(y_b)
            preds = (out.sigmoid() > 0.5)
            correct += (preds == y_b.bool()).sum().item()
            total += len(y_b)
    return total_loss / total, correct / total

# ===== 5. 训练 + 早停 =====
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
patience, wait = 10, 0

for epoch in range(100):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(val_acc)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: "
              f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/tmp/best_mlp.pt')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ===== 6. 加载最优模型评估 =====
model.load_state_dict(torch.load('/tmp/best_mlp.pt'))
model.eval()
with torch.no_grad():
    logits = model(X_te_t)
    proba = logits.sigmoid()
    preds = (proba > 0.5).int()

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
print("\n===== 测试集评估 =====")
print(classification_report(y_test.astype(int), preds.numpy(),
                             target_names=['恶性','良性']))
print(f"AUC: {roc_auc_score(y_test, proba.numpy()):.4f}")

# ===== 7. 可视化 =====
epochs_run = len(history['train_loss'])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='训练 Loss')
axes[0].plot(history['val_loss'], label='验证 Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('损失曲线')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['train_acc'], label='训练准确率')
axes[1].plot(history['val_acc'], label='验证准确率')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('准确率曲线')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Week 6 · PyTorch MLP 全流程', fontsize=14, fontweight='bold')
plt.tight_layout()

print("""
===== 本周工程实践总结 =====
✅ DataLoader：批量加载、shuffle、pin_memory 加速
✅ BCEWithLogitsLoss：比 BCE+Sigmoid 数值更稳定（log-sum-exp trick）
✅ AdamW + OneCycleLR：现代深度学习标准配置
✅ 梯度裁剪：clip_grad_norm_ 防梯度爆炸
✅ 早停 + 保存最优模型：防过拟合的工程实践
✅ BatchNorm + Dropout：正则化双保险
""")
```
