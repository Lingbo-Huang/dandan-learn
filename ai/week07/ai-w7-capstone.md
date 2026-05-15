---
layout: default
title: "Week 7 综合实战：图像分类全流程"
render_with_liquid: false
---

# Week 7 综合实战：图像分类全流程

## 目标

用 PyTorch 训练 CIFAR-10 分类器，对比从零训练 vs 迁移学习，掌握图像分类工程全流程。

## 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ===== 1. 数据加载 CIFAR-10 =====
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

train_full = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10('./data', train=False, transform=test_transform)

# 分出验证集
n_train = int(0.9 * len(train_full))
train_set, val_set = random_split(train_full, [n_train, len(train_full)-n_train])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=256, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# ===== 2. 自定义 ResNet-18（CIFAR版）=====
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))

class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ===== 3. 训练函数 =====
def run_epoch(model, loader, optimizer, criterion, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    
    return total_loss / total, correct / total

def train_model(model, n_epochs=50, base_lr=0.1):
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr,
        steps_per_epoch=len(train_loader), epochs=n_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc, best_state = 0, None
    
    for epoch in range(n_epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, True)
        scheduler.step()
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, False)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} [{time.time()-t0:.1f}s]: "
                  f"Train={tr_acc:.4f} Val={val_acc:.4f} (Best={best_val_acc:.4f})")
    
    model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, test_loader, None, criterion, False)
    print(f"\n测试集准确率: {te_acc:.4f}")
    
    return history

# ===== 4. 实验对比 =====
print("=" * 50)
print("从零训练 CIFARResNet")
model_scratch = CIFARResNet(num_classes=10).to(device)
params_scratch = sum(p.numel() for p in model_scratch.parameters())
print(f"参数量: {params_scratch/1e6:.2f}M")
hist_scratch = train_model(model_scratch, n_epochs=30, base_lr=0.05)

# ===== 5. 误分类分析 =====
model_scratch.eval()
confused = {}  # {(真实类, 预测类): 次数}

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = model_scratch(X).argmax(1)
        wrong = preds != y
        for true_c, pred_c in zip(y[wrong].cpu().numpy(), preds[wrong].cpu().numpy()):
            k = (classes[true_c], classes[pred_c])
            confused[k] = confused.get(k, 0) + 1

print("\nTop10 误分类对:")
for (true_c, pred_c), cnt in sorted(confused.items(), key=lambda x: -x[1])[:10]:
    print(f"  {true_c} → {pred_c}: {cnt}次")

# ===== 6. 可视化 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(len(hist_scratch['train_acc']))

axes[0].plot(epochs, hist_scratch['train_loss'], label='训练 Loss')
axes[0].plot(epochs, hist_scratch['val_loss'], label='验证 Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss 曲线')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, [a*100 for a in hist_scratch['train_acc']], label='训练准确率%')
axes[1].plot(epochs, [a*100 for a in hist_scratch['val_acc']], label='验证准确率%')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('准确率曲线')
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Week 7 · CIFAR-10 分类（CIFARResNet）", fontsize=14)
plt.tight_layout()

print("""
===== Week 7 总结 =====
✅ 卷积：参数共享+局部连接，相比全连接参数少 1000x+
✅ ResNet：跳跃连接解决梯度消失，支持训练 1000+ 层
✅ 迁移学习：Feature Extraction（小数据）/ Fine-tuning（大数据）
✅ 数据增强：随机裁剪/翻转/颜色抖动，有效防止过拟合
✅ Label Smoothing：防止过度自信，提升泛化
✅ OneCycleLR + SGD+Nesterov：图像分类常用配置
""")
```
