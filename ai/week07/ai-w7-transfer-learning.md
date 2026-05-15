---
layout: default
title: "迁移学习"
render_with_liquid: false
---

# 迁移学习

## 核心思想

在 ImageNet 上训练的网络，前几层学到了通用特征（边缘、纹理、形状），深层学到了语义特征。这些特征对其他视觉任务也有用。

与其从头训练（需要百万级数据），不如复用这些知识。

## 迁移策略

| 场景 | 目标数据集大小 | 与源域相似度 | 策略 |
|------|-------------|------------|------|
| A | 小 | 相似 | 只训练分类头（Feature Extraction） |
| B | 小 | 差异大 | 微调高层 + 训练分类头 |
| C | 大 | 相似 | 微调全部层（Fine-tuning） |
| D | 大 | 差异大 | 从头训练（或Fine-tune全部层） |

## PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ===== 数据预处理（针对 ImageNet 预训练模型的标准化）=====
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# ===== 策略1：Feature Extraction（冻结预训练权重）=====
def feature_extraction(num_classes=10):
    model = models.resnet50(pretrained=True)
    
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换分类头（自动可训练）
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    
    # 只有新头部需要训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Feature Extraction 模式: 可训练参数 {trainable:,} / 总参数 {total:,}")
    
    return model

# ===== 策略2：Fine-tuning（微调全部或部分层）=====
def fine_tuning(num_classes=10, freeze_layers=None):
    model = models.resnet50(pretrained=True)
    
    if freeze_layers:
        # 冻结指定层（如只训练 layer3 及之后）
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
    
    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# ===== 差异化学习率（预训练层小lr，新层大lr）=====
def get_optimizer(model, base_lr=1e-4):
    # 分层学习率
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': base_lr},       # 预训练层：小学习率
        {'params': head_params, 'lr': base_lr * 10},     # 新头部：大学习率
    ], weight_decay=1e-4)
    
    return optimizer

# ===== 完整训练流程（模拟）=====
def train_transfer(model, num_epochs=10, num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = get_optimizer(model, base_lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 模拟数据（实际应换成真实数据集）
    X = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, num_classes, (100,))
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = 80
    train_ds, val_ds = random_split(dataset, [train_size, 20])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
        
        scheduler.step()
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(correct / len(val_ds))
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss={history['train_loss'][-1]:.3f}, "
                  f"Val Acc={history['val_acc'][-1]:.3f}")
    
    return history

# 演示（小规模，不加载真实预训练权重）
print("策略1：Feature Extraction")
model_fe = fine_tuning(num_classes=5, freeze_layers=['layer1', 'layer2', 'layer3'])
print(f"可训练参数: {sum(p.numel() for p in model_fe.parameters() if p.requires_grad):,}")

# ===== 特征可视化（t-SNE）=====
import torch
from torchvision import models

def extract_features(model, loader, device='cpu'):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            feat = model(X_b)
            features.append(feat.cpu())
            labels.append(y_b)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()

# 演示特征提取
backbone = models.resnet18(pretrained=False)
backbone.fc = nn.Identity()  # 去掉分类头，输出 512 维特征

X_demo = torch.randn(50, 3, 32, 32)
y_demo = torch.randint(0, 5, (50,))
ds_demo = torch.utils.data.TensorDataset(
    torch.nn.functional.interpolate(X_demo, size=224), y_demo
)
loader_demo = DataLoader(ds_demo, batch_size=10)

feats, labs = extract_features(backbone, loader_demo)
print(f"\n特征形状: {feats.shape}")

# t-SNE 可视化
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
feats_2d = tsne.fit_transform(feats)

plt.figure(figsize=(7, 5))
for c in range(5):
    mask = labs == c
    plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                label=f'Class {c}', s=60, alpha=0.8)
plt.legend()
plt.title("预训练 ResNet-18 特征的 t-SNE 可视化")
plt.tight_layout()
```

## 迁移学习实践技巧

```python
# ===== 技巧1：渐进式解冻（Progressive Unfreezing）=====
# 先训练 head → 解冻最后 block → ... → 解冻全部
def progressive_unfreeze(model, stage):
    """
    stage 0: 只训练 fc
    stage 1: 解冻 layer4
    stage 2: 解冻 layer3+4
    stage 3: 解冻全部
    """
    layers_to_unfreeze = [[], ['layer4'], ['layer3','layer4'],
                           ['layer1','layer2','layer3','layer4']]
    
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if 'fc' in name or any(l in name for l in layers_to_unfreeze[stage]):
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage {stage}: {trainable:,} 个可训练参数")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 10)
for s in range(4):
    progressive_unfreeze(model, s)
```

## 面试要点

**Q: 迁移学习时为什么要对预训练层用更小的学习率？**

A: 预训练权重已经是很好的特征提取器，大学习率会破坏这些已学好的特征（catastrophic forgetting）。新头部需要从头学习，用大学习率更快收敛。Discriminative Fine-tuning 是 NLP 中的经典做法（ULMFiT/GPT）。

**Q: 什么情况下迁移学习效果差？**

A: ① 源域和目标域差异太大（如从自然图像迁移到医学图像、卫星图像）；② 目标域有大量标注数据，从头训练可能更好；③ 任务差异大（分类 → 密集预测）。此时考虑只用预训练的前几层或完全重头训练。
