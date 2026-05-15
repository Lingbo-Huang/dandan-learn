---
layout: default
title: "目标检测入门"
render_with_liquid: false
---

# 目标检测入门

## 任务定义

目标检测 = 图像分类 + 定位（找出图像中所有目标的类别和位置）。

输出：一组 $(x, y, w, h, \text{class}, \text{confidence})$。

## 核心概念

### IoU（Intersection over Union）

$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

- IoU > 0.5：通常认为是好的检测
- 用于 NMS 和评估

### NMS（Non-Maximum Suppression）

去除重叠检测框：
1. 按置信度排序
2. 选最高置信度框
3. 去除与其 IoU > 阈值的其他框
4. 对剩余框重复

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_iou(box1, box2):
    """box: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    """
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        best = order[0]
        keep.append(best)
        
        ious = np.array([compute_iou(boxes[best], boxes[i]) for i in order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]
    
    return keep

# NMS 示例
boxes = np.array([
    [100, 100, 300, 300],  # 主框
    [110, 105, 310, 305],  # 与主框高度重叠
    [150, 150, 350, 350],  # 稍微偏移
    [400, 400, 600, 600],  # 另一个目标
    [405, 402, 605, 602],  # 另一个目标的重复框
])
scores = np.array([0.9, 0.85, 0.7, 0.88, 0.82])

kept = nms(boxes, scores, iou_threshold=0.5)
print("NMS 保留的框索引:", kept)
print("NMS 后的框:", boxes[kept])
```

## 两阶段检测器：Faster R-CNN

### 核心组件

1. **Backbone**（如 ResNet）：提取特征图
2. **RPN（Region Proposal Network）**：生成候选区域
3. **RoI Pooling**：统一候选区域特征尺寸
4. **分类头 + 回归头**：对每个候选区域分类并精修位置

### RPN 工作原理

在特征图每个位置，生成 $k$ 个 anchor（不同尺度+比例）：
- 二分类：前景/背景
- 回归：anchor → proposal 的偏移量

```python
def generate_anchors(scales=[128, 256, 512], ratios=[0.5, 1, 2], base_size=16):
    """生成基础 anchor boxes"""
    anchors = []
    cx, cy = base_size // 2, base_size // 2
    
    for scale in scales:
        for ratio in ratios:
            w = scale * np.sqrt(ratio)
            h = scale / np.sqrt(ratio)
            anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    return np.array(anchors)

anchors = generate_anchors()
print(f"Anchor 形状: {anchors.shape}")  # (9, 4) = 3 scales × 3 ratios

# 可视化 anchors
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(-400, 700)
ax.set_ylim(-400, 700)
colors = ['red', 'blue', 'green'] * 3
for anchor, color in zip(anchors, colors):
    x1, y1, x2, y2 = anchor
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
ax.scatter([256//2], [256//2], s=200, c='k', zorder=5, label='中心点')
ax.legend()
ax.set_title("9个基础 Anchor Boxes")
ax.set_aspect('equal')
ax.grid(True)
plt.tight_layout()
```

## 单阶段检测器：YOLO

### YOLO 核心思想（v1）

将图像分成 $S \times S$ 网格，每个格子预测：
- $B$ 个候选框（$x, y, w, h, \text{confidence}$）
- $C$ 个类别概率

输出张量：$S \times S \times (5B + C)$

**一次前向传播直接出框**，速度极快！

```python
# YOLO 简化版：理解输出格式
S = 7   # 网格大小
B = 2   # 每格候选框数
C = 20  # VOC类别数

# 输出张量形状
output_shape = (S, S, 5*B + C)
print(f"YOLO v1 输出张量: {output_shape}")  # (7, 7, 30)

# 解码单个格子的预测
def decode_cell(pred, cell_i, cell_j, S=7, B=2, C=20, img_size=448):
    """解码 YOLO 某个格子的预测"""
    boxes = []
    for b in range(B):
        offset = b * 5
        cx = (pred[offset + 0] + cell_j) / S * img_size
        cy = (pred[offset + 1] + cell_i) / S * img_size
        w = pred[offset + 2] ** 2 * img_size  # v1 用平方预测宽高（相对于整图）
        h = pred[offset + 3] ** 2 * img_size
        conf = pred[offset + 4]
        boxes.append((cx, cy, w, h, conf))
    
    class_probs = pred[B*5:B*5+C]
    predicted_class = class_probs.argmax()
    
    return boxes, predicted_class
```

## 检测框架对比

| 模型 | 类型 | 速度(FPS) | mAP | 特点 |
|------|------|----------|-----|------|
| Faster R-CNN | 两阶段 | ~5-17 | 高 | 精度高，适合离线场景 |
| SSD | 单阶段 | ~58 | 中 | 多尺度检测，平衡速度精度 |
| YOLOv3 | 单阶段 | ~45 | 中高 | 实时场景 |
| YOLOv8 | 单阶段 | 实时 | 高 | 目前工业首选 |
| DETR | Transformer | ~28 | 高 | 无 NMS，端到端 |

## 评估指标：mAP

$$\text{AP} = \int_0^1 p(r) dr \approx \sum_{k=1}^N P(k) \cdot \Delta R(k)$$

$$\text{mAP} = \frac{1}{|\text{classes}|} \sum_c \text{AP}_c$$

## 面试要点

**Q: 两阶段检测器和单阶段检测器的核心权衡是什么？**

A: 两阶段（Faster RCNN）：先生成候选框再分类，精度高（小目标好），速度慢；单阶段（YOLO）：一次前向直接预测所有框，速度快，小目标稍差。实时场景用 YOLO，高精度离线场景用 Cascade RCNN 等。

**Q: anchor-free 检测器（FCOS/CenterPoint）有什么优势？**

A: 传统 anchor 需要人工设计尺度比例，对特殊比例目标检测差，且 anchor 数量多导致计算量大。Anchor-free 直接预测目标中心点+偏移量，更简洁，泛化性更好，是当前趋势。
