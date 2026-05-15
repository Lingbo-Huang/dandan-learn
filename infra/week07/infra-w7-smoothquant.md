---
layout: default
title: "D2 · SmoothQuant：激活量化的挑战与解决"
render_with_liquid: false
---

# D2 · SmoothQuant：激活量化的挑战与解决

## 激活量化的难题：Outlier 问题

LLM 中存在少量"巨型激活值"（activation outlier）现象：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b", 
    torch_dtype=torch.float16, 
    device_map='cuda'
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt").to('cuda')

activation_stats = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            x = output.detach().float()
            activation_stats[name] = {
                'max': x.abs().max().item(),
                'mean': x.abs().mean().item(),
                'outlier_ratio': (x.abs() > x.abs().mean() * 10).float().mean().item()
            }
    return hook

# 注册 hook 观察激活值分布
hooks = []
for name, module in model.named_modules():
    if 'fc1' in name or 'fc2' in name:
        hooks.append(module.register_forward_hook(hook_fn(name)))

with torch.no_grad():
    model(**inputs)

for hook in hooks:
    hook.remove()

# 打印统计
for name, stats in list(activation_stats.items())[:5]:
    print(f"{name}: max={stats['max']:.1f}, mean={stats['mean']:.3f}, "
          f"outlier_ratio={stats['outlier_ratio']:.2%}")

# 典型输出：
# transformer.h.0.fc1: max=187.2, mean=0.823, outlier_ratio=0.12%
# transformer.h.1.fc1: max=243.5, mean=0.891, outlier_ratio=0.08%
# 可以看到：max/mean ≈ 200-300×，但 outlier 只占 0.1%
```

**问题**：如果 max=243，mean=0.89：
- INT8 scale = 243/127 ≈ 1.91
- 大多数激活值量化后 = round(0.89/1.91) = round(0.47) = 0（精度几乎全失）

## SmoothQuant 的核心思想

**数学等价变换**：
$$Y = (X \cdot W) = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

其中 $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$（每个通道的平滑因子）。

**直觉**：将激活值中"大值"所在通道除以 $s_j$，同时将权重矩阵对应列乘以 $s_j$，使激活值分布更均匀，同时不改变输出。

```python
import torch
import numpy as np

def compute_smooth_scale(
    activation_samples: torch.Tensor,  # [num_samples, seq_len, hidden]
    weight: torch.Tensor,               # [out, in]
    alpha: float = 0.5
) -> torch.Tensor:
    """
    计算 SmoothQuant 的平滑因子
    
    alpha: 平滑强度（0=全给激活，1=全给权重）
           0.5 = 均匀分配（推荐）
    """
    # 激活值每个通道的最大绝对值
    act_max = activation_samples.abs().reshape(-1, activation_samples.shape[-1]).max(dim=0).values
    # [hidden]
    
    # 权重每个输入通道的最大绝对值（通常 per-output-channel 取 max）
    weight_max = weight.abs().max(dim=0).values  # [in]
    
    # 平滑因子
    scale = act_max.pow(alpha) / weight_max.pow(1 - alpha)
    scale = scale.clamp(min=1e-5)
    
    return scale


def smooth_quantize_linear(
    module: torch.nn.Linear,
    activation_samples: torch.Tensor,
    alpha: float = 0.5
) -> torch.nn.Linear:
    """
    对线性层应用 SmoothQuant 变换
    将变换"吸收"到权重中，推理时激活自动缩放
    """
    scale = compute_smooth_scale(activation_samples, module.weight, alpha)
    
    # 修改权重：W_new = diag(s) @ W（每列乘以 s_j）
    module.weight.data = module.weight.data * scale.unsqueeze(0)
    
    # 修改 bias（如果有）
    if module.bias is not None:
        module.bias.data = module.bias.data  # bias 不变，因为 s 作用于 X
    
    # 返回缩放因子（用于在前一层的输出处除以 s）
    return module, scale


def apply_smoothquant_to_model(model, calibration_data, alpha=0.5):
    """
    对整个模型应用 SmoothQuant
    需要：1. 收集校准数据的激活值
         2. 计算平滑因子
         3. 将平滑因子吸收到相邻层中
    """
    # 步骤 1：收集激活值统计
    activation_stats = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(input[0].detach())
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # 运行校准数据
    with torch.no_grad():
        for batch in calibration_data:
            model(**batch)
    
    for hook in hooks:
        hook.remove()
    
    # 步骤 2 & 3：计算并吸收平滑因子
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in activation_stats:
            samples = torch.stack(activation_stats[name][:32])  # 用前 32 个 batch
            module, scale = smooth_quantize_linear(module, samples, alpha)
            
            # 将 1/scale 传递给前一层（LayerNorm 的 weight/bias 吸收）
            # ... 实际实现需要找到前驱节点
    
    return model
```

## SmoothQuant 前后的分布变化

```python
import matplotlib.pyplot as plt

def visualize_smooth_effect():
    """可视化 SmoothQuant 前后的激活值分布"""
    
    # 模拟有 outlier 的激活值（OPT 风格）
    torch.manual_seed(42)
    num_tokens, hidden = 100, 512
    
    # 99% 的值是正常的，1% 是 outlier（特定 channel）
    x = torch.randn(num_tokens, hidden) * 0.5
    outlier_channels = [20, 150, 300]  # 少数 outlier channel
    for ch in outlier_channels:
        x[:, ch] = torch.randn(num_tokens) * 50  # 100× 大
    
    # 模拟权重
    W = torch.randn(hidden, hidden) * 0.01
    
    # 计算平滑因子
    act_max = x.abs().max(dim=0).values
    weight_max = W.abs().max(dim=0).values
    scale = act_max.pow(0.5) / weight_max.pow(0.5)
    
    # 平滑后的激活值
    x_smooth = x / scale.unsqueeze(0)
    
    print("平滑前激活值统计：")
    print(f"  max={x.abs().max():.1f}, mean={x.abs().mean():.3f}, "
          f"max/mean={x.abs().max()/x.abs().mean():.0f}×")
    
    print("平滑后激活值统计：")
    print(f"  max={x_smooth.abs().max():.1f}, mean={x_smooth.abs().mean():.3f}, "
          f"max/mean={x_smooth.abs().max()/x_smooth.abs().mean():.0f}×")
    
    # INT8 量化误差对比
    def quant_error(tensor):
        max_val = tensor.abs().max()
        scale = max_val / 127
        q = torch.clamp(torch.round(tensor / scale), -127, 127)
        dq = q * scale
        return (tensor - dq).abs().mean().item()
    
    print(f"\nINT8 量化误差（越小越好）：")
    print(f"  平滑前: {quant_error(x):.4f}")
    print(f"  平滑后: {quant_error(x_smooth):.4f}")

visualize_smooth_effect()
# 典型输出：
# 平滑前激活值统计：
#   max=172.3, mean=0.402, max/mean=428×
# 平滑后激活值统计：
#   max=8.2, mean=0.395, max/mean=21×
# 
# INT8 量化误差（越小越好）：
#   平滑前: 0.3185
#   平滑后: 0.0089   → 36× 改善！
```

## SmoothQuant 性能数据

论文数据（OPT-175B, A100）：

| 方法 | 模型准确率（PPL↓）| 推理加速 |
|------|----------------|---------|
| FP16 基线 | 8.34 | 1× |
| 朴素 W8A8 | 9.21 (+0.87) | 1.5× |
| SmoothQuant W8A8 | 8.40 (+0.06) | **1.56×** |

**关键**：SmoothQuant 的 W8A8 量化（权重和激活都量化为 INT8）几乎无损，同时获得 1.5× 加速。

## 实战：使用 SmoothQuant

```python
# 安装
# pip install smoothquant

from smoothquant import smooth_lm, get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-13b",
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

# 收集激活值统计（校准步骤）
# 使用 512 条校准数据
dataset = load_dataset("pile", split="validation", streaming=True)
calibration_samples = []
for i, example in enumerate(dataset):
    if i >= 512:
        break
    inputs = tokenizer(example['text'][:512], return_tensors='pt', truncation=True)
    calibration_samples.append(inputs['input_ids'])

act_scales = get_act_scales(model, calibration_samples)

# 应用 SmoothQuant
alpha = 0.5
smooth_lm(model, act_scales, alpha)

# 量化权重（W8A8）
from smoothquant import W8A8Linear
model = model.to(torch.int8)  # 简化，实际需要逐层替换
```

## 面试题

**Q: SmoothQuant 的数学原理是什么？为什么 alpha=0.5 是个好的默认值？**

A: SmoothQuant 利用矩阵乘法的等价变换：Y = XW = (X / s)(sW) = X̂Ŵ，其中 s 是每个通道的缩放因子，既减小激活值的 outlier，又不使权重变化太大。alpha 控制量化难度在激活值和权重之间的分配：alpha=0 把所有难度推给权重（激活完全平滑），alpha=1 不平滑（等于原始量化）。alpha=0.5 是平衡点，使得激活值和权重的量化难度大致相当，实践中效果最好。具体地，scale = act_max^alpha / weight_max^(1-alpha)，当两边难度相等时 alpha=0.5 最优。
