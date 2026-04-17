# 混合精度训练：FP16 / BF16 / FP8 实战

> **Week 2 · Day 6**  
> 目标：理解精度格式的区别，掌握混合精度训练的实现与稳定性技巧

---

## 一、浮点格式大对比

### 1.1 格式结构图解

```
FP32（单精度，4 bytes）:
  符号  指数（8位）    尾数（23位）
  [S] [EEEEEEEE] [FFFFFFFFFFFFFFFFFFFFFFF]
  范围: ±3.4×10^38    精度: ~7位十进制

FP16（半精度，2 bytes）:
  符号  指数（5位）  尾数（10位）
  [S] [EEEEE] [FFFFFFFFFF]
  范围: ±65504       精度: ~3位十进制
  ⚠️ 动态范围小，容易溢出（>65504 = inf）

BF16（Brain Float 16，2 bytes）:
  符号  指数（8位）   尾数（7位）
  [S] [EEEEEEEE] [FFFFFFF]
  范围: ±3.4×10^38   精度: ~2位十进制
  ✅ 动态范围 = FP32，不容易溢出，但精度低

FP8（新格式，1 byte）:
  E4M3: [S][EEEE][MMM]  范围小，精度中，适合前向
  E5M2: [S][EEEEE][MM]  范围大，精度低，适合梯度
  ⚠️ 需要 H100 或更新 GPU 才支持
```

### 1.2 动态范围可视化

```
可表示范围（log10 刻度）：

FP32: ──────────────────────────────────────── [-38, +38]
BF16: ──────────────────────────────────────── [-38, +38]（同 FP32）
FP16: ─────────────────── [-4.5, +4.5]（约 ±65504）
FP8 E5M2: ──────── [-15, +15]
FP8 E4M3: ───── [-9, +9]

精度（尾数位数）：
FP32: ████████████████████████ (23位)
BF16: ███████ (7位)
FP16: ██████████ (10位)
FP8 : ███ (3位)
```

---

## 二、混合精度训练原理

### 2.1 为什么"混合"？

```
纯 FP16 训练的问题：
  梯度值通常 << 1（如 1e-4），而 FP16 精度约到 6e-5
  → 极小梯度会下溢（underflow）变成 0 → 参数不更新 → 训练失败

解决方案：混合使用 FP16 和 FP32
  FP16 用于：前向计算、梯度计算（速度快，显存小）
  FP32 用于：优化器状态、主参数（精度高，数值稳定）
```

### 2.2 混合精度训练流程

```
混合精度训练完整流程：

FP32 主参数 W_fp32
      │ 拷贝并转换
      ▼
FP16 参数副本 W_fp16
      │ 前向传播（FP16，高速）
      ▼
FP16 激活值 → Loss（FP16 或 FP32）
      │
      ▼ × loss_scale (防下溢)
Scaled Loss
      │ 反向传播（FP16）
      ▼
FP16 梯度 G_fp16 × loss_scale
      │ ÷ loss_scale，转为 FP32
      ▼
FP32 梯度 G_fp32
      │ 用 Adam 更新
      ▼
FP32 主参数 W_fp32 (更新)
      │ 下一 iteration 时再拷贝为 FP16
```

### 2.3 Loss Scaling（损失缩放）

```
问题：FP16 的最小正数 ≈ 6e-5，梯度常常更小 → 下溢为 0

解法：在反向传播前乘以一个大数（loss_scale），反向后除回来

动态 Loss Scaling 策略（PyTorch GradScaler）：

  初始 scale = 2^16 = 65536

  每轮检查：
    ├── 梯度有 NaN/Inf？→ 跳过更新，scale ÷= 2
    └── 连续 2000 步无溢出？→ scale × 2

  目标：让 scale 自动维持在梯度不下溢的最小值
```

---

## 三、代码示例

### 3.1 PyTorch 原生混合精度（AMP）

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # 负责 loss scaling

for batch in dataloader:
    inputs, labels = batch[0].cuda(), batch[1].cuda()
    
    optimizer.zero_grad()
    
    # 前向在 FP16 下执行
    with autocast(dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # scaler 自动处理 loss scaling + 梯度 unscale
    scaler.scale(loss).backward()
    
    # unscale 后检查梯度，有 NaN 则跳过更新
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

### 3.2 BF16 训练（A100 / H100 推荐）

```python
# BF16 不需要 GradScaler（动态范围 = FP32，无下溢问题）
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

loss.backward()
optimizer.step()
```

### 3.3 DeepSpeed 混合精度配置

```json
{
  "fp16": {
    "enabled": "auto",           // 根据 GPU 类型自动选择
    "loss_scale": 0,             // 0 = 动态
    "loss_scale_window": 1000,   // 多少步后尝试扩大 scale
    "initial_scale_power": 16,   // 初始 scale = 2^16
    "hysteresis": 2,             // 溢出 N 次才缩小 scale
    "min_loss_scale": 1          // 最小 scale
  }
}

// 或者使用 BF16（Ampere+ GPU）
{
  "bf16": {
    "enabled": true
  }
}
```

### 3.4 FP8 训练（H100，实验性）

```python
# 使用 Transformer Engine（NVIDIA 官方 FP8 库）
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_format = Format.HYBRID  # E4M3 前向 + E5M2 反向
fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=fp8_format,
    amax_history_len=16,
    amax_compute_algo="max",
)

model = te.Linear(hidden_size, ffn_size)  # TE 提供的算子

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    outputs = model(inputs)
```

---

## 四、FP16 vs BF16 vs FP8 性能对比

### 4.1 显存节省

```
同一模型（LLaMA-7B）不同精度的显存占用：

精度     参数显存   训练时总显存   相对 FP32
───────────────────────────────────────────
FP32      28 GB       90+ GB       基准
FP16      14 GB       ~45 GB       ×0.5
BF16      14 GB       ~45 GB       ×0.5
FP8        7 GB       ~30 GB       ×0.33
```

### 4.2 计算速度（A100 峰值 FLOPS）

```
A100 80GB 理论峰值：
  FP32 Tensor Core: 312 TFLOPS
  FP16 Tensor Core: 624 TFLOPS  (2× FP32)
  BF16 Tensor Core: 624 TFLOPS  (2× FP32)
  INT8:            1248 TOPS    (4× FP32)
  FP8 (H100):     2000 TFLOPS  (H100 专属)

实测速度提升（vs FP32）：
  FP16/BF16: 约 1.5-2× (受显存带宽限制)
  INT8 推理:  约 2-3×
  FP8 训练:  约 2× (H100 实测)
```

### 4.3 数值稳定性对比

| 精度 | 溢出风险 | 精度损失 | 需要 Loss Scale | 适用场景 |
|------|---------|---------|----------------|---------|
| FP32 | 无 | 无 | 否 | 参考基准 |
| BF16 | 极低 | 轻微 | 否 | A100+ 首选 |
| FP16 | 中等 | 轻微 | 是 | V100/T4 |
| FP8 E4M3 | 高 | 明显 | 是（每层） | H100 前向 |
| FP8 E5M2 | 中 | 明显 | 是 | H100 梯度 |

---

## 五、数值稳定性技巧

```python
# 技巧 1：初始化很重要（避免激活值过大）
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

# 技巧 2：LayerNorm 保持 FP32（精度关键路径）
# autocast 默认会让 LayerNorm 在 FP32 运行（无需手动处理）

# 技巧 3：梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 技巧 4：排查 NaN（调试时）
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")

# 技巧 5：BF16 做 LLM 训练时减小学习率
# BF16 精度低 → Adam 的参数更新可能较粗糙 → 适当降低 lr
```

---

## 六、选择精度策略

```
决策流程：

GPU 类型？
  ├── H100 → FP8 训练（前沿，需要 TE 库）
  ├── A100 / A6000 → BF16（首选，稳定无需 GradScaler）
  ├── V100 / T4 → FP16 + GradScaler
  └── 消费级（RTX 系列）→ FP16 + GradScaler

任务类型？
  ├── 预训练（长时间训练）→ 优先 BF16（稳定性优先）
  ├── 微调（短时间）→ FP16/BF16 均可
  └── 推理 → INT8 / FP8（速度优先）
```

---

## 小结

- **FP16**：动态范围小，需要 Loss Scaling，老 GPU（V100）首选
- **BF16**：动态范围与 FP32 相同，无需 Loss Scaling，A100+ 首选
- **FP8**：H100 专属，速度翻倍但需要额外的量化感知处理
- **混合精度的核心**：前向/反向用低精度，优化器状态用高精度
- **Loss Scaling** 是 FP16 稳定训练的关键，PyTorch GradScaler 自动处理
