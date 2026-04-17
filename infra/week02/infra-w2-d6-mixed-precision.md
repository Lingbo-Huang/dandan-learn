# D6：混合精度训练（FP16 / BF16 / Loss Scaling）

> AI Infra Week2 · Day 6 | 作者：🥚🥚5号 ai infra

---

## 一、为什么需要混合精度训练

传统深度学习训练使用 **FP32**（32位浮点数）存储所有参数和梯度。混合精度（Mixed Precision）训练则在前向和后向传播中使用低精度（FP16 或 BF16），仅在参数更新时使用 FP32。

**收益巨大：**

| 指标 | FP32 | FP16 混合精度 | 提升 |
|------|------|-------------|------|
| 参数显存 | 4 bytes/param | 2 bytes/param | 2x |
| 梯度显存 | 4 bytes/param | 2 bytes/param | 2x |
| 计算速度（A100）| 1x | ~3x（Tensor Core） | 3x |
| 峰值显存 | 高 | 低（更大 batch） | 视情况 |

NVIDIA A100 的 Tensor Core 在 FP16/BF16 下峰值算力是 312 TFLOPS，FP32 下仅 77 TFLOPS，差距接近 4 倍。

---

## 二、浮点数格式深度解析

### 2.1 IEEE 754 浮点格式

```
FP32（单精度）：
  [符号 1位][指数 8位][尾数 23位]
  指数偏置 127
  范围：约 ±3.4 × 10^38
  精度：约 7 位十进制有效数字

FP16（半精度）：
  [符号 1位][指数 5位][尾数 10位]
  指数偏置 15
  范围：约 ±65504（最大值！）
  精度：约 3 位十进制有效数字

BF16（Brain Float 16）：
  [符号 1位][指数 8位][尾数 7位]
  指数偏置 127（与 FP32 相同！）
  范围：约 ±3.4 × 10^38（与 FP32 相同！）
  精度：约 2-3 位十进制有效数字（低于 FP16）
```

### 2.2 格式对比图

```
FP32:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM   (32 bits)
FP16:  S EEEEE MMMMMMMMMM                   (16 bits)
BF16:  S EEEEEEEE MMMMMMM                   (16 bits)

关键区别：
  - FP16 的指数域只有 5 位 → 最大值约 65504
  - BF16 的指数域与 FP32 相同 → 动态范围与 FP32 一致
  - FP16 精度更高（10位尾数 vs BF16的7位）
  - BF16 稳定性更好（不溢出），但精度更低
```

### 2.3 FP16 的两大风险

#### 风险 1：上溢（Overflow）

梯度值超过 FP16 最大值 65504 → 变成 `inf`（正无穷）→ 传播为 `nan`

```
典型场景：
  - 训练初期，参数初始化不合理导致大梯度
  - BatchNorm 在某些 batch 出现数值异常
  - 激活函数（如 exp）在大输入时爆炸
```

#### 风险 2：下溢（Underflow）

梯度值太小（如 1e-8），低于 FP16 最小正规数（约 6×10^-5）→ 变成 0 → 梯度消失

```
FP16 次正规数（subnormal）范围：约 5.96×10^-8 到 6.10×10^-5
小于该范围的正数被舍入为 0
```

---

## 三、Loss Scaling：解决下溢问题

### 3.1 原理

Loss Scaling 的核心思想：**在反向传播前，将 loss 乘以一个大的缩放因子 S，使梯度也被放大 S 倍，从而避免下溢；参数更新前再除以 S。**

```
正常流程（FP16 下溢）：
  Loss → 梯度 1e-8 → FP16 舍入为 0 → 参数不更新 ✗

Loss Scaling 流程：
  Loss × S(=65536) → 梯度 65536 × 1e-8 = 6.5536×10^-4
  → FP16 可表示！→ 参数更新前除以 S → 正确梯度 ✓
```

### 3.2 数学表达

```
标准训练：
  g = ∂L/∂θ
  θ ← θ - lr × g

Loss Scaling 训练：
  前向：L_scaled = L × S
  后向：g_scaled = ∂L_scaled/∂θ = S × g
  裁剪梯度：检查 g_scaled 是否含 inf/nan
  更新：g = g_scaled / S
        θ ← θ - lr × g （等价于原来的更新）
```

### 3.3 静态 vs 动态 Loss Scaling

**静态 Loss Scaling**：固定缩放因子，简单但不灵活

```python
scaler_value = 512.0
loss_scaled = loss * scaler_value
loss_scaled.backward()

for param in model.parameters():
    if param.grad is not None:
        param.grad /= scaler_value

optimizer.step()
```

**动态 Loss Scaling（GradScaler）**：自动调整缩放因子

```
算法：
  初始化：scale = 2^16 = 65536
  每 growth_interval 步（默认2000步）：
    若无 inf/nan：scale *= growth_factor（默认2.0）
  若出现 inf/nan：
    scale /= backoff_factor（默认0.5）
    跳过本次参数更新
```

---

## 四、PyTorch AMP（自动混合精度）实战

### 4.1 最基本的 AMP 用法

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 创建 GradScaler，初始 scale = 2^16
scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True,  # 设为 False 退化为 FP32（方便调试）
)

for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()

    optimizer.zero_grad()

    # autocast 上下文：自动将部分算子切换为 FP16
    with autocast(dtype=torch.float16):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    # scale loss，调用 backward
    scaler.scale(loss).backward()

    # 在 optimizer.step() 前 unscale 梯度
    # 如果梯度包含 inf/nan，则跳过更新
    scaler.unscale_(optimizer)

    # 梯度裁剪（必须在 unscale 后）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 实际更新参数（内部检查梯度有效性）
    scaler.step(optimizer)

    # 更新 scale 因子
    scaler.update()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Scale: {scaler.get_scale():.0f}")
```

### 4.2 BF16 AMP（PyTorch 1.10+）

```python
# BF16 不需要 GradScaler！
with autocast(dtype=torch.bfloat16):
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 4.3 autocast 内部做了什么

`autocast` 会自动将部分算子切换为低精度，保留需要高精度的算子：

```python
# autocast 自动选择精度的部分算子（示例）

# 切换为 FP16/BF16（计算密集型，精度影响小）：
torch.nn.Linear      # 矩阵乘法 → Tensor Core 加速
torch.nn.Conv2d      # 卷积
torch.nn.MultiheadAttention

# 保持 FP32（精度敏感型）：
torch.nn.LayerNorm   # 归一化需要精度
torch.nn.BatchNorm   # 同上
torch.nn.Softmax     # 指数运算容易溢出
torch.nn.CrossEntropyLoss
torch.log
torch.exp
```

---

## 五、Master Weight（主权重）技术

混合精度训练的完整架构：

```
┌────────────────────────────────────────────────────────────┐
│                    混合精度训练流程                          │
│                                                            │
│  FP16 参数副本（GPU）←──────────────────────────────────┐  │
│       ↓                                                 │  │
│  FP16 前向传播（autocast）                               │  │
│       ↓                                                 │  │
│  FP16 Loss × scale_factor                               │  │
│       ↓                                                 │  │
│  FP16 后向传播 → FP16 梯度                              │  │
│       ↓                                                 │  │
│  检查 inf/nan（GradScaler）                              │  │
│       ↓                                                 │  │
│  转换为 FP32 梯度 / scale_factor                        │  │
│       ↓                                                 │  │
│  FP32 Adam 更新 → FP32 Master Weights                  │  │
│       ↓                                                 │  │
│  FP32 → FP16 转换 ────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**为什么需要 FP32 Master Weights？**

假设某参数 `θ = 1.0`，梯度更新量 `Δθ = 1e-5`（学习率 × 梯度）：

```
FP32 下：
  1.0 + 1e-5 = 1.00001  ✓ 可表示

FP16 下：
  1.0 的 FP16 表示精度约为 1/1024 ≈ 9.77e-4
  1e-5 << 9.77e-4，会被舍入为 0
  结果：1.0 + 0 = 1.0  ✗ 参数没有更新！
```

FP32 Master Weights 保留了累积的微小更新，确保参数收敛。

---

## 六、完整混合精度训练示例

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class MixedPrecisionTrainer:
    """封装混合精度训练的 Trainer 类"""

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        use_fp16: bool = True,
        use_bf16: bool = False,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 精度设置
        assert not (use_fp16 and use_bf16), "只能二选一"
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)
        self.amp_enabled = use_fp16 or use_bf16

        # GradScaler 只对 FP16 有意义
        self.scaler = GradScaler(enabled=use_fp16)

        self.global_step = 0

    def train_step(self, batch, accumulate: bool = False):
        """
        单步训练
        accumulate=True 时跳过参数更新（梯度累积模式）
        """
        # 前向传播（低精度）
        with autocast(dtype=self.dtype, enabled=self.amp_enabled):
            loss = self.model(**batch)["loss"]
            # 梯度累积：缩小 loss 保持等效学习率
            loss = loss / self.gradient_accumulation_steps

        # 后向传播
        if self.use_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # 仅在累积完成时更新参数
        if not accumulate:
            self._update_params()
            self.global_step += 1

        return loss.item() * self.gradient_accumulation_steps

    def _update_params(self):
        if self.use_fp16:
            # unscale 梯度（FP16 模式）
            self.scaler.unscale_(self.optimizer)

        # 梯度裁剪
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        if self.use_fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    @property
    def current_scale(self):
        return self.scaler.get_scale() if self.use_fp16 else None


# ──── 使用示例 ────────────────────────────────────────────

model = MyGPTModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# FP16 训练
trainer = MixedPrecisionTrainer(
    model=model,
    optimizer=optimizer,
    use_fp16=True,
    gradient_accumulation_steps=4,
)

for step, batch in enumerate(dataloader):
    batch = {k: v.cuda() for k, v in batch.items()}
    is_last_accum_step = (step + 1) % 4 == 0

    loss = trainer.train_step(batch, accumulate=not is_last_accum_step)

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss:.4f} | Scale: {trainer.current_scale}")
```

---

## 七、FP16 vs BF16 选择指南

```
┌─────────────────────────────────────────────────────────┐
│                  选择混合精度格式                         │
│                                                         │
│  你的 GPU 支持 BF16 硬件加速？                          │
│  （A100, A800, H100, 4090, AMD MI300X, etc.）           │
│          ↓ 是              ↓ 否                         │
│      使用 BF16         使用 FP16                        │
│   - 无需 Loss Scale   - 需要 GradScaler                │
│   - 训练更稳定         - 精度略高                       │
│   - 推荐用于大模型     - 老 GPU 必选（V100 等）         │
└─────────────────────────────────────────────────────────┘
```

**实际经验：**

| 场景 | 推荐 |
|------|------|
| A100/H100 + LLM 预训练 | BF16（更稳定，推荐） |
| A100/H100 + 微调 | BF16 或 FP16 均可 |
| V100/T4 + 任意任务 | FP16（没有 BF16 支持） |
| 数值敏感任务（如精确计算） | FP32 |
| 推理部署 | FP16 或 INT8 |

---

## 八、调试混合精度的常见问题

### 8.1 NaN Loss 排查

```python
# 方法一：临时禁用混合精度，确认是精度问题
with autocast(enabled=False):
    loss = model(...)

# 方法二：检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")

# 方法三：降低 loss scale 的初始值
scaler = GradScaler(init_scale=2**10)  # 从较小值开始

# 方法四：降低学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 减小 10x
```

### 8.2 监控 Loss Scale

```python
# 记录 scale 的变化趋势
scales = []
for step, batch in enumerate(dataloader):
    ...
    scaler.update()
    scales.append(scaler.get_scale())

    # scale 不断下降是不正常的信号
    if step % 100 == 0 and len(scales) >= 100:
        trend = scales[-1] / scales[-100]
        if trend < 0.5:
            print(f"WARNING: Loss scale dropping fast! Current: {scaler.get_scale()}")
```

### 8.3 某些层需要 FP32

对于数值敏感的自定义层，可以强制 FP32：

```python
class FP32LayerNorm(nn.LayerNorm):
    """总是以 FP32 精度运行的 LayerNorm"""
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)

# 或者使用 autocast 白名单
class CustomLayer(nn.Module):
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        # 这里的计算总是 FP32
        return some_sensitive_computation(x)
```

---

## 九、本章小结

| 知识点 | 核心内容 |
|--------|---------|
| FP16 问题 | 动态范围小（最大 65504），易上溢下溢 |
| BF16 优势 | 动态范围与 FP32 相同，训练更稳定 |
| Loss Scaling | 放大梯度解决 FP16 下溢，动态调整 scale |
| GradScaler | PyTorch 内置动态 Loss Scaling 工具 |
| Master Weights | FP32 主权重积累微小更新，保证收敛 |
| autocast | 自动为不同算子选择合适精度 |
| 选择原则 | A100+ 用 BF16；老 GPU 用 FP16 + GradScaler |

**下一篇（D7）**将把所有知识点汇聚，完成一个完整的 GPT-2 DeepSpeed 训练实战，包括从数据准备到模型评估的全流程。
