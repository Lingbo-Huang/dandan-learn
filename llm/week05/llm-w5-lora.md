---
layout: default
title: "D2 · LoRA 与 QLoRA"
render_with_liquid: false
---

# D2 · LoRA 与 QLoRA

> **一句话**：LoRA 用低秩矩阵近似权重更新，让百亿参数模型的微调能在一张消费级 GPU 上完成。

---

## 一、LoRA 原理

### 1.1 动机

全量微调 7B 模型需要 ~14GB 显存（BF16）+ 梯度 + 优化器状态 ≈ 56GB。

LoRA 的假设：**微调的权重变化 ΔW 是低秩的**。

$$W' = W + \Delta W = W + AB$$

其中：
- $W \in \mathbb{R}^{d \times k}$：冻结的原始权重
- $A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times k}$：可训练的低秩矩阵
- $r \ll \min(d, k)$：秩，通常 4-64

**参数量对比**：
- 原始：$d \times k$ 参数
- LoRA：$(d + k) \times r$ 参数
- 对于 $d=k=4096, r=16$：原始 16.7M vs LoRA 131K，压缩 127倍

### 1.2 实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA 包装的线性层"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        d_out, d_in = original_layer.weight.shape
        
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA 参数（可训练）
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 初始化：A 用高斯，B 用零（确保初始 ΔW = 0）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出（冻结权重）
        original_output = self.original_layer(x)
        
        # LoRA 增量：x @ A^T @ B^T * scaling
        lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """将 LoRA 权重合并回原始层（推理加速）"""
        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None
        )
        merged.weight.data = (
            self.original_layer.weight.data + 
            (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data
        return merged


def add_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """给模型添加 LoRA"""
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 找到父模块并替换
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                
                lora_layer = LoRALinear(
                    module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
                setattr(parent, child_name, lora_layer)
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")
    
    return model
```

### 1.3 使用 PEFT 库

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # 秩
    lora_alpha=32,     # 缩放因子（通常设为 2*r）
    lora_dropout=0.1,
    # 应用 LoRA 的目标模块（attention 层的 Q/V 投影）
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    # 只对 lm_head 的部分也可选
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 83,886,080 || all params: 7,615,324,160 || trainable%: 1.10%
```

---

## 二、QLoRA：在 4bit 量化上微调

QLoRA（Dettmers et al., 2023）= NF4 量化 + Double Quantization + LoRA

### 2.1 NF4 量化原理

```python
import torch
import numpy as np

def quantize_nf4(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    NF4（NormalFloat4）量化
    
    核心：NF4 的 16 个量化点是根据正态分布的分位数设计的，
    对正态分布的权重有最小量化误差
    """
    # NF4 的 16 个量化点（来自 QLoRA 论文）
    NF4_TABLE = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
        -0.09105003625154495, 0.0, 0.07958029955625534,
        0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # 缩放到 [-1, 1]
    abs_max = tensor.abs().max()
    normalized = tensor / abs_max
    
    # 找最近的 NF4 量化点（量化）
    distances = (normalized.unsqueeze(-1) - NF4_TABLE).abs()
    quantized_indices = distances.argmin(dim=-1).to(torch.uint8)
    
    return quantized_indices, abs_max.item()

def dequantize_nf4(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """NF4 反量化"""
    NF4_TABLE = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
        -0.09105003625154495, 0.0, 0.07958029955625534,
        0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    return NF4_TABLE[quantized.long()] * scale

# 演示量化误差
W = torch.randn(64, 64)
q, scale = quantize_nf4(W)
W_dequant = dequantize_nf4(q, scale)
error = (W - W_dequant).abs().mean()
print(f"平均量化误差: {error.item():.6f}")
print(f"存储压缩: {W.element_size()*W.numel()} bytes → {q.numel()//2} bytes (4bit)")
```

### 2.2 QLoRA 完整训练脚本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, TrainingArguments
import torch

def run_qlora(
    model_name: str = "meta-llama/Llama-3.2-8B",
    output_dir: str = "./qlora_output"
):
    """QLoRA 微调（4bit 量化 + LoRA）"""
    
    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                           # 4bit 加载
        bnb_4bit_quant_type="nf4",                   # NF4 量化
        bnb_4bit_compute_dtype=torch.bfloat16,       # 计算时反量化为 BF16
        bnb_4bit_use_double_quant=True,              # Double Quantization
    )
    
    # 加载量化模型（8B 模型约占 5GB 显存，而非 16GB）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # 准备模型用于 kbit 训练（冻结量化层，启用梯度检查点）
    model = prepare_model_for_kbit_training(model)
    
    # 添加 LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练（实际只更新 LoRA 参数，约 80M）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # 等效 batch_size=16
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",        # 8bit 优化器（进一步省显存）
        logging_steps=10,
        gradient_checkpointing=True,      # 用时间换显存
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )
    
    # 加载数据集...（略）
    
    print("QLoRA 配置完成，可以开始训练")
    return model
```

---

## 三、LoRA 秩的选择

```python
def lora_rank_experiment():
    """不同秩 r 对参数量和效果的影响"""
    d_model = 4096  # Llama-2-7B 的 d_model
    
    print(f"{'秩 r':<8} {'参数量(M)':<15} {'适用场景'}")
    print("-" * 50)
    
    scenarios = {
        4: "快速验证，数据少（<1K）",
        8: "轻量微调，资源紧张",
        16: "通用推荐（默认）",
        32: "复杂任务，多领域",
        64: "较大任务，接近全量",
        128: "接近全量微调效果",
    }
    
    for r, desc in scenarios.items():
        # 每个注意力层的 LoRA 参数（Q+V）
        attn_params = 2 * (d_model * r + r * d_model)
        # 假设 32 层
        total_params = attn_params * 32 / 1e6
        print(f"r={r:<6} {total_params:<15.1f} {desc}")

lora_rank_experiment()
```

---

## 四、面试题精讲

**Q: LoRA 为什么将 B 初始化为零？**

A: 确保训练开始时 $\Delta W = AB = 0$，即 LoRA 初始时不改变原始模型的行为。如果 A 和 B 都随机初始化，训练初期的梯度会很嘈杂，破坏预训练模型已有的能力。

**Q: alpha/r 的比值有什么意义？**

A: `scaling = alpha/r` 是 LoRA 增量的缩放因子。常见设置 `alpha = 2*r`，此时 scaling=2。当改变 r 时，保持 alpha/r 不变（等效 lr），方便超参比较。

**Q: QLoRA 的 4bit 量化为什么不影响最终精度？**

A: 关键在于**只有权重是 4bit 的，计算时会反量化为 BF16**（前向传播和反向传播都在 BF16 精度下进行）。同时，LoRA 本身的参数是全精度 BF16，所以梯度更新是准确的。量化只影响存储，不影响计算精度。

---

## 小结

| | LoRA | QLoRA |
|--|------|-------|
| 显存（7B）| ~14GB | ~5GB |
| 速度 | 快 | 慢 30% |
| 精度 | 接近全量 | 略低于全量 |
| 适用 | 单卡 A100 | 单卡 3090/4090 |
