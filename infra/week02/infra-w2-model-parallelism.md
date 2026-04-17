# 模型并行：张量并行 & 流水线并行

> **Week 2 · Day 3**  
> 目标：理解当模型放不进单卡时的两种切分策略

---

## 一、为什么需要模型并行？

```
问题场景：

  模型大小：70B (LLaMA-2-70B)
  所需显存（FP16 + 梯度 + Adam）：
    参数: 70B × 2 bytes = 140 GB
    梯度: 70B × 2 bytes = 140 GB
    Adam: 70B × 8 bytes = 560 GB
    激活值（估算）: ≈ 100+ GB
    ─────────────────────────────
    合计: ≈ 940 GB

  A100 80GB × 8 = 640 GB → 仍然不够！

→ 必须把模型本身切分到多张 GPU 上
```

---

## 二、张量并行 (Tensor Parallelism)

### 2.1 核心思想

把单个层内的矩阵运算切分到多张 GPU，每张 GPU 算矩阵乘法的一部分。

### 2.2 列切分 vs 行切分

```
原始矩阵乘法：Y = X × W
  X: [B, N]  W: [N, M]  Y: [B, M]

─────────────────────────────────────────
方案 A：按列切分 W（Column Parallel）

  GPU 0: Y0 = X × W[:, :M/2]    → Y0: [B, M/2]
  GPU 1: Y1 = X × W[:, M/2:]    → Y1: [B, M/2]
  
  X 需要被广播到所有 GPU（All-Gather X）
  最终拼接: Y = [Y0 | Y1]        → Y: [B, M]

─────────────────────────────────────────
方案 B：按行切分 W（Row Parallel）

  先切分 X：
  GPU 0: X0 = X[:, :N/2]  GPU 1: X1 = X[:, N/2:]
  
  GPU 0: Y0 = X0 × W[:N/2, :]   → Y0: [B, M]
  GPU 1: Y1 = X1 × W[N/2:, :]   → Y1: [B, M]
  
  做 All-Reduce：Y = Y0 + Y1     → Y: [B, M]
```

### 2.3 Transformer 中的张量并行（Megatron 方案）

```
Self-Attention 层张量并行（4头切分，TP=2）：

     Input X
        │
   ─────┴─────
   │          │
 GPU 0      GPU 1
 Q0,K0,V0  Q1,K1,V1  ← 按注意力头切分
   │          │
 Attn Head   Attn Head
 0,1         2,3
   │          │
   └────┬─────┘
   All-Reduce
        │
     Output Y

FFN 层张量并行：
  W1 列切分（GPU0: 左半, GPU1: 右半）→ 激活函数 →
  W2 行切分（GPU0: 上半, GPU1: 下半）→ All-Reduce → 输出
```

### 2.4 张量并行配置示例（Megatron-LM 风格）

```python
# 使用 Megatron-Core 的张量并行
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_size, tp_size):
        super().__init__()
        # 列切分：每张 GPU 持有 ffn_size/tp_size 列
        self.fc1 = ColumnParallelLinear(
            hidden_size, ffn_size,
            gather_output=False  # 不汇总，直接传下一层
        )
        # 行切分：每张 GPU 持有 ffn_size/tp_size 行
        self.fc2 = RowParallelLinear(
            ffn_size, hidden_size,
            input_is_parallel=True  # 输入已经是切分状态
        )
    
    def forward(self, x):
        x, _ = self.fc1(x)  # 列并行，无需通信
        x = F.gelu(x)
        x, _ = self.fc2(x)  # 行并行，内部做 All-Reduce
        return x

# 启动时初始化 TP 进程组
tensor_parallel.initialize_model_parallel(
    tensor_model_parallel_size=4
)
```

---

## 三、流水线并行 (Pipeline Parallelism)

### 3.1 朴素流水线（Naive Pipeline）

```
模型 24 层，切4段，每段6层：

GPU 0 (L1-6):   [  F  ][  B  ]   [  F  ][  B  ]
GPU 1 (L7-12):       [  F  ][  B  ]   [  F  ][  B  ]
GPU 2 (L13-18):          [  F  ][  B  ]   [  F  ][  B  ]
GPU 3 (L19-24):              [  F  ][  B  ]   [  F  ][  B  ]
                ──────────────────────────────────────────▶ time

问题：大量 idle（气泡），GPU 利用率低！
气泡比例 = (p-1) / (m + p - 1)  其中 p=阶段数，m=micro-batch数
```

### 3.2 GPipe（micro-batch 流水线）

```
将 batch 切成 m=4 个 micro-batch：

GPU 0: [mb1_F][mb2_F][mb3_F][mb4_F][mb4_B][mb3_B][mb2_B][mb1_B]
GPU 1:        [mb1_F][mb2_F][mb3_F][mb4_F][mb4_B][mb3_B][mb2_B][mb1_B]
GPU 2:               [mb1_F][mb2_F][mb3_F][mb4_F][mb4_B][mb3_B][mb2_B][mb1_B]
GPU 3:                      [mb1_F][mb2_F][mb3_F][mb4_F][mb4_B][mb3_B][mb2_B][mb1_B]
       |←── bubble ──→|                                         |←── bubble ──→|

气泡比例 = (p-1)/(m+p-1) = 3/(4+3) ≈ 43%（micro-batch越多气泡越小）
```

### 3.3 1F1B（Interleaved 调度，PipeDream 改进）

```
交替前向反向，减少气泡：

GPU 0: [1F][2F][3F][4F][4B][3B][2B][1B]
GPU 1:    [1F][2F][3F][4F][4B][3B][2B][1B]
GPU 2:       [1F][2F][3F][1B][4F][4B][3B][2B]  ← 提前开始反向！
GPU 3:          [1F][1B][2F][2B][3F][3B][4F][4B]

气泡比例降低约 50%，但需要同时保存更多激活值（显存代价）
```

### 3.4 DeepSpeed Pipeline 配置示例

```python
import deepspeed
from deepspeed.pipe import PipelineModule

# 将模型转为流水线模块
layers = [
    *[TransformerLayer(config) for _ in range(24)]
]

model = PipelineModule(
    layers=layers,
    num_stages=4,          # 4 张 GPU，各负责 6 层
    partition_method='uniform',  # 均匀切分
    loss_fn=cross_entropy_loss
)

# DeepSpeed 引擎启动
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,  # 等价 micro-batch 数
        "pipeline": {
            "activation_checkpoint_interval": 1
        }
    }
)

# 训练（引擎内部处理流水线调度）
for batch in dataloader:
    loss = engine.train_batch(data_iter=iter([batch]))
```

---

## 四、张量并行 vs 流水线并行

### 4.1 特性对比

| 维度 | 张量并行 (TP) | 流水线并行 (PP) |
|------|-------------|---------------|
| 切分粒度 | 层内（矩阵切分） | 层间（层切分） |
| 通信频率 | 每层前向/反向都通信 | 每个 stage 边界通信 |
| 通信量 | 激活值大小 × 2 | 中间激活值 × micro-batch 数 |
| 显存节省 | 参数按 TP 等比减少 | 参数按 PP 等比减少 |
| 气泡浪费 | 无气泡 | 有气泡（1F1B 可减少） |
| 适用网络 | NVLink 高速互联（带宽要求高） | PCIe 或 IB 也可接受 |
| 实现复杂度 | 高（需修改模型） | 中（模型按层划分即可） |

### 4.2 性能对比数据

```
GPT-3 规模模型（175B），96 × A100 80GB，比较不同并行策略：

配置          | 吞吐量 (TFLOPS/GPU) | 显存利用率
─────────────────────────────────────────────────
纯 DP         | 无法运行（单卡 OOM）
TP=8, PP=1   | 148                | 72%
TP=4, PP=2   | 159                | 68%
TP=1, PP=8   | 131                | 65%
TP=4, PP=4 (最优 3D) | 163       | 71%
```

### 4.3 3D 并行（DP + TP + PP）

```
示例：96 GPU = DP(3) × TP(4) × PP(8)

3D 并行拓扑：
  8 个流水线阶段（每阶段 12 张 GPU）
  每个阶段内 4 路张量并行
  3 路数据并行（3份数据复制）

通信层级：
  TP: NVLink（同节点内，带宽最高）
  PP: Infiniband（节点间，低延迟）
  DP: Infiniband（节点间，高带宽）
```

---

## 五、实践建议

```
选择指南：

1. 同节点内 GPU（NVLink）→ 优先张量并行（带宽够）
2. 跨节点 GPU（Infiniband/以太网）→ 流水线并行（通信量小）
3. 超大规模（100B+）→ 3D 混合并行

经验公式：
  TP 维度 ≤ 节点内 GPU 数（NVLink 带宽）
  PP 维度取决于流水线 bubble 可接受程度
  DP 维度 = 总 GPU / (TP × PP)
```

---

## 小结

- **张量并行**：切矩阵，层内并行，通信频繁（需要 NVLink），无气泡
- **流水线并行**：切层，层间并行，通信少，有气泡（micro-batch + 1F1B 调度可缓解）
- **最优方案**：TP × PP × DP 三维组合，TP 用 NVLink，PP 跨节点
- 实际工程中显存和通信带宽的约束共同决定最优并行策略
