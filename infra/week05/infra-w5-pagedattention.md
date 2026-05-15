---
layout: default
title: "D2 · PagedAttention：虚拟内存思想在 GPU 上的应用"
render_with_liquid: false
---

# D2 · PagedAttention：虚拟内存思想在 GPU 上的应用

## 操作系统虚拟内存回顾

操作系统用虚拟内存解决物理内存碎片问题：
- 物理内存被切分成固定大小的**页（Page）**（通常 4KB）
- 每个进程有虚拟地址空间，通过**页表（Page Table）**映射到物理页
- 进程看到的是连续虚拟地址，物理页可以不连续
- **按需分配**：只在实际访问时才分配物理页

PagedAttention 将这个思想搬到 GPU KV Cache 管理！

## PagedAttention 核心概念

**Block**（块）：KV Cache 的基本分配单位
- 固定大小，例如 16 tokens 的 K+V
- 类比操作系统的物理页

**Block Table**（块表）：每个序列的块映射表
- 记录序列的第 0-15 个 token 在哪个物理块
- 类比操作系统的页表

```
传统 KV Cache（连续分配）：
  请求 A: [K0,V0 | K1,V1 | K2,V2 | ... | K2999,V2999 | 空 空 空...空]
           ← 3000 tokens 已用 ──────────────────────────────────────────────────→← 1096 slots 浪费 →
  
PagedAttention（分页分配）：
  请求 A 的块表：[Block#3, Block#7, Block#12, ...]
               ↓         ↓          ↓
  物理块: Block#3:[K0..K15,V0..V15]  Block#7:[K16..K31,V16..V31]  Block#12:[...]
  
  不同请求可以共享物理块！（用于 Prefix Sharing / RadixAttention）
```

## 内存利用率提升

vLLM 论文数据：

| 方法 | 内存利用率 | 最大并发 | 吞吐量 |
|------|-----------|---------|-------|
| 朴素实现 | ~24% | 33 | 1× |
| PagedAttention | ~96% | ~150+ | 2-4× |

**关键**：只有最后一个 block 存在内部碎片（平均 0.5 blocks），无外部碎片。

## PagedAttention 的数据结构

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch

@dataclass
class KVBlock:
    """物理 KV Cache 块"""
    block_id: int
    ref_count: int = 0      # 引用计数（用于 Copy-on-Write）
    
class BlockTable:
    """单个序列的块映射表"""
    def __init__(self):
        self.physical_blocks: List[int] = []  # 物理块 ID 列表
    
    def get_block_id(self, token_idx: int, block_size: int) -> int:
        """获取第 token_idx 个 token 所在的物理块 ID"""
        block_idx = token_idx // block_size
        return self.physical_blocks[block_idx]
    
    def get_slot(self, token_idx: int, block_size: int) -> int:
        """获取 token 在物理块内的槽位（绝对偏移）"""
        block_id = self.get_block_id(token_idx, block_size)
        slot_in_block = token_idx % block_size
        return block_id * block_size + slot_in_block

class KVCacheManager:
    """
    PagedAttention 的 KV Cache 管理器
    类似操作系统的物理内存管理器
    """
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, 
                 num_kv_heads: int, head_dim: int, device: str = 'cuda'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # 物理 KV Cache 存储
        # [num_blocks, num_layers, 2(K/V), num_kv_heads, block_size, head_dim]
        self.k_cache = torch.zeros(
            num_blocks, num_layers, num_kv_heads, block_size, head_dim,
            device=device, dtype=torch.float16
        )
        self.v_cache = torch.zeros(
            num_blocks, num_layers, num_kv_heads, block_size, head_dim,
            device=device, dtype=torch.float16
        )
        
        # 空闲块列表
        self.free_blocks: List[int] = list(range(num_blocks))
        self.blocks: List[KVBlock] = [KVBlock(i) for i in range(num_blocks)]
        
        # 序列块表
        self.block_tables: Dict[int, BlockTable] = {}  # seq_id → BlockTable
    
    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """为序列分配初始块"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            return False  # OOM
        
        table = BlockTable()
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.blocks[block_id].ref_count = 1
            table.physical_blocks.append(block_id)
        
        self.block_tables[seq_id] = table
        return True
    
    def append_slot(self, seq_id: int, new_token_count: int = 1) -> bool:
        """追加新 token 的槽位（可能需要新分配块）"""
        table = self.block_tables[seq_id]
        current_tokens = len(table.physical_blocks) * self.block_size
        # 检查最后一个块是否有空间
        last_block_used = current_tokens % self.block_size
        
        if last_block_used == 0 and len(table.physical_blocks) > 0:
            # 上一个块已满，需要新块
            if not self.free_blocks:
                return False
            block_id = self.free_blocks.pop()
            self.blocks[block_id].ref_count = 1
            table.physical_blocks.append(block_id)
        
        return True
    
    def get_block_table_tensor(self, seq_id: int) -> torch.Tensor:
        """返回块表的 GPU 张量（用于 CUDA Kernel）"""
        table = self.block_tables[seq_id]
        return torch.tensor(table.physical_blocks, dtype=torch.int32, device='cuda')
    
    def free(self, seq_id: int):
        """释放序列的所有块"""
        if seq_id not in self.block_tables:
            return
        table = self.block_tables.pop(seq_id)
        for block_id in table.physical_blocks:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.free_blocks.append(block_id)
```

## PagedAttention CUDA Kernel

核心挑战：如何在不连续的物理块上做 Attention？

```cuda
// paged_attention_kernel.cu
// 关键思想：通过 block_table 将 logical 地址翻译为 physical 地址

__global__ void paged_attention_v1(
    float* __restrict__ out,          // [batch, heads, head_dim]
    const float* __restrict__ q,      // [batch, heads, head_dim]
    const float* __restrict__ k_cache, // [num_blocks, num_kv_heads, block_size, head_dim]
    const float* __restrict__ v_cache, // [num_blocks, num_kv_heads, block_size, head_dim]
    const int* __restrict__ block_tables, // [batch, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [batch] 每个序列的当前长度
    int num_kv_heads, int head_dim, int block_size, int max_num_blocks
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    int kv_head_idx = head_idx / (gridDim.x / num_kv_heads);  // GQA 映射
    
    // 加载 Query
    float q_val[128];  // 假设 head_dim <= 128
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_val[d] = q[batch_idx * gridDim.x * head_dim + head_idx * head_dim + d];
    }
    __syncthreads();
    
    int seq_len = seq_lens[batch_idx];
    
    float m = -INFINITY;  // online softmax: max
    float l = 0.0f;       // online softmax: sum of exp
    float acc[128] = {0.0f};  // output accumulator
    
    // 遍历所有历史 token（通过 block table）
    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        int block_idx   = token_idx / block_size;
        int slot_offset = token_idx % block_size;
        
        // 通过 block table 找到物理块
        int physical_block = block_tables[batch_idx * max_num_blocks + block_idx];
        
        // 加载 K
        float k_val[128];
        for (int d = 0; d < head_dim; d++) {
            k_val[d] = k_cache[
                physical_block * num_kv_heads * block_size * head_dim
                + kv_head_idx * block_size * head_dim
                + slot_offset * head_dim
                + d
            ];
        }
        
        // 计算 attention score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_val[d] * k_val[d];
        }
        score /= sqrtf((float)head_dim);
        
        // Online softmax 更新
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        l = alpha * l + expf(score - m_new);
        
        // 加载 V 并更新输出
        for (int d = 0; d < head_dim; d++) {
            float v_val = v_cache[
                physical_block * num_kv_heads * block_size * head_dim
                + kv_head_idx * block_size * head_dim
                + slot_offset * head_dim
                + d
            ];
            acc[d] = alpha * acc[d] + expf(score - m_new) * v_val;
        }
        m = m_new;
    }
    
    // 归一化并写入输出
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[batch_idx * gridDim.x * head_dim + head_idx * head_dim + d] = acc[d] / l;
    }
}
```

## Prefix Sharing（前缀共享）

PagedAttention 的一个重要扩展：**共享 KV Cache**。

场景：多个请求有相同的系统 Prompt（system prompt）：

```
请求 A: [系统 Prompt 1000 tokens] + [用户问题 A 100 tokens]
请求 B: [系统 Prompt 1000 tokens] + [用户问题 B 150 tokens]

传统：两个请求各占 1100, 1150 tokens 的 KV Cache
PagedAttention：系统 Prompt 的 KV Cache 共享！

节省：1000 tokens × 2 bytes × 2(KV) × num_layers × kv_heads × d ≈ 节省 50%+
```

实现：使用引用计数（ref_count > 1 时 Copy-on-Write）

```python
def fork_sequence(self, parent_seq_id: int, child_seq_id: int):
    """
    序列 fork（用于 beam search 或 prefix sharing）
    共享所有父序列的块，使用 Copy-on-Write
    """
    parent_table = self.block_tables[parent_seq_id]
    child_table = BlockTable()
    
    for block_id in parent_table.physical_blocks:
        # 增加引用计数
        self.blocks[block_id].ref_count += 1
        child_table.physical_blocks.append(block_id)
    
    self.block_tables[child_seq_id] = child_table

def copy_on_write(self, seq_id: int) -> int:
    """
    写时复制：当要修改一个共享块时，先复制一份
    """
    table = self.block_tables[seq_id]
    last_block_id = table.physical_blocks[-1]
    
    if self.blocks[last_block_id].ref_count > 1:
        # 块被共享，需要 CoW
        new_block_id = self.free_blocks.pop()
        # 复制数据
        self.k_cache[new_block_id] = self.k_cache[last_block_id].clone()
        self.v_cache[new_block_id] = self.v_cache[last_block_id].clone()
        # 更新引用
        self.blocks[last_block_id].ref_count -= 1
        self.blocks[new_block_id].ref_count = 1
        table.physical_blocks[-1] = new_block_id
        return new_block_id
    return last_block_id
```

## 实战：使用 vLLM

```python
from vllm import LLM, SamplingParams

# vLLM 使用 PagedAttention 管理 KV Cache
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # 单卡
    gpu_memory_utilization=0.9,  # 预留 10% 给 PyTorch overhead
    max_num_seqs=256,           # 最大并发序列数
    max_num_batched_tokens=32768,  # 每次批处理的最大 token 数
    block_size=16,              # 每个 KV Cache 块的 token 数
    enable_prefix_caching=True, # 启用前缀共享
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

prompts = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What is the capital of France?",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:100]}")
    print()
```

## 本节小结

| 概念 | 传统 KV Cache | PagedAttention |
|------|-------------|----------------|
| 内存分配 | 连续，预留最大 | 分块，按需分配 |
| 碎片化 | 严重（~76%浪费）| 几乎没有（<4%）|
| 前缀共享 | 不支持 | 支持（CoW）|
| 最大并发 | 低 | 高（~4×）|
| 实现复杂度 | 简单 | 复杂（需要块表） |

## 面试题

**Q: PagedAttention 和操作系统虚拟内存的类比是什么？**

A: 类比非常直接：OS 的物理内存页 = PagedAttention 的 KV Cache Block；OS 的页表 = PagedAttention 的 Block Table；OS 的逻辑地址 = 序列中的 token 位置；OS 的写时复制（CoW）= PagedAttention 的 prefix sharing CoW。两者都解决了碎片化问题：OS 允许进程使用不连续的物理内存，PagedAttention 允许序列使用不连续的 GPU 显存块，大幅提升内存利用率。
