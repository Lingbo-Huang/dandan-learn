---
layout: default
title: "D3 · vLLM 与 PagedAttention"
render_with_liquid: false
---

# D3 · vLLM 与 PagedAttention

> **问题**：传统 LLM 服务的 KV Cache 使用静态预分配，导致严重的内存碎片（约 60-80% 内存浪费）。vLLM 的 PagedAttention 借鉴操作系统的虚拟内存管理，解决了这个问题。

---

## 一、传统 KV Cache 的问题

### 内存浪费的根源

```
传统方式：每个请求预分配 max_seq_len 的 KV Cache 空间

请求 A（实际用 200 tokens）: [USED: 200] [WASTED: 1824]  ←── 预分配 2048
请求 B（实际用 500 tokens）: [USED: 500] [WASTED: 1548]
请求 C（实际用 50 tokens） : [USED: 50 ] [WASTED: 1998]

平均内存利用率：~20%
```

### PagedAttention 的解法

```
PagedAttention：将 KV Cache 分成固定大小的"页"（block）

Block Pool（物理内存）:
[Block 0] [Block 1] [Block 2] [Block 3] [Block 4] ...

请求 A 的 Block Table:  [0, 3, 7]  → 按需分配，用完再要
请求 B 的 Block Table:  [1, 4]     → 不同请求不连续但无碎片
请求 C 的 Block Table:  [2]        → 只占一个 block

内存利用率：~95%
```

---

## 二、PagedAttention 核心机制

```python
"""
PagedAttention 的关键数据结构和算法

类比操作系统虚拟内存：
- 物理页（Physical Block）← 实际的 GPU 内存块
- 页表（Block Table）← 逻辑序列到物理块的映射
- 按需分页（On-demand Paging）← 需要时才分配
- 写时复制（Copy-on-Write）← 前缀缓存共享
"""

from typing import Optional
import torch

BLOCK_SIZE = 16  # 每个 block 存 16 个 token 的 KV

class BlockAllocator:
    """物理块分配器"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))  # 空闲块列表
        self.ref_counts = [0] * num_blocks           # 引用计数（用于 prefix caching）
    
    def allocate(self) -> Optional[int]:
        """分配一个空闲块"""
        if not self.free_blocks:
            return None  # OOM
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id
    
    def free(self, block_id: int) -> None:
        """释放块"""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)
    
    def fork(self, block_id: int) -> int:
        """写时复制（用于 beam search 等需要复制 KV 的场景）"""
        self.ref_counts[block_id] += 1
        return block_id  # 返回同一个块（引用计数 +1）


class SequenceKVCache:
    """单个序列的 KV Cache 管理"""
    
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_table: list[int] = []  # 逻辑块 → 物理块 ID
        self.num_tokens: int = 0
    
    @property
    def num_blocks(self) -> int:
        return len(self.block_table)
    
    def ensure_capacity(self, num_tokens: int) -> bool:
        """确保有足够的块存储 num_tokens 个 token 的 KV"""
        needed_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        while len(self.block_table) < needed_blocks:
            new_block = self.allocator.allocate()
            if new_block is None:
                return False  # 内存不足
            self.block_table.append(new_block)
        
        self.num_tokens = num_tokens
        return True
    
    def get_physical_block_id(self, logical_block: int) -> int:
        """逻辑块号 → 物理块 ID（通过 block table）"""
        return self.block_table[logical_block]
    
    def release(self) -> None:
        """释放所有块"""
        for block_id in self.block_table:
            self.allocator.free(block_id)
        self.block_table.clear()
        self.num_tokens = 0


class PagedKVCache:
    """分页 KV Cache（简化实现）"""
    
    def __init__(
        self,
        num_blocks: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        device: str = 'cuda',
    ):
        self.block_size = BLOCK_SIZE
        self.allocator = BlockAllocator(num_blocks)
        
        # 物理 KV Cache 存储（预分配全部内存）
        # key_cache[layer]: [num_blocks, n_heads, block_size, d_head]
        self.key_cache = torch.zeros(
            n_layers, num_blocks, n_heads, BLOCK_SIZE, d_head,
            device=device, dtype=torch.float16
        )
        self.val_cache = torch.zeros(
            n_layers, num_blocks, n_heads, BLOCK_SIZE, d_head,
            device=device, dtype=torch.float16
        )
    
    def write_kv(
        self,
        layer: int,
        seq_cache: SequenceKVCache,
        position: int,
        k: torch.Tensor,  # [n_heads, d_head]
        v: torch.Tensor,
    ) -> None:
        """写入一个 token 的 KV"""
        block_idx = position // BLOCK_SIZE
        block_offset = position % BLOCK_SIZE
        physical_block = seq_cache.get_physical_block_id(block_idx)
        
        self.key_cache[layer, physical_block, :, block_offset, :] = k
        self.val_cache[layer, physical_block, :, block_offset, :] = v
    
    def gather_kv(
        self,
        layer: int,
        seq_cache: SequenceKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """收集序列的所有 KV（用于 Attention 计算）"""
        all_keys = []
        all_vals = []
        
        for i, physical_block in enumerate(seq_cache.block_table):
            # 最后一个 block 可能未满
            if i == len(seq_cache.block_table) - 1:
                valid_tokens = seq_cache.num_tokens % BLOCK_SIZE
                if valid_tokens == 0:
                    valid_tokens = BLOCK_SIZE
            else:
                valid_tokens = BLOCK_SIZE
            
            all_keys.append(self.key_cache[layer, physical_block, :, :valid_tokens, :])
            all_vals.append(self.val_cache[layer, physical_block, :, :valid_tokens, :])
        
        return torch.cat(all_keys, dim=1), torch.cat(all_vals, dim=1)
```

---

## 三、连续批处理（Continuous Batching）

```python
"""
传统批处理 vs 连续批处理

传统：等所有请求都完成，才处理下一批
  [Req A: 生成中......]
  [Req B: 生成中..]
  等待 Req A 完成 → [Req C 才能开始]

连续批处理（iteration-level scheduling）：
  每个 step 后，已完成的请求退出，新请求立即加入
  Step 1: [A B C D]
  Step 2: [A B C D E]  ← E 新加入
  Step 3: [A _ C D E]  ← B 完成退出
  ...

吞吐量提升：2-4x
"""

class ContinuousBatchingScheduler:
    """连续批处理调度器（简化实现）"""
    
    def __init__(self, max_batch_size: int, max_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.running: list = []   # 当前运行中的序列
        self.waiting: list = []   # 等待队列
    
    def schedule(self) -> list:
        """决定本 step 处理哪些序列"""
        # 优先处理已经在运行的序列
        schedulable = list(self.running)
        
        # 尝试添加新序列
        total_tokens = sum(len(seq) for seq in schedulable)
        for new_seq in list(self.waiting):
            if (len(schedulable) < self.max_batch_size and 
                total_tokens + len(new_seq) <= self.max_tokens):
                schedulable.append(new_seq)
                self.waiting.remove(new_seq)
                total_tokens += len(new_seq)
        
        return schedulable
    
    def finish(self, seq) -> None:
        """序列生成完成"""
        if seq in self.running:
            self.running.remove(seq)
```

---

## 四、vLLM 实战

```python
from vllm import LLM, SamplingParams

# 启动 vLLM 服务
def demo_vllm():
    # 加载模型
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.90,  # 使用 90% GPU 显存作为 KV Cache
        max_model_len=4096,
        tensor_parallel_size=1,       # 单卡
        # tensor_parallel_size=4,     # 4 卡张量并行
        quantization="awq",           # 可选量化
    )
    
    # 批量推理（自动连续批处理）
    prompts = [
        "什么是注意力机制？",
        "解释 Transformer 架构",
        "LoRA 微调的原理是什么？",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Response: {output.outputs[0].text[:100]}...")
        print()

# 启动 OpenAI API 兼容服务
# vllm serve Qwen/Qwen2.5-7B-Instruct \
#   --host 0.0.0.0 --port 8000 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 4096

# 客户端调用
import openai

def call_vllm_api():
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-vllm"
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "什么是大语言模型？"}],
        max_tokens=256,
        temperature=0.7,
    )
    return response.choices[0].message.content
```

---

## 五、面试题精讲

**Q: vLLM 的 PagedAttention 和操作系统的虚拟内存有什么相似之处？**

A:
- OS 的物理页 ↔ PagedAttention 的 KV Block
- 页表（Page Table）↔ Block Table
- 按需分页 ↔ 按需分配 KV Block
- Copy-on-Write（fork 共享页）↔ Prefix Caching（共享前缀的 KV Block）
- OOM Killer ↔ vLLM 的 beam search 抢占

**Q: 为什么连续批处理比传统批处理吞吐量高很多？**

A: 传统批处理中，一个批次内最长序列决定等待时间，短序列生成完毕后 GPU 资源空置等待。连续批处理（iteration-level）每个 decode step 后都可以调整批次组成：已完成的序列退出，新请求立即加入，GPU 利用率接近 100%。

---

## 小结

| 特性 | 传统服务 | vLLM |
|------|---------|------|
| KV Cache 分配 | 静态预分配 | 动态分页 |
| 内存利用率 | ~20% | ~95% |
| 批处理 | 同步批 | 连续批 |
| 吞吐量 | 基线 | 2-4x |
| Prefix Caching | 无 | 支持（CoW）|
