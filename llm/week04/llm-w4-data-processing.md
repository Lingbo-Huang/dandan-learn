---
layout: default
title: "D1 · 预训练数据处理"
render_with_liquid: false
---

# D1 · 预训练数据处理

> **核心问题**：GPT-4 的 trillion 级别 token 从哪里来？怎么清洗？怎么配比？

---

## 一、预训练数据全景

大模型的能力上限由数据质量决定。LLaMA-3 使用了 15T tokens，Qwen2.5 使用了 18T tokens。这些数据来自：

| 数据源 | 比例（典型） | 特点 |
|--------|------------|------|
| Common Crawl | 40-70% | 覆盖广，但噪声大 |
| GitHub / 代码 | 10-20% | 提升推理能力 |
| 书籍/论文 | 10-20% | 高质量长文本 |
| Wikipedia | 3-5% | 结构化知识 |
| 对话数据 | 1-5% | 提升对话能力 |

### 数据配比的重要性

Llama-2 实验表明：代码数据从 4.5% 提升到 65% 后，数学推理分数提升了 30%。配比是影响模型能力的关键超参。

---

## 二、数据清洗流水线

```
原始爬取 → 语言识别 → 质量过滤 → 去重 → PII过滤 → Tokenize
```

### 2.1 质量过滤

```python
class QualityFilter:
    """预训练数据质量过滤"""
    
    def __init__(self):
        self.min_length = 100        # 最少字符数
        self.max_length = 100000     # 最多字符数
        self.min_word_ratio = 0.7    # 非字母字符比例阈值
        self.max_symbol_ratio = 0.1  # 特殊符号比例上限
    
    def is_valid(self, text: str) -> tuple[bool, str]:
        """返回 (是否通过, 过滤原因)"""
        # 1. 长度检查
        if len(text) < self.min_length:
            return False, "too_short"
        if len(text) > self.max_length:
            return False, "too_long"
        
        # 2. 单词比例（拦截乱码）
        words = text.split()
        if len(words) == 0:
            return False, "no_words"
        
        # 3. 符号比例（拦截 HTML/代码混杂页面）
        symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if symbol_count / len(text) > self.max_symbol_ratio:
            return False, "too_many_symbols"
        
        # 4. 重复行检查
        lines = text.split('\n')
        unique_lines = set(lines)
        if len(unique_lines) / max(len(lines), 1) < 0.5:
            return False, "too_many_repeated_lines"
        
        return True, "pass"

# 使用示例
qf = QualityFilter()
texts = [
    "This is a normal paragraph about machine learning...",
    "a",  # 太短
    "!!!@@@###$$$%%%^^^&&&***" * 100,  # 符号过多
]

for text in texts:
    valid, reason = qf.is_valid(text)
    print(f"Valid: {valid}, Reason: {reason}")
```

### 2.2 精确去重（MinHash LSH）

文本去重是预训练数据处理最重要的步骤之一。CC-Net 发现 CommonCrawl 中 30%+ 的内容是重复的。

```python
import hashlib
from dataclasses import dataclass
from typing import List, Set

def get_ngrams(text: str, n: int = 13) -> Set[str]:
    """提取字符级 n-gram"""
    words = text.lower().split()
    return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

class MinHashDeduplicator:
    """MinHash LSH 去重器（简化实现）"""
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.seen_hashes = {}
    
    def _minhash(self, text: str) -> List[int]:
        """计算 MinHash 签名"""
        ngrams = get_ngrams(text)
        if not ngrams:
            return [0] * self.num_perm
        
        signatures = []
        for i in range(self.num_perm):
            min_hash = float('inf')
            for gram in ngrams:
                h = int(hashlib.md5(f"{i}:{gram}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signatures.append(min_hash)
        return signatures
    
    def _jaccard_estimate(self, sig1: List[int], sig2: List[int]) -> float:
        """用 MinHash 估算 Jaccard 相似度"""
        matches = sum(a == b for a, b in zip(sig1, sig2))
        return matches / len(sig1)
    
    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """判断文本是否重复"""
        sig = self._minhash(text)
        
        # 简化：只与已见签名比较
        for seen_id, seen_sig in self.seen_hashes.items():
            similarity = self._jaccard_estimate(sig, seen_sig)
            if similarity >= self.threshold:
                return True  # 发现重复
        
        self.seen_hashes[doc_id] = sig
        return False


# 实际生产中使用 datasketch 库
# from datasketch import MinHash, MinHashLSH
# 
# lsh = MinHashLSH(threshold=0.8, num_perm=128)
# for doc_id, text in documents:
#     m = MinHash(num_perm=128)
#     for ngram in get_ngrams(text):
#         m.update(ngram.encode('utf8'))
#     result = lsh.query(m)
#     if not result:  # 不重复
#         lsh.insert(doc_id, m)
```

### 2.3 数据流水线完整实现

```python
from pathlib import Path
import json

class PretrainDataPipeline:
    """完整预训练数据处理流水线"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_filter = QualityFilter()
        self.deduplicator = MinHashDeduplicator(threshold=0.8)
        
        self.stats = {
            'total': 0, 'passed_quality': 0,
            'passed_dedup': 0, 'written': 0
        }
    
    def process_file(self, input_path: str) -> None:
        """处理单个 JSONL 文件"""
        with open(input_path) as f_in:
            out_path = self.output_dir / Path(input_path).name
            with open(out_path, 'w') as f_out:
                for line in f_in:
                    self.stats['total'] += 1
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    
                    # Step 1: 质量过滤
                    valid, reason = self.quality_filter.is_valid(text)
                    if not valid:
                        continue
                    self.stats['passed_quality'] += 1
                    
                    # Step 2: 去重
                    if self.deduplicator.is_duplicate(text, doc.get('id', '')):
                        continue
                    self.stats['passed_dedup'] += 1
                    
                    # Step 3: 写出
                    f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    self.stats['written'] += 1
    
    def report(self) -> None:
        s = self.stats
        print(f"Total: {s['total']:,}")
        print(f"After quality filter: {s['passed_quality']:,} "
              f"({100*s['passed_quality']/max(s['total'],1):.1f}%)")
        print(f"After dedup: {s['passed_dedup']:,} "
              f"({100*s['passed_dedup']/max(s['total'],1):.1f}%)")
```

---

## 三、数据配比策略

### DoReMi 方法（自动学习最优配比）

```python
# 伪代码：DoReMi 配比优化
# 1. 训练一个小的参照模型（uniform sampling）
# 2. 计算各域的 excess loss（当前域损失 - 参照模型损失）
# 3. 根据 excess loss 更新各域权重
# 4. 用新权重重采样，训练真正的模型

def doremi_weight_update(domain_losses, ref_losses, domain_weights, alpha=1.0):
    """
    domain_losses: 当前步各域的 loss
    ref_losses: 参照模型各域的 loss
    domain_weights: 当前各域权重
    """
    import numpy as np
    excess_losses = np.maximum(domain_losses - ref_losses, 0)
    # Hedge 算法更新
    log_weights = np.log(domain_weights) + alpha * excess_losses
    log_weights -= np.max(log_weights)  # 数值稳定
    new_weights = np.exp(log_weights)
    new_weights /= new_weights.sum()
    return new_weights
```

---

## 四、实战：构建中文预训练数据集

```python
"""
从 Common Crawl 中文子集构建预训练数据
工具链：
  - CCNet (Meta) - CC 处理工具
  - langdetect - 语言识别
  - jieba - 中文分词（用于质量检测）
"""

import langdetect
from langdetect import detect, LangDetectException

def is_chinese(text: str, threshold: float = 0.8) -> bool:
    """判断文本是否主要是中文"""
    try:
        lang = detect(text)
        return lang in ('zh-cn', 'zh-tw', 'zh')
    except LangDetectException:
        return False

def chinese_quality_score(text: str) -> float:
    """中文文本质量评分 (0-1)"""
    if not text:
        return 0.0
    
    # 中文字符比例
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    chinese_ratio = chinese_chars / len(text)
    
    # 标点符号比例（过高说明格式异常）
    punct_chars = sum(1 for c in text if c in '。，、；：？！""''（）【】')
    punct_ratio = punct_chars / len(text)
    
    # 综合评分
    score = chinese_ratio * 0.7 + (0.1 - min(punct_ratio, 0.1)) * 3
    return min(1.0, score)

# 处理示例
texts = [
    "人工智能技术正在快速发展，大模型的出现改变了整个行业的格局。",
    "AI technology is developing rapidly.",
    "。。。！！！###@@@",
]

for text in texts:
    print(f"Chinese: {is_chinese(text)}, Score: {chinese_quality_score(text):.2f}")
    print(f"  Text: {text[:50]}")
```

---

## 五、面试题精讲

**Q: 为什么预训练数据去重这么重要？**

A: 重复数据会导致模型"记忆"而非"学习"。实验（Lee et al. 2022）表明，去重后模型困惑度下降约 10%，且减少了数据记忆（隐私风险）。此外，重复数据浪费了有限的训练预算。

**Q: 如何估算需要多少训练 token？**

A: 根据 Chinchilla Scaling Law：最优 token 数 ≈ 20 × 参数量。一个 7B 模型最优约需 140B tokens，但实际（LLaMA-2）用了 2T tokens（计算过量但推理效率更高）。

**Q: 数据集中代码的比例为什么重要？**

A: 代码数据结构严谨、逻辑清晰，能显著提升模型的推理能力。Code Llama 和 StarCoder 的实验均证明，在基础语言模型继续在代码上训练，数学推理能力也会提升（涌现效应）。

---

## 小结

| 步骤 | 工具 | 关键点 |
|------|------|--------|
| 爬取 | Common Crawl, C4 | 覆盖度 vs 质量 |
| 语言识别 | langdetect, fastText | 阈值设置 |
| 质量过滤 | 规则 + 分类器 | 长度、符号比、重复行 |
| 去重 | MinHash LSH | 相似度阈值 0.7-0.8 |
| 配比 | DoReMi, 手工 | 代码/书籍占比关键 |
