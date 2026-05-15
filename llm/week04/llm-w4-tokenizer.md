---
layout: default
title: "D2 · Tokenizer 原理与实战"
render_with_liquid: false
---

# D2 · Tokenizer 原理与实战

> **Tokenizer 是模型与文本世界的接口**——同样一句话，不同的分词方式会给模型"看到"完全不同的东西。

---

## 一、为什么需要 Tokenizer？

神经网络只能处理数字，Tokenizer 将文本转换为 token ID 序列：

```
"Hello, World!" → [15496, 11, 2159, 0]   # GPT-2 BPE
"你好，世界！"   → [2, 2804, 8024, 5]     # 某中文 Tokenizer
```

### 三种分词粒度的对比

| 粒度 | 例子 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | ['H','e','l','l','o'] | 零 OOV | 序列太长 |
| 词级 | ['Hello', 'World'] | 语义完整 | OOV 严重 |
| 子词级(BPE) | ['Hell','o', 'World'] | 均衡，主流方案 | 需要训练 |

---

## 二、BPE (Byte Pair Encoding) 深度解析

### 2.1 训练过程

BPE 从字符开始，反复合并最频繁的相邻对：

```python
from collections import defaultdict
from typing import Dict, List, Tuple

class BPETrainer:
    """BPE Tokenizer 训练器（完整实现）"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则列表
        self.vocab = {}   # 最终词表
    
    def get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """将语料转换为词频字典，词内字符用空格分隔，词末加 </w>"""
        vocab = defaultdict(int)
        for text in corpus:
            for word in text.split():
                # 'hello' -> 'h e l l o </w>'
                chars = ' '.join(list(word)) + ' </w>'
                vocab[chars] += 1
        return dict(vocab)
    
    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple, int]:
        """统计所有相邻字节对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return dict(pairs)
    
    def merge_vocab(self, best_pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """合并词表中最优的字节对"""
        new_vocab = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab
    
    def train(self, corpus: List[str]) -> List[Tuple[str, str]]:
        """训练 BPE，返回合并规则"""
        # 初始词表：字符级
        vocab = self.get_vocab(corpus)
        
        # 收集所有初始字符
        initial_vocab = set()
        for word in vocab:
            for char in word.split():
                initial_vocab.add(char)
        
        print(f"初始词表大小: {len(initial_vocab)}")
        print(f"目标词表大小: {self.vocab_size}")
        
        merges = []
        num_merges = self.vocab_size - len(initial_vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            # 找最高频的字节对
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]
            
            if best_freq < 2:  # 频率过低，停止
                break
            
            # 合并
            vocab = self.merge_vocab(best_pair, vocab)
            merges.append(best_pair)
            
            if i % 100 == 0:
                print(f"  Merge {i}: {best_pair} (freq={best_freq})")
        
        self.merges = merges
        return merges


# 演示训练过程
corpus = [
    "low lower newest widest",
    "low lower low newest widest newer",
    "the quick brown fox jumps over the lazy dog",
    "the fox the fox the fox"
]

trainer = BPETrainer(vocab_size=50)
merges = trainer.train(corpus)
print(f"\n学到的合并规则: {merges[:10]}")
```

### 2.2 编码过程

```python
class BPEEncoder:
    """BPE 编码器（应用学到的合并规则）"""
    
    def __init__(self, merges: List[Tuple[str, str]], vocab: Dict[str, int]):
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab
        self.bpe_cache = {}
    
    def get_pairs(self, word: List[str]) -> set:
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs
    
    def bpe(self, word: str) -> str:
        """对单个词应用 BPE"""
        if word in self.bpe_cache:
            return self.bpe_cache[word]
        
        chars = list(word) + ['</w>']
        
        while True:
            pairs = self.get_pairs(chars)
            if not pairs:
                break
            
            # 找优先级最高（合并编号最小）的对
            valid_pairs = {p: self.merges[p] for p in pairs if p in self.merges}
            if not valid_pairs:
                break
            
            best = min(valid_pairs, key=valid_pairs.get)
            
            # 执行合并
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        result = ' '.join(chars)
        self.bpe_cache[word] = result
        return result
    
    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.split():
            bpe_result = self.bpe(word)
            for token in bpe_result.split():
                tokens.append(self.vocab.get(token, self.vocab.get('<unk>', 0)))
        return tokens
```

---

## 三、WordPiece vs BPE vs Unigram

```python
"""
三种主流子词算法对比

BPE (GPT 系列):
- 自底向上合并，贪心选择最高频对
- 确定性编码（同一字符串结果唯一）

WordPiece (BERT):
- 类似 BPE，但合并标准是最大化语言模型概率
- 子词前缀用 ## 标记，如 "playing" -> ["play", "##ing"]
- score = freq(AB) / (freq(A) * freq(B))

Unigram (SentencePiece/LLaMA):
- 自顶向下，从大词表开始裁剪
- 基于概率模型，支持多种分词结果采样
- 更适合多语言
"""

# 用 tokenizers 库演示
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(corpus_file: str, vocab_size: int = 30000) -> Tokenizer:
    """使用 HuggingFace tokenizers 训练 BPE"""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2
    )
    
    tokenizer.train(files=[corpus_file], trainer=trainer)
    return tokenizer
```

---

## 四、Byte-level BPE（GPT-2/GPT-4 方案）

Byte-level BPE 直接在字节（0-255）上操作，天然支持任意语言：

```python
import tiktoken

# GPT-4 使用的 cl100k_base 编码器
enc = tiktoken.get_encoding("cl100k_base")

# 编码
tokens = enc.encode("Hello, 你好，世界！")
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")

# 解码
text = enc.decode(tokens)
print(f"Decoded: {text}")

# 查看每个 token 对应的文本
for tok_id in tokens:
    token_bytes = enc.decode_single_token_bytes(tok_id)
    print(f"  {tok_id}: {token_bytes}")

# 不同语言的 token 效率对比
texts = {
    "English": "The quick brown fox jumps over the lazy dog",
    "中文": "快速的棕色狐狸跳过了懒惰的狗",
    "日本語": "素早い茶色のキツネが怠け者の犬を飛び越えた",
    "Code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
}

for lang, text in texts.items():
    toks = enc.encode(text)
    chars_per_tok = len(text) / len(toks)
    print(f"{lang}: {len(toks)} tokens, {chars_per_tok:.1f} chars/token")
```

---

## 五、SentencePiece（LLaMA/Qwen/Gemma 使用）

```python
import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='llama_tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,   # 覆盖 99.95% 的字符
    model_type='bpe',            # 也可用 'unigram'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
)

# 使用
sp = spm.SentencePieceProcessor()
sp.load('llama_tokenizer.model')

text = "预训练是大模型能力的基础"
tokens = sp.encode(text, out_type=str)
print(f"Tokens: {tokens}")
ids = sp.encode(text, out_type=int)
print(f"IDs: {ids}")
decoded = sp.decode(ids)
print(f"Decoded: {decoded}")
```

---

## 六、面试题精讲

**Q: BPE 和 WordPiece 的核心区别？**

A: 合并标准不同。BPE 选择**频率最高**的相邻对合并；WordPiece 选择**合并后使语言模型概率最大**的对（即 `log P(AB) - log P(A) - log P(B)` 最大）。WordPiece 更"语言学合理"，但 BPE 更快。

**Q: 中文的 token 效率为什么比英文低？**

A: 中文字符在 Unicode 空间中分散，BPE 需要更多合并步骤才能形成有效中文子词。GPT-4 的 cl100k 编码中，一个中文汉字平均约占 1.5-2 个 token，而英文一个单词只需 1-2 个 token（但英文单词字符更长）。Qwen 专门为中文设计了 tokenizer，中文效率更高。

**Q: 词表大小如何选择？**

A: 典型范围：
- 小模型/单语：16K-32K（GPT-2: 50K, LLaMA: 32K）
- 多语言模型：64K-200K（LLaMA-3: 128K, Qwen: 150K）
- 更大词表减少序列长度（对长文本和代码友好），但增加 embedding 层参数量

---

## 小结

```
字符级 → 词级 → 子词级（BPE）→ Byte-level BPE
                                    ↑
                              现代 LLM 的主流选择

BPE      : 简单高效，GPT 系列
WordPiece: 概率最优，BERT 系列
Unigram  : 多解采样，SentencePiece
```
