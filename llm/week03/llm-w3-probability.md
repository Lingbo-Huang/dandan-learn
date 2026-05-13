---
layout: default
title: "D3 · 概率论：贝叶斯定理与条件概率"
---

# D3 · 概率论基础

> **LLM Week 3**  
> LLM 的本质是概率模型：$P(\text{下一个词} | \text{已有文本})$。理解概率论，才能理解大模型在"想什么"。

---

## 一、大模型的概率本质

GPT 等自回归语言模型做的事情很简单：

$$P(\text{"今天天气很好"}) = P(\text{今}) \cdot P(\text{天}|\text{今}) \cdot P(\text{天}|\text{今天}) \cdot P(\text{气}|\text{今天天}) \cdots$$

这就是**条件概率**的链式法则（Product Rule）。

训练大模型 = 调整参数，让模型分配给真实文本的概率最大。

---

## 二、概率基础

**随机变量**：结果不确定的量。

- 离散型：投骰子 $X \in \{1,2,3,4,5,6\}$
- 连续型：身高 $X \in [0, \infty)$

**概率分布**：描述各结果的可能性。

$$P(X=x) \geq 0, \quad \sum_x P(X=x) = 1$$

**联合概率**：

$$P(X=x, Y=y) = P(\text{同时发生 } X=x \text{ 和 } Y=y)$$

**边际概率**（把另一个变量求和掉）：

$$P(X=x) = \sum_y P(X=x, Y=y)$$

---

## 三、条件概率

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

读作："在 $B$ 已经发生的条件下，$A$ 发生的概率"。

**语言模型示例**：

```
P("好"|"天气很") = P("天气很好") / P("天气很")
```

---

## 四、贝叶斯定理

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

各部分的名称：
- $P(A)$：**先验（Prior）**，不考虑 $B$ 时 $A$ 的概率
- $P(B|A)$：**似然（Likelihood）**，$A$ 发生时 $B$ 的概率
- $P(A|B)$：**后验（Posterior）**，看到 $B$ 后对 $A$ 的更新概率
- $P(B)$：**归一化常数（Evidence）**

**例子**：医疗诊断

```
疾病 D 的发病率（先验）：P(D) = 0.001
检测准确率（似然）：P(+|D) = 0.99，P(+|¬D) = 0.01

检测阳性（+）的条件下，真的患病的概率？

P(D|+) = P(+|D) × P(D) / P(+)
       = 0.99 × 0.001 / (0.99×0.001 + 0.01×0.999)
       ≈ 0.09   ← 只有 9%！
```

> 这就是"低发病率疾病，即使高准确率检测也会有大量假阳性"的数学来源。

---

## 五、Python 验证贝叶斯定理

```python
import numpy as np

# 参数
p_disease = 0.001           # 先验：发病率
p_pos_given_disease = 0.99  # 真阳率（敏感度）
p_pos_given_healthy = 0.01  # 假阳率（1-特异度）

# 后验：贝叶斯公式
p_pos = (p_pos_given_disease * p_disease + 
         p_pos_given_healthy * (1 - p_disease))

p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"检测阳性后患病概率: {p_disease_given_pos:.4f} = {p_disease_given_pos*100:.2f}%")

# 蒙特卡洛模拟验证
np.random.seed(42)
N = 1_000_000  # 100万人

true_positive = 0
false_positive = 0

for _ in range(N):
    is_sick = np.random.rand() < p_disease
    if is_sick:
        test_positive = np.random.rand() < p_pos_given_disease
    else:
        test_positive = np.random.rand() < p_pos_given_healthy
    
    if test_positive:
        if is_sick:
            true_positive += 1
        else:
            false_positive += 1

total_positive = true_positive + false_positive
empirical = true_positive / total_positive
print(f"蒙特卡洛验证: {empirical:.4f}（应接近 {p_disease_given_pos:.4f}）")
```

---

## 六、大模型中的条件概率链

语言模型的完整展开：

```python
def language_model_probability(tokens, lm):
    """
    计算一段文本的概率
    P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ...
    """
    log_prob = 0.0
    context = []
    
    for token in tokens:
        # 模型输出的是每个词的概率分布（softmax 输出）
        probs = lm.predict(context)
        log_prob += np.log(probs[token] + 1e-10)
        context.append(token)
    
    return np.exp(log_prob)  # 转回概率

# Perplexity：评估语言模型质量的指标
# 直觉：模型平均对每个词"有多困惑"
# Perplexity = exp(- 1/N × Σ log P(wᵢ|w₁...wᵢ₋₁))
# 越低越好：GPT-4 在某些基准上 Perplexity < 5
```

---

## 七、独立性与相关性

**条件独立**：

$$P(A|B, C) = P(A|C) \quad \Rightarrow \quad A \text{ 与 } B \text{ 在给定 } C \text{ 的条件下独立}$$

这在朴素贝叶斯中非常重要（假设特征之间条件独立）。

**相关性 ≠ 因果性**：

```python
# 冰淇淋销量 vs 溺水人数：高度相关
# 但原因是共同的混淆变量：气温

# 大模型也会学到虚假相关！
# 比如训练数据里"某人名"经常出现在"成功"旁边
# 并不意味着有因果关系
```

---

## 今天的关键认识

1. **大模型本质 = 条件概率预测**：$P(\text{下一个词}|\text{上文})$
2. **贝叶斯定理**：用新证据更新信念，先验 × 似然 ∝ 后验
3. **低基率问题**：即使精确度很高，罕见事件的预测也容易有大量假阳性
4. **Perplexity**：语言模型质量的标准度量，越低越好

---

## 明天预告

D4：**信息论**——熵、交叉熵、KL 散度，这些是损失函数的数学基础。
