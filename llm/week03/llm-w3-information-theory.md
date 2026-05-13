---
layout: default
title: "D4 · 信息论：熵、交叉熵、KL散度"
---

# D4 · 信息论：熵、交叉熵、KL 散度

> **LLM Week 3**  
> 为什么训练语言模型用"交叉熵损失"？今天从信息论出发彻底搞清楚。

---

## 一、信息量：惊喜程度

**直觉**：越不可能发生的事情，发生了越"惊喜"，携带的信息量越大。

$$I(x) = -\log_2 P(x) \quad \text{（单位：bits）}$$

| 事件 | 概率 | 信息量 |
|------|------|--------|
| 抛硬币出现正面 | 0.5 | 1 bit |
| 掷骰子出现 6 | 1/6 | 2.58 bits |
| 太阳明天升起 | ≈1 | ≈0 bits |
| 彩票中头奖 | 10⁻⁷ | 23.25 bits |

---

## 二、熵（Entropy）：平均信息量

$$H(X) = -\sum_x P(x) \log_2 P(x) = \mathbb{E}[-\log P(X)]$$

**直觉**：分布越均匀（越不确定），熵越大；分布越集中，熵越小。

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """计算离散分布的熵（单位：nats，用 ln 计算）"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # 避免 log(0)
    return -np.sum(probs * np.log(probs))

# 示例：两个词的语言模型
print(f"均匀分布 [0.5, 0.5]: {entropy([0.5, 0.5]):.4f} nats")  # 最大熵
print(f"偏斜分布 [0.9, 0.1]: {entropy([0.9, 0.1]):.4f} nats")  # 较小熵
print(f"确定分布 [1.0, 0.0]: {entropy([1.0]):.4f} nats")        # 0（无不确定性）

# 熵 vs 分布偏斜程度
p_values = np.linspace(0.001, 0.999, 100)
entropies = [entropy([p, 1-p]) for p in p_values]

plt.plot(p_values, entropies)
plt.xlabel('P(X=1)')
plt.ylabel('熵 (nats)')
plt.title('二元分布的熵')
plt.axvline(x=0.5, color='r', linestyle='--', label='最大熵点')
plt.legend()
plt.show()
```

---

## 三、交叉熵（Cross-Entropy）：用错误分布编码的代价

$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{KL}(P \| Q)$$

**直觉**：
- 真实分布是 $P$（真实标签）
- 模型预测是 $Q$（模型输出的概率）
- 交叉熵 = 用 $Q$ 来编码 $P$ 分布的数据所需的平均比特数

**关键性质**：
- 当 $Q = P$ 时，$H(P,Q) = H(P)$（最小值）
- 当 $Q \neq P$ 时，$H(P,Q) > H(P)$（总有额外代价）
- **最小化交叉熵 = 让模型的概率分布逼近真实分布**

```python
def cross_entropy(p_true, q_pred):
    """交叉熵损失"""
    p_true = np.array(p_true)
    q_pred = np.array(q_pred) + 1e-15  # 避免 log(0)
    return -np.sum(p_true * np.log(q_pred))

# 分类示例（3个类别）
p_true = [0, 1, 0]          # 真实标签：类别 1（one-hot）
q_good = [0.05, 0.90, 0.05]  # 好的预测：高置信度预测正确类
q_bad  = [0.33, 0.34, 0.33]  # 差的预测：几乎均匀

print(f"好预测的交叉熵: {cross_entropy(p_true, q_good):.4f}")  # 应较小
print(f"差预测的交叉熵: {cross_entropy(p_true, q_bad):.4f}")   # 应较大
```

---

## 四、为什么语言模型用交叉熵损失？

**训练目标**：最大化模型分配给真实文本的概率

$$\max_\theta \prod_{t=1}^T P_\theta(w_t | w_{<t})$$

取对数（更稳定，乘法变加法）：

$$\max_\theta \sum_{t=1}^T \log P_\theta(w_t | w_{<t})$$

等价于：

$$\min_\theta -\frac{1}{T} \sum_{t=1}^T \log P_\theta(w_t | w_{<t}) = \min_\theta H(P_{data}, P_\theta)$$

**这正是交叉熵损失！**

---

## 五、KL 散度：分布之间的距离

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

**性质**：
- $D_{KL}(P\|Q) \geq 0$（总是非负）
- $D_{KL}(P\|Q) = 0 \Leftrightarrow P = Q$（相等时为 0）
- **不对称**：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$（不是距离！）

```python
def kl_divergence(p, q):
    """KL(P||Q)：用 Q 近似 P 的信息损失"""
    p = np.array(p) + 1e-15
    q = np.array(q) + 1e-15
    return np.sum(p * np.log(p / q))

p = [0.4, 0.4, 0.2]
q = [0.3, 0.4, 0.3]

print(f"KL(P||Q) = {kl_divergence(p, q):.4f}")
print(f"KL(Q||P) = {kl_divergence(q, p):.4f}")  # 不同！
print(f"KL(P||P) = {kl_divergence(p, p):.4f}")  # = 0
```

---

## 六、LLM 训练中的 KL 散度

**RLHF（人类反馈强化学习）中的 KL 约束**：

$$\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

- $\pi_{ref}$：SFT 参考模型
- $\pi_\theta$：当前 RL 训练的模型
- KL 项防止模型"跑偏"太远（避免奖励黑客攻击）

**DPO（直接偏好优化）中的 KL 散度**也是核心——把强化学习转化为监督学习的关键数学工具。

---

## 七、Perplexity：信息论视角

$$\text{PPL} = \exp(H(P_{data}, P_{model})) = \exp\left(-\frac{1}{T}\sum_t \log P(w_t|w_{<t})\right)$$

**直觉**：如果 Perplexity = 10，意味着模型对每个词的预测相当于在 10 个词里猜。

```python
# Perplexity 计算
def compute_perplexity(log_probs):
    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)

# 示例
log_probs = [-0.5, -1.2, -0.3, -0.8, -0.6]  # 模型对5个词的 log 概率
ppl = compute_perplexity(log_probs)
print(f"Perplexity = {ppl:.2f}")  # 约 2.1，模型不错
```

---

## 今天的关键认识

| 概念 | 公式 | 在 LLM 中的作用 |
|------|------|----------------|
| 熵 | $H(P) = -\sum P \log P$ | 衡量分布的不确定性 |
| 交叉熵 | $H(P,Q) = -\sum P \log Q$ | **训练损失函数** |
| KL 散度 | $D_{KL}(P\|Q) = H(P,Q)-H(P)$ | RLHF/DPO 中的约束项 |
| 困惑度 | $\text{PPL} = e^{H(P,Q)}$ | **模型质量评估指标** |

---

## 明天预告

D5：**极大似然估计**——从统计角度理解为什么训练就是"找最可能生成数据的参数"。
