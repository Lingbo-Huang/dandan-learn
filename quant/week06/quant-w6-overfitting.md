---
layout: default
title: "D6 · 过拟合陷阱与防范"
render_with_liquid: false
---

# D6 · 过拟合陷阱与防范

> 量化领域最大的杀手不是选错方向，而是过拟合。所有看起来很好的回测，你都应该怀疑它。

---

## 1. 过拟合的本质

**过拟合（Overfitting）**：模型学到了训练数据中的噪声，而不是真实的规律，导致样本外表现远差于样本内。

在金融数据中，过拟合尤其危险：
- 信噪比极低（真实 alpha 远小于噪声）
- 参数空间巨大（158个因子的组合有无限可能）
- 人类天生会"讲故事"，任何虚假规律都能被合理化

---

## 2. 过拟合的识别

```python
import pandas as pd
import numpy as np

def detect_overfitting(in_sample_ic, out_of_sample_ic, threshold=0.6):
    """
    通过样本内外 IC 比较检测过拟合
    threshold: OOS IC / IS IC 的最低比率
    """
    is_mean = np.mean(in_sample_ic)
    oos_mean = np.mean(out_of_sample_ic)
    
    ratio = oos_mean / is_mean if is_mean != 0 else 0
    
    print(f"样本内平均 IC: {is_mean:.4f}")
    print(f"样本外平均 IC: {oos_mean:.4f}")
    print(f"OOS/IS 比率: {ratio:.2%}")
    
    if ratio < threshold:
        print(f"⚠️ 严重过拟合！OOS 表现不足样本内的 {threshold:.0%}")
    elif ratio < 0.8:
        print("⚠️ 轻度过拟合，样本外有所衰减")
    else:
        print("✅ 样本内外表现相近，过拟合风险可控")
    
    return ratio

def plot_is_oos_comparison(is_ret, oos_ret):
    """可视化样本内外收益对比"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 样本内
    (1 + is_ret).cumprod().plot(ax=ax1, label='样本内', color='blue')
    ax1.set_title('样本内累积收益')
    
    # 样本外
    (1 + oos_ret).cumprod().plot(ax=ax2, label='样本外', color='red')
    ax2.set_title('样本外累积收益')
    
    plt.tight_layout()
    return fig
```

---

## 3. 主要过拟合机制

### 3.1 参数搜索过拟合

```
测试 100 个参数组合 → 选最好的
-> 即使所有参数都是随机策略，最好的看起来也很好
```

**Deflated Sharpe Ratio (DSR)**：
$$
SR^* = SR \\times \\sqrt{\\frac{T-1}{T}} \\times \\frac{1}{\\sqrt{1 - \\hat{\\rho} \\cdot SR^2}}
$$

Lopez de Prado (2018) 提出的校正公式，考虑了多重测试的影响。

### 3.2 多重测试偏差

```python
def bonferroni_correction(pvalues, alpha=0.05):
    """
    Bonferroni 校正：控制族错误率（FWER）
    """
    n_tests = len(pvalues)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in pvalues]
    
    print(f"原始显著性水平: {alpha}")
    print(f"校正后显著性水平: {corrected_alpha:.6f}")
    print(f"通过校正的策略数: {sum(significant)}/{n_tests}")
    return significant

def fdr_correction(pvalues, alpha=0.05):
    """
    Benjamini-Hochberg FDR 校正：控制错误发现率
    比 Bonferroni 更宽松，量化界常用
    """
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = np.array(pvalues)[sorted_idx]
    
    threshold = np.arange(1, n+1) * alpha / n
    reject = sorted_pvals <= threshold
    
    # 找最大的 k 使得 p_(k) <= k*alpha/n
    if np.any(reject):
        max_reject = np.max(np.where(reject)[0])
        final_reject = np.zeros(n, dtype=bool)
        final_reject[sorted_idx[:max_reject+1]] = True
    else:
        final_reject = np.zeros(n, dtype=bool)
    
    return final_reject
```

### 3.3 数据窥视（Data Snooping）

```python
class DataSnoopingProtector:
    """
    防止数据窥视的严格开发流程
    """
    
    def __init__(self, data, test_ratio=0.3):
        # 测试集在初始化时就封存，不允许再访问
        n = len(data)
        test_start = int(n * (1 - test_ratio))
        
        self._test_data = data.iloc[test_start:]
        self.train_data = data.iloc[:test_start]
        self._test_accessed = False
        self._evaluation_count = 0
        self.MAX_EVALUATIONS = 3  # 最多使用测试集 3 次
    
    def get_test_data(self):
        """获取测试集（有限次数）"""
        self._evaluation_count += 1
        if self._evaluation_count > self.MAX_EVALUATIONS:
            raise RuntimeError("超过最大测试集使用次数！策略存在数据窥视风险。")
        
        self._test_accessed = True
        print(f"警告：第 {self._evaluation_count} 次使用测试集（最多 {self.MAX_EVALUATIONS} 次）")
        return self._test_data
```

---

## 4. 防过拟合实践清单

### 开发阶段

- [ ] 严格分离训练/验证/测试数据（时间顺序）
- [ ] 使用 Walk-Forward 验证而非随机 CV
- [ ] 限制超参数搜索次数，或使用贝叶斯优化
- [ ] 每个信号都要有经济学逻辑支撑（不只是统计显著）
- [ ] 正则化：L1/L2、Dropout、Max Depth

### 评估阶段

```python
def comprehensive_oos_test(model, train_data, test_data):
    """
    综合样本外测试
    """
    # 1. 时间样本外
    time_oos_ic = evaluate_on_period(model, test_data)
    
    # 2. 参数敏感性
    param_sensitivity = {}
    for noise_level in [0.01, 0.05, 0.1]:
        # 在特征上加噪声
        X_noisy = test_data['X'] + np.random.normal(0, noise_level, test_data['X'].shape)
        noisy_ic = evaluate_ic(model, X_noisy, test_data['y'])
        param_sensitivity[noise_level] = noisy_ic
    
    # 3. 市场环境分段
    bull_ic = evaluate_on_period(model, test_data.loc[test_data['bull_market']])
    bear_ic = evaluate_on_period(model, test_data.loc[~test_data['bull_market']])
    
    return {
        'time_oos_ic': time_oos_ic,
        'param_sensitivity': param_sensitivity,
        'bull_market_ic': bull_ic,
        'bear_market_ic': bear_ic
    }
```

---

## 5. Harvey et al. (2016) 的黄金法则

在量化领域，新发现的策略需要满足：

**t 统计量 > 3.0**（对应 p < 0.001）

而不是传统统计学的 t > 1.96（p < 0.05）。

原因：量化领域每年发表数百个"有效"因子，但大多数是多重测试的假阳性。

```python
def required_t_stat(n_tests, alpha=0.05):
    """
    给定测试次数，计算所需的最低 t 统计量
    """
    from scipy.stats import norm
    # Bonferroni 校正后的 p 值
    adjusted_alpha = alpha / n_tests
    # 对应的 z 分位数（双侧）
    z = norm.ppf(1 - adjusted_alpha / 2)
    return z

# 示例：测试了 100 个因子
print(f"测试 100 个因子，需要 t > {required_t_stat(100):.2f}")
print(f"测试 1000 个因子，需要 t > {required_t_stat(1000):.2f}")
```

---

## 小结

| 过拟合类型 | 症状 | 应对 |
|----------|------|------|
| 参数过拟合 | 样本外大幅衰减 | Bonferroni/FDR 校正 |
| 数据窥视 | 测试集反复使用 | 封存测试集，Walk-Forward |
| 特征过多 | 模型复杂度高 | 特征选择、正则化 |
| 时段过拟合 | 只在特定时期有效 | 多时段分段验证 |
| 讲故事偏差 | 事后合理化 | 先有经济逻辑再找数据 |
