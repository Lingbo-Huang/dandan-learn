# D5：资产间相关性分析（Pearson / Spearman / 滚动相关）

> **Week 2 · Day 5** | 量化金融基础系列

---

## 1. 为什么要研究相关性？

相关性是现代投资组合理论（MPT）的核心：Markowitz 的均值-方差框架表明，**资产间的相关性决定了分散化的效果**。完全正相关（$\rho=1$）无法分散风险；完全负相关（$\rho=-1$）可以完全对冲。

实践中，相关性分析用于：
- **组合构建**：寻找低相关资产以降低整体波动
- **风险归因**：识别哪些资产在危机时同步暴跌
- **配对交易**：利用历史相关性寻找套利机会
- **因子分析**：检测策略暴露是否存在冗余因子

---

## 2. Pearson 相关系数

### 2.1 定义

Pearson 积矩相关系数衡量两变量**线性关系**的强度：

$$
\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}
$$

样本估计量：

$$
\hat{\rho}_{XY} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \cdot \sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

取值范围 $\rho \in [-1, 1]$。

### 2.2 显著性检验

在 $H_0: \rho = 0$ 下，检验统计量：

$$
t = \frac{\hat{\rho}\sqrt{n-2}}{\sqrt{1-\hat{\rho}^2}} \sim t(n-2)
$$

### 2.3 局限性

- 仅捕捉**线性**关系，对非线性关系失效
- 对**异常值**极度敏感
- 在金融数据中，极端市场（危机期）的相关性往往急剧升高，而 Pearson 用全样本均值掩盖了这种动态性

---

## 3. Spearman 等级相关系数

Spearman 相关系数是 Pearson 系数的非参数版本，基于**秩（排名）**而非原始值：

$$
\rho_s = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2-1)}
$$

其中 $d_i = \text{rank}(x_i) - \text{rank}(y_i)$ 为两变量秩差。

等价地，Spearman 相关 = 对排名序列计算 Pearson 相关。

**优势**：
- 对单调非线性关系有效
- 对异常值鲁棒
- 无需分布假设（非参数）

**示例**：若两资产的收益率大小排名高度一致（一个涨另一个也涨），即便不成比例，Spearman 相关也会很高。

---

## 4. Kendall τ 相关系数

$$
\tau = \frac{C - D}{\binom{n}{2}}
$$

其中 $C$ 为一致对（concordant pairs）数，$D$ 为不一致对（discordant pairs）数。

Kendall τ 比 Spearman 更具有鲁棒性，但计算复杂度为 $O(n^2)$，大样本时较慢。

---

## 5. 动态相关性：滚动窗口相关

静态相关假设相关性不随时间变化，但实证表明**金融资产相关性是时变的**：
- 危机期（2008、2020）相关性急剧趋向 1
- 牛市期资产间相关性相对较低

滚动相关系数：

$$
\hat{\rho}_t(h) = \text{Corr}(r_{X,t-h:t},\; r_{Y,t-h:t})
$$

其中 $h$ 为窗口长度（如 60 个交易日）。

---

## 6. 相关矩阵与特征值分解

对多资产组合，用**相关矩阵**（Correlation Matrix）$\mathbf{C}$ 描述全局相关结构：

$$
C_{ij} = \frac{Cov(r_i, r_j)}{\sigma_i \sigma_j}
$$

对 $\mathbf{C}$ 做特征值分解（PCA）：

$$
\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T
$$

其中 $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_p)$ 为特征值（从大到小排列）。

**第一主成分**的方差解释比例为 $\lambda_1 / \sum_i \lambda_i$，反映市场整体联动程度（系统性风险）。危机期 $\lambda_1$ 通常急剧放大。

---

## 7. 代码实战

### 7.1 生成多资产收益率数据

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns

np.random.seed(42)
n = 1000  # 交易日数

# 构造相关矩阵（5只股票）
assets = ['沪深300', '中证500', '恒生指数', '纳斯达克', '黄金']

# 设定真实相关矩阵
true_corr = np.array([
    [1.00, 0.80, 0.45, 0.30, -0.20],
    [0.80, 1.00, 0.40, 0.25, -0.15],
    [0.45, 0.40, 1.00, 0.35, -0.10],
    [0.30, 0.25, 0.35, 1.00,  0.05],
    [-0.20, -0.15, -0.10, 0.05, 1.00]
])

# 设定波动率
vols = np.array([0.012, 0.015, 0.014, 0.013, 0.008])

# Cholesky 分解生成相关随机收益率
L = np.linalg.cholesky(true_corr)
z = np.random.normal(0, 1, (n, 5))
corr_returns = z @ L.T * vols

df = pd.DataFrame(corr_returns, columns=assets)
print("=== 收益率基本统计 ===")
print(df.describe().round(6))
```

### 7.2 三种相关系数对比

```python
print("\n=== 三种相关系数对比（沪深300 vs 中证500）===")
x, y = df['沪深300'].values, df['中证500'].values

pearson_r, pearson_p = pearsonr(x, y)
spearman_r, spearman_p = spearmanr(x, y)
kendall_tau, kendall_p = kendalltau(x, y)

print(f"Pearson  ρ = {pearson_r:.4f}  (p={pearson_p:.4e})")
print(f"Spearman ρ = {spearman_r:.4f}  (p={spearman_p:.4e})")
print(f"Kendall  τ = {kendall_tau:.4f}  (p={kendall_p:.4e})")
print(f"\n真实相关系数（设定值）: 0.80")

# 加入异常值后对比
x_outlier = x.copy()
y_outlier = y.copy()
x_outlier[-5:] = [0.15, -0.12, 0.20, -0.18, 0.25]  # 极端值
y_outlier[-5:] = [-0.10, 0.08, -0.15, 0.12, -0.20]  # 反向极端值

p_out, _ = pearsonr(x_outlier, y_outlier)
s_out, _ = spearmanr(x_outlier, y_outlier)
print(f"\n=== 加入5个异常值后 ===")
print(f"Pearson  ρ = {p_out:.4f}  (变化: {p_out-pearson_r:+.4f})")
print(f"Spearman ρ = {s_out:.4f}  (变化: {s_out-spearman_r:+.4f})")
print("→ Spearman 对异常值更鲁棒")
```

### 7.3 相关矩阵热力图

```python
# 计算样本相关矩阵
corr_matrix = df.corr(method='pearson')
spearman_matrix = df.corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pearson 热力图
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=axes[0],
            linewidths=0.5, annot_kws={'size': 9})
axes[0].set_title('Pearson 相关矩阵')

# Spearman 热力图
sns.heatmap(spearman_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=axes[1],
            linewidths=0.5, annot_kws={'size': 9})
axes[1].set_title('Spearman 相关矩阵')

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
```

### 7.4 滚动相关性分析

```python
# 沪深300 vs 恒生指数 的滚动相关
ret1 = df['沪深300']
ret2 = df['恒生指数']

windows = [30, 60, 120]
fig, ax = plt.subplots(figsize=(14, 5))

colors = ['steelblue', 'orange', 'green']
for window, color in zip(windows, colors):
    rolling_corr = ret1.rolling(window).corr(ret2)
    ax.plot(rolling_corr.values, label=f'{window}日滚动相关', color=color, linewidth=1.2)

ax.axhline(y=corr_matrix.loc['沪深300', '恒生指数'], 
           color='red', linestyle='--', linewidth=1.5,
           label=f'全期静态相关 = {corr_matrix.loc["沪深300", "恒生指数"]:.3f}')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('沪深300 vs 恒生指数：滚动相关系数')
ax.set_xlabel('交易日')
ax.set_ylabel('相关系数')
ax.legend()
ax.set_ylim(-1, 1)
plt.tight_layout()
plt.savefig('rolling_correlation.png', dpi=150)
plt.show()

# 相关性统计分析
rolling_60 = ret1.rolling(60).corr(ret2).dropna()
print(f"\n=== 60日滚动相关性统计（沪深300 vs 恒生）===")
print(f"均值:  {rolling_60.mean():.4f}")
print(f"标准差: {rolling_60.std():.4f}")
print(f"最小值（危机期）: {rolling_60.min():.4f}")
print(f"最大值（同步期）: {rolling_60.max():.4f}")
print(f"变异系数: {rolling_60.std()/rolling_60.mean():.4f}")
```

### 7.5 相关矩阵的 PCA 分析

```python
from numpy.linalg import eig

# 相关矩阵特征值分解
corr_arr = corr_matrix.values
eigenvalues, eigenvectors = eig(corr_arr)

# 排序
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 方差解释比例
explained_ratio = eigenvalues / eigenvalues.sum()
cumulative_ratio = np.cumsum(explained_ratio)

print("=== 相关矩阵 PCA ===")
print(f"{'PC':>4} | {'特征值':>8} | {'方差解释':>10} | {'累计解释':>10}")
print("-" * 40)
for i, (ev, er, cr) in enumerate(zip(eigenvalues, explained_ratio, cumulative_ratio)):
    print(f"PC{i+1:2d} | {ev:8.4f} | {er:10.4f} | {cr:10.4f}")

print(f"\n第1主成分（市场因子）解释方差比例: {explained_ratio[0]*100:.1f}%")
print(f"前2主成分合计: {cumulative_ratio[1]*100:.1f}%")

# 第一主成分因子载荷
print(f"\n第一主成分因子载荷（市场因子）:")
for asset, loading in zip(assets, eigenvectors[:, 0]):
    print(f"  {asset}: {loading:.4f}")
```

### 7.6 相关性与组合风险的关系

```python
# 演示相关性对组合波动率的影响
sigma_A = 0.012
sigma_B = 0.015
w_A = 0.5
w_B = 0.5

rho_range = np.linspace(-1, 1, 201)
port_vol = np.sqrt(w_A**2 * sigma_A**2 + w_B**2 * sigma_B**2 + 
                   2 * w_A * w_B * rho_range * sigma_A * sigma_B)

plt.figure(figsize=(10, 5))
plt.plot(rho_range, port_vol * np.sqrt(252) * 100, 
         color='steelblue', linewidth=2)
plt.axhline(y=(w_A * sigma_A + w_B * sigma_B) * np.sqrt(252) * 100, 
            color='red', linestyle='--', label='无分散化效益上界')
plt.axhline(y=abs(w_A * sigma_A - w_B * sigma_B) * np.sqrt(252) * 100,
            color='green', linestyle='--', label='完全对冲下界')
plt.xlabel('Pearson 相关系数 ρ')
plt.ylabel('组合年化波动率 (%)')
plt.title('相关性对组合波动率的影响（等权组合）')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('correlation_portfolio.png', dpi=150)
plt.show()

print("\n=== 关键节点 ===")
for rho in [-1.0, -0.5, 0.0, 0.5, 0.8, 1.0]:
    vol = np.sqrt(w_A**2*sigma_A**2 + w_B**2*sigma_B**2 + 
                  2*w_A*w_B*rho*sigma_A*sigma_B)
    print(f"ρ={rho:+.1f}: 组合年化波动率 = {vol*np.sqrt(252)*100:.2f}%")
```

---

## 8. 相关性陷阱

| 陷阱 | 说明 | 应对 |
|------|------|------|
| 伪相关 | 两变量共同受第三变量驱动 | 偏相关分析 |
| 非线性关系 | Pearson 为零但存在曲线关系 | 散点图 + Spearman |
| 危机传染 | 危机期相关性趋向1 | 滚动相关 + 条件相关 |
| 滞后相关 | 一个资产领先另一个 | 交叉相关函数（CCF） |
| 样本估计误差 | 小样本相关估计不稳定 | 压缩估计（Ledoit-Wolf） |

---

## 9. 关键公式汇总

$$
\boxed{\hat{\rho}_{XY} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}}
$$

$$
\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}, \quad d_i = \text{rank}(x_i) - \text{rank}(y_i)
$$

$$
\sigma_P^2 = w_A^2\sigma_A^2 + w_B^2\sigma_B^2 + 2w_Aw_B\rho_{AB}\sigma_A\sigma_B
$$

---

## 10. 小结

- Pearson 相关系数衡量线性关系，是最常用的相关度量，但对非线性和异常值敏感
- Spearman/Kendall 是非参数替代，对单调非线性关系和异常值更鲁棒
- 相关性是时变的，滚动相关揭示动态联动特征（危机传染效应）
- PCA 分解相关矩阵，量化系统性风险（市场因子）的影响比重
- 分散化的核心是利用低相关资产，但危机期相关性的"收敛"是组合构建的主要挑战

> 下一篇：D6 完整数据处理实战（Tushare获取→清洗→统计）——端到端的量化数据工作流
