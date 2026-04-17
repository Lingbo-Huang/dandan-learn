# D7：综合实战——沪深300风险收益画像

> **Week 2 · Day 7** | 量化金融基础系列 · 综合实战

---

## 1. 实战目标

本篇将 Week2 全部六个主题（对数收益率、正态检验、胖尾统计、风险度量、相关性、数据处理）整合为一个**完整的指数分析框架**，以沪深300指数（模拟数据）为对象，输出一份端到端的**风险收益画像报告**。

**输出物包含**：
- 完整的统计检验结果
- VaR / CVaR / 最大回撤三维风险图
- 与其他资产的动态相关性热图
- 多维度的量化结论与投资含义

---

## 2. 分析框架

```
数据获取与清洗
    ↓
收益率计算（对数/简单）
    ↓
描述统计（矩统计）
    ↓
正态假设检验（QQ图 + JB检验）
    ↓
胖尾统计（t分布拟合 + 尾部指数）
    ↓
风险度量（VaR + CVaR + MDD）
    ↓
跨资产相关性（静态 + 滚动）
    ↓
综合画像报告
```

---

## 3. 完整代码实战

### 3.1 数据准备

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from scipy.stats import norm, t, jarque_bera, shapiro, genpareto
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 中文字体（可选）
try:
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

np.random.seed(2024)

# ====================================================
# 生成多资产模拟数据（沪深300 + 对比资产）
# ====================================================
def generate_multi_asset_data(n=2520, seed=2024):
    """
    模拟 10 年日频数据（约2520个交易日）
    资产：沪深300、中证500、恒生指数、纳斯达克100、黄金
    """
    np.random.seed(seed)
    
    assets_config = {
        'HS300':  {'mu': 0.08/252, 'sigma': 0.20/np.sqrt(252)},
        'ZZ500':  {'mu': 0.10/252, 'sigma': 0.25/np.sqrt(252)},
        'HSI':    {'mu': 0.05/252, 'sigma': 0.22/np.sqrt(252)},
        'NDX':    {'mu': 0.15/252, 'sigma': 0.20/np.sqrt(252)},
        'GOLD':   {'mu': 0.06/252, 'sigma': 0.12/np.sqrt(252)},
    }
    
    corr_matrix = np.array([
        [1.00,  0.82,  0.45,  0.28, -0.18],
        [0.82,  1.00,  0.38,  0.22, -0.12],
        [0.45,  0.38,  1.00,  0.32, -0.08],
        [0.28,  0.22,  0.32,  1.00,  0.05],
        [-0.18, -0.12, -0.08,  0.05,  1.00],
    ])
    
    # Cholesky 分解生成相关收益率
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.normal(0, 1, (n, 5))
    
    names = list(assets_config.keys())
    vols = np.array([c['sigma'] for c in assets_config.values()])
    mus = np.array([c['mu'] for c in assets_config.values()])
    
    log_returns = z @ L.T * vols + mus
    
    # 加入跳跃（模拟市场崩溃）
    crisis_prob = 0.012
    crises = np.random.choice([0, 1], n, p=[1-crisis_prob, crisis_prob])
    crisis_ret = np.random.normal(-0.05, 0.02, (n, 5))
    log_returns += crises.reshape(-1, 1) * crisis_ret
    
    # 转换为价格序列
    prices = {}
    start_prices = [4000, 6000, 22000, 12000, 280]
    for i, name in enumerate(names):
        prices[name] = start_prices[i] * np.exp(np.cumsum(log_returns[:, i]))
    
    dates = pd.bdate_range('2014-01-01', periods=n, freq='B')
    
    price_df = pd.DataFrame(prices, index=dates)
    ret_df = pd.DataFrame(log_returns, columns=names, index=dates)
    
    return price_df, ret_df

price_df, ret_df = generate_multi_asset_data()
hs300_ret = ret_df['HS300']
hs300_price = price_df['HS300']

print(f"数据概况: {len(hs300_ret)} 个交易日")
print(f"时间范围: {hs300_ret.index[0].date()} ~ {hs300_ret.index[-1].date()}")
print(f"资产列表: {list(ret_df.columns)}")
```

### 3.2 描述统计与矩分析

```python
def full_descriptive_stats(returns, name='资产'):
    """完整描述统计"""
    r = returns.dropna()
    n = len(r)
    
    result = {
        '资产名称': name,
        '样本量': n,
        '年化收益率': r.mean() * 252,
        '年化波动率': r.std() * np.sqrt(252),
        'Sharpe比率': (r.mean() * 252 - 0.02) / (r.std() * np.sqrt(252)),
        '偏度': stats.skew(r),
        '超额峰度': stats.kurtosis(r),
        '日最大涨幅': r.max(),
        '日最大跌幅': r.min(),
        '正收益率占比': (r > 0).mean(),
    }
    
    return result

hs300_stats = full_descriptive_stats(hs300_ret, '沪深300')
print("\n" + "="*55)
print("       沪深300 描述统计报告")
print("="*55)
for k, v in hs300_stats.items():
    if isinstance(v, float):
        print(f"  {k:18s}: {v:>12.6f}")
    else:
        print(f"  {k:18s}: {v}")
```

### 3.3 正态假设全面检验

```python
def normality_full_test(returns):
    """全面的正态性检验"""
    r = returns.dropna().values
    n = len(r)
    
    # JB 检验
    jb_stat, jb_p = jarque_bera(r)
    
    # Shapiro-Wilk（取子样本）
    sw_sample = r[:5000] if n > 5000 else r
    sw_stat, sw_p = shapiro(sw_sample)
    
    # D'Agostino-Pearson
    da_stat, da_p = stats.normaltest(r)
    
    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(r, 'norm', args=(r.mean(), r.std()))
    
    print("\n=== 正态性检验汇总 ===")
    print(f"{'检验方法':20s} {'统计量':>12} {'p值':>12} {'结论':>15}")
    print("-" * 62)
    
    tests = [
        ('Jarque-Bera', jb_stat, jb_p),
        ('Shapiro-Wilk', sw_stat, sw_p),
        ("D'Agostino-Pearson", da_stat, da_p),
        ('Kolmogorov-Smirnov', ks_stat, ks_p),
    ]
    
    for name, stat, p in tests:
        conclusion = "❌ 拒绝正态" if p < 0.05 else "✅ 不拒绝正态"
        print(f"{name:20s} {stat:>12.4f} {p:>12.4e} {conclusion:>15s}")
    
    return {'jb_stat': jb_stat, 'jb_p': jb_p}

test_results = normality_full_test(hs300_ret)
```

### 3.4 胖尾建模

```python
def fit_fat_tail(returns):
    """拟合胖尾分布并对比"""
    r = returns.dropna().values
    
    # 正态分布拟合
    mu_n, sigma_n = norm.fit(r)
    ll_norm = norm.logpdf(r, mu_n, sigma_n).sum()
    aic_norm = -2 * ll_norm + 2 * 2
    
    # t 分布拟合
    df_t, loc_t, scale_t = t.fit(r)
    ll_t = t.logpdf(r, df_t, loc_t, scale_t).sum()
    aic_t = -2 * ll_t + 2 * 3
    
    print("\n=== 分布拟合与模型选择 ===")
    print(f"正态分布: μ={mu_n:.6f}, σ={sigma_n:.6f}")
    print(f"  对数似然={ll_norm:.2f}, AIC={aic_norm:.2f}")
    print(f"\nt分布拟合: df={df_t:.3f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
    print(f"  对数似然={ll_t:.2f}, AIC={aic_t:.2f}")
    print(f"\nΔAIC（t vs 正态）= {aic_t - aic_norm:.2f}")
    print("→ AIC 越小越好，t 分布" + ("更优 ✅" if aic_t < aic_norm else "更差 ❌"))
    
    if df_t > 4:
        excess_kurt_theory = 6 / (df_t - 4)
        excess_kurt_sample = stats.kurtosis(r)
        print(f"\n超额峰度对比: 理论={excess_kurt_theory:.4f}, 样本={excess_kurt_sample:.4f}")
    
    return {'df_t': df_t, 'loc_t': loc_t, 'scale_t': scale_t,
            'mu_n': mu_n, 'sigma_n': sigma_n}

fit_params = fit_fat_tail(hs300_ret)
```

### 3.5 全维度风险度量

```python
def comprehensive_risk_measures(returns, prices):
    """全维度风险度量"""
    r = returns.dropna()
    
    print("\n=== 风险度量全景 ===")
    
    # VaR（三种方法）
    confidence_levels = [0.90, 0.95, 0.99]
    print(f"\n{'置信水平':>8} | {'历史VaR':>10} | {'正态VaR':>10} | {'t分布VaR':>10} | {'CVaR':>10}")
    print("-" * 58)
    
    df_t, loc_t, scale_t = t.fit(r)
    mu_hat, sigma_hat = r.mean(), r.std()
    
    for alpha in confidence_levels:
        var_hist = -np.percentile(r, (1-alpha)*100)
        var_norm = -(mu_hat + norm.ppf(1-alpha) * sigma_hat)
        var_t = -t.ppf(1-alpha, df_t, loc_t, scale_t)
        
        tail = -r[r < -var_hist]
        cvar = tail.mean() if len(tail) > 0 else var_hist
        
        print(f"{alpha:>8.1%} | {var_hist:>10.4f} | {var_norm:>10.4f} | "
              f"{var_t:>10.4f} | {cvar:>10.4f}")
    
    # 最大回撤分析
    nav = (1 + r).cumprod()
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # 年化指标
    annual_ret = r.mean() * 252
    annual_vol = r.std() * np.sqrt(252)
    sharpe = (annual_ret - 0.02) / annual_vol
    calmar = annual_ret / abs(mdd)
    
    print(f"\n=== 综合评分 ===")
    print(f"年化收益率: {annual_ret*100:.2f}%")
    print(f"年化波动率: {annual_vol*100:.2f}%")
    print(f"最大回撤:   {mdd*100:.2f}%")
    print(f"Sharpe比率: {sharpe:.4f}")
    print(f"Calmar比率: {calmar:.4f}")
    
    return {
        'annual_ret': annual_ret,
        'annual_vol': annual_vol,
        'mdd': mdd,
        'sharpe': sharpe,
        'calmar': calmar,
        'drawdown': drawdown,
        'nav': nav
    }

risk_results = comprehensive_risk_measures(hs300_ret, hs300_price)
```

### 3.6 跨资产相关性分析

```python
def multi_asset_correlation(ret_df, target='HS300', window=60):
    """多资产相关性分析"""
    
    # 静态相关矩阵
    pearson_corr = ret_df.corr(method='pearson')
    spearman_corr = ret_df.corr(method='spearman')
    
    print(f"\n=== Pearson 相关矩阵 ===")
    print(pearson_corr.round(4))
    
    print(f"\n=== 沪深300 与其他资产相关性 ===")
    print(f"{'资产':>12} | {'Pearson':>10} | {'Spearman':>10} | {'相关强度':>10}")
    print("-" * 48)
    
    corr_labels = {
        (0.8, 1.0): '强正相关',
        (0.5, 0.8): '中正相关',
        (0.2, 0.5): '弱正相关',
        (-0.2, 0.2): '不相关',
        (-0.5, -0.2): '弱负相关',
        (-1.0, -0.5): '强负相关',
    }
    
    for asset in ret_df.columns:
        if asset == target:
            continue
        pr = pearson_corr.loc[target, asset]
        sr = spearman_corr.loc[target, asset]
        label = next((v for (lo, hi), v in corr_labels.items() if lo <= pr < hi), '未知')
        print(f"{asset:>12} | {pr:>10.4f} | {sr:>10.4f} | {label:>10}")
    
    # 滚动相关
    rolling_corrs = {}
    for asset in [c for c in ret_df.columns if c != target]:
        rolling_corrs[asset] = ret_df[target].rolling(window).corr(ret_df[asset])
    
    return pearson_corr, rolling_corrs

pearson_corr, rolling_corrs = multi_asset_correlation(ret_df)
```

### 3.7 综合画像图表

```python
def plot_comprehensive_portrait(ret_df, hs300_ret, hs300_price, 
                                 risk_results, fit_params, rolling_corrs):
    """生成沪深300完整风险收益画像图"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    r = hs300_ret.dropna()
    nav = risk_results['nav']
    drawdown = risk_results['drawdown']
    
    # 1. 价格走势
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(hs300_price.values, color='steelblue', linewidth=1.2, label='沪深300（模拟）')
    ax1.fill_between(range(len(hs300_price)), hs300_price.values, 
                     hs300_price.values.min(), alpha=0.1, color='steelblue')
    ax1.set_title('沪深300 价格走势（2014-2024 模拟数据）', fontsize=13, fontweight='bold')
    ax1.set_ylabel('指数点位')
    ax1.legend()
    
    # 2. 收益率序列
    ax2 = fig.add_subplot(gs[1, :2])
    colors_bar = ['#d62728' if x < 0 else '#2ca02c' for x in r.values]
    ax2.bar(range(len(r)), r.values, color=colors_bar, alpha=0.7, width=1)
    ax2.set_title('日对数收益率序列（红跌绿涨）')
    ax2.set_ylabel('对数收益率')
    ax2.axhline(0, color='black', linewidth=0.5)
    
    # 3. 滚动年化波动率
    ax3 = fig.add_subplot(gs[1, 2])
    vol_60 = r.rolling(60).std() * np.sqrt(252) * 100
    ax3.plot(vol_60.values, color='purple', linewidth=1)
    ax3.fill_between(range(len(vol_60)), vol_60.values, alpha=0.2, color='purple')
    ax3.set_title('60日滚动年化波动率(%)')
    ax3.set_ylabel('%')
    
    # 4. 收益率分布（含多分布拟合）
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(r.values, bins=80, density=True, color='steelblue', 
             alpha=0.6, edgecolor='white', label='实际分布')
    x = np.linspace(r.min(), r.max(), 300)
    mu_n, sigma_n = fit_params['mu_n'], fit_params['sigma_n']
    df_t, loc_t, scale_t = fit_params['df_t'], fit_params['loc_t'], fit_params['scale_t']
    ax4.plot(x, norm.pdf(x, mu_n, sigma_n), 'r-', linewidth=2, label=f'正态拟合')
    ax4.plot(x, t.pdf(x, df_t, loc_t, scale_t), 'g-', linewidth=2, label=f't(df={df_t:.1f})')
    ax4.set_title('收益率分布与拟合对比')
    ax4.legend(fontsize=8)
    ax4.set_xlabel('收益率')
    
    # 5. QQ 图
    ax5 = fig.add_subplot(gs[2, 1])
    (osm, osr), (slope, intercept, r_sq) = stats.probplot(r.values, dist="norm")
    ax5.scatter(osm, osr, alpha=0.2, s=3, color='steelblue')
    ax5.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
    ax5.set_title(f'QQ图 (正态检验, R²={r_sq**2:.4f})')
    ax5.set_xlabel('理论分位数')
    ax5.set_ylabel('样本分位数')
    
    # 6. 回撤序列
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.fill_between(range(len(drawdown)), drawdown.values, 0, 
                     color='red', alpha=0.4)
    ax6.axhline(y=risk_results['mdd'], color='darkred', linestyle='--',
                label=f'MDD={risk_results["mdd"]*100:.1f}%')
    ax6.set_title('净值回撤序列')
    ax6.set_ylabel('回撤幅度')
    ax6.legend(fontsize=8)
    
    # 7. 滚动相关性
    ax7 = fig.add_subplot(gs[3, :2])
    color_map = {'ZZ500': 'orange', 'HSI': 'green', 'NDX': 'purple', 'GOLD': 'brown'}
    for asset, rc in rolling_corrs.items():
        ax7.plot(rc.values, label=f'HS300 vs {asset}', 
                 color=color_map.get(asset, 'gray'), linewidth=1, alpha=0.8)
    ax7.axhline(0, color='black', linewidth=0.5)
    ax7.set_title('60日滚动相关系数（沪深300 vs 各资产）')
    ax7.set_ylabel('相关系数')
    ax7.legend(loc='lower right', fontsize=8)
    ax7.set_ylim(-1, 1)
    
    # 8. 风险收益散点
    ax8 = fig.add_subplot(gs[3, 2])
    annual_rets = ret_df.mean() * 252 * 100
    annual_vols = ret_df.std() * np.sqrt(252) * 100
    colors_assets = ['steelblue', 'orange', 'green', 'purple', 'brown']
    
    for i, (asset, ret, vol) in enumerate(zip(ret_df.columns, annual_rets, annual_vols)):
        ax8.scatter(vol, ret, s=120, color=colors_assets[i], zorder=5)
        ax8.annotate(asset, (vol, ret), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    
    # 有效前沿参考线（简单）
    ax8.set_xlabel('年化波动率 (%)')
    ax8.set_ylabel('年化收益率 (%)')
    ax8.set_title('各资产风险收益散点图')
    ax8.axhline(0, color='black', linewidth=0.3)
    ax8.grid(alpha=0.3)
    
    plt.suptitle('沪深300 风险收益综合画像报告（模拟数据）', 
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig('hs300_portrait.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("综合画像图表已保存: hs300_portrait.png")

plot_comprehensive_portrait(ret_df, hs300_ret, hs300_price, 
                              risk_results, fit_params, rolling_corrs)
```

### 3.8 汇总报告生成

```python
def generate_summary_report(hs300_stats, test_results, fit_params, risk_results):
    """生成文字版汇总报告"""
    
    df_t = fit_params['df_t']
    
    report = f"""
{'='*60}
     沪深300 风险收益画像 · 综合报告
     数据期间：2014-01-01 ~ 2024-01-01（模拟）
{'='*60}

【一、收益概况】
  年化收益率  : {hs300_stats['年化收益率']*100:.2f}%
  年化波动率  : {hs300_stats['年化波动率']*100:.2f}%
  Sharpe 比率 : {hs300_stats['Sharpe比率']:.4f}
  正收益率占比: {hs300_stats['正收益率占比']*100:.1f}%

【二、分布特征】
  偏度        : {hs300_stats['偏度']:.4f}  （负偏 = 左尾风险更大）
  超额峰度    : {hs300_stats['超额峰度']:.4f}  （>0 = 胖尾）
  JB 检验 p值 : {test_results['jb_p']:.2e}
  结论        : {'强烈拒绝正态假设，存在显著胖尾' if test_results['jb_p'] < 0.01 else '不拒绝正态假设'}

【三、胖尾模型】
  t分布自由度 : {df_t:.2f}
  隐含超额峰度: {6/(df_t-4):.4f if df_t > 4 else 'N/A（df≤4）'}
  评价        : {'中度胖尾' if df_t > 5 else '重度胖尾'} — t 分布显著优于正态拟合

【四、风险度量（日度）】
  95% VaR    : {-np.percentile(hs300_ret, 5)*100:.3f}%
  99% VaR    : {-np.percentile(hs300_ret, 1)*100:.3f}%
  99% CVaR   : {-hs300_ret[hs300_ret < np.percentile(hs300_ret, 1)].mean()*100:.3f}%
  最大回撤    : {risk_results['mdd']*100:.2f}%
  Calmar 比率 : {risk_results['calmar']:.4f}

【五、跨资产相关性】
  HS300-ZZ500 (同涨同跌，高度相关，分散效益低)
  HS300-HSI   (中度相关，港股可提供一定分散效益)
  HS300-NDX   (弱正相关，美股A股分化)
  HS300-GOLD  (弱负相关，黄金具有一定对冲价值)

【六、投资含义】
  1. 收益率分布显著偏离正态，尾部风险被正态假设严重低估
  2. 基于正态假设的 VaR 会低估极端风险，建议使用 t-VaR 或历史模拟
  3. 沪深300与中证500高度相关，难以通过A股内部实现有效分散
  4. 黄金资产与A股负相关，在组合中配置黄金可降低尾部风险
  5. 最大回撤较大，需要设置明确的止损机制

{'='*60}
"""
    print(report)
    return report

summary_report = generate_summary_report(hs300_stats, test_results, fit_params, risk_results)
```

---

## 4. Week2 知识体系回顾

| 主题 | 核心概念 | 在本次实战中的应用 |
|------|----------|-------------------|
| D1 对数收益率 | 时间可加性、年化公式 | 基础收益率序列构建 |
| D2 正态检验 | JB检验、QQ图 | 验证分布假设，发现胖尾 |
| D3 胖尾统计 | t分布、尾部指数 | 拟合更优分布，AIC对比 |
| D4 风险度量 | VaR、CVaR、MDD | 三维风险刻画 |
| D5 相关性 | Pearson、Spearman、滚动 | 跨资产联动分析 |
| D6 数据处理 | 清洗、特征工程、统计 | 端到端工作流 |

---

## 5. 关键公式体系

$$
r_t = \ln\frac{P_t}{P_{t-1}} \xrightarrow{\text{年化}} \hat{\mu}_{ann} = \bar{r} \times 252, \quad \hat{\sigma}_{ann} = s_r\sqrt{252}
$$

$$
\text{JB} = \frac{n}{6}\left(\gamma_1^2 + \frac{\kappa^2}{4}\right) \sim \chi^2(2) \quad \Rightarrow \quad \text{检验正态性}
$$

$$
\text{VaR}_\alpha = -Q_{1-\alpha}(r) \leq \text{CVaR}_\alpha = -E[r|r<-\text{VaR}_\alpha]
$$

$$
\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}, \quad \text{Calmar} = \frac{R_{ann}}{|\text{MDD}|}
$$

$$
\sigma_P^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}, \quad \mathbf{\Sigma}_{ij} = \rho_{ij}\sigma_i\sigma_j
$$

---

## 6. 总结

本次综合实战将 Week2 的七个主题融为一体，完成了对沪深300指数的全维度风险收益画像：

1. **数据层**：模拟真实数据管道，包含跳跃过程和市场崩溃情景
2. **统计层**：全面的矩分析，强烈拒绝正态假设，发现显著胖尾
3. **模型层**：t 分布 AIC 优于正态，为后续 VaR 计算提供更准确的基础
4. **风险层**：三种方法对比 VaR，结合 CVaR 和最大回撤形成完整风险画像
5. **组合层**：跨资产相关性揭示分散化机会（黄金 vs A股负相关）

**Week2 到此全部完成。** 下一步（Week3）将进入因子模型与多资产组合优化，在这些统计基础上构建真正的量化策略。
