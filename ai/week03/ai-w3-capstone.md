---
layout: default
title: "D7 · 综合实战：多变量回归项目"
---

# D7 · Week 3 综合实战

> **Week 3 · AI 基础线**  
> 把这周学的所有东西串成一个完整项目，养成规范的工程习惯。

---

## 本周回顾

| 天 | 核心概念 | 一句话 |
|----|---------|--------|
| D1 | 什么是机器学习 | 从数据里自动学规则：监督/无监督/强化 |
| D2 | 线性回归原理 | 正规方程 = $(X^TX)^{-1}X^Ty$，最小二乘 = 正交投影 |
| D3 | 梯度下降 | 沿梯度反方向迭代更新参数，Mini-batch 是实践默认 |
| D4 | 多项式回归 & 过拟合 | 偏差-方差权衡，学习曲线是最好的诊断工具 |
| D5 | 正则化 | Ridge(L2)压缩系数，Lasso(L1)稀疏化，λ 用交叉验证选 |
| D6 | sklearn 实战 | EDA→分割→预处理→多模型对比→GridSearch→评估 |

---

## 实战项目：二手车价格预测

> **场景**：给定二手车的各项参数，预测合理的市场价格。  
> **数据**：我们自己生成带真实规律的模拟数据，更清楚因果关系。

---

### 一、数据生成

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(42)
n = 1000

# 特征
age = np.random.randint(1, 15, n)              # 车龄（年）
mileage = age * 15000 + np.random.randn(n) * 5000  # 里程（公里）
mileage = np.clip(mileage, 0, None)
engine_cc = np.random.choice([1000, 1500, 2000, 2500, 3000], n)  # 排量
fuel_type = np.random.choice(['汽油', '柴油', '混动'], n, p=[0.6, 0.2, 0.2])
brand_tier = np.random.choice(['经济', '中端', '豪华'], n, p=[0.5, 0.35, 0.15])

# 价格公式（含噪声）
base_price = {
    ('经济', '汽油'): 80000, ('经济', '柴油'): 90000, ('经济', '混动'): 100000,
    ('中端', '汽油'): 150000, ('中端', '柴油'): 160000, ('中端', '混动'): 175000,
    ('豪华', '汽油'): 350000, ('豪华', '柴油'): 370000, ('豪华', '混动'): 420000,
}

price = np.array([
    base_price[(brand_tier[i], fuel_type[i])]
    * (0.95 ** age[i])                          # 每年折旧 5%
    * (1 - mileage[i] / 500000)                 # 里程折旧
    * (1 + engine_cc[i] / 10000)                # 排量加成
    + np.random.randn() * 5000                  # 噪声
    for i in range(n)
])
price = np.clip(price, 5000, None)

df = pd.DataFrame({
    '车龄': age, '里程': mileage, '排量': engine_cc,
    '燃料类型': fuel_type, '品牌档次': brand_tier, '价格': price
})

print("数据前5行:")
print(df.head())
print("\n价格统计:")
print(df['价格'].describe())
```

---

### 二、探索性分析

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 数值特征 vs 价格
for i, col in enumerate(['车龄', '里程', '排量']):
    axes[0, i].scatter(df[col], df['价格'], alpha=0.3, s=10)
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('价格（元）')
    axes[0, i].set_title(f'{col} vs 价格 (r={df[col].corr(df["价格"]):.3f})')

# 类别特征 vs 价格（箱线图）
for i, col in enumerate(['燃料类型', '品牌档次']):
    df.boxplot(column='价格', by=col, ax=axes[1, i])
    axes[1, i].set_title(f'{col} vs 价格')

# 价格分布
axes[1, 2].hist(df['价格'], bins=50, edgecolor='black')
axes[1, 2].set_xlabel('价格')
axes[1, 2].set_title('价格分布')

plt.tight_layout()
plt.show()
```

---

### 三、数据预处理（处理类别特征）

```python
X = df.drop('价格', axis=1)
y = df['价格']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数值特征 + 类别特征分开处理
numerical_features = ['车龄', '里程', '排量']
categorical_features = ['燃料类型', '品牌档次']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# 预处理后特征维度
X_train_processed = preprocessor.fit_transform(X_train)
print(f"处理后特征数量: {X_train_processed.shape[1]}")
# 3 数值 + (3-1) 燃料 + (3-1) 品牌 = 7 个特征
```

---

### 四、特征工程：交叉项

```python
from sklearn.preprocessing import PolynomialFeatures

# 添加交叉项：车龄 × 里程，车龄²
class CarFeatureEngineer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['车龄²'] = X['车龄'] ** 2
        X['车龄×里程'] = X['车龄'] * X['里程']
        X['log里程'] = np.log1p(X['里程'])
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

from sklearn.base import BaseEstimator, TransformerMixin

class CarFeatureEngineerSK(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # 由于 ColumnTransformer 已经处理，这里针对数值列添加交叉项
        return X
```

---

### 五、模型训练与对比

```python
from sklearn.linear_model import RidgeCV

# 方案一：简单 Ridge Pipeline
pipeline_ridge = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5))
])

pipeline_ridge.fit(X_train, y_train)

y_pred = pipeline_ridge.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge 回归结果:")
print(f"  RMSE: {rmse:,.0f} 元")
print(f"  MAE:  {mae:,.0f} 元")
print(f"  R²:   {r2:.4f}")
print(f"  最优 λ: {pipeline_ridge.named_steps['model'].alpha_:.4f}")
```

---

### 六、预测分析与可视化

```python
# 预测 vs 真实
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.4, s=20)
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
plt.xlabel('真实价格（元）')
plt.ylabel('预测价格（元）')
plt.title(f'预测 vs 真实 (R²={r2:.3f})')

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=40, edgecolor='black', color='steelblue')
plt.xlabel('残差（真实-预测）')
plt.ylabel('频次')
plt.title(f'残差分布（均值={residuals.mean():,.0f}）')
plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()

# 各品牌档次的预测误差
df_test = X_test.copy()
df_test['真实价格'] = y_test.values
df_test['预测价格'] = y_pred
df_test['误差率'] = (df_test['预测价格'] - df_test['真实价格']) / df_test['真实价格']

print("\n各品牌档次误差分析:")
print(df_test.groupby('品牌档次')['误差率'].describe().round(3))
```

---

### 七、What's next

```python
# 本周线性模型的局限：非线性关系学不好
# 下周（分类）开始，Week 5 开始学树模型，能显著提升效果

# 参考：随机森林通常能把这个数据集的 R² 从 0.6x 提升到 0.9x+
from sklearn.ensemble import RandomForestRegressor  # Week 5 会讲

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_processed, y_train)
rf_pred = rf.predict(preprocessor.transform(X_test))
print(f"随机森林 R²: {r2_score(y_test, rf_pred):.4f}")  # 期待更高分
```

---

## Week 3 完成！

🎉 恭喜完成 Week 3！这周掌握了：

- ✅ 机器学习的核心思路（模型+损失+优化）
- ✅ 线性回归：正规方程 + 梯度下降两条路
- ✅ 过拟合诊断：学习曲线、偏差-方差分解
- ✅ 正则化：Ridge、Lasso 的原理和适用场景
- ✅ 完整 sklearn 工作流：EDA→预处理→训练→评估

**Week 4 预告**：监督学习·分类——逻辑回归、决策边界、混淆矩阵
