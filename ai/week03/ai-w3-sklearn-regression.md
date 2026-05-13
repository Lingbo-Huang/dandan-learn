---
layout: default
title: "D6 · sklearn 实战：房价预测"
---

# D6 · sklearn 实战：加州房价预测

> **Week 3 · AI 基础线**  
> 把这周学到的一切串起来——完整的机器学习项目流程。

---

## 一、项目目标

**数据集**：加州房价数据集（sklearn 内置）  
**特征**：8 个数值特征（收入中位数、房龄、房间数等）  
**目标**：预测房价中位数（单位：10万美元）  
**评估指标**：RMSE、R²

---

## 二、第一步：数据探索（EDA）

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# 加载数据
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())

print("\n基本统计:")
print(df.describe())

print("\n缺失值:")
print(df.isnull().sum())
```

**特征说明**：

| 特征 | 含义 |
|------|------|
| MedInc | 街区收入中位数 |
| HouseAge | 房屋年龄中位数 |
| AveRooms | 平均房间数 |
| AveBedrms | 平均卧室数 |
| Population | 街区人口 |
| AveOccup | 平均居住人数 |
| Latitude | 纬度 |
| Longitude | 经度 |

---

## 三、第二步：可视化分析

```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, col in enumerate(housing.feature_names):
    ax = axes[i // 4, i % 4]
    ax.scatter(df[col], df['MedHouseVal'], alpha=0.1, s=5)
    ax.set_xlabel(col)
    ax.set_ylabel('价格')
    
    # 计算相关系数
    corr = df[col].corr(df['MedHouseVal'])
    ax.set_title(f'{col} (r={corr:.2f})')

plt.tight_layout()
plt.show()

# 相关矩阵热图
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('特征相关矩阵')
plt.show()

# 目标变量分布
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df['MedHouseVal'].hist(bins=50)
plt.xlabel('房价')
plt.title('房价分布')

plt.subplot(1, 2, 2)
np.log1p(df['MedHouseVal']).hist(bins=50)
plt.xlabel('log(房价)')
plt.title('log 房价分布（更接近正态）')
plt.tight_layout()
plt.show()
```

---

## 四、第三步：数据预处理

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = housing.data
y = housing.target

# 分割数据（固定 random_state 保证可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 特征标准化（务必先 fit 训练集，再 transform 测试集）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled = scaler.transform(X_test)          # 只 transform，绝对不 fit！

# 验证标准化效果
print(f"\n标准化后训练集均值: {X_train_scaled.mean(axis=0).round(3)}")   # 应接近 0
print(f"标准化后训练集标准差: {X_train_scaled.std(axis=0).round(3)}")    # 应接近 1
```

---

## 五、第四步：训练多个模型对比

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

models = {
    '线性回归': LinearRegression(),
    'Ridge (λ=1)': Ridge(alpha=1.0),
    'Ridge (λ=10)': Ridge(alpha=10.0),
    'Lasso (λ=0.01)': Lasso(alpha=0.01),
    'Lasso (λ=0.1)': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    results[name] = {
        '训练RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        '测试RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        '训练R²': r2_score(y_train, y_train_pred),
        '测试R²': r2_score(y_test, y_test_pred),
    }

results_df = pd.DataFrame(results).T
print(results_df.round(4))
```

---

## 六、第五步：交叉验证 + 超参数调优

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# 用 GridSearchCV 搜索最优 λ
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5,                          # 5折交叉验证
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,                     # 使用所有 CPU 核心
    verbose=1
)

# 注意：用未标准化的数据（Pipeline 会自动处理）
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优CV RMSE: {-grid_search.best_score_:.4f}")
print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test))):.4f}")
print(f"测试集 R²: {grid_search.score(X_test, y_test):.4f}")
```

---

## 七、第六步：分析结果

```python
best_model = grid_search.best_estimator_
ridge_model = best_model.named_steps['model']

# 查看特征重要性（系数绝对值越大，特征越重要）
feature_importance = pd.DataFrame({
    '特征': housing.feature_names,
    '系数': ridge_model.coef_,
    '|系数|': np.abs(ridge_model.coef_)
}).sort_values('|系数|', ascending=False)

print("特征重要性排序:")
print(feature_importance)

# 可视化：预测值 vs 真实值
y_pred_final = grid_search.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_final, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='完美预测线')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('预测值 vs 真实值')
plt.legend()
plt.show()

# 残差分布（好的模型残差应接近正态分布、无系统性偏差）
residuals = y_test - y_pred_final
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_final, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50)
plt.xlabel('残差')
plt.title('残差分布')
plt.tight_layout()
plt.show()
```

---

## 八、完整流程总结

```
数据加载 → 数据探索 → 可视化分析
    ↓
数据分割（先分！）
    ↓
预处理（仅用训练集 fit）
    ↓
模型对比（多个基线）
    ↓
交叉验证 + 超参数搜索
    ↓
最终评估（只用一次测试集）
    ↓
结果分析（特征重要性 + 残差诊断）
```

**最终结果（参考）**：
- 测试集 RMSE ≈ 0.73（万美元）
- 测试集 R² ≈ 0.60

这个结果告诉我们：线性模型能解释约 60% 的房价方差。想提升，需要非线性模型（下周分类，Week 5 开始讲树模型和神经网络）。

---

## 明天预告

D7：**综合实战**——完整项目，从数据探索到特征工程到调参，产出一份完整的分析报告。
