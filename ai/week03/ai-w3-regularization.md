---
layout: default
title: "D5 · 正则化：Ridge & Lasso"
---

# D5 · 正则化：Ridge & Lasso

> **Week 3 · AI 基础线**  
> 今天学如何系统地对抗过拟合——正则化的数学原理与实践。

---

## 一、正则化的本质

过拟合的根本原因：**参数 $\theta$ 的值可以变得很大**，从而拟合训练集的每一个噪声点。

正则化的思路：**在损失函数里加一个惩罚项，让大参数"代价更高"**。

$$J_{reg}(\theta) = J(\theta) + \lambda \cdot \Omega(\theta)$$

- $\lambda$：正则化强度（越大，对大参数的惩罚越重）
- $\Omega(\theta)$：惩罚项（不同选择对应不同的正则化方式）

---

## 二、Ridge 回归（L2 正则化）

$$J_{Ridge}(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^n \theta_j^2$$

**关键点**：
- 惩罚参数的**平方和**
- 梯度更新：$\theta_j := \theta_j(1 - \frac{\alpha\lambda}{m}) - \frac{\alpha}{m}\sum_{i=1}^m(h(x^{(i)})-y^{(i)})x_j^{(i)}$
- 每次更新前先把 $\theta_j$ 乘以一个小于 1 的系数（"权重衰减"）
- 通常**不对截距 $\theta_0$ 正则化**

**效果**：让所有参数都趋向 0，但不会变成恰好 0（参数只是小，不是消失）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 生成过拟合场景的数据
np.random.seed(42)
X = np.sort(np.random.rand(30) * 6 - 3)
y = X**3 - 2*X**2 + X + np.random.randn(30) * 3

X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=0.3, random_state=42
)

# 对比：无正则化 vs Ridge
for name, model in [
    ("无正则化 (d=10)", make_pipeline(PolynomialFeatures(10), LinearRegression())),
    ("Ridge λ=0.1 (d=10)", make_pipeline(PolynomialFeatures(10), StandardScaler(), Ridge(alpha=0.1))),
    ("Ridge λ=10 (d=10)", make_pipeline(PolynomialFeatures(10), StandardScaler(), Ridge(alpha=10))),
]:
    model.fit(X_train, y_train)
    print(f"{name}: 训练R²={model.score(X_train, y_train):.3f}, 测试R²={model.score(X_test, y_test):.3f}")
```

---

## 三、Lasso 回归（L1 正则化）

$$J_{Lasso}(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^n |\theta_j|$$

**与 Ridge 的关键区别**：

| | Ridge (L2) | Lasso (L1) |
|--|-----------|-----------|
| 惩罚项 | $\sum \theta_j^2$ | $\sum \|\theta_j\|$ |
| 参数是否为 0 | 趋向 0，但不为 0 | **可以精确变为 0** |
| 特征选择 | 不会 | **会自动做特征选择** |
| 几何解释 | 约束区域是球形 | 约束区域是菱形（有角） |

**Lasso 的稀疏性来自哪里？**

```
L2 的约束区域是圆形（光滑），损失函数等高线很少恰好在轴上相切
L1 的约束区域是菱形（有尖角），损失函数等高线很容易在角上相切

角上 = 某个参数精确等于 0 = 该特征被"删掉"了
```

---

## 四、正则化强度 λ 的影响

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge 不同 λ 值下的系数变化
alphas = np.logspace(-3, 3, 100)
X_poly = PolynomialFeatures(10).fit_transform(StandardScaler().fit_transform(X_train))

coef_path = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly, y_train)
    coef_path.append(ridge.coef_)

coef_path = np.array(coef_path)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for j in range(coef_path.shape[1]):
    plt.plot(np.log10(alphas), coef_path[:, j])
plt.xlabel('log10(λ)')
plt.ylabel('系数值')
plt.title('Ridge 正则化路径')
plt.axvline(x=0, color='gray', linestyle='--')

# Lasso 会有些系数提前变成 0
lasso_coefs = []
for alpha in np.logspace(-3, 1, 50):
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_poly, y_train)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)
plt.subplot(1, 2, 2)
for j in range(lasso_coefs.shape[1]):
    plt.plot(np.log10(np.logspace(-3, 1, 50)), lasso_coefs[:, j])
plt.xlabel('log10(λ)')
plt.ylabel('系数值')
plt.title('Lasso 正则化路径（稀疏性）')
plt.tight_layout()
plt.show()
```

---

## 五、ElasticNet：两者的结合

$$J_{EN}(\theta) = \frac{1}{2m}\sum(h - y)^2 + \lambda_1 \sum|\theta_j| + \lambda_2 \sum\theta_j^2$$

```python
from sklearn.linear_model import ElasticNet

# r 控制 L1/L2 的比例：r=1 退化为 Lasso，r=0 退化为 Ridge
en = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1 + 50% L2
```

**什么时候用哪个**：

| 场景 | 推荐 |
|------|------|
| 特征很多，但只有少数相关 | Lasso |
| 特征相关性强（多重共线性） | Ridge |
| 不确定 | ElasticNet |
| 特征少，过拟合轻微 | Ridge |

---

## 六、用交叉验证自动选 λ

```python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline

# RidgeCV 自动在给定范围内选最优 λ
ridge_cv = make_pipeline(
    PolynomialFeatures(10),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
)
ridge_cv.fit(X_train, y_train)

best_alpha = ridge_cv.named_steps['ridgecv'].alpha_
print(f"Ridge 最优 λ = {best_alpha:.4f}")
print(f"测试集 R² = {ridge_cv.score(X_test, y_test):.4f}")
```

---

## 七、贝叶斯视角：正则化 = 先验分布

正则化还有一个深刻的统计解释：

- **Ridge（L2）**= 给参数加高斯先验 $\theta \sim \mathcal{N}(0, \sigma^2)$
- **Lasso（L1）**= 给参数加拉普拉斯先验 $\theta \sim \text{Laplace}(0, b)$

参数越大，先验概率越小 → 后验概率越小 → 最大后验估计（MAP）会自然压制大参数。

这说明正则化不是"trick"，而是有坚实统计基础的方法。

---

## 八、今天的核心认识

1. **正则化 = 在损失函数里加惩罚项**，让大参数代价更高
2. **Ridge（L2）**：参数变小，不消失，适合特征都相关的场景
3. **Lasso（L1）**：参数可以变成 0，自动做特征选择
4. **$\lambda$ 是超参数**：太小不起作用，太大欠拟合，用交叉验证选

---

## 明天预告

D6：**sklearn 实战**——完整流程跑一遍：数据探索、预处理、训练、评估、调参。

> 💡 **思考题**：如果两个特征高度相关，Lasso 会倾向于只保留其中一个，另一个变成 0。这是优点还是缺点？
