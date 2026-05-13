---
layout: default
title: "D4 · 多项式回归 & 过拟合"
---

# D4 · 多项式回归 & 过拟合

> **Week 3 · AI 基础线**  
> 今天学模型复杂度的核心权衡：欠拟合 vs 过拟合，以及偏差-方差分解。

---

## 一、为什么需要多项式回归？

现实数据往往不是线性的：

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(-3, 3, 100)
# 真实关系：y = 0.5x³ - 2x² + x + 2
y = 0.5*X**3 - 2*X**2 + X + 2 + np.random.randn(100) * 1.5

plt.scatter(X, y, alpha=0.6)
plt.title('非线性数据')
plt.show()
```

强行用直线拟合这样的数据，永远学不好（高偏差）。

**解决方案**：添加多项式特征

$$h(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3$$

关键认识：**这本质上还是线性回归**——只不过特征从 $[x]$ 变成了 $[x, x^2, x^3]$。

---

## 二、多项式特征构造

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X_reshaped = X.reshape(-1, 1)

# 对比不同次数的多项式
degrees = [1, 2, 3, 10]
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, degree in enumerate(degrees):
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    train_mse = np.mean((y_pred - y)**2)
    
    axes[i].scatter(X, y, alpha=0.4, s=15)
    axes[i].plot(X_plot, model.predict(X_plot), 'r-', linewidth=2)
    axes[i].set_title(f'degree={degree}\nMSE={train_mse:.2f}')
    axes[i].set_ylim(-15, 15)

plt.tight_layout()
plt.show()
```

---

## 三、过拟合与欠拟合

| 情况 | 表现 | 原因 |
|------|------|------|
| **欠拟合** (Underfitting) | 训练集误差大，测试集也大 | 模型太简单，容量不够 |
| **恰好** | 训练集误差小，测试集接近 | 模型容量合适 |
| **过拟合** (Overfitting) | 训练集误差极小，测试集误差大 | 模型太复杂，记住了噪声 |

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

train_errors = []
test_errors = []

for degree in range(1, 15):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    train_errors.append(np.mean((model.predict(X_train) - y_train)**2))
    test_errors.append(np.mean((model.predict(X_test) - y_test)**2))

plt.semilogy(range(1, 15), train_errors, 'b-o', label='训练误差')
plt.semilogy(range(1, 15), test_errors, 'r-o', label='测试误差')
plt.xlabel('多项式次数')
plt.ylabel('MSE (log scale)')
plt.legend()
plt.axvline(x=3, color='g', linestyle='--', label='真实度数')
plt.title('偏差-方差曲线')
plt.show()
```

---

## 四、偏差-方差分解

这是机器学习里最重要的分析框架之一。

**期望测试误差 = 偏差² + 方差 + 不可减少的噪声**

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

**直觉理解**：

```
偏差（Bias）：瞄准中心的能力
方差（Variance）：不同训练集上结果的稳定性

偏差高 = 模型太简单，学不到规律 = 欠拟合
方差高 = 模型太复杂，对训练数据太敏感 = 过拟合
```

可视化：

```
              低偏差        高偏差
低方差   |  ● ●          |    ●
         |●     ●        |  ●   ●
         |  ● ●          |    ●
                            
高方差   |        ●      |●         ●
         |  ●    ● ●    |    ●●
         | ● ●  ●       |       ●
```

---

## 五、如何判断是欠拟合还是过拟合？

**学习曲线（Learning Curve）**是最好的诊断工具：

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_mse, 'b-o', label='训练误差')
    plt.plot(train_sizes, val_mse, 'r-o', label='验证误差')
    plt.xlabel('训练样本数')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (degree, name) in enumerate([(1, '欠拟合(d=1)'), (3, '合适(d=3)'), (15, '过拟合(d=15)')]):
    plt.sca(axes[i])
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    plot_learning_curve(model, X_reshaped, y, name)

plt.tight_layout()
plt.show()
```

**解读**：
- **欠拟合**：训练误差和验证误差都高，两者差距小（两条线都在高处贴近）
- **过拟合**：训练误差很低，验证误差很高，两者差距大（两条线分得很开）
- **合适**：两条线都低，且差距小

---

## 六、解决方案

| 问题 | 解决方案 |
|------|---------|
| 欠拟合 | 增加特征、增加模型复杂度、减少正则化 |
| 过拟合 | **加正则化**（明天讲）、减少特征、增加数据量、使用 Dropout |

---

## 七、特征工程小技巧

多项式回归不只是 $x^2, x^3$，还可以：

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 多特征的交叉项
X_2d = np.array([[1, 2], [3, 4]])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_2d)

# 输出特征名
print(poly.get_feature_names_out(['x1', 'x2']))
# ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
# 自动包含交叉项！
```

---

## 今天的核心认识

> **没有免费的午餐**：更复杂的模型不一定更好。
> 
> 模型容量 = 剪刀，数据量 = 布料。  
> 剪刀太大（模型太复杂）会把布料（数据）剪碎（过拟合）。  
> 剪刀太小（模型太简单）剪不出花样（欠拟合）。

---

## 明天预告

D5：**正则化**——系统化解决过拟合问题，Ridge、Lasso 背后的数学。

> 💡 **思考题**：如果模型过拟合了，增加更多训练数据能解决问题吗？
