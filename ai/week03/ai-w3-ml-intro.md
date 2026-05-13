---
layout: default
title: "D1 · 什么是机器学习？"
---

# D1 · 什么是机器学习？

> **Week 3 · AI 基础线**  
> 今天我们正式踏入机器学习的大门——从"写死的规则"到"让数据说话"。

---

## 一、为什么需要机器学习？

传统编程的逻辑是：

```
输入 + 规则 → 输出
```

比如：写一个判断邮件是否是垃圾邮件的程序，你需要手动写几百条规则："如果包含'中奖'就是垃圾邮件"……

但现实问题太复杂，规则根本写不完。机器学习换了个思路：

```
输入 + 输出 → 规则（模型）
```

**让机器从数据里自己学规则。**

---

## 二、机器学习的三大类型

### 1. 监督学习（Supervised Learning）

**有标签**：每个训练样本都有"正确答案"。

```
数据：(房屋面积 → 价格)
      (邮件文本 → 垃圾/正常)
      (图片 → 猫/狗)
```

分为两类：
- **回归**：预测连续值（价格、温度、股价）
- **分类**：预测类别（垃圾邮件？猫还是狗？）

本周重点：**回归**

### 2. 无监督学习（Unsupervised Learning）

**无标签**：数据没有"正确答案"，让机器自己发现规律。

典型任务：
- **聚类**（K-Means）：把相似的样本分到一组
- **降维**（PCA）：把高维数据压缩到低维
- **异常检测**：找出和其他样本差异最大的点

### 3. 强化学习（Reinforcement Learning）

**有奖惩**：智能体通过与环境交互，最大化累积奖励。

典型场景：AlphaGo、游戏 AI、机器人控制

---

## 三、机器学习的核心三要素

任何一个机器学习问题，都可以分解为三个部分：

### 1. 模型（Model）

**假设函数**：描述输入和输出之间的关系。

线性回归的模型假设：

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

用向量表示：

$$h_\theta(\mathbf{x}) = \mathbf{\theta}^T \mathbf{x}$$

### 2. 损失函数（Loss Function）

**衡量模型有多差**：预测值和真实值之间的差距。

回归常用：**均方误差（MSE）**

$$L(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### 3. 优化算法（Optimizer）

**找到最好的参数**：让损失函数最小化。

最常用：**梯度下降（Gradient Descent）**（明天详细讲）

---

## 四、训练 / 验证 / 测试：为什么要分？

这是机器学习里最容易被忽视、却最重要的概念之一。

```
全部数据
├── 训练集（70%）：用来训练模型，调整参数
├── 验证集（15%）：用来调超参数，选模型
└── 测试集（15%）：最后一次性评估，不能用来做决策
```

**为什么不能用训练集评估？**

因为模型可能只是"记住了"训练数据（过拟合），在新数据上表现很差。

---

## 五、泛化：机器学习的终极目标

```python
# 过拟合：训练集误差很小，测试集误差很大
train_error = 0.001   # "记住了"训练数据
test_error  = 0.45    # 遇到新数据就傻了

# 欠拟合：训练集和测试集误差都很大
train_error = 0.38    # 模型太简单，学不到规律
test_error  = 0.42

# 恰好合适：两者都小
train_error = 0.05
test_error  = 0.06    # ✅ 泛化好
```

**泛化能力** = 模型在未见过数据上的表现。这才是我们真正追求的。

---

## 六、偏差-方差权衡（预告）

```
模型太简单（高偏差）→ 欠拟合 → 训练/测试都差
模型太复杂（高方差）→ 过拟合 → 训练好/测试差
                    ↕
              找到最优点
```

Week 3 D4 会详细讲，这周会反复提到这个概念。

---

## 七、第一个机器学习代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 生成简单数据：y = 2x + 1 + 噪声
np.random.seed(42)
X = np.random.rand(100, 1) * 10          # 100个样本，1个特征
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 1.5   # 真实关系 + 噪声

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 评估
print(f"训练集 R²: {model.score(X_train, y_train):.4f}")
print(f"测试集 R²: {model.score(X_test, y_test):.4f}")
print(f"学到的参数: θ₁={model.coef_[0]:.4f}, θ₀={model.intercept_:.4f}")
# 输出：接近 θ₁=2.0, θ₀=1.0（真实值）

# 可视化
plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
plt.scatter(X_test, y_test, alpha=0.6, color='orange', label='测试数据')
plt.plot(X, model.predict(X), 'r-', linewidth=2, label='拟合直线')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('第一个线性回归模型')
plt.show()
```

---

## 八、今天的关键词

| 概念 | 一句话 |
|------|--------|
| 监督学习 | 有标签，学规则 |
| 无监督学习 | 无标签，找规律 |
| 模型 | 描述输入→输出关系的函数 |
| 损失函数 | 衡量模型预测有多差 |
| 泛化 | 在新数据上表现好才算真的好 |
| 过拟合 | 记住了训练数据，不会推广 |
| 欠拟合 | 模型太简单，连训练数据都学不好 |

---

## 明天预告

D2：**线性回归原理**——推导损失函数，解析解（正规方程），理解"最小二乘法"是怎么回事。

> 💡 **思考题**：如果损失函数不用 MSE，用平均绝对误差（MAE）会怎样？有什么优缺点？
