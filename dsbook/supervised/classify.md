---
file_format: mystnb
kernelspec:
  name: python3
---

# Classification with Regression Concepts

In regression, we typically aim to predict a continuous outcome (like a numerical value). However, we can extend the same concepts of regression to **classification tasks**, where the goal is to predict a discrete class label. By using specific target values (like $y = \pm 1$), we can transform regression into a method for classifying data.

In this chapter, we'll explore how regression-based techniques can be adapted to solve classification problems.

## The Concept of Classification with Regression

In classification, the target variable $y_i$ represents the class label. For binary classification, we often use the labels $y_i = +1$ and $y_i = -1$ to represent the two different classes. The goal is to assign the correct class to new data points based on features $\mathbf{x}_i$.

We can approach this problem similarly to regression by fitting a function $f(\mathbf{x})$, but instead of predicting a continuous variable, we focus on predicting whether $f(\mathbf{x})$ is positive or negative:

- If $f(\mathbf{x}) > 0$, predict class $+1$.
- If $f(\mathbf{x}) < 0$, predict class $-1$.

This can be achieved by minimizing a **loss function**, which represents the error between the predicted values and the true class labels.

## Classification Loss Functions

When applying regression techniques to classification, the choice of loss function is critical. In regression, we typically minimize the **sum of squared residuals**, but in classification, we use loss functions that are designed to penalize misclassifications. Below are two common loss functions used in classification tasks:

### Hinge Loss (Used in Support Vector Machines)

The **hinge loss** is commonly used in support vector machines (SVMs) and is defined as:

```{math}
\mathcal{L}_{\text{hinge}} = \sum_i \max(0, 1 - y_i f(\mathbf{x}_i))
```

This loss penalizes any data points where the predicted value $f(\mathbf{x}_i)$ does not match the true label $y_i$. If $y_i f(\mathbf{x}_i) \geq 1$, the prediction is correct and there is no penalty. If $y_i f(\mathbf{x}_i) < 1$, there is a penalty proportional to the misclassification.

### Logistic Loss (Used in Logistic Regression)

The **logistic loss** is used in logistic regression and is defined as:

```{math}
\mathcal{L}_{\text{logistic}} = \sum_i \log(1 + \exp(-y_i f(\mathbf{x}_i)))
```

This loss function provides a smooth gradient and penalizes incorrect predictions by increasing the loss for large errors. Logistic regression aims to minimize this loss while interpreting $f(\mathbf{x}_i)$ as a probability.

## Classification Example: Logistic Regression

Let’s look at a simple example where we use the concepts of regression to perform a classification task. In logistic regression, we aim to model the probability that a given data point belongs to class $+1$ or class $-1$.

The logistic function, also called the **sigmoid function**, is used to map the output of the linear function $f(\mathbf{x})$ to a probability between 0 and 1:

```{math}
P(y_i = +1 | \mathbf{x}_i) = \frac{1}{1 + \exp(-f(\mathbf{x}_i))}
```

Where $f(\mathbf{x}_i) = \mathbf{x}_i^\top \beta$ is the linear function that we optimize. The logistic loss function is then minimized to find the best-fitting model.

## Optimizing the Model for Classification

To solve the classification task, we follow the same procedure used in regression:

1. **Define a loss function**: For classification, this could be the hinge loss or logistic loss.
2. **Optimize the loss function**: We minimize the loss function using methods like gradient descent or optimization tools.
3. **Make predictions**: After fitting the model, we predict the class label based on the sign of $f(\mathbf{x})$.

### Example of Logistic Regression in Code

```{code-cell}
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sns.set_style("whitegrid")

# Generate a simple dataset
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.sign(X[:, 0] + X[:, 1])  # Simple linear boundary

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic loss function
def logistic_loss(beta, X, y):
    z = X @ beta
    return np.sum(np.log(1 + np.exp(-y * z)))

# Initial guess for beta (weights)
initial_beta = np.zeros(X.shape[1])

# Minimize the logistic loss
result = minimize(logistic_loss, initial_beta, args=(X, y), method='BFGS')
optimized_beta = result.x

# Generate a grid of values for plotting the decision boundary
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Compute the logistic regression decision boundary (p = 0.5)
z = optimized_beta[0] * x1_grid + optimized_beta[1] * x2_grid
probability = sigmoid(z)

# Create a plot with the data points and decision boundary
plt.figure(figsize=(8, 6))

# Scatter plot of the dataset with class labels
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", s=60, edgecolor='k')

# Plot the decision boundary where probability = 0.5
plt.contour(x1_grid, x2_grid, probability, levels=[0.5], colors='black', linestyles='--')

plt.title("Logistic Regression with Decision Boundary (p = 0.5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
```

In this example:
- **Logistic Loss** is minimized to fit a classification model.
- The model predicts class labels $y = \pm 1$, based on the sign of $f(\mathbf{x})$.
- The classification accuracy is calculated by comparing the predicted labels with the actual labels.

## Summary

By extending the concepts of regression to classification, we can use similar techniques, such as minimizing a loss function, to build classification models. By using specific loss functions like **hinge loss** or **logistic loss**, we adapt regression methods to classify data. This approach demonstrates how foundational ideas from regression can be applied to a wide range of machine learning tasks.

In the next chapter, we will explore **regularization techniques** in classification, which are essential for improving the generalization of models, particularly in high-dimensional data.
