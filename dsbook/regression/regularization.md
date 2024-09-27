---
file_format: mystnb
kernelspec:
  name: python3
---

# Regularization in Linear Regression

## Introduction to Regularization

While **linear regression** works well in many cases, it can suffer from **overfitting** when the model becomes too complex or the dataset contains noise. Overfitting occurs when a model captures not only the true underlying pattern but also random fluctuations or noise in the data. This leads to poor generalization on unseen data.

One way to combat overfitting is to apply **regularization**, which modifies the loss function by adding a penalty for large model coefficients. The idea is to constrain the model's flexibility and encourage simpler models that generalize better.

In this chapter, we will introduce two commonly used regularization techniques:

- **Ridge Regression** (also known as L2 regularization)
- **LASSO Regression** (also known as L1 regularization)

## Ridge Regression (L2 Regularization)

In **ridge regression**, we modify the ordinary least squares loss function by adding a penalty proportional to the square of the coefficients' magnitudes. This penalty discourages large coefficients, leading to a smoother model that is less likely to overfit.

The objective function for ridge regression is:

```{math}
\mathcal{l}_{ridge} = \sum_i \left( f(\mathbf{x}_i) - y_i \right)^2 + \lambda \sum_j \beta_j^2
```

Where:
- $\lambda$ is a **regularization parameter** that controls the strength of the penalty. When $\lambda = 0$, ridge regression reduces to ordinary least squares. As $\lambda$ increases, the penalty becomes stronger.
- $\beta_j$ are the model coefficients (excluding the intercept).

The key idea here is that by penalizing the size of the coefficients, we shrink them toward zero, which can help mitigate overfitting.

### Ridge Regression Using Scikit-learn

The **scikit-learn** package provides a `Ridge` class that implements ridge regression. Below is an example of how to apply ridge regression to a dataset.

```{code-cell} ipython3
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the loss function for Ridge regression
def ridge_loss(params, x, y, reg_param):
    slope, intercept = params
    predictions = slope * x + intercept
    residual_sum_squares = np.sum((y - predictions) ** 2)
    l2_penalty = reg_param * np.sum(slope**2)
    return residual_sum_squares + l2_penalty

# Generate synthetic data
rng = np.random.RandomState(42)
x = rng.rand(50)
y = 2 * x - 0.5 + 0.2 * rng.randn(50)

# Minimize the Ridge loss function
reg_param = 1.0  # Regularization parameter (λ)
initial_params = np.random.randn(2)  # Initial guess for slope and intercept
result_ridge = minimize(ridge_loss, initial_params, args=(x, y, reg_param))

# Optimized slope and intercept
slope_ridge, intercept_ridge = result_ridge.x

# Generate predictions
xfit = np.linspace(0, 1, 1000)
yfit_ridge = slope_ridge * xfit + intercept_ridge

# Plot results
sns.scatterplot(x=x, y=y)
plt.plot(xfit, yfit_ridge, color='g', label="Ridge")
plt.legend()
plt.show()

# Print optimized parameters
print(f"Optimized slope (Ridge): {slope_ridge:.4f}, Optimized intercept: {intercept_ridge:.4f}")
```

In this example, `alpha` corresponds to $\lambda$ and controls the regularization strength. By tuning this parameter, we can adjust the model's complexity.

## LASSO Regression (L1 Regularization)

**LASSO regression** (Least Absolute Shrinkage and Selection Operator) is another regularization technique. Instead of penalizing the sum of squared coefficients, LASSO penalizes the sum of the absolute values of the coefficients. This results in sparse solutions, where some of the coefficients may become exactly zero.

The objective function for LASSO regression is:

```{math}
\mathcal{l}_{lasso} = \sum_i \left( f(\mathbf{x}_i) - y_i \right)^2 + \lambda \sum_j |\beta_j|
```

Where:
- $\lambda$ again controls the regularization strength.
- The absolute value penalty leads to **sparse models**, meaning that LASSO can perform **feature selection** by setting some coefficients to zero.

LASSO is particularly useful when we have many features, as it can identify and retain only the most important ones.

### LASSO Regression Using Scikit-learn

The **scikit-learn** package provides a `Lasso` class for L1-regularized regression. Below is an example of LASSO applied to a dataset.

```{code-cell} ipython3
# Define the loss function for LASSO regression
def lasso_loss(params, x, y, reg_param):
    slope, intercept = params
    predictions = slope * x + intercept
    residual_sum_squares = np.sum((y - predictions) ** 2)
    l1_penalty = reg_param * np.sum(np.abs(slope))
    return residual_sum_squares + l1_penalty

# Minimize the LASSO loss function
reg_param_lasso = 0.1  # Regularization parameter (λ)
initial_params_lasso = np.random.randn(2)  # Initial guess for slope and intercept
result_lasso = minimize(lasso_loss, initial_params_lasso, args=(x, y, reg_param_lasso))

# Optimized slope and intercept
slope_lasso, intercept_lasso = result_lasso.x

# Generate predictions
yfit_lasso = slope_lasso * xfit + intercept_lasso

# Plot results
sns.scatterplot(x=x, y=y)
plt.plot(xfit, yfit_lasso, color='b', label="LASSO")
plt.legend()
plt.show()

# Print optimized parameters
print(f"Optimized slope (LASSO): {slope_lasso:.4f}, Optimized intercept: {intercept_lasso:.4f}")
```

As with ridge regression, `alpha` controls the strength of regularization. When $\alpha$ is large, more coefficients will be set to zero, resulting in a simpler model.

## Comparing Ridge and LASSO

While both **ridge** and **LASSO** regression aim to reduce overfitting by penalizing large coefficients, they behave differently:

- **Ridge regression** tends to shrink coefficients but rarely sets them exactly to zero. This means that it keeps all features in the model, but the influence of less important features is reduced.
- **LASSO regression**, on the other hand, can shrink coefficients all the way to zero, effectively performing feature selection.

The choice between ridge and LASSO depends on the problem:
- If you believe that all features are relevant and you want to shrink their influence, **ridge** might be the better option.
- If you suspect that only a few features are truly important, **LASSO** can help by selecting those features automatically.

## Elastic Net: Combining Ridge and LASSO

In some cases, it can be beneficial to combine the strengths of ridge and LASSO regression. This is achieved with **Elastic Net**, which includes both L1 and L2 penalties.

The objective function for Elastic Net is:

```{math}
\mathcal{l}_{elasticnet} = \sum_i \left( f(\mathbf{x}_i) - y_i \right)^2 + \lambda_1 \sum_j |\beta_j| + \lambda_2 \sum_j \beta_j^2
```

Here, both $\lambda_1$ and $\lambda_2$ control the balance between L1 and L2 regularization. This method is particularly useful when there are correlations between features, as it can encourage a grouping effect, where correlated features tend to have similar coefficients.

### Elastic Net Using Scikit-learn

The **scikit-learn** package also provides an `ElasticNet` class, which combines ridge and LASSO regularization.

```{code-cell} ipython3
from sklearn.linear_model import ElasticNet

# Fit Elastic Net model
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio controls the balance between L1 and L2 regularization
elastic_net_model.fit(x[:, np.newaxis], y)

# Predict and plot
yfit_elastic_net = elastic_net_model.predict(xfit[:, np.newaxis])

sns.scatterplot(x=x, y=y)
plt.plot(xfit, yfit_elastic_net, color='m')
plt.show()
```

By tuning both the `alpha` and `l1_ratio` parameters, we can control the balance between ridge and LASSO regularization.

## Conclusion

Regularization techniques such as **ridge regression**, **LASSO**, and **Elastic Net** are powerful tools for improving the generalization of linear regression models. By introducing a penalty for large coefficients, these methods prevent overfitting and can even perform feature selection in the case of LASSO. The choice of regularization depends on the nature of the problem and the dataset, but by understanding the trade-offs, we can choose the method that best fits our needs.

