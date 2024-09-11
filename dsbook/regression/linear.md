---
file_format: mystnb
kernelspec:
  name: python3
---

# Linear Regression

## General

The purpose of **regression** is to model the relationship between a **dependent variable** (output) and one or more **independent variables** (inputs). This is one of the most common techniques in statistical modeling, helping us understand and predict the behavior of a dependent variable based on independent variables.

### Independent and Dependent Variables

- **Independent variable**: Denoted as \(\mathbf{x}_i\), these are the input variables or predictors.
- **Dependent variable**: Denoted as \(y_i\), this is the output variable or response we are trying to predict.

## Model

In a simple case of a linear relationship, we want to find a function that models this relationship. One such model could be linear: \(f(\mathbf{x}_i) = a\mathbf{x}_i + b\), where \(a\) is the slope of the line and \(b\) is the intercept.

We can generalize this model as:

\[
f(\mathbf{x}_i') = \mathbf{x}_i'^T \beta
\]

Where:
- \(\beta = (b, a)^T\) are the model parameters (intercept and slope),
- \(\mathbf{x}_i' = (1, \mathbf{x}_i^T)^T\) is the extended input vector including a constant 1 for the intercept,
- The goal is to find \(\beta\) that best fits the data.

### Least Squares Estimation

To fit a function \(f(\mathbf{x})\) to the data points \(\{(\mathbf{x}_i, y_i)\}_i\), we minimize the sum of squared residuals:

\[
\mathcal{l}=\sum_i e_i^2 = \sum_i (f(\mathbf{x}_i) - y_i)^2
\]

Where \(e_i = f(\mathbf{x}_i) - y_i\) is the error or residual for the \(i\)-th data point. This process is known as **least squares**. $\mathcal{l}$ is known as a **loss function**. By minimizing the loss function, we find the best-fitting function for the data.

If the errors \(e_i\) follow a normal distribution, the **maximum likelihood estimation (MLE)** is equivalent to least squares minimization, making this approach optimal under these assumptions.


### Example code

The sk-learn package contains some functions for linear regression, that can be applied out of the box. Here is an example where we first generate random data with a linear trend and then fit a linear model to that data

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Seaborn settings
sns.set(style="whitegrid")

# Generating random data
rng = np.random.RandomState(4711)
x = rng.rand(50)
y_linear = 2 * x - 3 + 0.2*rng.randn(50)


model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y_linear)

xfit = np.linspace(0, 1, 1000)
yfit = model.predict(xfit[:, np.newaxis])

# Scatter plot with seaborn
sns.scatterplot(x=x, y=y_linear)
plt.plot(xfit, yfit, color='r')
plt.show()
```

In the next section we will investigate how to fit models to data.

### Connection to Hypothesis Testing

FIXME: Move this

The idea behind OLS regression is related to categorical tests. In both cases, we want to determine whether the observed data can be explained by chance or whether there is an association between the input and output variables. In regression, we fit a function \(f(\mathbf{x}_i)\approx y_i\) and test whether the sum of squared residuals \(\sum_i e_i^2\) is significantly smaller than what would be expected by chance if there were no relationship between \(\mathbf{x}_i\) and \(y_i\).

