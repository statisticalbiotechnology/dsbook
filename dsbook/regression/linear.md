---
file_format: mystnb
kernelspec:
  name: python3
---

# Linear Regression

## General

The purpose of **regression** is to model the relationship between a **dependent variable** (output) and one or more **independent variables** (inputs). This is one of the most common techniques in statistical modeling, helping us understand and predict the behavior of a dependent variable based on independent variables.

### Independent and Dependent Variables

- **Independent variable**: Denoted as $\mathbf{x}_i$, these are the input variables or predictors.
- **Dependent variable**: Denoted as $y_i$, this is the output variable or response we are trying to predict.

## Model

In a simple case of a linear relationship, we want to find a function that models this relationship. One such model could be linear: $f(\mathbf{x}_i) = a\mathbf{x}_i + b$, where $a$ is the slope of the line and $b$ is the intercept.

We can generalize this model as:

```{math}
f(\mathbf{x}_i') = \mathbf{x}_i'^T \beta
```

Where:
- $\beta = (b, a)^T$ are the model parameters (intercept and slope),
- $\mathbf{x}_i' = (1, \mathbf{x}_i^T)^T$ is the extended input vector including a constant 1 for the intercept,
- The goal is to find $\beta$ that best fits the data.

### Least Squares Estimation

To fit a function $f(\mathbf{x})$ to the data points $\{(\mathbf{x}_i, y_i)\}_i$, we minimize the sum of squared residuals:

```{math}
\mathcal{l}=\sum_i e_i^2 = \sum_i (f(\mathbf{x}_i) - y_i)^2
```

Where $e_i = f(\mathbf{x}_i) - y_i$ is the error or residual for the $i$-th data point. This process is known as **least squares**. $\mathcal{l}$ is known as a **loss function**. By minimizing the loss function, we find the best-fitting function for the data.

If the errors $e_i$ follow a normal distribution, the **maximum likelihood estimation (MLE)** is equivalent to least squares minimization, making this approach optimal under these assumptions.


### Linear Regression using Scikit-learn.

TThe **scikit-learn** package offers efficient functions for performing linear regression right out of the box. Below is an example where we generate random data following a linear trend, then fit a linear model using the `LinearRegression` class.

This approach is both fast and highly optimized, making it easy to apply linear regression with minimal setup.

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
y_linear = 2 * x - 0.5 + 0.2*rng.randn(50)


model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y_linear)

xfit = np.linspace(0, 1, 1000)
yfit = model.predict(xfit[:, np.newaxis])

# Scatter plot with seaborn
sns.scatterplot(x=x, y=y_linear)
plt.plot(xfit, yfit, color='r')
plt.show()
```

## An Alternative Way to Fit a Model

The scikit-learn linear regression uses an **analytical solution** to the least squares problem, [a well-known method for deriving the optimal parameters](https://en.wikipedia.org/wiki/Least_squares#Solving_the_least_squares_problem). While this approach is efficient, it can be generalized by using alternative fitting techniques that allow for more complex regression models and custom loss functions.

One such method is to manually minimize the **loss function**, which is the error between the predicted and actual values, using optimization techniques. This approach opens the door to advanced regression models and custom loss functions that aren't necessarily linear.

Here, we use the `scipy.optimize.minimize` function to achieve the same linear regression result by explicitly defining a **loss function** and minimizing it.

```{code-cell} ipython3
from scipy.optimize import minimize
import numpy as np

# Define a generic loss function for both linear and polynomial models
def loss(params, model_function, x, y):
    return np.sum((y - model_function(x, params))**2)

# Define the linear model function
def linear_model(x, params):
    slope, intercept = params
    return slope * x + intercept

# A function searching for a set of parameters that minimize a given loss function over data
def minimize_loss(loss, model, x, y, num_params):
    initial_params = np.random.randn(num_params)
    return minimize(loss, initial_params, args=(model, x, y))

# Minimize the loss function
result_linear = minimize_loss(loss, linear_model, x, y_linear, 2)

# Optimized parameters
slope, intercept = result_linear.x

# Print the optimized slope and intercept
print(f"Optimized slope: {slope}, Optimized intercept: {intercept}")

# Generate prediction line
xfit = np.linspace(0, 1.0, 1000)
yfit_linear = linear_model(xfit, (slope, intercept))

# Plot data and model
sns.scatterplot(x=x, y=y_linear)
plt.plot(xfit, yfit_linear, color='r')
plt.show()
```

In this alternative approach, we use the same underlying concept of minimizing a **loss function**, but the technique is more flexible, allowing us to extend the method to more advanced models such as polynomial or kernel-based regressions, which we'll explore in later sections.

The concept of minimizing a loss function provides the foundation for generalizing regression to other types of models and fitting techniques, allowing us to tackle more complex relationships between variables.

### Connection to Hypothesis Testing

FIXME: Move this

The idea behind OLS regression is related to categorical tests. In both cases, we want to determine whether the observed data can be explained by chance or whether there is an association between the input and output variables. In regression, we fit a function $f(\mathbf{x}_i)\approx y_i$ and test whether the sum of squared residuals $\sum_i e_i^2$ is significantly smaller than what would be expected by chance if there were no relationship between $\mathbf{x}_i$ and $y_i$.


