---
file_format: mystnb
kernelspec:
  name: python3
---

# Linear Regression

Linear regression is one of the simplest and most widely used models for supervised learning. In this chapter, we will explore the linear regression model and show how it can be used.

## Simple Linear Regression

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Seaborn settings
sns.set(style="whitegrid")

# Generating random data
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

# Scatter plot with seaborn
sns.scatterplot(x=x, y=y)
plt.show()
```

The linear regression model assumes that the relationship between two variables is approximately linear.

# Fitting the Model

To fit a linear model, we can use the LinearRegression estimator from scikit-learn.

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

# Minimize the loss function
initial_guess_linear = [0, 0]
result_linear = minimize(loss, initial_guess_linear, args=(linear_model, x, y))
# Optimized parameters
slope, intercept = result_linear.x

# Print the optimized slope and intercept
print(f"Optimized slope: {slope}, Optimized intercept: {intercept}")

# Generate prediction line
xfit = np.linspace(0, 10, 1000)
yfit_linear = linear_model(xfit, (slope, intercept))

# Plot data and model
sns.scatterplot(x=x, y=y)
plt.plot(xfit, yfit_linear, color='r')
plt.show()
```

This model fits a line to the data and makes predictions based on this line.

Polynomial Regression
Linear regression can be extended to handle more complex relationships by transforming the input data. Here, we demonstrate polynomial regression.

```{code-cell} ipython3
from sklearn.preprocessing import PolynomialFeatures


# Define the polynomial model function
def polynomial_model(x, params, degree=7):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x[:, np.newaxis])
    return np.dot(x_poly, params)


# Generate random data
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y_non_linear = np.sin(x) + 0.1 * rng.randn(50)

# Polynomial basis function for x
poly = PolynomialFeatures(degree=7)
x_poly = poly.fit_transform(x[:, np.newaxis])
initial_guess_poly = np.zeros(x_poly.shape[1])

result_poly = minimize(loss, initial_guess_poly, args=(polynomial_model, x, y_non_linear))

poly_params = result_poly.x

# Generate fit lines
yfit_poly = polynomial_model(xfit, poly_params)

sns.scatterplot(x=x, y=y_non_linear)
plt.plot(xfit, yfit_poly, color='r')
plt.title("Polynomial Regression Fit")
plt.show()
```