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

### Residuals

In regression, the **residual** for each data point is the difference between the observed value \(y_i\) and the value predicted by the model \(f(\mathbf{x}_i)\). The residual for the \(i\)-th point can be written as:

```{math}
e_i = y_i - f(\mathbf{x}_i)
```

Residuals give us a way to evaluate how well the model fits the data. A smaller residual indicates that the model's prediction is close to the actual value, while a larger residual suggests a larger error.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Seaborn settings
sns.set(style="whitegrid")

# Generating random data
rng = np.random.RandomState(4711)
x = rng.rand(30)
y_linear = 2 * x - 0.5 + 0.2*rng.randn(30)

# Define a generic loss function for linear models
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

# Generate prediction line
xfit = np.linspace(0, 1.0, 1000)
yfit_linear = linear_model(xfit, (slope, intercept))

# Plot data, model, and residuals
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y_linear, label='Data')
plt.plot(xfit, yfit_linear, color='r', label='$f(\mathbf{x})$')

# Create a custom legend entry for the residual lines
plt.plot([x[0], x[0]], [y_linear[0], linear_model(x[0], (slope, intercept))], color='blue', label='Residual')

# Plot bidirectional residual lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y_linear[i], linear_model(x[i], (slope, intercept))], color='blue')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```
Above is an illustration of the concept of residuals. Given the a model, $f(\mathbf{x})$, of the data points $\{(\mathbf{x}_i, y_i)\}_i$, how much of the data "remains to be explained".


### Least Squares Estimation

To fit a function $f(\mathbf{x})$ to the data points $\{(\mathbf{x}_i, y_i)\}_i$, we minimize the sum of squared errors (SSE):

```{math}
\mathcal{l}=\sum_i e_i^2 = \sum_i (y_i - f(\mathbf{x}_i) )^2
```

This process is known as **least squares**. $\mathcal{l}$ is known as a **loss function**. By minimizing the loss function, we find the best-fitting function for the data.

If the errors $e_i$ follow a normal distribution, the **maximum likelihood estimation (MLE)** is equivalent to least squares minimization, making this approach optimal under these assumptions.


### Fit a Linear Model to Data

A general pattern for machine learning, is the utilization of an optimizer for the minimization of a loss function.

One such method is to manually minimize the **loss function**, which is the error between the predicted and actual values, using optimization techniques. This approach opens the door to advanced regression models and custom loss functions that aren't necessarily linear.

Here, we use the `scipy.optimize.minimize` function to achieve the same linear regression result by explicitly defining a **loss function** and minimizing it.

```{code-cell} ipython3
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Seaborn settings
sns.set(style="whitegrid")

# Generating random data
rng = np.random.RandomState(4711) #Here we could give any number a a seed for the random number generator
x = rng.rand(50)
y_linear = 2 * x - 0.5 + 0.2*rng.randn(50)

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

In this approach, we use the same underlying concept of minimizing a **loss function**, but the technique is more flexible, allowing us to extend the method to more advanced models such as polynomial or kernel-based regressions, which we'll explore in later sections.

The concept of minimizing a loss function provides the foundation for generalizing regression to other types of models and fitting techniques, allowing us to tackle more complex relationships between variables.

## Scikit-learn

It should be noted that the method described above, by explicitly defining an loss function and minimizing it with a direct call to an optimizer, is selected for explaining something about machine learning, and not the preferd practical approach to e.g. linear regression. Instead linear regression uses an **analytical solution** to the least squares problem, [a well-known method for deriving the optimal parameters](https://en.wikipedia.org/wiki/Least_squares#Solving_the_least_squares_problem). 

Instead of thinking too hard on the implementation, we would use a package for the task. Here, [scikit-learn](https://scikit-learn.org/) is one of the most popular open-source machine learning libraries in Python. It provides simple and efficient tools for data mining and data analysis, making it ideal for academic and industrial applications alike. Built on top of well-established libraries like NumPy, SciPy, and Matplotlib, scikit-learn offers a wide range of machine learning algorithms for both supervised and unsupervised learning tasks.

### Key Features:

- **Easy to use**: scikit-learn’s consistent API and clear documentation make it accessible to beginners.
- **Wide range of algorithms**: It includes support for classification, regression, clustering, dimensionality reduction, and model selection techniques.
- **Interoperable**: scikit-learn integrates seamlessly with other Python libraries like pandas and NumPy, allowing for smooth data handling and preprocessing.

### Linear Regression using Scikit-learn.

The **scikit-learn** package offers efficient functions for performing linear regression right out of the box. Below is an example where we generate random data following a linear trend, then fit a linear model using the `LinearRegression` class.

This approach is both fast and highly optimized, making it easy to apply linear regression with minimal setup.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y_linear)

xfit = np.linspace(0, 1, 1000)
yfit = model.predict(xfit[:, np.newaxis])

# Scatter plot with seaborn
sns.scatterplot(x=x, y=y_linear)
plt.plot(xfit, yfit, color='r')
plt.show()
```
