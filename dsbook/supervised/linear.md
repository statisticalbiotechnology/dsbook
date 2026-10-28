---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
downloads:
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/linear.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/linear.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/linear.ipynb
    title: Run on Binder
---

# Regression

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/linear.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/linear.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/linear.ipynb)

## General

The purpose of **regression** is to model the relationship between a **dependent variable** (output) and one or more **independent variables** (inputs). It is one of the most common techniques in statistical modeling, used both to understand and to predict the dependent variable.

### Independent and Dependent Variables

- **Independent variable**: Denoted as $\mathbf{x}_i$, these are the input variables or predictors.
- **Dependent variable**: Denoted as $y_i$, this is the output variable or response we are trying to predict.

## A Linear Regression Model

With a single independent variable $x_i$, the simplest model is a straight line, $f(x_i) = a x_i + b$, where $a$ is the slope of the line and $b$ is the intercept.

We can write this model in a more general form:

```{math}
f(\mathbf{x}_i') = \mathbf{x}_i'^T \boldsymbol{\beta}
```

Where:
- $\boldsymbol{\beta} = (b, a)^T$ are the model parameters (intercept and slope),
- $\mathbf{x}_i' = (1, x_i)^T$ is the extended input vector, which includes a constant 1 for the intercept,
- The goal is to find the $\boldsymbol{\beta}$ that best fits the data.

The same form covers several independent variables: then $\mathbf{x}_i' = (1, \mathbf{x}_i^T)^T$, and $\boldsymbol{\beta}$ holds the intercept and one slope per variable.

### Residuals

In regression, the **residual** for each data point is the difference between the observed value $y_i$ and the value predicted by the model $f(\mathbf{x}_i)$. The residual for the $i$-th point can be written as:

```{math}
e_i = y_i - f(\mathbf{x}_i)
```

Residuals give us a way to evaluate how well the model fits the data. A residual close to zero means that the model's prediction is close to the observed value, while a residual far from zero indicates a larger error.

```{code-cell}
:tags: [hide-cell]

# When the page runs in the browser (JupyterLite), seaborn is not preinstalled, so we install it first
import sys
if sys.platform == "emscripten":
    import micropip
    await micropip.install("seaborn")
```

```{code-cell}
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Seaborn settings
sns.set_theme(style="whitegrid")

N = 30  # Number of data points

# Generating random data
rng = np.random.RandomState(0)
x = rng.rand(N)
y_linear = 2 * x - 0.5 + 0.2*rng.randn(N)

# Define a generic loss function for linear models
def loss(params, model_function, x, y):
    return np.sum((y - model_function(x, params))**2)

# Define the linear model function
def linear_model(x, params):
    slope, intercept = params
    return slope * x + intercept

# A function searching for a set of parameters that minimize a given loss function over data,
# starting from a random, but reproducible, initial guess
def minimize_loss(loss, model, x, y, num_params):
    initial_params = np.random.RandomState(0).randn(num_params)
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
plt.plot(xfit, yfit_linear, color='r', label=r'$f(\mathbf{x})$')

# Create a custom legend entry for the residual lines
plt.plot([x[0], x[0]], [y_linear[0], linear_model(x[0], (slope, intercept))], color='blue', label='Residual')

# Draw the residual of each data point as a vertical line
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y_linear[i], linear_model(x[i], (slope, intercept))], color='blue')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

In the figure above, the vertical blue lines are the residuals: they show how much of each data point the model $f(\mathbf{x})$ leaves unexplained.

### Least Squares Estimation

To fit a function $f(\mathbf{x})$ to the data points $\{(\mathbf{x}_i, y_i)\}_i$, we minimize the sum of squared errors (SSE):

```{math}
\mathcal{L}_{\text{SSE}}=\sum_{i=1}^N e_i^2 = \sum_{i=1}^N (y_i - f(\mathbf{x}_i) )^2
```
This process is known as **least squares**. $\mathcal{L}$ is known as a **loss function**. The loss measures how poorly a model fits the data, and different regression lines have different losses. By minimizing the loss function, we find the function that best fits the data according to the selected loss function.

If the errors $e_i$ are independent and normally distributed with the same variance, the **maximum likelihood estimation (MLE)** is [equivalent](https://stats.stackexchange.com/questions/143705/maximum-likelihood-method-vs-least-squares-method) to least squares minimization, making this approach optimal under these assumptions.

Minimizing the SSE gives the same parameters as minimizing the more commonly reported mean squared error (MSE), which divides the SSE by the number of data points $N$:

```{math}
\mathcal{L}_{\text{MSE}}=\frac{1}{N}\sum_{i=1}^N e_i^2 = \frac{1}{N}\sum_{i=1}^N (y_i - f(\mathbf{x}_i) )^2
```

### Fit a Linear Model to Data

A general pattern in machine learning is to use an optimizer to minimize a loss function. Here, we define the SSE loss explicitly and minimize it with the `scipy.optimize.minimize` function. Nothing in this procedure is specific to straight lines, so later in this chapter we use the same code to fit non-linear models. Try changing the number of data points `N` in the code, and see how the fitted slope and intercept change.

```{code-cell}
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Seaborn settings
sns.set_theme(style="whitegrid")

N = 50  # Number of data points

# Generating random data
rng = np.random.RandomState(0) # Any number can be used as seed for the random number generator
x = rng.rand(N)
y_linear = 2 * x - 0.5 + 0.2*rng.randn(N)

# Define a generic loss function for both linear and polynomial models
def loss(params, model_function, x, y):
    return np.sum((y - model_function(x, params))**2)

# Define the linear model function
def linear_model(x, params):
    slope, intercept = params
    return slope * x + intercept

# A function searching for a set of parameters that minimize a given loss function over data,
# starting from a random, but reproducible, initial guess
def minimize_loss(loss, model, x, y, num_params):
    initial_params = np.random.RandomState(0).randn(num_params)
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

## Notes on Optimization

By default, `scipy.optimize.minimize` uses the [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm), a refinement of gradient descent, to find an optimal set of parameters $(a, b)$. Gradient descent is an iterative algorithm for finding the minimum of a function; here, the function is the loss, and we search for the parameters that minimize it.

### Concept of Gradient Descent

The basic idea of gradient descent is to update the parameters in the opposite direction of the gradient of the loss function with respect to the parameters. This is because the gradient points in the direction of the steepest ascent, and we want to go in the opposite direction to find the minimum.

Below, we use gradient descent to find the optimal $a$ and $b$ for a linear regression model, and draw its steps on a contour plot of the loss as a function of $a$ and $b$.

```{code-cell}
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Loss evaluated on a grid of (a, b) values
def compute_loss(a, b, x, y):
    # Add an axis to a and b so that they broadcast against the data points
    return np.sum((y - (a[..., np.newaxis] * x + b[..., np.newaxis]))**2, axis=-1)

# Define the gradient of the loss function
def compute_gradient(a, b, x, y):
    grad_a = -2 * np.sum(x * (y - (a * x + b)))
    grad_b = -2 * np.sum(y - (a * x + b))
    return grad_a, grad_b

# Gradient descent parameters
learning_rate = 0.01
n_iterations = 100

# Initial guess
a, b = 0.0, 0.0

# Data
rng = np.random.RandomState(0)
x_ = np.array(rng.rand(30))
y_ = np.array(2 * x_ - 0.5 + 0.2*rng.randn(30))

# Store the steps
steps_a = [a]
steps_b = [b]

# Gradient descent loop
for i in range(n_iterations):
    grad_a, grad_b = compute_gradient(a, b, x_, y_)
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    steps_a.append(a)
    steps_b.append(b)

# The minimum of the same loss, found with scipy's optimizer
slope_min, intercept_min = minimize_loss(loss, linear_model, x_, y_, 2).x

# Create a grid for contour plot
a_values = np.linspace(-0.5, 3.0, 100)
b_values = np.linspace(-1, 0.5, 100)
a_grid, b_grid = np.meshgrid(a_values, b_values)
loss_grid = compute_loss(a_grid, b_grid, x_, y_)

# Plotting the contour plot of the loss function
plt.figure(figsize=(10, 6))
plt.contourf(a_grid, b_grid, loss_grid, levels=50, cmap='viridis')
plt.colorbar(label='Loss magnitude')
plt.scatter(steps_a, steps_b, color='red', s=50, label=f'Gradient descent ({n_iterations} steps)')
plt.plot(steps_a, steps_b, color='white', linestyle='--', linewidth=1)
plt.scatter([slope_min], [intercept_min], color='white', edgecolor='black', s=60, label='Minimum of the loss')
plt.title('Contour plot of the loss function with Gradient Descent Steps')
plt.xlabel('Slope (a)')
plt.ylabel('Intercept (b)')
plt.legend()
plt.show()

# Output the parameters
print(f"After {n_iterations} steps: slope {a:.3f}, intercept {b:.3f}")
print(f"Minimum of the loss: slope {slope_min:.3f}, intercept {intercept_min:.3f}")
```

After 100 steps, gradient descent is still some distance from the minimum: the loss forms a long, narrow valley, and the steps shrink as they follow its nearly flat floor.

````{note}
### Mathematical Formulation of Gradient Descent

To understand gradient descent mathematically, let’s start with a simple one-dimensional case. Suppose we have a function $f(\beta)$ and want to find its minimum. The derivative of $f$ with respect to $\beta$, denoted as $f'(\beta)$, gives the slope of the function at any point $\beta$. Gradient descent updates $\beta$ iteratively in the opposite direction of $f'(\beta)$ to move toward the minimum:

```{math}
\beta_{\text{new}} = \beta - \eta f'(\beta)
```

where $\eta$, the learning rate, is a small positive number that sets the step size. This process continues until $f'(\beta) \approx 0$, indicating that we have reached a local minimum.

#### Extending to Multiple Dimensions

In higher dimensions, we consider a multivariable function $\mathcal{L}(\boldsymbol{\beta})$, where $\boldsymbol{\beta} = \begin{bmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n \end{bmatrix}$ represents the parameters of the model in column vector form, and $\mathcal{L}$ is the loss function we aim to minimize. Instead of a single derivative, we use the **gradient**, denoted $\nabla \mathcal{L}(\boldsymbol{\beta})$, which is a column vector of partial derivatives with respect to each parameter:

```{math}
\nabla \mathcal{L}(\boldsymbol{\beta}) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \beta_1} \\ \frac{\partial \mathcal{L}}{\partial \beta_2} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial \beta_n} \end{bmatrix}
```

Each component $\frac{\partial \mathcal{L}}{\partial \beta_i}$ represents the rate of change of $\mathcal{L}$ with respect to $\beta_i$. The gradient $\nabla \mathcal{L}(\boldsymbol{\beta})$ points in the direction of the steepest increase of $\mathcal{L}$; therefore, moving in the opposite direction of $\nabla \mathcal{L}(\boldsymbol{\beta})$ decreases $\mathcal{L}$.

The gradient descent update rule for multiple parameters is:

```{math}
\boldsymbol{\beta}_{\text{new}} = \boldsymbol{\beta} - \eta \nabla \mathcal{L}(\boldsymbol{\beta})
```

#### Applying Gradient Descent to Linear Regression

In our linear regression model, we minimize the sum of squared errors $\mathcal{L}(a, b)$ for the parameters $a$ (slope) and $b$ (intercept). The gradient of $\mathcal{L}$ with respect to $a$ and $b$ consists of the partial derivatives $\frac{\partial \mathcal{L}}{\partial a}$ and $\frac{\partial \mathcal{L}}{\partial b}$, which we can express in column vector form, in the same order as $\boldsymbol{\beta} = (b, a)^T$, as:

```{math}
\nabla \mathcal{L}(a, b) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial b} \\ \frac{\partial \mathcal{L}}{\partial a} \end{bmatrix} = \begin{bmatrix} -2 \sum_i \left( y_i - (a x_i + b) \right) \\ -2 \sum_i x_i \left( y_i - (a x_i + b) \right) \end{bmatrix}
```

The updates for $a$ and $b$ are then:

```{math}
\begin{bmatrix} b_{\text{new}} \\ a_{\text{new}} \end{bmatrix} = \begin{bmatrix} b \\ a \end{bmatrix} - \eta \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial b} \\ \frac{\partial \mathcal{L}}{\partial a} \end{bmatrix}
```
````

## Scikit-learn

Defining a loss function explicitly and handing it to a general-purpose optimizer is a good way to explain how machine learning works, but it is not how linear regression is usually done in practice. The least-squares problem has an [analytical solution](https://en.wikipedia.org/wiki/Least_squares#Solving_the_least_squares_problem), which software packages use instead. In practice, we use a package such as [scikit-learn](https://scikit-learn.org/), one of the most popular machine learning libraries in Python, which implements a wide range of supervised and unsupervised methods.

### Linear Regression using Scikit-learn

Below, we fit the same data as above with scikit-learn's `LinearRegression` class.

```{code-cell}
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

## Regression with Basis Functions

So far our model has been a straight line, but many relationships are curved. This section shows that the same recipe, choosing a model and minimizing its loss, also fits curved relationships. The trick is to turn each input $x$ into several new features, such as $x$, $x^2$ and $x^3$, and to fit a weighted sum of these features. The model is still linear in its parameters, so the fitting does not change; only the features do. The functions that create the features are called *basis functions*.

### Polynomial Basis Functions

A common choice is **polynomial basis functions**: from each input $x$ we create the terms $x^2$, $x^3$, and so on, which let the model capture curved patterns. Instead of fitting a linear model $y = a x + b$, we then fit a polynomial of degree $n$. This transforms the input space into a higher-dimensional space where the relationship between $x$ and $y$ may be linear, even if it appears non-linear in the original space. Below, we fit a polynomial of degree 7, which has 8 parameters, to 50 data points. Try changing the number of data points `N` and the degree of the polynomial `degree` in the code, and see how the fit changes.

```{code-cell}
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Seaborn settings
sns.set_theme(style="whitegrid")

N = 50      # Number of data points
degree = 7  # Degree of the polynomial

# Generating random data
rng = np.random.RandomState(1)
x = rng.rand(N)
y_non_linear = np.sin(10.*x) + 0.1 * rng.randn(N)

# Define the polynomial model function, where the degree is given by the number of parameters
def polynomial_model(x, params):
    degree = len(params) - 1
    x_poly = np.vstack([x**i for i in range(degree + 1)]).T
    return np.dot(x_poly, params)


# Optimize the parameters of our model using the same SSE loss function as before.
# A polynomial of degree n has n+1 parameters
result_poly = minimize_loss(loss, polynomial_model, x, y_non_linear, degree + 1)

poly_params = result_poly.x

# Generate fit lines
yfit_poly = polynomial_model(xfit, poly_params)

sns.scatterplot(x=x, y=y_non_linear)
plt.plot(xfit, yfit_poly, color='r')
plt.ylim(-2, 2)  # Fixed range, so that overfitted curves don't hide the data
plt.title("Polynomial Regression Fit")
plt.show()
```

### Optional: Gaussian Basis Functions

This last part of the chapter is optional reading. Another choice is **Gaussian basis functions**, also known as **radial basis functions (RBF)**. Each one is a bell-shaped bump around a center, so a weighted sum of them can follow localized, non-linear behavior in the data. Below, we use 11 Gaussian bases and let the optimizer fit not only their weights but also their centers and their common width. That makes the model non-linear in its parameters, but it is still fitted in exactly the same way, by minimizing the SSE.

```{code-cell}
# Define a Gaussian basis function
def gaussian_basis(x, centers, width):
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers[np.newaxis, :]) / width)**2)

# Define the Gaussian model
def gaussian_model(x, params):
    # Split the params into weights, centers and common width of the bases
    n_bases = len(params)//2
    weights = params[:n_bases]
    centers = params[n_bases:2*n_bases]
    width = params[-1]
    # Calculate the values of each basis for each x
    basis = gaussian_basis(x, centers, width)
    return np.dot(basis, weights)

n_bases = 11  # Number of Gaussian bases to fit to our data

# Optimize the parameters of our model using the same SSE loss function as before
result_gaussian = minimize_loss(loss, gaussian_model, x, y_non_linear, n_bases*2+1)
gaussian_params = result_gaussian.x

yfit_gauss = gaussian_model(xfit, gaussian_params)

# Plot the Gaussian fit
sns.scatterplot(x=x, y=y_non_linear)
plt.plot(xfit, yfit_gauss, color='r')
plt.ylim(-2, 2)
plt.show()
```

With many basis functions, such flexible models can fit the noise in the data as well as the underlying pattern. The next chapter introduces regularization, which controls model complexity and so counteracts such overfitting.

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Explain what regression is used for, and identify the dependent variable and the independent variables in a regression problem.
2. Write a linear model as $f(\mathbf{x}_i') = \mathbf{x}_i'^T \boldsymbol{\beta}$, and explain how adding a constant 1 to the input vector handles the intercept.
3. Define a residual. Calculate the residuals, the sum of squared errors (SSE) and the mean squared error (MSE) for a small dataset by hand.
4. Explain least squares as minimizing a loss function, and state under which assumption about the errors it is the same as maximum likelihood estimation.
5. Describe how gradient descent minimizes a loss: what the gradient is, the update rule $\boldsymbol{\beta}_{\text{new}} = \boldsymbol{\beta} - \eta \nabla \mathcal{L}(\boldsymbol{\beta})$, and the role of the learning rate $\eta$.
6. Explain how polynomial basis functions let a regression model fit non-linear relationships, while the fitting procedure (minimizing a loss) stays the same.
:::
