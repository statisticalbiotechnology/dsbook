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
rng = np.random.RandomState(4711)
x = rng.rand(50)
y_linear = 2 * x - 3 + 0.2*rng.randn(50)

# Scatter plot with seaborn
sns.scatterplot(x=x, y=y_linear)
plt.show()
```

The linear regression model assumes that the relationship between two variables is linear.

## Fitting the Model

To fit a linear model, we can use the optimize function from scipy. To do so we need to define a loss function, i.e. a function that we want to minimize in order to get a fit. A set of parameters that minimize the loss for a function is our definition of optimality.  

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

def minimize_loss(loss, model, x, y, num_params):
    initial_params = np.random.randn(num_params)
    return minimize(loss, initial_params, args=(model, x, y))

# Minimize the loss function
initial_guess_linear = [0, 0]
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

This model fits a line to the data and makes predictions based on this line.

Polynomial Regression
Linear regression can be extended to handle more complex relationships by transforming the input data. Here, we demonstrate polynomial regression.

```{code-cell} ipython3
# Define the polynomial model function
def polynomial_model(x, params, degree=7):
    x_poly = np.vstack([x**i for i in range(degree + 1)]).T
    return np.dot(x_poly, params)

# Generate random data (usingf same x as previous example)
y_non_linear = np.sin(10.*x) + 0.1 * rng.randn(50)

# Polynomial basis function for x
result_poly = minimize_loss(loss, polynomial_model, x, y_non_linear,8)

poly_params = result_poly.x

# Generate fit lines
yfit_poly = polynomial_model(xfit, poly_params)

sns.scatterplot(x=x, y=y_non_linear)
plt.plot(xfit, yfit_poly, color='r')
plt.title("Polynomial Regression Fit")
plt.show()
```

```{code-cell} ipython3

# Define a Gaussian basis function
def gaussian_basis(x, centers, width):
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers[np.newaxis, :]) / width)**2)

# Define the Gaussian model
def gaussian_model(x, params):
    # Split the params into weighths, centers and common width of the bases
    N = len(params)//2
    weights = params[:N]
    centers = params[N:2*N]
    width = params[-1]
    # Calculate the values of each basis for each x 
    basis = gaussian_basis(x, centers, width)
    return np.dot(basis, weights)

N = 15  # Number of Gaussian bases to fit to our data

result_gaussian = minimize_loss(loss, gaussian_model, x, y_non_linear, N*2+1)
gaussian_params = result_gaussian.x

yfit_gauss = gaussian_model(xfit, gaussian_params)

# Plot the gausian fit
sns.scatterplot(x=x, y=y_non_linear)
plt.plot(xfit, yfit_gauss, color='r')
plt.title("Gaussian Bases Fit To Nonlinear data")
plt.show()
```