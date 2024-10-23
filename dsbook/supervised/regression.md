---
file_format: mystnb
kernelspec:
  name: python3
---

# Non-Linear Regression with Kernel Methods

Linear regression is one of the simplest and most widely used models for supervised learning. In this chapter, we will explore how to expand the linear regression model to model non-linear data, by expanding our data into multiple features with *kernel functions*.

## Polynomial Kernels as Basis Functions

In the previous sections, we explored linear regression and saw that the goal is to minimize a **loss function** (such as the sum of squared residuals) to obtain the best-fit model. This concept is central to regression tasks, regardless of whether the model is linear or non-linear.

One powerful approach for non-linear regression is to use **polynomial basis functions**. Polynomial regression can be thought of as applying a kernel transformation to the data by creating higher-order polynomial terms, such as $x^2$, $x^3$, and so on. These terms allow us to capture more complex patterns in the data.

For example, instead of fitting a linear model $y = a x + b$, we can fit a polynomial of degree $n$, which effectively transforms the input space into a higher-dimensional space where the relationship between $x$ and $y$ may be linear, even if the relationship appears non-linear in the original space.

In fact, **polynomial kernels**  can be used to implicitly transform the data without explicitly computing the higher-dimensional features. In kernel-based models like Support Vector Machines (SVM) or kernel ridge regression, a polynomial kernel can be used to fit more complex functions to the data while still employing the same optimization techniques as in linear regression.


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
y_non_linear = np.sin(10.*x) + 0.1 * rng.randn(50)

# Define the polynomial model function
def polynomial_model(x, params, degree=7):
    x_poly = np.vstack([x**i for i in range(degree + 1)]).T
    return np.dot(x_poly, params)


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

### Gaussian Kernels

We can further extend this concept by introducing other types of kernel functions, such as **Gaussian kernels**. These kernels, also known as **radial basis functions (RBF)**, allow us to capture localized, non-linear behavior in the data. In essence, Gaussian kernels provide a more flexible model that can handle highly non-linear patterns.

### Unifying Loss Minimization Across Models

What is essential in all these models—whether linear, polynomial, or Gaussian—is that we continue to use the same framework of **minimizing a loss function** to find the best-fit parameters. The kernel-based approach simply transforms the input space, allowing us to apply non-linear transformations without changing the fundamental concept of minimizing error.

In the next chapter, we will explore **more advanced loss functions** and **regularization techniques**, which help control model complexity and prevent overfitting, particularly important in high-dimensional kernel-transformed spaces.

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

By focusing on the unified concept of **loss minimization**, we can see that even complex, non-linear models follow the same principles as basic linear regression—only the structure of the model and the basis functions (kernels) change.
