---
file_format: mystnb
kernelspec:
  name: python3
---

# Assessing Significance Using Permutation Testing

## Introduction

In regression models, we often want to assess the significance of the model fit and individual coefficients. Instead of relying on classical statistical tests like the **F-test**, we can use **permutation testing** to evaluate significance. This approach involves creating a **null distribution** by permuting the dependent variable (the target) and calculating the corresponding loss values. We then compare the observed loss to this distribution to compute a **p-value**.

### Why Permutation Testing?

Permutation testing is non-parametric, meaning it doesn’t assume a specific distribution of the residuals or the coefficients. This makes it a powerful and flexible method, especially in cases where the assumptions of parametric tests (e.g., normality) might not hold.

## Permutation Testing for Model Significance

In permutation testing, we test the null hypothesis that there is no relationship between the dependent and independent variables. To do this, we randomly shuffle the dependent variable **y** while keeping the independent variable **x** unchanged, refit the model, and compute the loss. Repeating this process many times creates a **null distribution** of losses. The p-value is calculated by comparing the observed loss to this null distribution.

### Step-by-step Procedure

1. **Fit the model** on the original data and calculate the observed loss.
2. **Permute** the dependent variable (shuffle **y**) and refit the model on the permuted data to calculate the loss under the null hypothesis.
3. **Repeat** the permutation process many times to build a distribution of losses under the null hypothesis.
4. Calculate the **p-value** by comparing the observed loss to this distribution.

### Code Example: Permutation Test for Model Significance

Below is an example of how to implement permutation testing for assessing the significance of a regression model.

```{code-cell} ipython3
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic data
rng = np.random.RandomState(42)
x = rng.rand(50)
y = 2 * x - 0.5 + 0.2 * rng.randn(50)

# Define the linear model
def linear_model(params, x):
    slope, intercept = params
    return slope * x + intercept

# Define the loss function (sum of squared residuals)
def loss(params, x, y):
    return np.sum((y - linear_model(params, x))**2)

# Fit the model to the original data
initial_params = np.random.randn(2)
result = minimize(loss, initial_params, args=(x, y))
observed_loss = result.fun

# Permutation test
n_permutations = 1000
permuted_losses = []

for _ in range(n_permutations):
    # Shuffle y and fit the model to the permuted data
    y_permuted = np.random.permutation(y)
    permuted_result = minimize(loss, initial_params, args=(x, y_permuted))
    permuted_losses.append(permuted_result.fun)

# Compute the p-value by comparing the observed loss to the null distribution
permuted_losses = np.array(permuted_losses)
p_value = np.mean(permuted_losses <= observed_loss)

# Plot the null distribution and observed loss
plt.hist(permuted_losses, bins=30, alpha=0.7, label="Permuted Losses")
plt.axvline(observed_loss, color='r', linestyle='--', label=f"Observed Loss = {observed_loss:.2f}")
plt.title(f"Permutation Test for Model Significance (p-value = {p_value:.4f})")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Observed loss: {observed_loss:.4f}")
print(f"p-value from permutation test: {p_value:.4f}")
```

### Explanation of the Code

1. **Observed Loss**: The model is fit to the original data, and the observed loss is computed.
2. **Permutation Procedure**: The dependent variable **y** is shuffled, and a new model is fit to the permuted data. The loss is calculated for each permuted model.
3. **Null Distribution**: After performing many permutations (in this case, 1000), we have a distribution of losses under the null hypothesis.
4. **p-value**: The p-value is computed as the proportion of permuted losses that are less than or equal to the observed loss. A small p-value indicates that the observed loss is significantly smaller than what we would expect under the null hypothesis, suggesting that the model is meaningful.

## Interpreting the Results

- If the **p-value** is small (typically less than 0.05), we reject the null hypothesis, concluding that the model is statistically significant and the relationship between the independent and dependent variables is unlikely to have occurred by chance.
- If the p-value is large, we fail to reject the null hypothesis, suggesting that the model does not capture a meaningful relationship.

## Permutation Test for Coefficients

We can extend this approach to test the significance of individual coefficients. Instead of permuting the entire **y** variable, we could permute only one feature or generate a sampling distribution for each coefficient to assess its significance.

Here’s how we can conduct a permutation test for the slope coefficient.

### Code Example: Permutation Test for Slope Significance

```{code-cell} ipython3
# Define a function to compute the slope from the regression parameters
def extract_slope(params):
    return params[0]  # The slope is the first parameter in the model

# Observed slope from the original data
observed_slope = extract_slope(result.x)

# Permutation test for slope significance
permuted_slopes = []

for _ in range(n_permutations):
    y_permuted = np.random.permutation(y)
    permuted_result = minimize(loss, initial_params, args=(x, y_permuted))
    permuted_slope = extract_slope(permuted_result.x)
    permuted_slopes.append(permuted_slope)

# Compute the p-value by comparing the observed slope to the null distribution of slopes
permuted_slopes = np.array(permuted_slopes)
p_value_slope = np.mean(np.abs(permuted_slopes) >= np.abs(observed_slope))

# Plot the null distribution of slopes and observed slope
plt.hist(permuted_slopes, bins=30, alpha=0.7, label="Permuted Slopes")
plt.axvline(observed_slope, color='r', linestyle='--', label=f"Observed Slope = {observed_slope:.2f}")
plt.title(f"Permutation Test for Slope Significance (p-value = {p_value_slope:.4f})")
plt.xlabel("Slope")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Observed slope: {observed_slope:.4f}")
print(f"p-value for slope significance: {p_value_slope:.4f}")
```

### Interpretation of Slope Permutation Test

- The **permutation test for the slope** generates a null distribution of slopes by permuting the dependent variable and calculating the slope for each permuted dataset.
- The **p-value** is computed by comparing the observed slope to this null distribution. If the p-value is small, it suggests that the slope is significantly different from zero, meaning the independent variable has a meaningful effect on the dependent variable.

## Conclusion

Using **permutation testing**, we can assess the significance of regression models and individual coefficients in a flexible, non-parametric way. By generating a null distribution of losses (or coefficients) through random permutations, we can compute p-values and determine whether the observed model captures a meaningful relationship between the independent and dependent variables.

Permutation testing is a powerful alternative to traditional parametric tests like the **F-test** and **t-tests**, especially when the underlying assumptions of those tests may not hold.