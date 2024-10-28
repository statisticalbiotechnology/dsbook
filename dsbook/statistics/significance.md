---
file_format: mystnb
kernelspec:
  name: python3
---

# Hypothesis Testing

Hypothesis testing is a statistical procedure used to determine if a sample data set provides sufficient evidence to reject a stated null hypothesis (*H₀*) in favor of an alternative hypothesis (*H₁*). This method is fundamental in science for drawing inferences about populations based on sample data. It allows researchers to make data-driven decisions and evaluate the likelihood that their observations are due to random chance or a genuine effect.

## Null Hypothesis (*H₀*) and Alternative Hypothesis (*H₁*)

In hypothesis testing, we begin by stating two competing hypotheses:
- **Null Hypothesis (*H₀*)**: This is a statement suggesting there is no effect, relationship, or difference in the population. It represents the status quo or a baseline assumption. For example, in a clinical trial, *H₀* might state that a new drug has no effect compared to a placebo.
- **Alternative Hypothesis (*H₁*)**: This hypothesis reflects what the researcher aims to prove. It suggests that there is an effect, a relationship, or a difference. In the clinical trial example, *H₁* would state that the new drug has a beneficial effect compared to the placebo.

The purpose of hypothesis testing is to assess whether the data provides enough evidence to reject *H₀* in favor of *H₁*. This process involves comparing the observed data to what we would expect under *H₀*.

## Test Statistics

A **test statistic** is a value calculated from sample data that allows us to make a decision about the hypotheses. One commonly used test statistic is the **difference in means** between two groups. For example, if we want to compare the average effect of a treatment versus a placebo, we calculate the difference in the sample means for the two groups. The test statistic helps determine how far the observed data deviates from what we would expect under *H₀*, which typically assumes that there is no difference in means between the groups.

## Sampling Distribution under the Null Hypothesis

The **sampling distribution under the null hypothesis** is the distribution of the test statistic assuming that *H₀* is true. This distribution helps us understand the range of possible values the test statistic can take if the null hypothesis is correct. By comparing the observed test statistic to this distribution, we can determine how likely it is to observe such a value by random chance alone. This comparison is essential for calculating the *p* value.

## $p$ value

The **$p$ value** represents the probability of obtaining the observed data, or something more extreme, if the null hypothesis were true. It is used as a measure of evidence against *H₀*:
- A smaller $p$ value suggests that the observed data would be unlikely under *H₀*, providing stronger evidence against the null hypothesis.
- For example, a $p$ value of 0.03 means there is a 3% chance of observing the data (or something more extreme) if *H₀* is true. This small probability indicates that the data is not consistent with *H₀*, leading us to consider rejecting it.

## Significance Level (α)

The **significance level** (denoted as α) is a predetermined threshold that determines whether the *p* value is considered small enough to reject the null hypothesis. A common choice for α is 0.05, which means we are willing to accept a 5% chance of incorrectly rejecting *H₀* (Type I error):
- If the *p* value is less than α, the results are considered **statistically significant**, and we reject *H₀*.
- The choice of α depends on the context of the study and the consequences of making an error. For example, in medical research, a smaller α (e.g., 0.01) might be chosen to reduce the risk of false positives.

## False Positives and False Negatives

In hypothesis testing, two types of errors can occur:
- **False Positive (Type I Error)**: Rejecting *H₀* when it is actually true. This error can lead to incorrect conclusions, such as believing a treatment is effective when it is not.
- **False Negative (Type II Error)**: Failing to reject *H₀* when the alternative hypothesis is true. This error can result in missed discoveries, such as failing to detect a real effect or relationship.

The balance between Type I and Type II errors is crucial in hypothesis testing. Researchers often need to consider the trade-offs between these errors when designing studies and choosing significance levels.

## Limitations & Misinterpretations

While hypothesis testing is powerful, it has its limitations and can be easily misinterpreted:
- **$p$ values** provide the strength of evidence against *H₀*, but they do not indicate the size of the effect or its practical significance. A small $p$ value might suggest a statistically significant result, but the actual effect size could be trivial.
- A **$p$ value** does not give the probability that either hypothesis is true. It only tells us how compatible the observed data is with *H₀*.
- Results can be **statistically significant** without being **practically significant**, and vice versa. Thus, it is crucial to interpret the results in the context of the research question and the real-world implications. For instance, a medical treatment might show a statistically significant improvement, but the actual benefit to patients might be minimal.

## Assessing Significance Using Permutation Testing

In regression models, we often want to assess the significance of the model fit and individual coefficients. Instead of relying on classical statistical tests like the **F-test**, we can use **permutation testing** to evaluate significance. This approach involves creating a **null distribution** by permuting the dependent variable (the target) and calculating the corresponding loss values. We then compare the observed loss to this distribution to compute a **$p$ value**.

### Why Permutation Testing?

Permutation testing is non-parametric, meaning it doesn’t assume a specific distribution of the residuals or the coefficients. This makes it a powerful and flexible method, especially in cases where the assumptions of parametric tests (e.g., normality) might not hold.

## Permutation Testing for Model Significance

In permutation testing, we test the null hypothesis that there is no relationship between the dependent and independent variables. To do this, we randomly shuffle the dependent variable **y** while keeping the independent variable **x** unchanged, refit the model, and compute the loss. Repeating this process many times creates a **null distribution** of losses. The $p$ value is calculated by comparing the observed loss to this null distribution.

### Step-by-step Procedure

1. **Fit the model** on the original data and calculate the observed loss.
2. **Permute** the dependent variable (shuffle **y**) and refit the model on the permuted data to calculate the loss under the null hypothesis.
3. **Repeat** the permutation process many times to build a distribution of losses under the null hypothesis.
4. Calculate the **$p$ value** by comparing the observed loss to this distribution.

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
y = 1.2 * x - 0.5 + 1.1 * rng.randn(50)

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
print(f"p value from permutation test: {p_value:.4f}")
```

### Explanation of the Code

1. **Observed Loss**: The model is fit to the original data, and the observed loss is computed.
2. **Permutation Procedure**: The dependent variable **y** is shuffled, and a new model is fit to the permuted data. The loss is calculated for each permuted model.
3. **Null Distribution**: After performing many permutations (in this case, 1000), we have a distribution of losses under the null hypothesis.
4. **$p$ value**: The $p$ value is computed as the proportion of permuted losses that are less than or equal to the observed loss. A small $p$ value indicates that the observed loss is significantly smaller than what we would expect under the null hypothesis, suggesting that the model is meaningful.

## Interpreting the Results

- If the **$p$ value** is small (typically less than 0.05), we reject the null hypothesis, concluding that the model is statistically significant and the relationship between the independent and dependent variables is unlikely to have occurred by chance.
- If the $p$ value is large, we fail to reject the null hypothesis, suggesting that the model does not capture a meaningful relationship.

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