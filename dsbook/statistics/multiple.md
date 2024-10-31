# Multiple Hypothesis Corrections

## Introduction

In data science, particularly in the context of biotechnology, the analysis often involves testing a large number of hypotheses simultaneously. This is common in fields such as genomics, transcriptomics, and proteomics, where experiments might involve thousands of genes or proteins being tested for associations with a condition or trait. The challenge of multiple hypothesis testing arises because, when many tests are conducted, the likelihood of obtaining false positives (incorrectly rejecting a null hypothesis) increases.

To understand this, imagine testing 1,000 hypotheses with a significance threshold ($\alpha$) of 0.05. If the null hypothesis is true for all these tests, we would still expect about 5% of them, or 50 tests, to be significant simply by chance. Without any correction for multiple comparisons, the results are prone to contain many false positives, reducing the reliability of the conclusions drawn from the data.

Multiple hypothesis correction methods are therefore crucial in order to control error rates and ensure that the results we report are genuinely significant and not just due to random variation. Different correction techniques control different types of error rates, such as the Family-Wise Error Rate (FWER) or the False Discovery Rate (FDR), which help us balance the trade-off between sensitivity (finding true effects) and specificity (avoiding false positives).

### Bonferroni Correction

The Bonferroni correction is one of the simplest and most conservative methods for controlling the Family-Wise Error Rate (FWER). The FWER is the probability of making at least one false positive among all the hypotheses tested. The Bonferroni correction aims to reduce the chance of any false positives by adjusting the significance threshold for each individual test.

The Bonferroni method works as follows:

1. Given a significance level $\alpha$, divide it by the number of hypotheses $m$ being tested to obtain the adjusted significance level:

```{math}
\alpha_{\text{adjusted}} = \frac{\alpha}{m}
```
2. Conduct each individual hypothesis test using this adjusted significance level $\alpha_{\text{adjusted}}$. If a p-value is less than $\alpha_{\text{adjusted}}$, the corresponding hypothesis is considered significant.

By using the Bonferroni correction, we control the probability of making at least one false positive across all the tests, thus providing a stringent control of the error rate. However, this method can be overly conservative, especially when dealing with a large number of hypotheses, leading to a higher chance of missing true effects (i.e., a loss of statistical power).

### Calculating the Upper Bound of the False Discovery Rate with the Benjamini-Hochberg Procedure

The Benjamini-Hochberg (BH) procedure is a popular method for controlling the False Discovery Rate (FDR), which is the expected proportion of false positives among the rejected hypotheses. Unlike more conservative approaches like the Bonferroni correction, which aims to eliminate all false positives at the cost of potentially missing true effects, the BH method aims to maintain a balance by controlling the rate of false discoveries rather than eliminating them entirely.

The BH procedure involves ranking the p-values from all the tests in ascending order. Let’s denote these ordered p-values as $p_{(1)}, p_{(2)}, \dots, p_{(m)}$, where $m$ is the total number of hypotheses. The method works as follows:

1. Rank all the p-values such that $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Choose a desired FDR level, denoted as $q$.
3. For each p-value $p_{(i)}$, calculate the threshold value $\frac{i}{m} \times q$, where $i$ is the rank of the p-value and $m$ is the total number of hypotheses.
4. Find the largest $i$ for which $p_{(i)} \le \frac{i}{m} \times q$. Reject all hypotheses with p-values less than or equal to $p_{(i)}$.

By using this approach, the BH procedure ensures that the expected proportion of false discoveries is controlled at the desired level $q$, allowing us to draw more reliable conclusions while still maintaining a reasonable sensitivity to detect true effects.

### Calculating q-values with Storey's Method

Storey introduced an alternative to the BH procedure for controlling the FDR, known as the q-value approach. The q-value can be thought of as the FDR analogue of the p-value; it provides a measure of significance that directly incorporates the concept of multiple hypothesis testing, offering an estimate of the proportion of false positives incurred when calling a particular test significant.

Storey's method for calculating q-values starts similarly to the BH procedure by ranking p-values. However, it also introduces an estimate of the proportion of null hypotheses, denoted as $\pi_0$, which represents the fraction of hypotheses for which the null hypothesis is true. This allows for a more adaptive and potentially less conservative estimate of the FDR.

Here are the steps to calculate q-values using Storey's approach:

1. **Estimate $\pi_0$**: This is done by evaluating the distribution of p-values. Storey suggested estimating $\pi_0$ by calculating the proportion of p-values greater than a threshold, typically chosen to be around 0.5 or higher. Formally, 
   
```{math}
\pi_0 = \frac{\#\{ p_i > \lambda \}}{m (1 - \lambda)}
```
   
where $\lambda$ is a tuning parameter, often set between 0.5 and 0.95.

2. **Calculate the q-value for each p-value**: After estimating $\pi_0$, the next step is to calculate the q-values. For each p-value $p_{(i)}$, the q-value is defined as:

```{math}
q_{(i)} = \min_{j \ge i} \left( \frac{\pi_0 m p_{(j)}}{j} \right)
```

This step ensures that the q-values are monotonically increasing, which is important for interpretability, as higher p-values should not have lower q-values.

3. **Interpretation**: Once calculated, the q-value for each test can be interpreted as the minimum FDR at which that particular test would be called significant. This allows researchers to directly assess the reliability of their findings, helping to identify the most robust results in the context of multiple hypothesis testing.

Storey's method provides a more flexible framework for controlling the FDR, especially in situations where a significant proportion of the tests are expected to be truly null. By adapting the estimate of $\pi_0$, Storey's approach can lead to more powerful inferences compared to traditional methods.
