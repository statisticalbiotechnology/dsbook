---
jupytext:
  formats: md:myst
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
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/multireg.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/multireg.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/multireg.ipynb
    title: Run on Binder
---

# Multiple Regression Example

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/multireg.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/multireg.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/multireg.ipynb)

In this example, we apply multiple regression to analyze the relationship between several proteins and a clinical variable, BMI (Body Mass Index). We fit the model with a ridge penalty and plot predicted against actual BMI.

## Data Loading and Preparation

The first step involves loading and preparing data from the CPTAC (Clinical Proteomic Tumor Analysis Consortium) database for Lung Squamous Cell Carcinoma (LSCC). We retrieve proteomics data and the clinical variable BMI, then merge these datasets based on matching patient records.

```{code-cell}
:tags: [hide-cell]

import importlib.util, sys
def _has(pkg): return importlib.util.find_spec(pkg) is not None
if not _has("cptac"):
    %pip install -q "cptac>=1.5.13"
```

```{code-cell}
:tags: [hide-output]

import pandas as pd
import cptac
import cptac.utils as ut
lscc = cptac.Lscc()
prot = lscc.get_proteomics("umich")
prot = ut.reduce_multiindex(df=prot, tuples=True)
clin_var = "bmi"
clinical = lscc.get_clinical('mssm')[[clin_var]]
clin_and_prot = clinical.join(prot, how='inner').dropna(subset=[clin_var])

relevant_prot = [('POLI', 'ENSP00000462664.1'), ('MYL4', 'ENSP00000347055.1'), ('NRP2', 'ENSP00000350432.5'), ('CFHR2', 'ENSP00000356385.4'), ('SMAD2', 'ENSP00000262160.6'), ('KIAA1328', 'ENSP00000280020.5')]
variables_df = clin_and_prot.loc[:, [clin_var] + relevant_prot ].dropna()
```

We use the levels of six proteins (POLI, MYL4, NRP2, CFHR2, SMAD2 and KIAA1328) as independent variables, and keep the 96 patients with complete data. Note that these six proteins were picked using the same data that we fit below: they are among the 25 proteins, out of the roughly 10,000 measured in nearly all patients, whose levels correlate most strongly with BMI in these patients.

## Defining the Ridge Regression Model

We standardize each protein to mean 0 and standard deviation 1, so that the ridge penalty treats all proteins alike, and minimize the ridge loss with `scipy.optimize.minimize`. The intercept, the last element of `beta`, is left out of the penalty, so that the fit does not depend on the average BMI. For comparison, we also fit the model without a penalty ($\lambda = 0$).

```{code-cell}
import numpy as np
from scipy.optimize import minimize

X = variables_df.drop(columns=[clin_var]).to_numpy()
y = variables_df[clin_var].to_numpy()

# Standardize features to mean 0 and standard deviation 1
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Set the regularization strength
lambda_ridge = 50

# Define the Ridge loss function
def ridge_loss(beta, X, y, lambda_ridge):
    # Prediction
    y_pred = X @ beta[:-1] + beta[-1]  # Last beta is intercept
    # Ridge loss: Sum of Squared Errors + penalty on the coefficients (not on the intercept)
    error = y - y_pred
    loss = np.sum(error ** 2) + lambda_ridge * np.sum(beta[:-1] ** 2)
    return loss

# Initial guess for beta (coefficients and intercept)
initial_beta = np.zeros(X.shape[1] + 1)

# Fit the model without a penalty, for comparison, and with the ridge penalty
unpenalized_beta = minimize(ridge_loss, initial_beta, args=(X, y, 0), method='L-BFGS-B').x
opt_result = minimize(ridge_loss, initial_beta, args=(X, y, lambda_ridge), method='L-BFGS-B')

# Extract optimized coefficients
optimized_beta = opt_result.x

# Separate coefficients and intercept
coefficients = optimized_beta[:-1]
intercept = optimized_beta[-1]

print(f"Coefficients without penalty: {np.round(unpenalized_beta[:-1], 2)}")
print(f"Ridge coefficients:           {np.round(coefficients, 2)}")
print(f"Ridge intercept: {intercept:.2f}")
```

As the proteins are standardized, each coefficient is the change in predicted BMI per standard deviation of the protein's level. The penalty shrinks the larger coefficients and makes them more similar to each other. Here we picked $\lambda = 50$ by hand; the [validation chapter](xval.md) shows how to choose it from the data.

## Visualization: Predicted vs Actual BMI

After fitting the model, we calculate the predicted BMI values and plot them against the actual BMI values.

```{code-cell}
import seaborn as sns
import matplotlib.pyplot as plt

# Predicted BMI using the ridge coefficients
y_pred = X @ coefficients + intercept

# Create a DataFrame for easy plotting
plot_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# Plot predicted vs actual using seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(data=plot_df, x='Actual', y='Predicted', color='blue', s=60)
sns.lineplot(x=[y.min(), y.max()], y=[y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.title('Predicted vs Actual BMI')
plt.grid(True)
plt.show()
```

The dashed red line marks perfect predictions. Because we evaluate the model on the same patients that it was fitted to, using proteins that were selected for their correlation with BMI in these patients, this plot overstates how well the model would predict BMI for new patients. The [validation chapter](xval.md) shows how to measure that properly.

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Set up a multiple regression where several predictors, here protein levels, together predict a clinical variable such as BMI.
2. Explain how such a model is fitted by minimizing a ridge loss, why the features are standardized first, and why the intercept is left out of the penalty.
3. Judge how well a fitted model predicts by plotting predicted against actual values, and explain why such a plot overstates the performance when the model, or the choice of its features, was fitted to the same data.
:::
