---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple Regression


```{code-cell} ipython3

import pandas as pd
import cptac
import cptac.utils as ut
en = cptac.Lscc()
en.list_data_sources()
prot = en.get_proteomics("umich")
prot = ut.reduce_multiindex(df=prot, tuples=True)
clin_var = "bmi"
clinical = en.get_clinical('mssm')[[clin_var]]
clin_and_prot = clinical.join(prot, how='inner').dropna(subset=[clin_var])

relevant_prot = [('POLI', 'ENSP00000462664.1'), ('MYL4', 'ENSP00000347055.1'), ('NRP2', 'ENSP00000350432.5'), ('CFHR2', 'ENSP00000356385.4'), ('SMAD2', 'ENSP00000262160.6'), ('KIAA1328', 'ENSP00000280020.5')]
variables_df = clin_and_prot.loc[:, [clin_var] + relevant_prot ].dropna()

```


```{code-cell} ipython3
import numpy as np
from scipy.optimize import minimize

X = variables_df.drop(columns=[clin_var])
y = variables_df[clin_var].values

# Standardize features

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Set the regularization strength 
lambda_ridge = 0.5

# Define the Ridge loss function
def ridge_loss(beta, X, y, lambda_ridge):
    # Prediction
    y_pred = X @ beta[:-1] + beta[-1]  # Last beta is intercept
    # Ridge loss: Sum of Squared Errors + regularization term
    error = y - y_pred
    loss = np.sum(error ** 2) + lambda_ridge * np.sum(beta[:-1] ** 2)
    return loss

# Initial guess for beta (coefficients and intercept)
initial_beta = np.zeros(X.shape[1] + 1)

# Optimize the loss function using scipy's minimize
opt_result = minimize(ridge_loss, initial_beta, args=(X, y, lambda_ridge), method='L-BFGS-B')

# Extract optimized coefficients
optimized_beta = opt_result.x

# Separate coefficients and intercept
coefficients = optimized_beta[:-1]
intercept = optimized_beta[-1]

print(f"Optimized coefficients: {coefficients}")
print(f"Optimized intercept: {intercept}")
```

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt

# Predicted tumor size using optimized coefficients
y_pred = X @ coefficients + intercept
#y_pred = X @ coefficients_sklearn + intercept_sklearn
# y_pred = X @ [0.,0.,0.,0.,1. ] #+ intercept_sklearn

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
