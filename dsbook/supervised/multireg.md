---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple Regression Example

In this example, we apply multiple regression to analyze the relationship between several proteins and a clinical variable, BMI (Body Mass Index). We perform Ridge Regression using regularization to prevent overfitting, and visualize how well the model predicts BMI from the selected protein features.

## Data Loading and Preparation

The first step involves loading and preparing data from the CPTAC (Clinical Proteomic Tumor Analysis Consortium) database for Lung Squamous Cell Carcinoma (LSCC). We retrieve proteomics data and a relevant clinical variable (BMI), then merge these datasets based on matching patient records.

```{code-cell} ipython3
:tags: [hide-output]
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

* **Clinical Data**: We extract BMI as the clinical variable.
* **Proteomics Data**: We use proteomic measurements from several proteins (POLI, MYL4, NRP2, CFHR2, SMAD2, and KIAA1328) as our independent variables.
* **Data Joining**: The clinical and proteomics data are merged into a single dataframe and missing values are handled.

## Defining the Ridge Regression Model

Ridge regression adds a regularization term to penalize large coefficients, helping to control model complexity and reduce overfitting. The following steps define the ridge regression loss function and optimize it using the `scipy.optimize.minimize` function.

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

* **Ridge Loss Function**: The loss function includes the sum of squared errors (SSE) and a regularization term (𝜆) applied to the coefficients.
* **Optimization**: We initialize the coefficients and intercept to zero and use minimize to find the optimal values by minimizing the ridge loss function.

## Visualization: Predicted vs Actual BMI

After fitting the model, we calculate the predicted BMI values and plot them against the actual BMI values to evaluate the model's performance.

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

* **Scatter Plot**: The plot compares the actual BMI values against the predicted values from the model. A red dashed line indicates the ideal scenario where predictions perfectly match the actual values.
* **Visual Evaluation**: If the points lie close to the red line, the model’s predictions are accurate. Deviations from this line represent prediction errors.

This example demonstrates how multiple regression can be extended with regularization to improve model generalization, particularly when working with clinical and proteomic datasets.

