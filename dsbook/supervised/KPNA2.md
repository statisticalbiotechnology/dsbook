---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: jb
  language: python
  name: python3
downloads:
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/KPNA2.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/KPNA2.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/KPNA2.ipynb
    title: Run on Binder
---

# Multiple Regression Analysis of KPNA2 Gene Expression

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/KPNA2.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/KPNA2.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/KPNA2.ipynb)

Breast cancer is a heterogeneous disease with varied clinical outcomes, and understanding the molecular factors involved in its progression is critical for improving prognostic capabilities and therapeutic strategies. One area of focus has been gene expression profiling, which provides insights into the molecular pathways associated with disease aggressiveness and progression.

The data used here come from [a study of the molecular basis of histologic grade in breast cancer](https://doi.org/10.1093/jnci/djj052). From that dataset we focus on a single gene, KPNA2, which encodes Karyopherin Subunit Alpha 2, a protein involved in nuclear transport. KPNA2 has been identified as a potential biomarker in breast cancer due to its association with poor prognosis and its role in various cellular processes that are critical for tumorigenesis, such as cell proliferation, differentiation, and DNA repair. Elevated levels of KPNA2 have been observed in more aggressive cancers and correlate with adverse clinical outcomes, including higher tumor grade and lymph node involvement.

The dataset used here includes KPNA2 gene expression measurements from breast cancer patients, along with clinical data on histologic grade, lymph node status, tumour size and patient age. Histologic grade categorizes tumor cells based on their appearance and organization, providing a measure of tumor aggressiveness. Lymph node status indicates whether the axillary lymph nodes were free of tumour cells (coded as 0, node-negative) or contained metastatic tumour cells (coded as 1, node-positive), which is a standard indicator of how far the tumour has spread.

In this analysis, we apply a multiple regression model to examine the relationship between KPNA2 expression and clinical predictors, specifically histologic grade, lymph node status and tumour size. By modeling these associations, we aim to quantify the extent to which these clinical factors are related to KPNA2 expression, thus enhancing our understanding of its role in breast cancer progression.

### Read data

We begin by reading a dataset, and plotting the first entries of the data set.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load data (assuming data URL or local file is accessible)
try:
    gene_table = pd.read_csv('brc.txt')
except:
    from urllib.request import urlopen
    url = 'https://raw.githubusercontent.com/statOmics/statisticalGenomicsCourse/master/tutorial1/gse2990BreastcancerOneGene.txt'
    gene_table = pd.read_table(urlopen(url), sep=" ")
    gene_table.to_csv('brc.txt')

# Clean data
gene_table.head()
```

We add a column with the log of the KNAP2 gene expression value

```{code-cell} ipython3
# Log-transform the KPNA2 gene expression values
gene_table["log_gene"] = np.log(gene_table["gene"])
gene_table.head()
```

Subsequently we plot the (logged) gene expression as function of node status and tumor grade.

```{code-cell} ipython3
# Visualize KPNA2 expression by histologic grade and lymph node status
plt.figure(figsize=(10, 6))
sns.boxplot(x='grade', y='log_gene', hue='node', data=gene_table)
plt.xlabel('Histologic Grade')
plt.ylabel('Log-transformed KPNA2 Expression')
plt.legend(title='Lymph Node Status')
plt.show() 
```

We then model the logged gene expression as a function on tumor size, as well as a function of Grade, Node status and size. As a first check we investigate the residual errors of the predictor.

```{code-cell} ipython3
# Perform two regressions: one with only 'size' and another with 'grade', 'node', and 'size'

# Regression with only 'size' as the independent variable
y = gene_table['log_gene']
X_size = gene_table[['size']]
model_size = LinearRegression()
model_size.fit(X_size, y)
gene_table['predicted_log_gene_size'] = model_size.predict(X_size)

# Regression with 'grade', 'node', and 'size' as independent variables
X_all = gene_table[['grade', 'node', 'size']]
model_all = LinearRegression()
model_all.fit(X_all, y)
gene_table['predicted_log_gene_all'] = model_all.predict(X_all)

# Calculate residuals for both regression models

# Residuals for model with only 'size' as the independent variable
gene_table['residuals_size'] = gene_table['log_gene'] - gene_table['predicted_log_gene_size']

# Residuals for model with 'grade', 'node', and 'size' as independent variables
gene_table['residuals_all'] = gene_table['log_gene'] - gene_table['predicted_log_gene_all']

# Plot the residuals for both models
plt.figure(figsize=(12, 6))

# Residuals for size-only model
plt.subplot(1, 2, 1)
plt.scatter(gene_table['size'], gene_table['residuals_size'], color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Tumor Size')
plt.ylabel('Residuals')
plt.ylim(-1,1)
plt.title('Residuals (Size-Only Model)')
plt.grid(True)

# Residuals for full model with grade, node, and size
plt.subplot(1, 2, 2)
plt.scatter(gene_table['size'], gene_table['residuals_all'], color='green', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Tumor Size')
plt.ylabel('Residuals')
plt.ylim(-1,1)
plt.title('Residuals (Full Model: Grade, Node, Size)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

The errors are smaller when we apply the full model compared to the regression based only on tumor size. We also plot the predictions as a function of actual value.

```{code-cell} ipython3
# Scatterplots of actual vs predicted log_gene values for both models

plt.figure(figsize=(12, 6))

# Scatterplot for size-only model predictions
plt.subplot(1, 2, 1)
plt.scatter(gene_table['log_gene'], gene_table['predicted_log_gene_size'], color='blue', alpha=0.5)
plt.plot([gene_table['log_gene'].min(), gene_table['log_gene'].max()], 
         [gene_table['log_gene'].min(), gene_table['log_gene'].max()], 
         'r--', linewidth=1)  # Line y=x for reference
plt.xlabel('Actual Log-transformed KPNA2 Expression')
plt.ylabel('Predicted Log-transformed KPNA2 (Size Only)')
plt.title('Actual vs Predicted (Size Only Model)')
plt.grid(True)

# Scatterplot for full model predictions
plt.subplot(1, 2, 2)
plt.scatter(gene_table['log_gene'], gene_table['predicted_log_gene_all'], color='green', alpha=0.5)
plt.plot([gene_table['log_gene'].min(), gene_table['log_gene'].max()], 
         [gene_table['log_gene'].min(), gene_table['log_gene'].max()], 
         'r--', linewidth=1)  # Line y=x for reference
plt.xlabel('Actual Log-transformed KPNA2 Expression')
plt.ylabel('Predicted Log-transformed KPNA2 (Grade, Node, Size)')
plt.title('Actual vs Predicted (Full Model: Grade, Node, Size)')
plt.grid(True)

plt.tight_layout()
plt.show()
```
