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
execute:
  skip: true  # built by hand: run `make` locally and commit the executed .ipynb
downloads:
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/supervisedML.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/supervisedML.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/supervisedML.ipynb
    title: Run on Binder
---

# Example of classification

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/supervised/supervisedML.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/supervised/supervisedML.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/supervised/supervisedML.ipynb)

We begin by reading in the TCGA Breast Cancer dataset, and calculate significance of the measured genes as differentially expressed when comparing Progesterone Positive with Progesterone Negative cancers.

```{code-cell} ipython3
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    ![ ! -f "dsbook/README.md" ] && git clone https://github.com/statisticalbiotechnology/dsbook.git
    my_path = "dsbook/dsbook/common/"
else:
    my_path = "../common/"
sys.path.append(my_path) # Read local modules for tcga access and qvalue calculations
import load_tcga as tcga
import qvalue 

brca = tcga.get_expression_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://datahub.assets.cbioportal.org/brca_tcga_pub2015.tar.gz',"data_mrna_seq_v2_rsem.txt")
```

Then we read the clinical information of the samples and extract the PR status of our samples.

```{code-cell} ipython3
brca_clin = tcga.get_clinical_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://datahub.assets.cbioportal.org/brca_tcga_pub2015.tar.gz',"data_clinical_sample.txt")
brca.dropna(axis=0, how='any', inplace=True)
brca = brca.loc[~(brca<=0.0).any(axis=1)]
brca = pd.DataFrame(data=np.log2(brca),index=brca.index,columns=brca.columns)
brca_clin.loc["PR"]= (brca_clin.loc["PR_STATUS_BY_IHC"]!="Negative") 
pr_bool = (brca_clin.loc["PR"] == True)
```

We then select the most differential genes. The procedure selecting such genes will hopefully be more clear after the statistics part of this course.

```{code-cell} ipython3
def get_significance_two_groups(row):
    log_fold_change = row[pr_bool].mean() - row[~pr_bool].mean()
    p = ttest_ind(row[pr_bool],row[~pr_bool],equal_var=False)[1]
    return [p,-np.log10(p),log_fold_change]

pvalues = brca.apply(get_significance_two_groups,axis=1,result_type="expand")
pvalues.rename(columns = {list(pvalues)[0]: 'p', list(pvalues)[1]: '-log_p', list(pvalues)[2]: 'log_FC'}, inplace = True)
qvalues = qvalue.qvalues(pvalues)
```

## The overoptimistic investigator

We begin with a case of supervised machine learning aimed as a warning, as it illustrates the importance of separating training from testing data.

Imagine a situation where we want to find the best combination of genes unrelated to a condition that still are telling of the condition. If that sounds like an impossibility, it is because it is impossible. However, there is nothing stopping us from trying.

So first we select the 1000 genes which are the least differentially expressed genes when comparing PR positive with PR negative breast cancers.

```{code-cell} ipython3
last1k=brca.loc[qvalues.iloc[-1000:,:].index]
```

Subsequently we standardize the data, i.e. we assure a standard deviation of 1 and a mean of zero for every gene among our 1k genes.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(last1k.values.T)  # Scale all gene expression values to stdv =1 and mean =0
y = 2*pr_bool.values.astype(int) - 1       # transform from bool to -1 and 1
```

We are now ready to try to train a linear SVM for the task of separating PR negatives from PR positives. We test the performance of our classifier on the training data.

```{code-cell} ipython3
from sklearn import svm
from sklearn.metrics import confusion_matrix
clf = svm.LinearSVC(C=1,max_iter=5000).fit(X, y)  # Train a SVM
y_pred = clf.predict(X)                        # Predict labels for the given features
pd.DataFrame(data = confusion_matrix(y, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

Fantastic! The classifier manages to use junk data to perfectly separate our PR+ from PR- cancers. 

However, before we call NEJM, let's try to see if we can separate an *independent* test set in the same manner. We use the function train_test_split to divide the data into 60% training data and 40% test data.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
clf = svm.LinearSVC(C=1,max_iter=5000).fit(X_train, y_train) # Train an SVM
y_pred = clf.predict(X_test)                              # Predict labels for the given features
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

In this setting, the classifier seems to have very little predictive power.  

The reason for the discrepancy between the two predictors is that in both cases the large number of variables makes the predictor overfit to the data. In the first instance, we could not detect the problem as we were testing on the overfitted data. However, when holding out a separate test set, the predictor's weak performance was blatantly visible.

+++

## A low dimensional classifier

Let's now focus on an alternative setting, where we instead select five separate genes which are among the most differentially expressed transcripts when comparing PR+ and PR-.

How would we combine their expression values optimally? 

Again we begin by standardizing our features.

Note that we select these genes using the labels of *all* samples, including the ones that will end up in the test set, and that the scaler below is likewise fitted on all samples. This is a mild form of data leakage, and it makes the performance we report below somewhat optimistic. A fully honest evaluation would repeat both the gene selection and the scaling inside every training fold.

```{code-cell} ipython3
top5=brca.loc[qvalues.iloc[[1,2,5,6,9],:].index]
scaler = StandardScaler()
X = scaler.fit_transform(top5.values.T) # Scale all gene expression values to stdv =1 and mean =0
y = 2*pr_bool.values.astype(int) - 1           # transform from bool to -1 and 1
```

We then separate 30% of our cancers into a separate test set. The function `GridSearchCV` uses cross validation (k=5) to select an optimal slack penalty $C$ out from a vector of different choices.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
param_grid = [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]
clf = GridSearchCV(svm.LinearSVC(max_iter=10000000,class_weight="balanced"), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)
print("Best parameters for the model: " + str(clf.best_params_))
y_pred = clf.predict(X_test)
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

Given the choice of penalty $C=0.1$, we can now perform a cross validation (k=5) on the full data set. Here we will train five separate classifiers on each cross validation training set, and subsequently merge each such predictor's prediction into one combined result.

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold

y_pred, y_real = np.array([]), np.array([])
skf = StratifiedKFold(n_splits=5)
for train_id, test_id in skf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_id,:], X[test_id,:], y[train_id],y[test_id]
    clf = svm.LinearSVC(C=0.1,max_iter=100000,class_weight="balanced").fit(X_train, y_train) # Train an SVM
    y_pred_fold = clf.predict(X_test)                                # Predict labels for the given features
    y_pred = np.concatenate([y_pred,y_pred_fold])
    y_real = np.concatenate([y_real,y_test])
pd.DataFrame(data = confusion_matrix(y_real, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

### Some study questions for the notebook

1. **Data Preprocessing Effects**:
   - Modify the data preprocessing steps (such as normalization or feature scaling) and observe how these changes affect the outcomes of the models. Which preprocessing step had the most significant impact?

2. **Feature Weights**:
    - For the overoptimistic predictor, investigate which features that are given the highest weight, by studying the `clf.coef_` vector. Do these genes have any particular relevance for the disease?

3. **Cross-validation**:
   - Vary the number of folds in the cross validation of the low dimensional classifier. Does this affect the outcome?

4. **Type of classifier**
    - Replace the LinearSVC classifier by any non-linear classifier from the sk-learn library, e.g. [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting). How does this affect the behaviour of the classifier?
  
5.  **Model Robustness**:
    - Introduce noise to the data (e.g., randomly flip labels or add outliers) and re-evaluate the model's performance. Which model handles noise better, and why do you think that is?

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Explain why a linear SVM trained on 1000 genes that are unrelated to PR status can still separate the training samples perfectly, and why only an independent test set reveals that this is overfitting.
2. Interpret a confusion matrix of actual against predicted classes, and calculate the accuracy, the fraction of correctly classified samples, from it.
3. Identify data leakage in a study design, such as selecting differentially expressed genes or fitting the standardization on all samples before the test set is split off, and explain why it makes the reported performance optimistic.
:::
