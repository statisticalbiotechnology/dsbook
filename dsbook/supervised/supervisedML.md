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

We begin by reading in the TCGA breast cancer dataset, in which we will compare progesterone receptor positive (PR+) with progesterone receptor negative (PR-) cancers.

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

Then we read the clinical information of the samples and extract their PR status. We keep only the samples whose PR status was determined as positive or negative, and drop the few gene symbols that occur more than once in the expression table. As we will log-transform the expression values, we also discard every gene with a zero in any sample. Note that this removes many genes, among them *PGR*, the gene encoding the progesterone receptor itself.

```{code-cell} ipython3
brca_clin = tcga.get_clinical_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://datahub.assets.cbioportal.org/brca_tcga_pub2015.tar.gz',"data_clinical_sample.txt")
pr_status = brca_clin.loc["PR_STATUS_BY_IHC"]
pr_known = pr_status.isin(["Positive", "Negative"])   # Drop samples with unknown or indeterminate PR status
brca = brca.loc[:, pr_known.index[pr_known]]
pr_bool = (pr_status[pr_known] == "Positive")
brca = brca.loc[~brca.index.duplicated(keep=False)]   # Drop gene symbols that occur more than once
brca.dropna(axis=0, how='any', inplace=True)
brca = brca.loc[~(brca<=0.0).any(axis=1)]
brca = np.log2(brca)
print(f"{brca.shape[1]} samples ({pr_bool.sum()} PR+ and {(~pr_bool).sum()} PR-), {brca.shape[0]} genes")
```

We then calculate how significantly each gene is differentially expressed between PR+ and PR- cancers, and sort the genes accordingly. The procedure will be explained in the statistics part of this book.

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
pd.DataFrame(data = confusion_matrix(y, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actual_PR+"])
```

Fantastic! The classifier manages to use junk data to perfectly separate our PR+ from PR- cancers. 

However, before we call NEJM, let's try to see if we can separate an *independent* test set in the same manner. We use the function `train_test_split` to divide the data into 60% training data and 40% test data.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
clf = svm.LinearSVC(C=1,max_iter=5000).fit(X_train, y_train) # Train an SVM
y_pred = clf.predict(X_test)                              # Predict labels for the given features
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actual_PR+"])
```

In this setting, the classifier has no predictive power: it classifies 44% of the test samples correctly, fewer than the 67% we would get by always predicting PR+.

The reason for the discrepancy between the two predictors is that in both cases the large number of variables makes the predictor overfit to the data. In the first instance, we could not detect the problem as we were testing on the overfitted data. However, when holding out a separate test set, the predictor's weak performance was blatantly visible.

+++

## A low dimensional classifier

Let's now focus on an alternative setting, where we instead select five genes which are among the most differentially expressed transcripts when comparing PR+ and PR-. How would we combine their expression values optimally? Again we begin by standardizing our features.

Note that we select these genes using the labels of *all* samples, including the ones that will end up in the test set, and that the scaler below is likewise fitted on all samples. Further, the penalty $C$ that we choose below on one training set is then reused in a cross validation over all samples, including that training set. These are mild forms of data leakage, and they make the performance we report below somewhat optimistic. A fully honest evaluation would repeat the gene selection, the scaling and the choice of $C$ inside every training fold, as in the nested cross validation of the [validation chapter](./xval.md).

```{code-cell} ipython3
top5=brca.loc[qvalues.iloc[[1,2,5,6,9],:].index]
scaler = StandardScaler()
X = scaler.fit_transform(top5.values.T) # Scale all gene expression values to stdv =1 and mean =0
y = 2*pr_bool.values.astype(int) - 1           # transform from bool to -1 and 1
```

We then separate 30% of our cancers into a separate test set. The function `GridSearchCV` uses cross validation (k=5) to select the penalty parameter $C$ from a list of candidate values.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
param_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
clf = GridSearchCV(svm.LinearSVC(max_iter=10000000,class_weight="balanced"), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)
best_C = clf.best_params_['C']
print("Best parameters for the model: " + str(clf.best_params_))
y_pred = clf.predict(X_test)
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actual_PR+"])
```

With the selected penalty $C=0.1$, we now perform a cross validation (k=5) on the full data set: we train one classifier on each of the five training sets, and merge their predictions on the corresponding validation folds into one combined result.

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold

y_pred, y_real = np.array([]), np.array([])
skf = StratifiedKFold(n_splits=5)
for train_id, test_id in skf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_id,:], X[test_id,:], y[train_id],y[test_id]
    clf = svm.LinearSVC(C=best_C,max_iter=100000,class_weight="balanced").fit(X_train, y_train) # Train an SVM
    y_pred_fold = clf.predict(X_test)                                # Predict labels for the given features
    y_pred = np.concatenate([y_pred,y_pred_fold])
    y_real = np.concatenate([y_real,y_test])
pd.DataFrame(data = confusion_matrix(y_real, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actual_PR+"])
```

## Study questions

1. **Data Preprocessing Effects**:
   - Modify the data preprocessing steps (such as normalization or feature scaling) and observe how these changes affect the outcomes of the models. Which preprocessing step had the most significant impact?

2. **Feature Weights**:
    - For the overoptimistic predictor, investigate which features are given the highest weight, by studying the `clf.coef_` vector. Do these genes have any particular relevance for the disease?

3. **Cross-validation**:
   - Vary the number of folds in the cross validation of the low dimensional classifier. Does this affect the outcome?

4. **Type of classifier**
    - Replace the LinearSVC classifier by any non-linear classifier from the scikit-learn library, e.g. [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting). How does this affect the behaviour of the classifier?

5. **Model Robustness**:
    - Introduce noise to the data (e.g., randomly flip labels or add outliers) and re-evaluate the model's performance. Which model handles noise better, and why do you think that is?

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Explain why a linear SVM trained on 1000 genes that are unrelated to PR status can still separate the training samples perfectly, and why only an independent test set reveals that this is overfitting.
2. Interpret a confusion matrix of actual against predicted classes, and calculate the accuracy, the fraction of correctly classified samples, from it.
3. Identify data leakage in a study design, such as selecting differentially expressed genes or fitting the standardization on all samples before the test set is split off, and explain why it makes the reported performance optimistic.
:::
