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
---

# Example of classification

We begin by reading in the TCGA Breast Cancer dataset, and calculate significance of the measured genes as differntial expressed when comparing Progesterone Positive with Progesterone Negative cancers.

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

brca = tcga.get_expression_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pub2015.tar.gz',"data_mrna_seq_v2_rsem.txt")
```

Then we read the clinical information of the samples and extract the PR status of our samples.

```{code-cell} ipython3
brca_clin = tcga.get_clinical_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pub2015.tar.gz',"data_clinical_sample.txt")
brca.dropna(axis=0, how='any', inplace=True)
brca = brca.loc[~(brca<=0.0).any(axis=1)]
brca = pd.DataFrame(data=np.log2(brca),index=brca.index,columns=brca.columns)
brca_clin.loc["PR"]= (brca_clin.loc["PR_STATUS_BY_IHC"]!="Negative") 
pr_bool = (brca_clin.loc["PR"] == True)
```

We then select the most differential genes. The procedure selecting such genes will hopefully be more cleare after the statistics part of this course

```{code-cell} ipython3
def get_significance_two_groups(row):
    log_fold_change = row[pr_bool].mean() - row[~pr_bool].mean()
    p = ttest_ind(row[pr_bool],row[~pr_bool],equal_var=False)[1]
    return [p,-np.log10(p),log_fold_change]

pvalues = brca.apply(get_significance_two_groups,axis=1,result_type="expand")
pvalues.rename(columns = {list(pvalues)[0]: 'p', list(pvalues)[1]: '-log_p', list(pvalues)[2]: 'log_FC'}, inplace = True)
qvalues = qvalue.qvalues(pvalues)
```

## The overoptimstic investigator

We begin with a case of supervised machine learning aimed as a warning, as it illustrates the importance of separating training from testing data.

Imagine a situation where we want to find the best combination of genes unrelated to a condition that still are telling of the condition. Does that sound like an imposibility, it is because it is imposible. However, there is nothing stopping us to try.

So first we select the 1000 genes which are the least differentialy expressed genes when comparing PR positive with PR negative breast cancers.

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

We are now ready to try to train a linear SVM for the task of predictin PR negatives from PR positives. We test the performance of our classifier on the training data.

```{code-cell} ipython3
from sklearn import svm
from sklearn.metrics import confusion_matrix
clf = svm.LinearSVC(C=1,max_iter=5000).fit(X, y)  # Train a SVM
y_pred = clf.predict(X)                        # Predict labels for the give features
pd.DataFrame(data = confusion_matrix(y, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

Fantastic! The classifier manage to use junk data to perfectly separate our PR+ from PR- cancers. 

However, before we call NEJM, lets try to see if we can sparate an *independent* test set in the same manner. We use the function train_test_split to divide the data into 60% training data and 40% test data.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
clf = svm.LinearSVC(C=1,max_iter=5000).fit(X_train, y_train) # Train an SVM
y_pred = clf.predict(X_test)                              # Predict labels for the give features
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

In this setting, the classifier seems to have very little predictive power.  

The reason for the discrepency of the two predictors are that in both cases the large number of variables makes the predictor to overfit to the data. In the first instance, we could not detect the problem as we were testing on the overfitted data. However, when holding out a separate test set, the predictors weak performance was blatantly visible.

+++

## A low dimensional classifier

Lets now focus on an alternative setting, where we instead select six separate genes which are among the most differentially expressed transcripts when comparing PR+ and PR-.

How would we combine their expression values optimaly? 

Again we begin by standardize our features.

```{code-cell} ipython3
top6=brca.loc[qvalues.iloc[[1,2,5,6,9],:].index]
scaler = StandardScaler()
X = scaler.fit_transform(top6.values.T) # Scale all gene expression values to stdv =1 and mean =0
y = 2*pr_bool.values.astype(int) - 1           # transform from bool to -1 and 1
```

We then separate 40% of our cancers into a separate test set. The function $GridSearchCV$ use cross validation (k=5) to select an optimal slack penalty $C$ out from a vector of differnt choices.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
param_grid = [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]
clf = GridSearchCV(svm.LinearSVC(max_iter=10000000,class_weight="balanced"), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)
print("Best cross validation accuracy for the model: " + str(clf.best_params_))
y_pred = clf.predict(X_test)
pd.DataFrame(data = confusion_matrix(y_test, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

Given the choise of penalty $C=0.1$, we can now perform a cross validation (k=5) on the full data set. Here we will train thee separate classifiers on ech cross validation training set, and subsequently merge each such predictor's prediction into one combined result.

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold

y_pred, y_real = np.array([]), np.array([])
skf = StratifiedKFold(n_splits=5)
for train_id, test_id in skf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_id,:], X[test_id,:], y[train_id],y[test_id]
    clf = svm.LinearSVC(C=0.1,max_iter=100000).fit(X_train, y_train) # Train an SVM
    y_pred_fold = clf.predict(X_test)                                # Predict labels for the give features
    y_pred = np.concatenate([y_pred,y_pred_fold])
    y_real = np.concatenate([y_real,y_test])
pd.DataFrame(data = confusion_matrix(y_real, y_pred),columns = ["predicted_PR-","predicted_PR+"],index=["actual_PR-","actualPR+"])
```

### Some study questions for the notebook

1. **Data Preprocessing Effects**:
   - Modify the data preprocessing steps (such as normalization or feature scaling) and observe how these changes affect the outcomes of the models. Which preprocessing step had the most significant impact?

2. **Feature Weights**:
    - Fot the overoptimistic predictor, investigate which features that are given the highest weight, by studying the `clf.coef_` vector. Do these genes have any particular relevance for the disaease?

3. **Cross-validation**:
   - Vary the number of folds in the cross validation of the low dimensional classifier. Does this effect the outcome?

4. **Type of classifier**
    - Replace the LinearSVM classifier by any non-linear classifier from the sk-learn library, e.g. [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting). How does this effect the behaviour of the classifier?
  
5.  **Model Robustness**:
    - Introduce noise to the data (e.g., randomly flip labels or add outliers) and re-evaluate the model's performance. Which model handles noise better, and why do you think that is?
