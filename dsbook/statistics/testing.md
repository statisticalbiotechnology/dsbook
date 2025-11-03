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

+++ {"slideshow": {"slide_type": "slide"}}

# Differential expression anlaysis of the TCGA breast cancer set

First we retrieve the breast cancer RNAseq data as well as the clinical classification of the sets from cbioportal.org. 

The gene expresion data is stored in the [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) `brca`, and the adherent clinical information of the cancers and their patients is stored in the [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) `brca_clin`. It can be woth exploring these data structures.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
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
brca_clin = tcga.get_clinical_data(my_path + "../data/brca_tcga_pub2015.tar.gz", 'https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pub2015.tar.gz',"data_clinical_sample.txt")
```

+++ {"slideshow": {"slide_type": "slide"}}

Before any further analysis we clean our data. This includes removal of genes where no transcripts were found for any of the samples , i.e. their values are either [NaN](https://en.wikipedia.org/wiki/NaN) or zero. 

The data is also log transformed. It is generally assumed that expression values follow a log-normal distribution, and hence the log transformation implies that the new values follow a nomal distribution.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
brca.dropna(axis=0, how='any', inplace=True)
brca = brca.loc[~(brca<=0.0).any(axis=1)]
brca = pd.DataFrame(data=np.log2(brca),index=brca.index,columns=brca.columns)
```

+++ {"slideshow": {"slide_type": "slide"}}

We can get an overview of the expression data, i.e differnt characterizations of the tumors from other sources (patient file, histological analysis etc) than the expression data:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
brca
```

+++ {"slideshow": {"slide_type": "slide"}}

and the clinical data:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
brca_clin
```

+++ {"slideshow": {"slide_type": "slide"}}

### Differential expression analysis

The goal of the excercise is to determine which genes that are differentially expressed in so called tripple negative cancers as compared to other cancers. A breast cancer is triple negative when it does not express either [Progesterone receptors](https://en.wikipedia.org/wiki/Progesterone_receptor), [Estrogen receptors](https://en.wikipedia.org/wiki/Estrogen_receptor) or [Epidermal growth factor receptor 2](https://en.wikipedia.org/wiki/HER2/neu). Such cancers are known to behave different than other cancers, and are not amendable to regular [hormonal theraphies](https://en.wikipedia.org/wiki/Hormonal_therapy_(oncology)).

We first create a vector of booleans, that track which cancers that are tripple negative. This will be needed as an input for subsequent significance estimation.

```{code-cell} ipython3
brca_clin.index
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
brca_clin.loc["3N"]= (brca_clin.loc["PR_STATUS_BY_IHC"]=="Negative") & (brca_clin.loc["ER_STATUS_BY_IHC"]=="Negative") & (brca_clin.loc["IHC_HER2"]=="Negative")
tripple_negative_bool = (brca_clin.loc["3N"] == True)
```

+++ {"slideshow": {"slide_type": "slide"}}

Next, for each transcript that has been measured, we calculate (1) log of the average Fold Change difference between tripple negative and other cancers, and (2) the significance of the difference between tripple negative and other cancers.

An easy way to do so is by defining a separate function, `get_significance_two_groups(row)`, that can do such calculations for any row of the `brca` DataFrame, and subsequently we use the function `apply` for the function to execute on each row of the DataFrame. For the significance test we use a $t$ test, which is provided by the function [`ttest_ind`.](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

This results in a new table with gene names and their $p$ values of differential concentration, and their fold changes.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def get_significance_two_groups(row):
    log_fold_change = row[tripple_negative_bool].mean() - row[~tripple_negative_bool].mean() # Calculate the log Fold Change
    p = ttest_ind(row[tripple_negative_bool],row[~tripple_negative_bool],equal_var=False)[1] # Calculate the significance
    return [p,-np.log10(p),log_fold_change]

pvalues = brca.apply(get_significance_two_groups,axis=1,result_type="expand")
pvalues.rename(columns = {list(pvalues)[0]: 'p', list(pvalues)[1]: '-log_p', list(pvalues)[2]: 'log_FC'}, inplace = True)
```

+++ {"slideshow": {"slide_type": "slide"}}

The resulting list can be further investigated.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
pvalues
```

+++ {"slideshow": {"slide_type": "slide"}}

A common way to illustrate the diffrential expression values are by plotting the negative log of the $p$ values, as a function of the mean [fold change](https://en.wikipedia.org/wiki/Fold_change) of each transcript. This is known as a [Volcano plot](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)).

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
sns.set_context("talk")
ax = sns.relplot(data=pvalues,x="log_FC",y="-log_p",aspect=1.5,height=6)
ax.set(xlabel="$log_2(TN/not TN)$", ylabel="$-log_{10}(p)$");
```

+++ {"slideshow": {"slide_type": "fragment"}}

The regular interpretation of a Volcano plot is that the ges in the top left and the top right corner are the most interesting ones, as the have a large fold change between the conditions as well as being very significant.
