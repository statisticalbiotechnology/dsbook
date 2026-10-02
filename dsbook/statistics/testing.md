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
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/testing.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/testing.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/testing.ipynb
    title: Run on Binder
---

+++ {"slideshow": {"slide_type": "slide"}}

# Differential expression analysis of a highly replicated yeast experiment

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/testing.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/testing.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/testing.ipynb)

Most RNA-seq experiments you will meet in the literature compare three samples against three samples. That is far too few replicates to see what a $p$ value *distribution* actually looks like. We will therefore work with an unusually generous data set: the yeast RNA-seq benchmark produced by the Barton group in Dundee, in which a *snf2* knockout mutant (`snf2`, a deletion of the [SNF2](https://www.yeastgenome.org/locus/S000005816) gene of the SWI/SNF chromatin remodelling complex) was compared to an isogenic wild type (`WT`), with **48 biological replicates of each condition**.

The experiment was designed exactly for the purpose of studying the statistics of differential expression, and is described in [Gierliński et al. (2015)](https://doi.org/10.1093/bioinformatics/btv425) and [Schurch et al. (2016)](https://doi.org/10.1261/rna.053959.115). The raw reads are deposited at ENA under accession [PRJEB5348](https://www.ebi.ac.uk/ena/browser/view/PRJEB5348), and the gene count matrices we use here are distributed in the authors' repository [bartongroup/profDGE48](https://github.com/bartongroup/profDGE48).

We download two small archives, one per condition, each holding 48 [HTSeq-count](https://htseq.readthedocs.io/) files with one read count per gene. The counts are collected into a single [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) `yeast`, whose rows are genes and whose columns are the 96 samples.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Cache the downloaded archives, so that we only fetch them once
data_dir = "../data" if os.path.isdir("../data") else "data"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://raw.githubusercontent.com/bartongroup/profDGE48/master/Preprocessed_data/"

def read_count_data(condition, file_name):
    "Download (once) and read the per-replicate count files of one condition."
    local_file = os.path.join(data_dir, file_name)
    if not os.path.exists(local_file):
        urllib.request.urlretrieve(base_url + file_name, local_file)
    columns = {}
    with tarfile.open(local_file) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            replicate = member.name.split("_")[1]          # e.g. "rep07"
            sample = condition + "_" + replicate           # e.g. "Snf2_rep07"
            counts = pd.read_csv(tar.extractfile(member), sep="\t",
                                 header=None, index_col=0, names=["gene", sample])
            columns[sample] = counts[sample]
    return pd.DataFrame(columns).sort_index(axis=1)

wt = read_count_data("WT", "WT_countdata.tar.gz")
snf2 = read_count_data("Snf2", "Snf2_countdata.tar.gz")
yeast = pd.concat([wt, snf2], axis=1)
```

+++ {"slideshow": {"slide_type": "slide"}}

Before any further analysis we clean our data. The count files end with a handful of book-keeping rows from the read counter (reads that could not be assigned to a gene, and so on) which we remove. We also remove genes that were not reliably measured in every sample, i.e. genes having a [NaN](https://en.wikipedia.org/wiki/NaN) or a zero count in at least one sample.

Sequencing libraries differ in depth, so a raw count is not comparable between samples. We divide each sample by a *size factor*, the median ratio of that sample's counts to the average sample, and then log transform. It is generally assumed that expression values follow a log-normal distribution, and hence the log transformation implies that the new values follow a normal distribution.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
counter_rows = ["no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique"]
yeast = yeast.drop(index=counter_rows)

yeast.dropna(axis=0, how="any", inplace=True)
yeast = yeast.loc[~(yeast <= 0.0).any(axis=1)]

log_counts = np.log(yeast)
size_factors = np.exp(log_counts.sub(log_counts.mean(axis=1), axis=0).median(axis=0))
expr = np.log2(yeast.div(size_factors, axis=1))
```

+++ {"slideshow": {"slide_type": "slide"}}

We can get an overview of the expression data:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
expr
```

+++ {"slideshow": {"slide_type": "slide"}}

and of the samples themselves, i.e. which condition each sample belongs to, how deeply it was sequenced, and the size factor we just estimated for it:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sample_info = pd.DataFrame({
    "condition": ["snf2" if s.startswith("Snf2") else "WT" for s in expr.columns],
    "replicate": [s.split("_")[1] for s in expr.columns],
    "library_size": yeast.sum(),
    "size_factor": size_factors,
})
sample_info
```

+++ {"slideshow": {"slide_type": "slide"}}

### Differential expression analysis

The goal of the exercise is to determine which genes that are differentially expressed in the *snf2* knockout as compared to the wild type. [Snf2](https://www.yeastgenome.org/locus/S000005816) is the ATPase subunit of the SWI/SNF chromatin remodelling complex, a transcriptional regulator, so we expect its deletion to change the expression of a large number of genes.

We first create a vector of booleans, that track which samples that are knockouts. This will be needed as an input for subsequent significance estimation.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
snf2_bool = (sample_info["condition"] == "snf2")
snf2_bool
```

+++ {"slideshow": {"slide_type": "slide"}}

Next, for each transcript that has been measured, we calculate (1) log of the average Fold Change difference between the knockout and the wild type, and (2) the significance of the difference between the knockout and the wild type.

An easy way to do so is by defining a separate function, `get_significance_two_groups(row)`, that can do such calculations for any row of the `expr` DataFrame, and subsequently we use the function `apply` for the function to execute on each row of the DataFrame. For the significance test we use a $t$ test, which is provided by the function [`ttest_ind`.](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

This results in a new table with gene names and their $p$ values of differential expression, and their fold changes.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def get_significance_two_groups(row):
    log_fold_change = row[snf2_bool].mean() - row[~snf2_bool].mean() # Calculate the log Fold Change
    p = ttest_ind(row[snf2_bool],row[~snf2_bool],equal_var=False)[1] # Calculate the significance
    return [p,-np.log10(p),log_fold_change]

pvalues = expr.apply(get_significance_two_groups,axis=1,result_type="expand")
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

A common way to illustrate the differential expression values are by plotting the negative log of the $p$ values, as a function of the mean [fold change](https://en.wikipedia.org/wiki/Fold_change) of each transcript. This is known as a [Volcano plot](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)).

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
ax.set(xlabel="$log_2(snf2/WT)$", ylabel="$-log_{10}(p)$");
```

+++ {"slideshow": {"slide_type": "fragment"}}

The regular interpretation of a Volcano plot is that the genes in the top left and the top right corner are the most interesting ones, as they have a large fold change between the conditions as well as being very significant.

With 48 replicates per condition the plot is dominated by a very large number of significant genes. We can count how many genes we would call significant at the conventional threshold $p<0.05$, and compare that to the number we would expect to see by pure chance if no gene at all were differentially expressed.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
m = pvalues.shape[0]
print("Number of genes tested:              ", m)
print("Genes with p < 0.05:                 ", int((pvalues["p"] < 0.05).sum()))
print("Expected by chance if all genes null:", int(round(0.05 * m)))
```

+++ {"slideshow": {"slide_type": "slide"}}

### How much does the number of replicates matter?

The analysis above used all 96 samples. Almost no real experiment does. The strength of this particular data set is that we can *pretend* to have run a smaller experiment, simply by drawing a subset of the replicates, and see what we would have concluded.

Below we draw $n$ wild type and $n$ knockout replicates at random, repeat the whole $t$ test for every gene, and record how many genes pass $p<0.05$. To keep the repeated analysis fast we use a vectorised version of the same Welch $t$ test as above, testing all genes in one call.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def two_group_p_values(data, group_a, group_b):
    "Welch t test of every gene (row), comparing the samples in group_a to those in group_b."
    return ttest_ind(data[group_a], data[group_b], axis=1, equal_var=False)[1]

wt_samples = list(expr.columns[~snf2_bool])
snf2_samples = list(expr.columns[snf2_bool])

rng = np.random.default_rng(42)
replicate_numbers = [3, 10, 48]
example_p = {}
summary = []

for n in replicate_numbers:
    repeats = 1 if n == len(wt_samples) else 20
    counts = []
    for repeat in range(repeats):
        subset_wt = list(rng.choice(wt_samples, n, replace=False))
        subset_snf2 = list(rng.choice(snf2_samples, n, replace=False))
        p = two_group_p_values(expr, subset_snf2, subset_wt)
        counts.append(int((p < 0.05).sum()))
        if repeat == 0:
            example_p[n] = p
    summary.append({"replicates per group": n,
                    "median genes with p<0.05": int(np.median(counts)),
                    "lowest": min(counts), "highest": max(counts)})

pd.DataFrame(summary).set_index("replicates per group")
```

+++ {"slideshow": {"slide_type": "slide"}}

The $p$ value histograms tell the same story in a more informative way. A histogram of $p$ values is a mixture of a flat, uniform part, stemming from the genes that follow the null hypothesis $H_0$, and a spike close to zero, stemming from the genes that are truly differentially expressed. The dashed line marks the height the histogram would have if *every* gene followed $H_0$.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sns.set_context("notebook")
fig, axes = plt.subplots(1, len(replicate_numbers), figsize=(15, 4), sharey=True)
bins = 20
for ax, n in zip(axes, replicate_numbers):
    ax.hist(example_p[n], bins=bins, range=(0, 1), color="steelblue")
    ax.axhline(m / bins, color="k", ls="--", lw=1)
    ax.set_title(f"n = {n} vs {n}")
    ax.set_xlabel("$p$")
axes[0].set_ylabel("Number of genes")
plt.tight_layout();
```

+++ {"slideshow": {"slide_type": "fragment"}}

Two things are worth noticing. First, the spike at low $p$ grows steadily with the number of replicates: the genes are just as differentially expressed in the small experiment as in the large one, we are simply unable to demonstrate it. Statistical *power* is bought with replicates, not with sequencing depth. Second, the result of a three-versus-three experiment is highly unstable — the `lowest` and `highest` columns of the table above show how much the number of findings varies between two equally valid draws of three replicates, whereas the 48-versus-48 comparison has no such lottery element.

This is also a warning about how to read the literature. A gene that appears in a three-versus-three study and not in another is not necessarily a contradiction; it may simply be two draws from the same lottery.

+++ {"slideshow": {"slide_type": "fragment"}}

Finally, note that even the flat part of the histograms contains a substantial number of genes below $p<0.05$ that are not differentially expressed. Out of $m$ tested genes we expect $0.05m$ false positives at that threshold, regardless of the number of replicates. How to deal with that problem is the topic of the [next chapter](./multiple.md).
