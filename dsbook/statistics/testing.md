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
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/testing.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/testing.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/testing.ipynb
    title: Run on Binder
---

+++ {"slideshow": {"slide_type": "slide"}}

# Differential expression analysis of two inbred mouse strains

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/testing.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/testing.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/testing.ipynb)

Most RNA-seq experiments you will meet in the literature compare three samples against three samples. That is too few replicates to see what a $p$ value *distribution* actually looks like. We will therefore work with a more generously replicated data set: the mouse RNA-seq experiment of [Bottomly et al. (2011)](https://doi.org/10.1371/journal.pone.0017820), which compares gene expression in the striatum, a region of the brain, between two inbred mouse strains, **10 C57BL/6J mice** and **11 DBA/2J mice**, all of them adult males.

C57BL/6J (abbreviated B6) and DBA/2J (D2) are two of the most widely used laboratory mouse strains. B6 is the strain the mouse reference genome was sequenced from, and the two strains have been extensively characterised for behavioural and other striatum-related traits. Genetically they differ at millions of positions in their genomes. Within a strain, on the other hand, the mice are inbred and therefore genetically almost identical. The variation between the replicates of one strain is consequently not genetic, but stems from everything else that differs between two animals and two sequencing libraries.

The data set has since been reused in several comparisons of differential expression methods. We use the gene count table prepared by the [ReCount](https://doi.org/10.1186/1471-2105-12-449) project, which holds one read count per gene and sample, together with a table describing the samples. Both are downloaded once and cached. (The raw reads are available from the Sequence Read Archive under accession SRA026846.)

```{code-cell}
:tags: [hide-cell]

# When the page runs in the browser (JupyterLite), we install seaborn, and fetch the data from a copy in the book's
# repository, as the ReCount server does not allow downloads from within a browser
import sys
if sys.platform == "emscripten":
    import os
    import micropip
    await micropip.install(["seaborn", "pyodide-http"])
    import pyodide_http
    pyodide_http.patch_all()
    from urllib.request import urlretrieve
    repo = "https://raw.githubusercontent.com/statisticalbiotechnology/dsbook/main/dsbook/statistics/data/"
    data_dir = "../data" if os.path.isdir("../data") else "data"
    os.makedirs(data_dir, exist_ok=True)
    for file_name in ["bottomly_count_table.txt", "bottomly_phenodata.txt"]:
        if not os.path.exists(os.path.join(data_dir, file_name)):
            urlretrieve(repo + file_name, os.path.join(data_dir, file_name))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Cache the downloaded tables, so that we only fetch them once
data_dir = "../data" if os.path.isdir("../data") else "data"
os.makedirs(data_dir, exist_ok=True)

recount_url = "https://bowtie-bio.sourceforge.net/recount/"
for folder, file_name in [("countTables/", "bottomly_count_table.txt"),
                          ("phenotypeTables/", "bottomly_phenodata.txt")]:
    local_file = os.path.join(data_dir, file_name)
    if not os.path.exists(local_file):
        urllib.request.urlretrieve(recount_url + folder + file_name, local_file)

mouse = pd.read_csv(os.path.join(data_dir, "bottomly_count_table.txt"), sep="\t", index_col=0)
pheno = pd.read_csv(os.path.join(data_dir, "bottomly_phenodata.txt"), sep=" ", index_col=0)
mouse.shape
```

+++ {"slideshow": {"slide_type": "slide"}}

The rows of `mouse` are the genes, named by their [Ensembl](https://www.ensembl.org/Mus_musculus/) identifiers, and the columns are the 21 samples, named by their accession numbers in the Sequence Read Archive.

Before any further analysis we clean our data. Most of the 36,536 genes in the table are not expressed in the striatum, or only at a level where a handful of reads decide the outcome. We keep the genes with at least 10 reads in every sample.

Sequencing libraries differ in depth, so a raw count is not comparable between samples. We divide each sample by a *size factor*, the median over all genes of the ratio between the sample's count and the gene's geometric mean count across all samples, and then log transform. It is generally assumed that expression values follow a log-normal distribution, and hence the log transformation implies that the new values follow a normal distribution.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
mouse = mouse.loc[(mouse >= 10).all(axis=1)]

log_counts = np.log(mouse)
size_factors = np.exp(log_counts.sub(log_counts.mean(axis=1), axis=0).median(axis=0))
expr = np.log2(mouse.div(size_factors, axis=1))
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

and of the samples themselves, i.e. which strain each sample belongs to, on which of the three sequencing flow cells (`experiment`) it was sequenced, how deeply it was sequenced, and the size factor we just estimated for it:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sample_info = pd.DataFrame({
    "strain": pheno.loc[expr.columns, "strain"].map({"C57BL/6J": "B6", "DBA/2J": "D2"}),
    "experiment": pheno.loc[expr.columns, "experiment.number"],
    "library_size": mouse.sum(),
    "size_factor": size_factors,
})
sample_info
```

+++ {"slideshow": {"slide_type": "slide"}}

## Differential expression analysis

The goal of the exercise is to determine which genes are differentially expressed in the striatum of D2 mice as compared to B6 mice.

We first create a vector of booleans that tracks which samples are from D2 mice. This will be needed as an input for subsequent significance estimation.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
d2_bool = (sample_info["strain"] == "D2")
d2_bool
```

+++ {"slideshow": {"slide_type": "slide"}}

Next, for each gene that has been measured, we calculate (1) the log fold change between the two strains, i.e. the difference between the mean log$_2$ expression values of the two strains, and (2) the significance of the difference between the strains.

An easy way to do so is by defining a separate function, `get_significance_two_groups(row)`, that can do such calculations for any row of the `expr` DataFrame, and subsequently we use the function `apply` for the function to execute on each row of the DataFrame. For the significance test we use a $t$ test, which is provided by the function [`ttest_ind`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html). With `equal_var=False` it performs *Welch's* $t$ test, which, unlike the pooled $t$ test of the previous chapter, does not assume that the expression of a gene varies equally much in the two strains.

This results in a new table with gene names, their $p$ values of differential expression, and their log fold changes.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def get_significance_two_groups(row):
    log_fold_change = row[d2_bool].mean() - row[~d2_bool].mean() # Calculate the log Fold Change
    p = ttest_ind(row[d2_bool],row[~d2_bool],equal_var=False)[1] # Calculate the significance
    return [p,-np.log10(p),log_fold_change]

pvalues = expr.apply(get_significance_two_groups,axis=1,result_type="expand")
pvalues.rename(columns = {list(pvalues)[0]: 'p', list(pvalues)[1]: '-log_p', list(pvalues)[2]: 'log_FC'}, inplace = True)
```

+++ {"slideshow": {"slide_type": "slide"}}

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
pvalues
```

+++ {"slideshow": {"slide_type": "slide"}}

A common way to illustrate the differential expression values is to plot the negative log of the $p$ values as a function of the $\log_2$ [fold change](https://en.wikipedia.org/wiki/Fold_change) of each gene. This is known as a [Volcano plot](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)).

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
ax.set(xlabel=r"$\log_2(D2/B6)$", ylabel=r"$-\log_{10}(p)$");
```

+++ {"slideshow": {"slide_type": "fragment"}}

The genes in the top left and top right corners are the most interesting ones, as they combine a large fold change with a small $p$ value.

We can count how many genes we would call significant at the conventional threshold $p<0.05$, and compare that to the number we would expect to see by pure chance if no gene at all were differentially expressed.

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

## How much does the number of replicates matter?

The analysis above used all 21 samples, which is more than most experiments can afford. With this data set we can *pretend* to have run a smaller experiment, simply by drawing a subset of the replicates, and see what we would have concluded.

Below we draw $n$ B6 and $n$ D2 replicates at random, repeat the whole $t$ test for every gene, and record how many genes pass $p<0.05$. To keep the repeated analysis fast we use a vectorised version of the same Welch $t$ test as above, testing all genes in one call.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def two_group_p_values(data, group_a, group_b):
    "Welch t test of every gene (row), comparing the samples in group_a to those in group_b."
    return ttest_ind(data[group_a], data[group_b], axis=1, equal_var=False)[1]

b6_samples = list(expr.columns[~d2_bool])
d2_samples = list(expr.columns[d2_bool])

rng = np.random.default_rng(42)
replicate_numbers = [3, 5, 10]
typical_p = {}
summary = []

for n in replicate_numbers:
    draws, counts = [], []
    for repeat in range(20):
        subset_b6 = list(rng.choice(b6_samples, n, replace=False))
        subset_d2 = list(rng.choice(d2_samples, n, replace=False))
        p = two_group_p_values(expr, subset_d2, subset_b6)
        draws.append(p)
        counts.append(int((p < 0.05).sum()))
    # keep the draw whose number of findings is closest to the median, for the histograms below
    typical = np.argmin(np.abs(np.array(counts) - np.median(counts)))
    typical_p[n] = draws[typical]
    summary.append({"replicates per group": n,
                    "median genes with p<0.05": int(np.median(counts)),
                    "lowest": min(counts), "highest": max(counts)})

pd.DataFrame(summary).set_index("replicates per group")
```

+++ {"slideshow": {"slide_type": "slide"}}

The $p$ value histograms tell the same story in a more informative way. For each $n$ we show the draw whose number of genes below $p<0.05$ is closest to the median of the 20 draws. A histogram of $p$ values is a mixture of a flat, uniform part, stemming from the genes that follow the null hypothesis $H_0$, and a spike close to zero, stemming from the genes that are truly differentially expressed. The dashed line marks the height the histogram would have if *every* gene followed $H_0$.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sns.set_context("notebook")
fig, axes = plt.subplots(1, len(replicate_numbers), figsize=(15, 4), sharey=True)
bins = 20
for ax, n in zip(axes, replicate_numbers):
    ax.hist(typical_p[n], bins=bins, range=(0, 1), color="steelblue")
    ax.axhline(m / bins, color="k", ls="--", lw=1)
    ax.set_title(f"n = {n} vs {n}")
    ax.set_xlabel("$p$")
axes[0].set_ylabel("Number of genes")
plt.tight_layout();
```

+++ {"slideshow": {"slide_type": "fragment"}}

Two things are worth noticing. First, the spike at low $p$ grows steadily with the number of replicates, and at three versus three it is barely visible. The genes are just as differentially expressed in the small experiment as in the large one; we are simply unable to demonstrate it. Here, statistical [power](#sec:statistics:power) is bought with replicates. Sequencing the same mice more deeply would reduce the counting noise, which mainly matters for lowly expressed genes, but not the variation between animals. Second, the result of a three-versus-three experiment is unstable — the `lowest` and `highest` columns of the table above show how much the number of findings varies between two equally valid draws of three replicates per strain.

This is also a warning about how to read the literature. A gene that appears in a three-versus-three study and not in another is not necessarily a contradiction; it may simply be two draws from the same lottery.

+++ {"slideshow": {"slide_type": "fragment"}}

Finally, note that the flat part of the histograms, from the genes that follow $H_0$, extends all the way down to $p=0$, so some of the genes below $p<0.05$ are not differentially expressed. If all $m$ tested genes were null, we would expect $0.05m$ false positives at that threshold, regardless of the number of replicates. As some of the genes are truly differentially expressed, $0.05m$ is an upper bound on the expected number of false positives. How to deal with that problem is the topic of the [next chapter](./multiple.md).

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Explain why the counts of an RNA-seq experiment are filtered for lowly expressed genes, divided by a per-sample size factor and log transformed before a $t$ test is applied to each gene.
2. Calculate how many genes you would expect below $p<0.05$ if all $m$ tested genes followed $H_0$, and compare it with the observed number of genes below that threshold.
3. Interpret a $p$ value histogram as a mixture of a flat part from genes that follow $H_0$ and a spike near zero from genes that are differentially expressed.
4. Explain how the number of replicates affects statistical power and the stability of the findings, and argue why a gene reported in one three-versus-three study but not in another is not necessarily a contradiction.
:::
