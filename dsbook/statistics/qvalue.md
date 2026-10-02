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
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/qvalue.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/qvalue.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/qvalue.ipynb
    title: Run on Binder
---

# $q$ Value calculations in a yeast knockout experiment

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/qvalue.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/qvalue.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/qvalue.ipynb)

+++

## Differential expression analysis with multiple testing

This notebook continues from where the previous notebook on [hypothesis testing](./testing.md) ended.

We again compare the *snf2* knockout yeast strain to its isogenic wild type, using the highly replicated RNA-seq benchmark of the Barton group, with 48 biological replicates of each condition ([Gierliński et al., 2015](https://doi.org/10.1093/bioinformatics/btv425); [Schurch et al., 2016](https://doi.org/10.1261/rna.053959.115); ENA accession [PRJEB5348](https://www.ebi.ac.uk/ena/browser/view/PRJEB5348); count matrices from [bartongroup/profDGE48](https://github.com/bartongroup/profDGE48)).

We first recreate the steps of the previous notebook. Since we will repeat the testing several times in this notebook, we write the $t$ test in a vectorised form that tests all genes in a single call, rather than looping over the rows of the table.

```{code-cell} ipython3
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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
            sample = condition + "_" + member.name.split("_")[1]
            counts = pd.read_csv(tar.extractfile(member), sep="\t",
                                 header=None, index_col=0, names=["gene", sample])
            columns[sample] = counts[sample]
    return pd.DataFrame(columns).sort_index(axis=1)

yeast = pd.concat([read_count_data("WT", "WT_countdata.tar.gz"),
                   read_count_data("Snf2", "Snf2_countdata.tar.gz")], axis=1)

counter_rows = ["no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique"]
yeast = yeast.drop(index=counter_rows)
yeast.dropna(axis=0, how="any", inplace=True)
yeast = yeast.loc[~(yeast <= 0.0).any(axis=1)]

log_counts = np.log(yeast)
size_factors = np.exp(log_counts.sub(log_counts.mean(axis=1), axis=0).median(axis=0))
expr = np.log2(yeast.div(size_factors, axis=1))

wt_samples = [s for s in expr.columns if s.startswith("WT")]
snf2_samples = [s for s in expr.columns if s.startswith("Snf2")]

def get_significance_two_groups(data, group_a, group_b):
    "Welch t test of every gene, comparing the samples of group_a to those of group_b."
    log_fold_change = data[group_a].mean(axis=1) - data[group_b].mean(axis=1)
    p = ttest_ind(data[group_a], data[group_b], axis=1, equal_var=False)[1]
    return pd.DataFrame({"p": p, "-log_p": -np.log10(p), "log_FC": log_fold_change.values},
                        index=data.index)

pvalues = get_significance_two_groups(expr, snf2_samples, wt_samples)
pvalues = pvalues.loc[~pvalues.index.duplicated(keep='first')]
```

When plotting the $p$ value distribution below, we see an enrichment of low $p$ values. These are the tests of the genes that adhere to the alternative hypothesis. We also see a uniform distribution of the $p$ values in the higher end of the distribution i.e. $p$ values of 0.3-1.0. These are likely stemming from genes adhering to $H_0$

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
sns.histplot(x=pvalues["p"],kde=False)
plt.xlim(0,1.0);
```

### $q$ value esitmation
We define a function for the calculation of $\pi_0$. Here we use a different method than the one described in Storey&Tibshirani. The details of this method, known as the *bootstrap method*, are given in [Storey 2002](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/1467-9868.00346)

```{code-cell} ipython3
import numpy as np
import numpy.random as npr

def bootstrap(invec):
    idx = npr.randint(0, len(invec), len(invec))
    return [invec[i] for i in idx]

def estimatePi0(p, numBoot=100, numLambda=100, maxLambda=0.95):
    p.sort()
    n=len(p)
    lambdas=np.linspace(maxLambda/numLambda,maxLambda,numLambda)
    Wls=np.array([n-np.argmax(p>=l) for l in lambdas])
    pi0s=np.array([Wls[i] / (n * (1 - lambdas[i])) for i in range(numLambda)])
    minPi0=np.min(pi0s)
    mse = np.zeros(numLambda)
    for boot in range(numBoot):
        pBoot = bootstrap(p)
        pBoot.sort()
        WlsBoot =np.array([n-np.argmax(pBoot>=l) for l in lambdas])
        pi0sBoot =np.array([WlsBoot[i] / (n *(1 - lambdas[i])) for i in range(numLambda)])
        mse = mse + np.square(pi0sBoot-minPi0)
    minIx = np.argmin(mse)
    return pi0s[minIx]
```

We subsequently use Storey&Tibshirani to calculate first calculate, 

$$ \hat{\rm FDR}(t) = \frac{\pi_0mt}{|\{p_i\le t\}|}, $$

and then smooth the $\rm FDR(t)$ estimates as, 

$$\hat{q}(p_i)=\min_{t \ge p_i}\hat{\rm FDR}(t).$$

```{code-cell} ipython3
def qvalues(pvalues):
    m = pvalues.shape[0] # The number of p-values
    pvalues.sort_values("p",inplace=True) # sort the pvalues in acending order
    pi0 = estimatePi0(list(pvalues["p"].values))
    print("pi_0 estimated to " + str(pi0))
    
    # calculate a FDR(t) as in Storey & Tibshirani
    num_p = 0.0
    for ix in pvalues.index:
        num_p += 1.0
        t = pvalues.loc[ix,"p"]
        fdr = pi0*t*m/num_p
        pvalues.loc[ix,"q"] = fdr
    
    # calculate a q(p) as the minimal FDR(t)
    old_q=1.0
    for ix in reversed(list(pvalues.index)):
        q = min(old_q,pvalues.loc[ix,"q"])
        old_q = q
        pvalues.loc[ix,"q"] = q
    return pvalues
```

```{code-cell} ipython3
qv = qvalues(pvalues)
```

We note a very low $\pi_0$, indicating that a large majority of all yeast genes respond to the deletion of *SNF2*. That is a biologically reasonable result — Snf2 is the ATPase of the SWI/SNF chromatin remodelling complex, a global regulator of transcription — but it is also a consequence of the enormous statistical power of 48 replicates per condition. With this many replicates, even very small and biologically uninteresting expression differences become detectable.

We can list the differential genes, in descending order of significance.

```{code-cell} ipython3
qv
```

## Displaying number of findings as a function of $q$ value (a $p$-$q$ plot)
A plot of the number of differentially expressed genes as a function of $q$ value gives the same message.

```{code-cell} ipython3
sns.lineplot(x=pvalues["q"],y=list(range(pvalues.shape[0])),errorbar=None,lw=3)
plt.xlim(0,0.1);
plt.ylim();
plt.ylabel("Number of differential genes");
```

## Volcano plots revisited
We often see that Volcano plots are complemented with FDR tresholds. Here we complement the previous lecture's volcano plot with coloring indicating if transcripts are significantly differentially abundant at a FDR-treshhold of $0.01$.

```{code-cell} ipython3
qv["Significant"] = qv["q"]<0.01
less_than_FDR_1 = qv[qv["q"]<0.01]
p_treshold = less_than_FDR_1.iloc[-1]["-log_p"]
print(f"{len(less_than_FDR_1)} of {len(qv)} genes have q < 0.01")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
sns.set_context("talk")
ax = sns.relplot(data=pvalues,x="log_FC",y="-log_p",hue="Significant",aspect=1.5,height=6)
plt.axhline(p_treshold)
ax.set(xlabel="$log_2(snf2/WT)$", ylabel="$-log_{10}(p)$");
```

+++

## A ground truth null experiment

A $p$ value histogram that is enriched near zero, and a $\pi_0$ well below one, are easy to produce. The harder question is whether we should *believe* them: how do we know that the machinery above is not manufacturing findings out of technical artefacts?

Most data sets cannot answer that question, because we never know which genes are truly differentially expressed. This data set can. The 48 wild type replicates are all the same strain, grown and sequenced in the same way. If we arbitrarily declare 24 of them to be "group A" and the remaining 24 to be "group B", we have built an experiment in which the null hypothesis is true for *every single gene*. Any finding is by construction a false positive.

```{code-cell} ipython3
sns.set_context("notebook")
rng = np.random.default_rng(7)
shuffled_wt = list(rng.permutation(wt_samples))
group_a, group_b = shuffled_wt[:24], shuffled_wt[24:]

null_pvalues = get_significance_two_groups(expr, group_a, group_b)
null_pvalues = null_pvalues.loc[~null_pvalues.index.duplicated(keep='first')]

m = null_pvalues.shape[0]
bins = 20
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
ax[0].hist(null_pvalues["p"], bins=bins, range=(0, 1), color="steelblue")
ax[0].axhline(m / bins, color="k", ls="--", lw=1)
ax[0].set_title("WT vs WT, 24 vs 24 (all $H_0$)")
ax[1].hist(pvalues["p"], bins=bins, range=(0, 1), color="indianred")
ax[1].axhline(m / bins, color="k", ls="--", lw=1)
ax[1].set_title("snf2 vs WT, 48 vs 48")
for a in ax:
    a.set_xlabel("$p$"); a.set_ylabel("Number of genes")
plt.tight_layout();
```

The difference is striking. The wild type against wild type comparison gives a flat histogram, at the height $m/20$ that a uniform distribution predicts (dashed line), with no spike at all near zero. That is exactly what the theory says a collection of true null hypotheses should look like, and it is a direct, empirical validation of the $t$ test we have been using on these data.

Running the same $q$ value machinery on this null experiment should therefore estimate $\pi_0$ close to one, and report essentially no findings.

```{code-cell} ipython3
null_qv = qvalues(null_pvalues)
for threshold in (0.01, 0.05, 0.10, 0.20):
    print(f"genes with q < {threshold:4.2f}: {int((null_qv['q'] < threshold).sum()):5d}"
          f"   (out of {m})")
print("smallest q value in the null experiment: %.3f" % null_qv["q"].min())
```

The estimated $\pi_0$ is close to one and no gene survives any reasonable $q$ value threshold, whereas the real comparison produced thousands of findings. The multiple hypothesis correction is not simply throwing away everything, nor is it rubber-stamping noise.

A caveat worth keeping in mind: individual splits of the wild type replicates fluctuate somewhat more than pure theory predicts. The 48 replicates were not all handled on the same day, so they are not perfectly exchangeable, and a split that happens to separate two batches will show a mild enrichment of small $p$ values. Try re-running the two cells above with a different seed to see this for yourself. This residual structure is precisely the kind of thing that a three-versus-three experiment has no way of detecting.

+++

## Spline estimation of $\pi_0$

Storey and Tibshirani outlines an other procedure for estimating $\pi_0$ than the bootstrap procedure used above, i.e. using 
$\hat{\pi_0}(\lambda) = \frac{|\{p>\lambda \}|}{m(1-\lambda)}$.

Below, we almost follow the article's described procedure (please try to find the difference on how we select which lambdas we evaluate). Furthermore we fit a [qubic spline](https://en.wikipedia.org/wiki/Smoothing_spline) to these $\pi_0$ estimates.

```{code-cell} ipython3
from scipy.interpolate import UnivariateSpline

m = pvalues.shape[0] # The number of p-values
pvalues.sort_values("p",inplace=True,ascending=False) # sort the pvalues in decending order
num_p = -1
for ix in pvalues.index:
    num_p += 1
    lambda_p = pvalues.loc[ix,"p"]
    pi0_hat = num_p/(m*(1-lambda_p))
    pvalues.loc[ix,"pi0_hat"] = pi0_hat

pvalues.sort_values("p",inplace=True) # sort the pvalues in ascending order
s = UnivariateSpline(pvalues["p"],pvalues["pi0_hat"], k=3,s=10)
```

We plot the estimates (blue) as well as the spline fit (red) for two different intevalls of $\lambda$. You will see the need of a smoother, particularly in the region near 1.0.

```{code-cell} ipython3
def plot_pi0_hat(p,s,xlow,xhigh,ylow,yhigh,ax):
  sns.lineplot(x=pvalues["p"],y=pvalues["pi0_hat"],errorbar=None,lw=3, ax=ax, color='b')
  sns.lineplot(x=pvalues["p"],y=s(pvalues["p"]),errorbar=None,lw=3,ax=ax, color='r')
  ax.set_xlim(xlow,xhigh);
  ax.set_ylim(ylow,yhigh);
  ax.set_xlabel("$\lambda $");
  ax.set_ylabel("$\pi_0(\lambda)$");

fig, ax = plt.subplots(1,2,figsize=(12, 4))
plot_pi0_hat(pvalues,s,0,1,0,0.6,ax[0])
plot_pi0_hat(pvalues,s,0.95,1,0.1,0.35,ax[1])
```

We can obtain a final estimate by evaluating the spline for $\lambda=1$, and compare the it to the bootstrapping estimate.

```{code-cell} ipython3
print("Spline estimate of pi_0: " + str(s(1)))
print("Bootstrap estimate of pi_0: " + str(estimatePi0(list(pvalues["p"].values))))
```
