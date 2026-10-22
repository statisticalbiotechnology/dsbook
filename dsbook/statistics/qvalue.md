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
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/qvalue.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/qvalue.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/qvalue.ipynb
    title: Run on Binder
---

# $q$ Value calculations in a comparison of two mouse strains

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/qvalue.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/qvalue.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/qvalue.ipynb)

+++

## Differential expression analysis with multiple testing

This notebook continues from where the previous notebook on [hypothesis testing](./testing.md) ended.

We again compare the striatum of the two inbred mouse strains C57BL/6J (B6, 10 mice) and DBA/2J (D2, 11 mice), from the RNA-seq experiment of [Bottomly et al. (2011)](https://doi.org/10.1371/journal.pone.0017820), using the gene count table prepared by the [ReCount](https://doi.org/10.1186/1471-2105-12-449) project.

We first recreate the steps of the previous notebook. Since we will repeat the testing several times in this notebook, we write the $t$ test in a vectorised form that tests all genes in a single call, rather than looping over the rows of the table.

```{code-cell} ipython3
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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

mouse = mouse.loc[(mouse >= 10).all(axis=1)]
log_counts = np.log(mouse)
size_factors = np.exp(log_counts.sub(log_counts.mean(axis=1), axis=0).median(axis=0))
expr = np.log2(mouse.div(size_factors, axis=1))

strain = pheno.loc[expr.columns, "strain"]
b6_samples = list(expr.columns[strain == "C57BL/6J"])
d2_samples = list(expr.columns[strain == "DBA/2J"])

def get_significance_two_groups(data, group_a, group_b):
    "Welch t test of every gene, comparing the samples of group_a to those of group_b."
    log_fold_change = data[group_a].mean(axis=1) - data[group_b].mean(axis=1)
    p = ttest_ind(data[group_a], data[group_b], axis=1, equal_var=False)[1]
    return pd.DataFrame({"p": p, "-log_p": -np.log10(p), "log_FC": log_fold_change.values},
                        index=data.index)

pvalues = get_significance_two_groups(expr, d2_samples, b6_samples)
```

When plotting the $p$ value distribution below, we see an enrichment of low $p$ values. These are the tests of the genes that adhere to the alternative hypothesis. We also see a uniform distribution of the $p$ values in the higher end of the distribution i.e. $p$ values of 0.5-1.0. These are likely stemming from genes adhering to $H_0$

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
sns.histplot(x=pvalues["p"],kde=False)
plt.xlim(0,1.0);
```

### $q$ value estimation
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

Following Storey&Tibshirani, we subsequently first calculate, 

$$ \hat{\rm FDR}(t) = \frac{\pi_0mt}{|\{p_i\le t\}|}, $$

and then make the $\hat{\rm FDR}(t)$ estimates monotone by taking the minimum over all thresholds at least as large as $p_i$, 

$$\hat{q}(p_i)=\min_{t \ge p_i}\hat{\rm FDR}(t).$$

```{code-cell} ipython3
def qvalues(pvalues):
    m = pvalues.shape[0] # The number of p-values
    pvalues.sort_values("p",inplace=True) # sort the pvalues in ascending order
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

The estimated $\pi_0$ of about 0.5 says that roughly half of the expressed genes differ in their expression between the two strains. Given that the strains differ at millions of positions in their genomes, many of them in sequences that regulate gene expression, that is not unreasonable. One might worry that some of the differences are technical: the reads were aligned to the reference genome, which is a B6 genome, so reads from D2 genes carrying many sequence differences could align less well and make those genes look lower expressed in D2. Bottomly et al. tested this by realigning the D2 reads to a D2 version of the genome, and found that the effect on their results was minimal.

Note that $\pi_0$ and the $q$ values answer different questions. $\pi_0$ estimates *how many* genes are differentially expressed in total, while a $q$ value tells us *which* genes we can single out with a controlled error rate. Many of the expression differences are too small to be pinned down gene by gene with ten mice per strain, even though their combined effect on the $p$ value distribution is clearly visible. (Bottomly et al. used a test designed for count data, and reported 1,727 differentially expressed genes at $q<0.01$ out of the 16,183 genes they tested.)

We can list the differential genes, in descending order of significance.

```{code-cell} ipython3
qv
```

## Displaying number of findings as a function of $q$ value
A plot of the number of differentially expressed genes as a function of the $q$ value threshold shows how many genes we can report at each FDR level.

```{code-cell} ipython3
sns.lineplot(x=pvalues["q"],y=list(range(1, pvalues.shape[0] + 1)),errorbar=None,lw=3)
plt.xlim(0,0.1);
plt.ylim(0, (pvalues["q"] < 0.1).sum());
plt.ylabel("Number of differential genes");
```

## Volcano plots revisited
We often see that Volcano plots are complemented with FDR thresholds. Here we complement the previous lecture's volcano plot with coloring indicating if transcripts are significantly differentially abundant at an FDR threshold of $0.01$.

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
ax.set(xlabel="$log_2(D2/B6)$", ylabel="$-log_{10}(p)$");
```

+++

## A ground truth null experiment

A $p$ value histogram that is enriched near zero, and a $\pi_0$ well below one, are easy to produce. The harder question is whether we should *believe* them: how do we know that the machinery above is not manufacturing findings out of technical artefacts?

Most data sets cannot answer that question, because we never know which genes are truly differentially expressed. This data set lets us construct such a case. The ten B6 mice are inbred, i.e. genetically almost identical, and were all treated the same way. If we split them into two groups of five, we have built an experiment in which the null hypothesis is true for *every single gene*. Any finding is by construction a false positive.

The B6 mice were sequenced on three different flow cells (the `experiment` column of the sample table in the previous notebook). To keep the comparison fair, we alternate the mice of each flow cell between the two groups, so that both groups contain mice from all three flow cells.

```{code-cell} ipython3
sns.set_context("notebook")
flow_cell = pheno["experiment.number"]
b6_by_flow_cell = sorted(b6_samples, key=lambda s: flow_cell[s])
group_a, group_b = b6_by_flow_cell[0::2], b6_by_flow_cell[1::2]

null_pvalues = get_significance_two_groups(expr, group_a, group_b)

m = null_pvalues.shape[0]
bins = 20
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
ax[0].hist(null_pvalues["p"], bins=bins, range=(0, 1), color="steelblue")
ax[0].axhline(m / bins, color="k", ls="--", lw=1)
ax[0].set_title("B6 vs B6, 5 vs 5 (all $H_0$)")
ax[1].hist(pvalues["p"], bins=bins, range=(0, 1), color="indianred")
ax[1].axhline(m / bins, color="k", ls="--", lw=1)
ax[1].set_title("D2 vs B6, 11 vs 10")
for a in ax:
    a.set_xlabel("$p$"); a.set_ylabel("Number of genes")
plt.tight_layout();
```

The difference is striking. The B6 against B6 comparison has no spike at all near zero, so the test does not manufacture findings out of nothing. The histogram is, however, not perfectly flat either: small $p$ values are *rarer* than the uniform distribution predicts (the dashed line marks the height $m/20$ of a uniform distribution), and intermediate ones more common. The reason is the flow cells. Each of our two groups contains mice from all three flow cells, so the differences between flow cells inflate the variance *within* each group, while they cancel out in the difference *between* the group means. The $t$ statistics therefore come out smaller than they should, and the $p$ values larger. In other words, the test becomes *conservative*. The same applies to the comparison between the strains, as Bottomly et al. spread the mice of both strains over all three flow cells, so we lose some power there too.

Running the same $q$ value machinery on this null experiment should therefore report no findings.

```{code-cell} ipython3
null_qv = qvalues(null_pvalues)
for threshold in (0.01, 0.05, 0.10, 0.20):
    print(f"genes with q < {threshold:4.2f}: {int((null_qv['q'] < threshold).sum()):5d}"
          f"   (out of {m})")
print("smallest q value in the null experiment: %.3f" % null_qv["q"].min())
```

No gene survives any reasonable $q$ value threshold, whereas the real comparison produced hundreds of findings at $q<0.01$. The multiple hypothesis correction is not simply throwing away everything, nor is it rubber-stamping noise. Note, though, that the estimated $\pi_0$ comes out below one, as the estimate assumes that the true nulls give a flat histogram, which they do not quite do here.

### A batch effect

The care we took to balance the flow cells was not a formality. Below we instead compare the four B6 mice sequenced on flow cell 6 to the six sequenced on flow cells 4 and 7. The null hypothesis is still true for every gene in a biological sense, as all ten mice are B6, but now the two groups also differ in *how* they were processed.

```{code-cell} ipython3
batch_a = [s for s in b6_samples if flow_cell[s] == 6]
batch_b = [s for s in b6_samples if flow_cell[s] != 6]
batch_pvalues = get_significance_two_groups(expr, batch_a, batch_b)

plt.figure(figsize=(6.5, 4.5))
plt.hist(batch_pvalues["p"], bins=bins, range=(0, 1), color="darkorange")
plt.axhline(m / bins, color="k", ls="--", lw=1)
plt.title("B6 flow cell 6 vs B6 flow cells 4 and 7")
plt.xlabel("$p$"); plt.ylabel("Number of genes");
batch_qv = qvalues(batch_pvalues)
print(f"genes with q < 0.05: {int((batch_qv['q'] < 0.05).sum())}")
```

Now the histogram shows a clear spike of small $p$ values, $\pi_0$ is estimated well below one, and the $q$ value machinery even reports a few significant genes, although none of them is differentially expressed in a biological sense. The $t$ test cannot tell a difference caused by biology from a difference caused by the processing of the samples. Batch effects thus cut both ways: when the batches are confounded with the groups they create false findings, and when they are spread over the groups they inflate the variance and cost power, as we saw above. This is why a good experimental design spreads the conditions evenly over the processing batches, as Bottomly et al. did for the two strains, and why more advanced analyses include the batch as an explanatory variable in the model, rather than leaving it in the noise.

+++

## Spline estimation of $\pi_0$

Storey and Tibshirani outline another procedure for estimating $\pi_0$ than the bootstrap procedure used above, i.e. using 
$\hat{\pi_0}(\lambda) = \frac{|\{p>\lambda \}|}{m(1-\lambda)}$.

Below, we almost follow the article's described procedure (please try to find the difference on how we select which lambdas we evaluate). Furthermore we fit a [cubic spline](https://en.wikipedia.org/wiki/Smoothing_spline) to these $\pi_0$ estimates.

```{code-cell} ipython3
from scipy.interpolate import UnivariateSpline

m = pvalues.shape[0] # The number of p-values
pvalues.sort_values("p",inplace=True,ascending=False) # sort the pvalues in descending order
num_p = -1
for ix in pvalues.index:
    num_p += 1
    lambda_p = pvalues.loc[ix,"p"]
    pi0_hat = num_p/(m*(1-lambda_p))
    pvalues.loc[ix,"pi0_hat"] = pi0_hat

pvalues.sort_values("p",inplace=True) # sort the pvalues in ascending order
s = UnivariateSpline(pvalues["p"],pvalues["pi0_hat"], k=3,s=10)
```

We plot the estimates (blue) as well as the spline fit (red) for two different intervals of $\lambda$. You will see the need of a smoother, particularly in the region near 1.0.

```{code-cell} ipython3
def plot_pi0_hat(p,s,xlow,xhigh,ylow,yhigh,ax):
  sns.lineplot(x=pvalues["p"],y=pvalues["pi0_hat"],errorbar=None,lw=3, ax=ax, color='b')
  sns.lineplot(x=pvalues["p"],y=s(pvalues["p"]),errorbar=None,lw=3,ax=ax, color='r')
  ax.set_xlim(xlow,xhigh);
  ax.set_ylim(ylow,yhigh);
  ax.set_xlabel(r"$\lambda $");
  ax.set_ylabel(r"$\pi_0(\lambda)$");

fig, ax = plt.subplots(1,2,figsize=(12, 4))
plot_pi0_hat(pvalues,s,0,1,0,1.05,ax[0])
plot_pi0_hat(pvalues,s,0.95,1,0,1.05,ax[1])
```

We can obtain a final estimate by evaluating the spline for $\lambda=1$, and compare it to the bootstrapping estimate.

```{code-cell} ipython3
print("Spline estimate of pi_0: " + str(s(1)))
print("Bootstrap estimate of pi_0: " + str(estimatePi0(list(pvalues["p"].values))))
```

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Interpret an estimated $\pi_0$ in its biological context, and contrast what $\pi_0$ tells you (how many genes differ in total) with what $q$ values tell you (which individual genes can be called at a controlled error rate).
2. Explain how a ground-truth null experiment, such as splitting genetically identical B6 mice into two groups, checks whether an analysis manufactures findings, and describe the $p$ value histogram and $q$ values you expect from it.
3. Explain why processing batches, such as flow cells, make a $t$ test conservative when they are spread evenly over the groups, but create false findings when they are confounded with the groups. Identify both situations from a $p$ value histogram.
4. Argue why a good experimental design spreads the conditions evenly over the processing batches, and judge whether a given design confounds batch with condition.
:::
