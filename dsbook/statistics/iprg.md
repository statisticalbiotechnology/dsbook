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
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/iprg.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/iprg.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/iprg.ipynb
    title: Run on Binder
---

# Checking $q$ values against a known truth

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/statistics/iprg.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/statistics/iprg.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/statistics/iprg.ipynb)

## Why we need a ground truth

In the [previous section](./qvalue.md) we *estimated* false discovery rates, but we could not
check the estimates: we do not know which genes are truly differentially expressed, so the
$q$ values had to be taken largely on faith. A procedure that reports $q=0.01$ makes a promise
about the long run frequency of errors, and that promise should be testable.

A *spike-in* experiment lets us test it. One prepares a complex but constant biological
background, and adds a small number of foreign proteins at concentrations chosen, and
therefore known, by the experimenter. Only the spiked-in proteins should differ between the
samples. We can then run a complete differential abundance analysis and afterwards count how
many of our findings were actually changed. The fraction of errors among the findings is the
*false discovery proportion* (FDP),

$$ {\rm FDP}(t) = \frac{|\{ \text{true nulls with } p_i \le t\}|}{|\{ p_i \le t \}|}, $$

which is what $\hat{\rm FDR}(t)$, and hence $\hat{q}$, is trying to predict. The FDR is an
expectation over hypothetical repetitions of the experiment; the FDP is a number we can read
off a single data set, *if* we know the truth.

## The iPRG 2015 spike-in study

We will use the benchmark set of the
[ABRF Proteome Informatics Research Group](https://abrf.org/research-group/proteome-informatics-research-group-iprg),
described by
[Choi et al., *J. Proteome Res.* 2017](https://doi.org/10.1021/acs.jproteome.6b00881), and
deposited at MassIVE under accession
[MSV000079843](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000079843).

A tryptic digest of a *Saccharomyces cerevisiae* lysate was used as background, identical in
all samples. Into it six proteins were spiked at four different combinations of concentrations,
giving four sample groups, each analysed in triplicate on a Q-Exactive instrument: twelve
LC-MS/MS runs in total. The amounts, in fmol per injection, were

| Protein | Accession | Condition 1 | Condition 2 | Condition 3 | Condition 4 |
| --- | --- | --- | --- | --- | --- |
| VAC2_YEAST | P44015 | 65 | 55 | 15 | 2 |
| ISCB_YEAST | P55752 | 55 | 15 | 2 | 65 |
| SFG2_YEAST | P44374 | 15 | 2 | 65 | 55 |
| UTR6_YEAST | P44983 | 2 | 65 | 55 | 15 |
| PGA4_YEAST | P44683 | 11 | 0.6 | 10 | 500 |
| ZRT4_YEAST | P55249 | 10 | 500 | 11 | 0.6 |

The six accessions carry yeast-like protein names, but they are in fact *Haemophilus
influenzae* proteins, so that participants in the study could not spot them by looking at the
protein list. Every other protein in the experiment is a yeast protein and should, by
construction, be unchanged.

We start from the protein-level table that the [MSstats](https://msstats.org) pipeline
produces after summarising the Skyline peptide intensities into one log$_2$ intensity per
protein and run.

```{code-cell} ipython3
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    ![ ! -f "dsbook/README.md" ] && git clone https://github.com/statisticalbiotechnology/dsbook.git
    my_path = "dsbook/dsbook/statistics/"
else:
    my_path = "./"
sys.path.append(my_path + "../common/") # Read the local module for q value calculations
import qvalue

# Protein-level run summary of the iPRG2015 study (MassIVE MSV000079843), as produced by
# MSstats from the Skyline output. Reshaped to a wide table from the teaching copy in
# https://github.com/ZenBrayn/asms_2020_fall_workshop (exercises/day3_basic_stat/iPRG_example_runsummary.csv)
iprg = pd.read_csv(my_path + "data/iprg2015_protein_runsummary.csv", index_col=0)
iprg = iprg.dropna()             # keep proteins quantified in all twelve runs
spiked = iprg["Spikein"]         # the ground truth: True for the six spiked proteins
intensity = iprg.drop(columns="Spikein")
print(intensity.shape, "proteins x runs, of which", spiked.sum(), "are spiked in")
intensity.head()
```

The twelve columns are named after the sample group and the technical replicate, which makes
the design easy to recover.

```{code-cell} ipython3
design = pd.DataFrame([c.split("_") for c in intensity.columns],
                      columns=["Condition", "Replicate"], index=intensity.columns)
design.T
```

## The design is visible in the data

Before doing any statistics we look at the quantities themselves. For every protein we
subtract its own mean log$_2$ intensity, putting all proteins on a comparable scale, and plot
the profile over the four conditions.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

condition_means = intensity.T.groupby(design["Condition"]).mean().T   # protein x condition
centered = condition_means.sub(condition_means.mean(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(9, 6))
rng = np.random.default_rng(1)
background = centered.loc[~spiked]
for prot in rng.choice(background.index, 200, replace=False):
    ax.plot(range(4), background.loc[prot], color="grey", alpha=0.15, lw=1)
for prot in centered.loc[spiked].index:
    ax.plot(range(4), centered.loc[prot], marker="o", lw=3, label=prot.split("|")[2])
ax.set_xticks(range(4), ["Cond 1", "Cond 2", "Cond 3", "Cond 4"])
ax.set_ylabel("centered $\\log_2$ intensity")
ax.legend(fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22));
```

The grey band of background proteins is essentially flat — 95% of them vary by less than a
factor of two between the extreme conditions — while the six coloured profiles swing by
factors of forty to five hundred, following the concentration table above.

## Differential abundance between two conditions

We now perform the analysis we would have performed without knowing the truth: a $t$ test of
every protein between condition 1 and condition 2, as in the
[differential expression](./testing.md) section. One detail differs. With only three replicates
per group we use the pooled-variance $t$ test rather than Welch's, since Welch's correction
buys robustness against unequal variances at a price in degrees of freedom that we cannot
afford at $n=3$.

```{code-cell} ipython3
def compare(cond_a, cond_b):
    """Two sample t test of every protein between two sample groups."""
    a = intensity.loc[:, design["Condition"] == cond_a]
    b = intensity.loc[:, design["Condition"] == cond_b]
    p = ttest_ind(a, b, axis=1)[1]
    return pd.DataFrame({"p": p,
                         "log_FC": b.mean(axis=1) - a.mean(axis=1),
                         "Spikein": spiked}, index=intensity.index)

results = compare("Condition1", "Condition2")
results.sort_values("p").head(8)
```

The $p$ value histogram of the background proteins is the diagnostic that the null
distribution behaves. If the yeast proteins really are unchanged, and the $t$ test is
correctly calibrated, their $p$ values should be uniformly distributed on $[0,1]$.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(x=results.loc[~results["Spikein"], "p"], bins=20, stat="density", ax=ax,
             color="steelblue", label="yeast background (true nulls)")
ax.axhline(1.0, color="black", ls="--", lw=2)
for p in results.loc[results["Spikein"], "p"]:
    ax.plot([p], [0.06], marker="^", ms=14, color="firebrick", clip_on=False)
ax.set_xlim(0, 1); ax.set_xlabel("$p$"); ax.legend(fontsize=13)
ax.set_title("red triangles: the six spiked proteins", fontsize=14);
print("fraction of background proteins with p < 0.05:",
      round((results.loc[~results["Spikein"], "p"] < 0.05).mean(), 3))
```

The histogram is close to flat, if slightly depleted near zero, and the fraction of background
proteins below $p=0.05$ is a little under the 5% that a uniform null would give. The
background is a proper — indeed mildly conservative — null. Five of the six spiked proteins
sit at the extreme left edge.
The sixth is VAC2, which was spiked at 65 and 55 fmol in the two conditions — a genuine but
only 1.2-fold difference that three replicates cannot resolve. It is a true positive that no
method could reasonably be expected to find.

## $q$ values

We estimate $\pi_0$ and the $q$ values with the same bootstrap procedure as in the
[previous section](./qvalue.md), here imported from the book's `common` module rather than
retyped.

```{code-cell} ipython3
np.random.seed(0)   # the pi_0 estimate is based on bootstrapping
print("pi_0 estimated to", round(qvalue.estimatePi0(list(results["p"].values)), 3))
results = qvalue.qvalues(results)
results.head(8)
```

The estimate $\hat{\pi}_0\approx 1$ says that essentially none of the proteins are
differentially abundant, which is the correct answer: 6 out of 3025 is $0.2\%$. Contrast this
with the knockout comparison of the previous section, where $\hat{\pi}_0$ came out far below
one.

## The payoff: estimated FDR against the true FDP

Because we know which proteins were spiked, we can walk down the list in order of increasing
$p$ value and, at every position, compute both what the procedure *claims* the error rate is
and what it *actually* is.

```{code-cell} ipython3
results["FDP"] = (~results["Spikein"]).cumsum() / np.arange(1, len(results) + 1)

top = results.head(15)
fig, ax = plt.subplots(figsize=(9, 6))
ax.step(range(1, 16), top["q"], where="mid", lw=3, label="estimated $q$ value")
ax.step(range(1, 16), top["FDP"], where="mid", lw=3, ls="--", label="true FDP")
ax.set_xlabel("number of proteins accepted")
ax.set_ylabel("false discovery rate")
ax.set_ylim(0, 1); ax.legend(fontsize=13);
top[["p", "log_FC", "Spikein", "q", "FDP"]]
```

The four highest ranked proteins are all spiked, and there the $q$ value and the FDP agree at
essentially zero. The first background protein enters at rank five, where the observed FDP
jumps to $1/5=0.2$ against an estimate of $0.04$. From then on the two curves track each other
loosely, with the estimate rising the faster of the two: by the tenth acceptance it has already
saturated at one, while the true FDP is only one half. Note also how *granular* the FDP
is: with six true positives it can take only the values $0, 1/k, 2/k, \ldots$, so one unlucky
background protein moves it by twenty percentage points. That granularity is a property of the
benchmark, not of the method.

To get a less noisy picture we repeat the analysis for all six pairs of conditions and record,
at a set of $q$ value thresholds, how many proteins were accepted and what fraction of them
were background.

```{code-cell} ipython3
import itertools
np.random.seed(0)
thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
rows = []
for a, b in itertools.combinations(range(1, 5), 2):
    r = qvalue.qvalues(compare(f"Condition{a}", f"Condition{b}"))
    for t in thresholds:
        accepted = r[r["q"] <= t]
        if len(accepted) > 0:
            rows.append({"comparison": f"{a} vs {b}", "threshold": t,
                         "accepted": len(accepted),
                         "true positives": int(accepted["Spikein"].sum()),
                         "FDP": 1 - accepted["Spikein"].mean()})
calibration = pd.DataFrame(rows)
calibration.pivot(index="threshold", columns="comparison", values="FDP").round(2)
```

```{code-cell} ipython3
comparisons = sorted(calibration["comparison"].unique())
# many comparisons share an FDP of exactly zero, so nudge them apart horizontally
jitter = {c: 1 + (i - 2.5) * 0.035 for i, c in enumerate(comparisons)}
calibration["x"] = calibration["threshold"] * calibration["comparison"].map(jitter)

fig, ax = plt.subplots(figsize=(7.5, 7))
sns.scatterplot(data=calibration, x="x", y="FDP", hue="comparison", s=120, ax=ax)
diagonal = np.geomspace(0.008, 0.26, 50)
ax.plot(diagonal, diagonal, color="black", ls="--", lw=2)
ax.set_xscale("log"); ax.minorticks_off()
ax.set_xticks(thresholds, [str(t) for t in thresholds])
ax.set_xlabel("estimated FDR ($q$ value threshold)")
ax.set_ylabel("true FDP")
ax.set_xlim(0.008, 0.28); ax.set_ylim(-0.03, 1.0)
ax.legend(fontsize=11, title="conditions");
```

Points on the dashed line are perfectly calibrated, points below it conservative and points
above it anti-conservative. At the strictest settings the procedure makes almost no mistakes:
five of the six comparisons have an FDP of exactly zero at $q\le0.02$, and four of them still
do at $q\le0.1$. Further out the points drift above the line, to FDPs between $0.14$ and
$0.33$ at $q\le0.2$. Part of that drift is unavoidable arithmetic rather than a real bias —
one false discovery among five accepted proteins is already an FDP of $0.2$, so a benchmark
with six true positives simply cannot resolve an FDR of $0.05$.

Condition 1 versus 3 is the real outlier. At $q\le0.1$ roughly two thirds of the accepted
proteins are background, and its $\hat{\pi}_0$ comes out around $0.6$ — the estimator
concluded that a third of the yeast proteins were differentially abundant. Printing $\pi_0$
and the behaviour of the background for every comparison shows where the problem lies.

```{code-cell} ipython3
np.random.seed(0)
for a, b in itertools.combinations(range(1, 5), 2):
    r = compare(f"Condition{a}", f"Condition{b}")
    bg = r.loc[~r["Spikein"], "p"]
    print(f"{a} vs {b}:  pi_0 = {qvalue.estimatePi0(list(r['p'].values)):.2f}"
          f"   background fraction with p<0.05 = {(bg < 0.05).mean():.3f}")
```

This is not a failure of the $q$ value machinery but of our assumption about the ground truth.
For condition 1 versus 3 more than one in ten of the yeast proteins falls below $p=0.05$,
about twice what a uniform null would give, and the estimator is right to conclude that
something is changing. The yeast background is simply not perfectly identical between those
two sample groups: the four mixtures were prepared and acquired separately, and small
differences in digestion, injected amount or acquisition order leave systematic traces in the
quantities. No FDR estimator can protect against a null that is not a null, and the proteins
it flags are not errors of statistics but of our labelling of the truth. A benchmark tells us
about the whole pipeline — sample preparation, quantification, normalisation and statistics
together — not only about the last step.

## What to take away

* Estimated $q$ values can be checked, and on a well behaved comparison of this benchmark they
  are accurate to within the resolution the benchmark allows.
* $\hat{\pi}_0$ is itself informative: a value near one is the signature of a data set in which
  almost nothing changes, and a much lower value on data where little *should* change warns
  that the experiment, not the biology, is producing signal.
* Spike-in benchmarks have limits. Six true positives is very few, their concentrations were
  chosen for convenience rather than realism, and peptides from the same protein do not give
  independent measurements, so the effective number of tests is smaller than the number of rows.
* FDR control in proteomics is in practice applied twice: to the peptide-spectrum matches
  during identification, where decoy databases provide an artificial null, and to the
  quantitative comparison, as here. Errors from the first stage propagate into the second,
  which is a further reason to validate whole pipelines on data such as this.
