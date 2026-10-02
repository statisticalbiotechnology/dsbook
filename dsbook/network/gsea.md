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
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/network/gsea.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/network/gsea.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/network/gsea.ipynb
    title: Run on Binder
---

# Example of ORA and GSEA

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/network/gsea.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/network/gsea.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/network/gsea.ipynb)

## The airway dataset

The methods of the previous chapter are easiest to appreciate on data where we already know what the answer should look like. We will therefore use the *airway* dataset of [Himes et al. (2014)](https://doi.org/10.1371/journal.pone.0099625), deposited at GEO as [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778).

Himes and co-workers cultured primary human airway smooth muscle (ASM) cells from four donors and treated them with dexamethasone, a synthetic glucocorticoid used to control asthma, or left them untreated. Each of the four cell lines contributes one treated and one untreated sample, so the experiment is a *paired* two-condition design with $4+4=8$ RNA-seq libraries. This is about as controlled as a human transcriptomics experiment gets: the only thing that differs within a pair is the drug.

It is also an experiment with a built-in positive control. Dexamethasone acts through the glucocorticoid receptor (*NR3C1*), a transcription factor with a well characterised set of direct target genes -- *FKBP5*, *PER1*, *TSC22D3* and *DUSP1* among them. If our pathway analysis is working, that signal should be impossible to miss.

```{code-cell}
:tags: [hide-cell]

import importlib.util, sys
def _has(pkg): return importlib.util.find_spec(pkg) is not None
if not _has("gseapy"):
    %pip install -q "gseapy"
```

GEO distributes the study as a gene-level FPKM matrix with one column per library. We download it once and cache it under `dsbook/data/`.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import os
import requests
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    ![ ! -f "dsbook/README.md" ] && git clone https://github.com/statisticalbiotechnology/dsbook.git
    my_path = "dsbook/dsbook/common/"
else:
    my_path = "../common/"
sys.path.append(my_path) # Read local modules for qvalue calculations
import qvalue

data_dir = my_path + "../data/"
os.makedirs(data_dir, exist_ok=True)
fpkm_path = data_dir + "GSE52778_All_Sample_FPKM_Matrix.txt.gz"
fpkm_url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/suppl/"
            "GSE52778_All_Sample_FPKM_Matrix.txt.gz")
if not os.path.exists(fpkm_path):
    print(f"Downloading {fpkm_path} ...")
    with open(fpkm_path, "wb") as fh:
        fh.write(requests.get(fpkm_url).content)

fpkm = pd.read_csv(fpkm_path, sep=r"\s+")
fpkm.iloc[:5, :6]
```

The full series also contains cells treated with the $\beta_2$-agonist albuterol, alone and in combination with dexamethasone. We ignore those and keep the eight libraries of the untreated/dexamethasone comparison. The library names carry the treatment, and the cell lines were sequenced in blocks of four, so `LL01`--`LL04` come from donor N61311, `LL05`--`LL08` from N052611, and so on. That lets us line up the samples pairwise.

```{code-cell} ipython3
untreated = ["Untreated_LL01", "Untreated_LL05", "Untreated_LL09", "Untreated_LL13"]
dex       = ["Dex_LL02",       "Dex_LL06",       "Dex_LL10",       "Dex_LL14"]
cell_line = ["N61311",         "N052611",        "N080611",        "N061011"]

airway = fpkm.set_index("gene_id")[dex + untreated]
airway = airway[(airway >= 1.0).all(axis=1)]   # keep genes expressed in every sample
airway = np.log2(airway)
airway.shape
```

Two remarks on that filter. First, the matrix is quantified as FPKM rather than as raw counts; FPKM is a poor basis for a count-based differential-expression model, but both the $t$-test we use below and the rank-based enrichment scores of GSEA only care about the relative ordering of samples and genes, so it serves our purpose. Second, the gene-level FPKM values contain a handful of all-or-nothing artefacts, where a gene is reported at several thousand FPKM in one condition and at exactly zero in the other. Requiring at least 1 FPKM in *all* eight libraries removes them and leaves us with around eleven thousand robustly expressed genes.

## Differential expression, dexamethasone versus untreated

Because the samples are paired by cell line, we use a paired $t$-test rather than the two-sample test used elsewhere in this book: each gene is tested on the four within-donor differences. This removes the donor-to-donor variation, which in primary cells is substantial.

```{code-cell} ipython3
def get_significance_paired(row):
    # the columns of airway are ordered as the four dexamethasone samples
    # followed by the four untreated samples, one pair per cell line
    treated, control = row.to_numpy()[:len(dex)], row.to_numpy()[len(dex):]
    log_fold_change = treated.mean() - control.mean()
    p = ttest_rel(treated, control)[1]
    return [p, -np.log10(p), log_fold_change]

pvalues = airway.apply(get_significance_paired, axis=1, result_type="expand")
pvalues.rename(columns = {list(pvalues)[0]: 'p', list(pvalues)[1]: '-log_p', list(pvalues)[2]: 'log_FC'}, inplace = True)
qvalues = qvalue.qvalues(pvalues)
```

A volcano plot shows a clear response: some four hundred genes move by more than a factor of two, with about twice as many going up as down.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(data=qvalues,x="log_FC",y="-log_p")
plt.xlabel("$log_2(FC)$")
plt.ylabel("$-log_{10}(p)$")
plt.show()
```

Before going further, it is worth checking the positive controls. The four textbook glucocorticoid-receptor target genes should be up-regulated by dexamethasone.

```{code-cell} ipython3
qvalues.loc[["FKBP5", "PER1", "TSC22D3", "DUSP1"]]
```

All four are up by roughly a factor of 8--16 with $q$ values around 1%. The experiment worked, and any pathway method we apply should recover this.

Note also how modest the $q$ values are. With four pairs the $t$ statistic rests on three degrees of freedom, so even a perfectly consistent eight-fold change cannot be declared significant with much confidence, and no gene in this experiment reaches a $q$ value below $10^{-2}$. That is a property of the sample size rather than of the effect sizes, and it is why the thresholds we use below are far less stringent than they would be for a cohort of hundreds of patients.

## Over-representation analysis

We use the [gseapy](https://gseapy.readthedocs.io/) module to run an overrepresentation analysis. The module is unfortunately not implementing pathway analysis itself. It instead call a remote webserver [Enrichr](https://maayanlab.cloud/Enrichr/).

In the analysis here we use the [KEGG](https://www.genome.jp/kegg/) database's definition of biological pathways. This choice can easily be changed to other databases such as GO.

Here we select the genes with $q$ values below $0.01$ as an input for the analysis. First we select this as our gene_list, and then we calculate the overlap of the gene list to all the pathways in KEGG. Note that the background is the set of genes we actually measured, not all human genes -- a pathway can only be over-represented relative to what could have been detected.

```{code-cell} ipython3
import gseapy as gp

pathway_db=['KEGG_2019_Human']
background=set(qvalues.index)
gene_list = list(qvalues.loc[qvalues["q"]<0.01,"q"].index)

output_enrichr=pd.DataFrame()
enr=gp.enrichr(
                gene_list=gene_list,
                gene_sets=pathway_db,
                background=background,
                outdir = None
            )
```

```{code-cell} ipython3
len(gene_list)
```

We clean up the results a bit by only keeping some of the resulting metrics. We also multiple hypothesis correct our results, and list the terms with a FDR less than 20%.

```{code-cell} ipython3
kegg_enr = enr.results[["P-value","Term"]].rename(columns={"P-value": "p"})
kegg_enr = qvalue.qvalues(kegg_enr)
kegg_enr.loc[kegg_enr["q"]<0.20]
```

The analysis seem to find overrepresentation of relatively few pathways, given how convincing the differential expression was. The strongest term, *Mineral absorption*, is driven by the metallothioneins *MT2A*, *MT1X* and *MT1E*, which are indeed classically induced by glucocorticoids -- but it is a somewhat oblique way of describing a drug response, and nothing that we would call a glucocorticoid pathway comes out at all.

```{code-cell} ipython3
enr.results.sort_values("P-value").head(5)[["Term", "P-value", "Genes"]]
```

This is the characteristic failure mode of ORA. The 200-odd genes that pass the threshold are a thin slice of a response in which thousands of genes shift a little, and by cutting at $q<0.01$ we threw away all the information about *how much* each gene moved and in which direction.

## Geneset Enrichment analysis

Subsequently we use gseapy to perform a geneset enrichment analysis (GSEA), which uses the whole ranked list instead of a thresholded one. This time we compare to the *hallmark* collection of [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp), fifty carefully curated sets that each summarise a coherent biological state.

We put the dexamethasone samples first so that a *positive* enrichment score means "up in dexamethasone".

```{code-cell} ipython3
classes = ["Dex" if sample_name.startswith("Dex") else "Untreated" for sample_name in airway.columns]
gs = gp.GSEA(data=airway,
                 gene_sets='MSigDB_Hallmark_2020',
                 classes=classes, # cls=class_vector
                 permutation_type='phenotype', # null from permutations of class labels
                 permutation_num=1000, # reduce number to speed up test
                 min_size=15, # minimal size of pathway
                 outdir=None,  # do not write output to disk
                 no_plot=True, # Skip plotting
                 method='signal_to_noise',
                 threads=4, # Number of allowed parallel processes
                 seed=42,
                 format='png',)
                 # ascending=True)
gs.run()
gs.res2d[["Term","ES","NES","NOM p-val","FDR q-val"]].head(8)
```

### Why the permutation null matters here

Look closely at the nominal $p$ values: several of them cluster tightly just above $0.016$, and none falls below it. That is not a coincidence. Permuting the phenotype labels of eight samples into two groups of four can produce at most $\binom{8}{4}=70$ distinct assignments, of which only 35 are distinct up to swapping the group names. The permutation $p$ value can therefore not be resolved beyond roughly $1/70$, no matter how many permutations we ask for, and the FDR estimates inherit that coarseness.

Phenotype permutation is the right null whenever it is affordable, because it preserves the correlation between genes. With very few samples per group -- the usual rule of thumb is fewer than seven -- it simply runs out of resolution, and the recommended fallback is to permute the *gene sets* instead: the ranked list is kept fixed and random gene sets of the same size are drawn to build the null. This is a weaker null, as it treats genes as independent and so tends to be anti-conservative for sets of co-regulated genes, but it is the only one with enough resolution here.

```{code-cell} ipython3
gs = gp.GSEA(data=airway,
                 gene_sets='MSigDB_Hallmark_2020',
                 classes=classes,
                 permutation_type='gene_set', # null from permutations of the gene sets
                 permutation_num=1000,
                 min_size=15,
                 outdir=None,
                 no_plot=True,
                 method='signal_to_noise',
                 threads=4,
                 seed=42,
                 format='png',)
gs.run()
gs_res = gs.res2d
```

We list the pathways with a FDR below 0.25, the threshold conventionally used for GSEA.

```{code-cell} ipython3
significant = gs_res[gs_res["FDR q-val"].astype(float)<0.25]
significant[["Term","ES","NES","NOM p-val","FDR q-val"]]
```

Where ORA found almost nothing, GSEA reports a couple of dozen hallmark sets, all of them up in dexamethasone. Two of the strongest are exactly the positive controls we hoped for, although they are not labelled the way a pharmacologist would label them:

* **Adipogenesis** is a glucocorticoid signature in disguise. Dexamethasone is a standard ingredient of the cocktail used to drive adipocyte differentiation in culture, so the hallmark set is populated with genes that the glucocorticoid receptor induces directly.
* **Androgen Response** is a steroid-receptor set, and the nuclear receptors share a large part of their target repertoire; *FKBP5* sits in its leading edge.

### Reading the leading edge, and reading the label

The second hit, **TNF-alpha Signaling via NF-kB**, deserves a closer look, because at face value it says the opposite of what one expects from an anti-inflammatory drug. Inspecting its leading edge explains the discrepancy.

```{code-cell} ipython3
term = "TNF-alpha Signaling via NF-kB"
lead = gs_res.loc[gs_res.Term == term, "Lead_genes"].iloc[0].split(";")
lead
```

These are not markers of an activated inflammatory programme. *PER1*, *DUSP1*, *KLF9*, *ATF3* and *NR4A1* are immediate-early genes that the glucocorticoid receptor induces directly, and further down the list *TNFAIP3* (A20) and *NFKBIA* ($I\kappa B\alpha$) are *inhibitors* of NF-$\kappa$B whose induction is one of the mechanisms by which glucocorticoids shut the pathway down. The gene set is named after the stimulus used to define it, not after the direction of regulation, and a set can be enriched because its brake was applied just as easily as because its accelerator was.

The cells here were never challenged with a cytokine, so there was no inflammatory tone to suppress; consistently, the *Inflammatory Response* hallmark itself is flat.

```{code-cell} ipython3
gs_res.loc[gs_res.Term.isin(["Inflammatory Response",
                             "Interferon Gamma Response"]),
           ["Term","NES","NOM p-val","FDR q-val"]]
```

The lesson is the one that makes GSEA worth the extra machinery over ORA: the enrichment score tells you *that* a coherent group of genes moved together, and the leading edge tells you *which* genes did it. Only the second of those can be interpreted biologically.

### Visualising the enrichment

The package we are using for accessing GSEA, gseapy, has some built in plotting routine for illustrating the enrichment for any given pathway.

```{code-cell} ipython3
axs = gs.plot(terms=gs_res.Term[0])
```

We can also compare the enrichment between multiple pathways

```{code-cell} ipython3
axs = gs.plot(list(significant.Term[:5]), show_ranking=False, legend_kws={'loc': (1.05, 0)}, )
```

as well as the heatmap (Normalized gene expression as a function of gene and sample) for the genes in a given pathway. The four columns on the left are the dexamethasone-treated cultures and the four on the right the untreated ones; the clean split of the leading-edge genes into two blocks is what the enrichment score is measuring.

```{code-cell} ipython3
from gseapy import heatmap
# plotting heatmap
i = 0
genes = gs_res.Lead_genes[i].split(";")[:25]
# Make sure that ``ofname`` is not None, if you want to save your figure to disk
ax = heatmap(df = gs.heatmat.loc[genes], z_score=0, title=gs_res.Term[i], figsize=(14,4))
```
