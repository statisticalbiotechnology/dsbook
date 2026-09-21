---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
downloads:
  - url: https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/network/fba.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/network/fba.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/network/fba.ipynb
    title: Run on Binder
---

# Flux Balance Analysis

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://193.10.159.40.nip.io/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/network/fba.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/network/fba.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/network/fba.ipynb)

## Introduction

In the previous chapters we treated pathways as *sets* of genes: a pathway was something a gene could be a member of, and we asked whether such sets were enriched among our experimental findings. This chapter takes the next step and treats metabolism as a *quantitative network*. Metabolic reactions do not merely group genes together — they consume and produce metabolites in fixed stoichiometric proportions, and the rates at which they run must be compatible with each other. By writing down these constraints explicitly, we can *simulate* metabolism: predict how fast a cell can grow, which nutrients it needs, which genes it cannot survive without, and how it will reroute its metabolism when a gene is knocked out or a nutrient runs out.

The rate at which a reaction converts substrates into products is called its **flux**, typically measured in mmol per gram dry weight of cells per hour (mmol gDW⁻¹ h⁻¹). In principle, fluxes could be computed from enzyme kinetics. For a single enzyme, Michaelis–Menten kinetics gives the flux as

$$v = \frac{\mathrm{d}[P]}{\mathrm{d}t} = V_{\max}\frac{[S]}{K_M + [S]} = k_{cat}[E]_0\frac{[S]}{K_M + [S]},$$

but this requires knowing $k_{cat}$, $K_M$, enzyme and metabolite concentrations for *every* reaction in the cell — thousands of parameters that are difficult or impossible to measure. **Flux balance analysis (FBA)** takes a radically more economical approach: instead of trying to compute the one true flux distribution from kinetics, it asks which flux distributions are *possible* given the stoichiometry of the network, and then selects among them with an optimality principle (see the [primer by Orth, Thiele and Palsson](https://doi.org/10.1038/nbt.1614)). The only inputs needed are the reaction stoichiometries and a handful of measured exchange rates — no kinetic parameters at all.

## Genome-Scale Metabolic Models

The networks on which FBA operates are called **genome-scale metabolic models (GEMs)**. A GEM is a mathematical reconstruction of all metabolic reactions known to occur in an organism, assembled from its genome annotation together with pathway databases such as [KEGG](https://www.kegg.jp/), [Reactome](https://reactome.org/) and organism-specific literature. The reconstruction process is analogous to genome assembly: just as a genome is assembled from sequence reads, an organism's *reactome* is assembled from individual biochemical reactions, each linked to the enzyme that catalyzes it and the gene that encodes the enzyme.

A GEM contains three layers of information:

1. **Reactions and metabolites**, with exact, mass-balanced stoichiometry. Metabolites in different cellular compartments (cytosol, mitochondria, extracellular space, …) are treated as distinct species, so transport steps between compartments are themselves reactions.
2. **Gene–protein–reaction (GPR) rules** that connect each reaction to the gene(s) whose products catalyze it. The rules use Boolean logic: `GENE1 and GENE2` means the enzyme is a complex requiring both subunits, while `GENE1 or GENE2` means two isozymes can each carry the reaction independently.
3. **Bounds and exchange reactions** that describe which metabolites the cell can import from or export to its environment, and how fast.

The first GEM, for *Haemophilus influenzae*, was published in 1999; today models exist for thousands of organisms, from *E. coli* (the best curated model, *i*ML1515) to yeast and human. Human metabolism is described by generic models such as [Recon3D](https://doi.org/10.1038/nbt.4072) and [Human-GEM](https://github.com/SysBioChalmers/Human-GEM) with on the order of 13,000 reactions, 8,000 metabolites and 3,000 genes. A recent overview of available models and their applications is given by [Gu et al.](https://doi.org/10.1186/s13059-019-1730-3).

It is worth stressing what distinguishes a GEM from the pathway representations of the previous chapters. A pathway diagram records the *topology* of metabolism — which metabolites are connected by which reactions. A GEM additionally records the *stoichiometry*, and this makes all the difference: a topological analysis of glycolysis would tell us that glucose is eventually converted to pyruvate, but only the stoichiometric model knows that each glucose yields exactly two pyruvate, two net ATP, and that the pathway stalls without a way to regenerate NAD⁺. Empirically, model-based flux predictions of gene essentiality correlate with experimental knockout screens where purely topological measures (degree, betweenness or closeness centrality of a gene in the network) do not.

## The Stoichiometric Matrix

The mathematical core of a GEM is its **stoichiometric matrix** $S$, with one row per metabolite and one column per reaction. The entry $S_{ij}$ is the stoichiometric coefficient of metabolite $i$ in reaction $j$: negative if the metabolite is consumed, positive if it is produced, and zero if it does not participate. Almost all entries are zero, since a typical reaction involves only a handful of metabolites.

Throughout this chapter we will use a deliberately small model of core energy metabolism — a cartoon of glycolysis, respiration and fermentation:

```{figure} ./img/fba_toy_model.svg
:name: fba-toy-model
:alt: Toy model of core energy metabolism, with oxygen and glucose uptake, glycolysis, respiration, fermentation through two isozymes, lactate secretion and ATP demand.
:width: 100%

A toy model of core energy metabolism. Blue boxes are internal metabolites, red boxes are metabolites outside the cell boundary (dashed), connected through exchange reactions. The gray arrow indicates that oxygen enters as a co-substrate of the respiration reaction.
```

The model has five metabolites and eight reactions:

| Reaction | Chemistry | Interpretation |
|----------|-----------|----------------|
| $v_1$ | $\rightarrow$ Glc | glucose uptake (exchange) |
| $v_2$ | $\rightarrow$ O₂ | oxygen uptake (exchange) |
| $v_3$ | Glc $\rightarrow$ 2 Pyr + 2 ATP | glycolysis (lumped) |
| $v_4$ | Pyr + 3 O₂ $\rightarrow$ 15 ATP | respiration (lumped) |
| $v_5$ | Pyr $\rightarrow$ Lac | fermentation, isozyme 1 |
| $v_6$ | Pyr $\rightarrow$ Lac | fermentation, isozyme 2 |
| $v_7$ | Lac $\rightarrow$ | lactate secretion (exchange) |
| $v_8$ | ATP $\rightarrow$ | ATP demand (exchange) |

Whole pathways are lumped into single reactions, and side metabolites (CO₂, water, NADH) are omitted, but the ATP yields are roughly right: complete oxidation of one glucose yields $2 + 2 \times 15 = 32$ ATP, while fermentation to lactate yields only the 2 ATP of glycolysis. Reactions $v_1$, $v_2$, $v_7$ and $v_8$ are **exchange reactions**: they appear chemically unbalanced ("Glc appears from nothing") because they represent transport across the boundary of the system. Every model needs them — without exchange reactions, the steady-state assumption introduced below would force all fluxes to zero. The two parallel fermentation reactions $v_5$ and $v_6$ represent two isozymes, i.e. a GPR rule of the form `LDHA or LDHB`.

The stoichiometric matrix of this model is

$$
S = \begin{array}{r|rrrrrrrr}
 & v_1 & v_2 & v_3 & v_4 & v_5 & v_6 & v_7 & v_8\\\hline
\text{Glc} & 1 & 0 & -1 & 0 & 0 & 0 & 0 & 0\\
\text{O}_2 & 0 & 1 & 0 & -3 & 0 & 0 & 0 & 0\\
\text{Pyr} & 0 & 0 & 2 & -1 & -1 & -1 & 0 & 0\\
\text{ATP} & 0 & 0 & 2 & 15 & 0 & 0 & 0 & -1\\
\text{Lac} & 0 & 0 & 0 & 0 & 1 & 1 & -1 & 0\\
\end{array}
$$

## The Steady-State Assumption

Let $x$ be the vector of metabolite concentrations and $v$ the vector of fluxes. The stoichiometric matrix connects the two: the concentrations change according to

$$\frac{\mathrm{d}x}{\mathrm{d}t} = S\,v.$$

FBA now makes its central assumption, the **pseudo-steady state**: over the time scale of interest, internal metabolite concentrations do not change, because metabolism operates much faster than the processes that alter cellular composition (growth, regulation). Every internal metabolite is produced exactly as fast as it is consumed:

$$S\,v = 0.$$

This is a strong simplification — it makes the model blind to metabolite accumulation and to dynamics — but it is what frees us from kinetics: the equation involves only stoichiometry and fluxes, not concentrations or rate constants.

Together with **bounds** on each flux,

$$lb_i \le v_i \le ub_i,$$

the steady-state equation defines the space of feasible flux distributions. Bounds encode reaction irreversibility ($lb_i = 0$ for irreversible reactions), maximal uptake rates of nutrients (e.g. glucose uptake at most 10 mmol gDW⁻¹ h⁻¹), and knockouts ($lb_i = ub_i = 0$).

Note that the system $Sv = 0$ is **underdetermined**: a genome-scale model has far more reactions (unknowns) than metabolites (equations) — the toy model has 8 unknowns and 5 equations, Human-GEM has some 13,000 unknowns and 8,000 equations. There is therefore no unique solution but a whole continuum of feasible flux distributions, geometrically a convex polytope in flux space. Constraint-based modeling embraces this: rather than pinning down *the* flux state, the constraints delimit everything the network *can* do.

## FBA as Linear Programming

To select a biologically meaningful flux distribution from the feasible space, FBA assumes that evolution has tuned metabolism to be (near-)optimal for some cellular **objective** $Z = c^T v$, a linear combination of fluxes. FBA is then the optimization problem

$$
\begin{aligned}
\max_v \quad & Z = c^T v\\
\text{subject to} \quad & S v = 0\\
& lb \le v \le ub,
\end{aligned}
$$

which is a **linear program (LP)**: the objective and all constraints are linear in $v$. Linear programs of enormous size can be solved efficiently and to global optimality — this is what makes FBA feasible at genome scale, in contrast to kinetic models.

The choice of objective is a modeling decision. For microbes, the standard objective is flux through a **biomass reaction** — an artificial reaction that drains amino acids, nucleotides, lipids and energy in the proportions needed to build one gram of cell, so that maximizing it means maximizing growth rate. Fast-growing cells such as bacteria or tumor cells are plausibly selected for growth; for differentiated human cells one instead uses objectives such as ATP production (muscle), neurotransmitter synthesis (neurons), or lipid storage (adipocytes). Note that the objective must be something the cell *produces*: maximizing a nutrient uptake flux is not meaningful, since the optimum would simply sit at whatever upper bound we imposed on the uptake, telling us nothing.

## A Worked Example

Let us put the toy model to work with nothing but `numpy` and `scipy`'s linear-programming routine. We maximize ATP production ($c = e_8$) under a fixed glucose supply and ask what happens when oxygen becomes scarce.

```{code-cell}
import numpy as np
import pandas as pd
from scipy.optimize import linprog

metabolites = ["Glc", "O2", "Pyr", "ATP", "Lac"]
reactions = ["glc_uptake", "o2_uptake", "glycolysis", "respiration",
             "ferm_iso1", "ferm_iso2", "lac_secretion", "atp_demand"]

S = np.array([
    # v1  v2  v3  v4  v5  v6  v7  v8
    [  1,  0, -1,  0,  0,  0,  0,  0],   # Glc
    [  0,  1,  0, -3,  0,  0,  0,  0],   # O2
    [  0,  0,  2, -1, -1, -1,  0,  0],   # Pyr
    [  0,  0,  2, 15,  0,  0,  0, -1],   # ATP
    [  0,  0,  0,  0,  1,  1, -1,  0],   # Lac
])

def fba(bounds, c):
    """Maximize c @ v subject to S v = 0 and lb <= v <= ub."""
    res = linprog(-np.asarray(c), A_eq=S, b_eq=np.zeros(len(metabolites)),
                  bounds=bounds, method="highs")
    assert res.success, res.message
    return res.x + 0.0, -res.fun + 0.0  # + 0.0 turns -0.0 into 0.0

objective = np.zeros(len(reactions))
objective[reactions.index("atp_demand")] = 1.0

def make_bounds(glc_max=10.0, o2_max=np.inf):
    bounds = [(0.0, None)] * len(reactions)  # all reactions irreversible
    bounds[reactions.index("glc_uptake")] = (0.0, glc_max)
    bounds[reactions.index("o2_uptake")] = (0.0, o2_max)
    return bounds

flux_aerobic, atp_aerobic = fba(make_bounds(o2_max=np.inf), objective)
flux_limited, atp_limited = fba(make_bounds(o2_max=15.0), objective)

print(pd.DataFrame({"plenty of O2": flux_aerobic,
                    "O2 limited": flux_limited}, index=reactions))
print(f"\nMaximal ATP production: {atp_aerobic:.0f} vs {atp_limited:.0f}")
```

With plenty of oxygen, the optimal strategy is pure respiration: all 20 pyruvate produced from 10 glucose are oxidized, yielding 320 ATP, and no lactate is secreted. When oxygen uptake is capped at 15, respiration can only handle 5 pyruvate; the remaining 15 *overflow* into fermentation and leave the cell as lactate, and the ATP yield collapses to 95. The model has rediscovered **overflow metabolism**: yeast fermenting to ethanol despite available oxygen, *E. coli* spilling acetate — and the **Warburg effect**, the observation that tumor cells secrete large amounts of lactate. In the tumor case the binding constraint is not necessarily oxygen itself but the limited capacity of respiratory machinery relative to the enormous demand for fast growth; the FBA logic is the same, and constraint-based models of cancer metabolism build directly on this effect.

```{code-cell}
import matplotlib.pyplot as plt

x = np.arange(len(reactions))
w = 0.38
plt.figure(figsize=(8, 4))
plt.bar(x - w/2, flux_aerobic, w, label="plenty of O2")
plt.bar(x + w/2, flux_limited, w, label="O2 limited")
plt.xticks(x, reactions, rotation=30, ha="right")
plt.ylabel("flux (mmol gDW$^{-1}$ h$^{-1}$)")
plt.legend()
plt.tight_layout()
plt.show()
```

## Alternate Optima and Flux Variability Analysis

A subtlety hides in the solution above: `linprog` returned *one* optimal flux distribution, but it is not the only one. Any split of the fermentation flux between the two isozymes $v_5$ and $v_6$ gives exactly the same ATP production. The optimal *value* of an LP is unique, but the optimal *solution* often is not — and at genome scale, with thousands of parallel routes, degeneracy is the rule rather than the exception. Interpreting a single arbitrary optimum as "the" predicted flux state is a common mistake.

**Flux variability analysis (FVA)** quantifies the degeneracy. For each reaction $i$, it solves two additional LPs — minimize and maximize $v_i$ — while constraining the original objective to stay at its optimal value $Z^{opt}$:

$$
\begin{aligned}
\min_v / \max_v \quad & v_i\\
\text{subject to} \quad & S v = 0, \quad lb \le v \le ub, \quad c^T v = Z^{opt}.
\end{aligned}
$$

The resulting interval $[v_i^{\min}, v_i^{\max}]$ tells us how well determined each flux is at the optimum:

```{code-cell}
def fva(bounds, c):
    _, z_opt = fba(bounds, c)
    # pin the objective at its optimum via an extra equality constraint
    A_eq = np.vstack([S, c])
    b_eq = np.append(np.zeros(len(metabolites)), z_opt)
    ranges = []
    for i in range(len(reactions)):
        e = np.zeros(len(reactions)); e[i] = 1.0
        lo = linprog(e, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        hi = linprog(-e, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        ranges.append((lo.fun, -hi.fun))
    return pd.DataFrame(ranges, index=reactions, columns=["min", "max"])

print(fva(make_bounds(o2_max=15.0), objective).round(2))
```

Most fluxes are uniquely determined by the optimum — the uptakes, glycolysis, respiration and total lactate secretion have identical minimum and maximum. But each isozyme flux can individually range between 0 and 15: the model can say *that* 15 units of pyruvate must be fermented, but not *which* enzyme carries the flux. In real applications FVA is used to classify reactions as **essential** (flux always nonzero at the optimum), **optional** (range includes zero) or **blocked** (flux zero in every feasible solution), and to compare the metabolic capabilities of, say, different gut bacteria under a shared diet.

A related refinement is **parsimonious FBA (pFBA)**: among all flux distributions achieving the optimal objective, pick one minimizing the total flux $\sum_i |v_i|$. The rationale is that flux costs enzyme, and enzyme costs resources — so of two equally productive states, the cell prefers the more economical one. pFBA removes wasteful loops from the reported solution (though it cannot break ties between exactly symmetric isozymes, as FVA just showed us).

## Predicting Knockouts: Gene Essentiality

Because a GEM links genes to reactions through its GPR rules, we can simulate genetics. To knock out a gene *in silico*, we disable every reaction whose GPR rule evaluates to false without that gene — an `and` rule loses its reaction when any member is deleted, an `or` rule only when all isozymes are gone. Then we re-run FBA and compare the optimal objective with and without the gene:

$$\text{gene essential} \iff Z^{KO} \le \theta \cdot Z^{WT},$$

for some cutoff $\theta$ (with $\theta = 0$ demanding complete loss of the objective; in practice a cutoff like $\theta = 0.5$ is used, since a knockout that costs half the growth rate will quickly be outcompeted).

In our toy model, let glycolysis be catalyzed by gene $G_1$, respiration by the complex $G_2$ `and` $G_3$, and the two fermentation isozymes by $G_4$ and $G_5$:

```{code-cell}
gpr = {
    "glycolysis": "G1",
    "respiration": "G2 and G3",
    "ferm_iso1": "G4",
    "ferm_iso2": "G5",
}
genes = ["G1", "G2", "G3", "G4", "G5"]

def knockout_bounds(bounds, ko_gene):
    """Disable all reactions whose GPR rule fails without ko_gene."""
    bounds = list(bounds)
    state = {g: (g != ko_gene) for g in genes}
    for rxn, rule in gpr.items():
        if not eval(rule, {}, state):
            bounds[reactions.index(rxn)] = (0.0, 0.0)
    return bounds

bounds = make_bounds(o2_max=15.0)
_, z_wt = fba(bounds, objective)

rows = []
for g in genes:
    _, z_ko = fba(knockout_bounds(bounds, g), objective)
    rows.append({"knockout": g, "ATP": z_ko, "ratio": z_ko / z_wt,
                 "essential (cutoff 0.5)": z_ko / z_wt <= 0.5})
print(pd.DataFrame(rows).round(2).to_string(index=False))
```

The simulation reproduces classical genetic logic. Deleting $G_1$ abolishes ATP production entirely — glycolysis is unconditionally essential. Deleting either subunit of the respiratory complex ($G_2$ or $G_3$) drops the objective to 20/95 ≈ 21% of wild type — essential under our cutoff, even though the cell survives on fermentation alone. The isozymes $G_4$ and $G_5$ are individually dispensable, because the `or` redundancy lets the other one take over; only the double knockout would matter. Note also that essentiality is *conditional*: with unlimited oxygen, the fermentation genes would be irrelevant and respiration would dominate — an essential gene is essential *for a given objective in a given environment*.

This simple loop, scaled up to genome size, is one of the most successful uses of GEMs: predicted essentiality agrees with experimental knockout screens for roughly 90% of genes in *E. coli* and yeast. Applications include finding drug targets (genes essential in a pathogen or a tumor model but dispensable in the host model), designing minimal genomes, and **metabolic engineering** — methods such as [OptKnock](https://doi.org/10.1002/bit.10803) search for knockout combinations that *couple* growth to the secretion of a desired product, so that evolution pushes production up rather than down. These design problems ask discrete questions (which reactions to remove?) and are formulated as mixed-integer linear programs — still solvable, if more expensive than plain LP.

## From Toy Models to Genome Scale

Everything we did above transfers directly to real models; only the bookkeeping grows. In practice nobody builds $S$ by hand — models are distributed in the standard SBML format and analyzed with toolboxes such as [COBRApy](https://opencobra.github.io/cobrapy/) in Python or [RAVEN](https://github.com/SysBioChalmers/RAVEN) in MATLAB, which provide `model.optimize()`, `flux_variability_analysis()` and `single_gene_deletion()` as one-liners, along with curated models like the *E. coli* core model — a natural next step if you want to experiment beyond this chapter.

Two extensions are particularly important for human data science:

**Context-specific models.** The generic human GEM describes what *any* human cell could do, but a hepatocyte and a neuron express different enzyme repertoires. By overlaying transcriptomics (or proteomics) on the GPR rules, algorithms such as tINIT extract the subnetwork actually available to a given cell type or patient sample, yielding cell-type- and even patient-specific models. This is the sense in which GEMs serve as an *integration platform* for omics data: the model supplies the mechanistic scaffold, the omics data selects the active parts. Personalized cancer models built this way have been used to find tumor-specific essential genes as candidate drug targets, and liver models of non-alcoholic fatty liver disease correctly predicted altered demand for serine and glycine in patients — a hypothesis subsequently tested by supplementation studies.

**Better constraints.** Plain FBA constrains only stoichiometry and uptake bounds; its modern descendants add further physics. Enzyme-constrained models (e.g. the GECKO framework) include the enzymes themselves in the stoichiometry, using $v_i \le k_{cat}\cdot[E_i]$ so that fluxes are limited by proteome capacity — with such constraints, phenomena like overflow metabolism emerge without imposing an oxygen cap by hand, exactly as our toy example hinted. Other variants integrate quantitative metabolomics to relax the steady-state assumption where measurements show metabolite levels changing.

For a deeper treatment, the primer [What is flux balance analysis?](https://doi.org/10.1038/nbt.1614) by Orth et al. is the canonical starting point; [O'Brien et al.](https://doi.org/10.1016/j.cell.2015.05.019) review the broader family of constraint-based methods; and [Gu et al.](https://doi.org/10.1186/s13059-019-1730-3) and [Zhang & Hua](https://doi.org/10.3389/fphys.2015.00413) survey applications in biotechnology and systems medicine.

## Discussion Questions

1. The pseudo-steady state assumption sets $Sv = 0$ for internal metabolites. What biological situations violate it, and what do we gain in exchange for accepting it?
2. Why are growth (biomass) and ATP production reasonable objective functions, while maximizing glucose uptake is not? What objective would you propose for a neuron, or for a liver cell exporting glucose during fasting?
3. In the FVA output, the two isozyme fluxes each ranged over $[0, 15]$. Does this mean either range endpoint is biologically plausible? What extra information would pin the split down?
4. Our knockout of the respiration complex left 21% of ATP production. Would you call it essential? Discuss how the essentiality cutoff $\theta$ trades false positives against false negatives, e.g. when nominating drug targets.
5. A metabolic network diagram (topology only) and a GEM (with stoichiometry) both contain glycolysis. Give a concrete question the GEM can answer that the diagram cannot.
