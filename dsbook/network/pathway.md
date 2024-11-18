# Pathway Analysis

## Introduction


A typical output from a high-throughput experiment is a list of genes, transcripts, or proteins. Given this list, one might want to identify common functions among these analytes. Here, pathway analysis has become a go-to method for associating functions with experimental findings. Such analysis provides essential insights into the complex relationships among biological molecules and how they contribute to specific cellular functions. A pathway represents a set of biochemical reactions and interactions that take place within a cell or organism, involving metabolites, genes, and proteins. Understanding these pathways allows researchers to link molecular changes to phenotypic outcomes, such as disease states or responses to treatment.

## Pathway Databases

Two of the most widely used databases for pathway analysis are KEGG and Reactome. These databases serve as repositories of curated information about biological pathways, providing researchers with access to comprehensive maps of cellular functions.

### KEGG

The **Kyoto Encyclopedia of Genes and Genomes (KEGG)** is a manually curated database that offers a collection of high-level maps integrating genomic, chemical, and systemic functional information. KEGG provides comprehensive pathway maps, including metabolic pathways, signal transduction pathways, and regulatory pathways. KEGG pathways are represented as graphical diagrams, which help in visualizing molecular interactions and their roles in specific biological functions.

### Reactome

**Reactome** is another prominent pathway database that provides detailed information about cellular processes, including metabolic reactions, signal transduction, immune system functions, and more. Reactome is an open-source, manually curated knowledge base, focusing on the relationships between genes, proteins, and other molecules in the context of biological pathways. Compared to KEGG, Reactome provides finer details about molecular interactions and is enriched by contributions from experts in the field.

## Over Representation Analysis (ORA)

One of the basic approaches to pathway analysis is **Over Representation Analysis (ORA)**. ORA identifies pathways that are statistically overrepresented in a list of differentially expressed genes or proteins compared to what would be expected by chance. ORA assumes that the input list contains genes of particular interest, such as those identified through an experiment involving transcriptomics or proteomics.

### Fisher's Exact Test

A common statistical method used for ORA is **Fisher's exact test**, which is designed to determine if there are nonrandom associations between two categorical variables. In the context of pathway analysis, Fisher's exact test can be used to assess whether the number of genes from a given pathway in the input list is significantly larger than what would be expected by random chance. The null hypothesis in this context is that a gene in the pathway is as probable to appear in the gene list as it is to appear in the non-list.

Consider the following contingency table:

|                   | In Pathway | Not in Pathway | **Sum** |
|-------------------|------------|----------------|---------|
| In Gene List      | $a$        | $b$            | $a+b$   |
| Not in Gene List  | $c$        | $d$            | $c+d$   |
| **All Genes**     | $a+c$      | $b+d$        | $a+b+c+d$ |

To calculate the significance of enrichment, we start by considering the number of combinations of ways to pick genes from the input list and pathway using choose notation. The probability of observing a particular outcome can be expressed as:

$$ P(X = a) = \frac{\binom{a+b}{a} \binom{c+d}{c}}{\binom{a+b+c+d}{a+c}} $$

Expanding the binomial expressions, we have:

$$ P(X = a) = \frac{\frac{(a+b)!}{a! b!} \cdot \frac{(c+d)!}{c! d!}}{\frac{(a+b+c+d)!}{(a+c)! (b+d)!}} $$

Simplifying this further, we get:

$$ P(X = a) = \frac{(a+b)! (c+d)! (a+c)! (b+d)!}{a! b! c! d! (a+b+c+d)!} $$

This represents the probability of picking $a$ genes in the pathway from the gene list and $c$ genes not in the pathway. To calculate the p-value, we need to consider not just this particular outcome, but also all more extreme outcomes, i.e., those with an equal or more imbalanced distribution. Thus, the p-value is obtained by summing the probabilities of all outcomes that are at least as extreme as the observed outcome:

$$ p = \sum_{x \geq a} P(X = x) $$

## Gene Set Enrichment Analysis (GSEA)

**Gene Set Enrichment Analysis (GSEA)** is a more sophisticated pathway analysis method compared to ORA. Unlike ORA, which relies on a predefined threshold to determine differentially expressed genes, GSEA considers the entire ranked list of genes, avoiding the need to impose an arbitrary cutoff.

In GSEA, gene sets corresponding to known biological pathways are tested for their enrichment at the top or bottom of a ranked gene list, typically based on differential expression scores. The idea is to determine whether the genes in a given pathway tend to be overrepresented among the most up- or down-regulated genes.

### Enrichment Score and Null Distribution

GSEA works by calculating an **enrichment score (ES)**, which measures how often genes from the gene set of interest appear in the ranked list. Starting at the top of the ranked gene list, an enrichment score is computed by walking down the list, increasing when a gene is in the gene set and decreasing otherwise.

To assess the statistical significance of the observed enrichment score, GSEA uses a **null distribution** obtained through permutation. The ranked gene list is shuffled many times to generate a background distribution of ES values, which can then be used to calculate the p-value for the observed enrichment score.

### Kolmogorov-Smirnov (KS) Test

The **Kolmogorov-Smirnov (KS) test** is used in GSEA to calculate the enrichment score. The KS test is a non-parametric test that measures the maximum deviation between the observed cumulative distribution of the gene set and the expected distribution under the null hypothesis. In GSEA, the enrichment score is effectively the maximum deviation encountered as we move down the ranked list, capturing whether genes from the pathway are found disproportionately at the top or bottom of the list. This score is then compared against the null distribution to determine statistical significance.

GSEA also corrects for multiple testing by calculating **false discovery rates (FDRs)**, thus allowing researchers to account for the number of pathways being tested simultaneously. A low FDR value indicates a significantly enriched pathway.

## Summary

Pathway analysis is a powerful tool for interpreting omics data in systems biology, enabling researchers to understand how changes in molecular profiles translate to biological processes. KEGG and Reactome are key resources that provide curated pathway information. Approaches like Over Representation Analysis and Gene Set Enrichment Analysis allow for the identification of pathways that are statistically enriched, shedding light on complex biological mechanisms. While ORA provides a straightforward way to identify overrepresented pathways, GSEA offers a more nuanced analysis that considers the complete distribution of gene ranks, reducing the bias introduced by arbitrary thresholds.
