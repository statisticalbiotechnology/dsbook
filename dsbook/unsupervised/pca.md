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
  - url: https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/unsupervised/pca.ipynb&branch=main
    title: Run on the KTH JupyterHub
  - url: https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/unsupervised/pca.ipynb
    title: Run on Google Colab
  - url: https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/unsupervised/pca.ipynb
    title: Run on Binder
---

# Principal Component Analysis (PCA)

<!-- launch-badges -->
[![KTH JupyterHub](https://img.shields.io/badge/launch-KTH%20JupyterHub-F37626?logo=jupyter&logoColor=white)](https://nbserv.duckdns.org/hub/user-redirect/git-pull?repo=https://github.com/statisticalbiotechnology/dsbook&urlpath=lab/tree/dsbook/dsbook/unsupervised/pca.ipynb&branch=main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/statisticalbiotechnology/dsbook/blob/main/dsbook/unsupervised/pca.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/statisticalbiotechnology/dsbook/main?labpath=dsbook/unsupervised/pca.ipynb)

Principal Component Analysis (PCA) is an unsupervised learning method that finds the axes of maximum variation in your data and projects the data onto them. In biotechnology it is used to find the major sources of variation in gene expression data, to visualize how samples relate to each other, to detect batch effects and outliers, and as a preprocessing step before clustering.

````{margin}
```{note}
A [projection](https://en.wikipedia.org/wiki/Projection_(linear_algebra)) is a transformation that maps a point onto a subspace, such as a line or plane, by finding the closest point in that subspace. Imagine shining a light perpendicular to a surface; the shadow cast is a projection of the original shape onto that surface. In mathematics, projections are often used to simplify complex data by reducing dimensions while retaining important relationships.
```
````

## Decomposition of a matrix

- From linear algebra, we know that we can multiply two vectors into a matrix.

```{math}
\mathbf{X} = \mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^{\textsf{T}} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix} \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix} = \begin{bmatrix} u_1 v_1 & u_1 v_2 & u_1 v_3 \\ u_2 v_1 & u_2 v_2 & u_2 v_3 \\ u_3 v_1 & u_3 v_2 & u_3 v_3 \\ u_4 v_1 & u_4 v_2 & u_4 v_3 \end{bmatrix}.
```

- What if we could do the opposite? I.e., given a matrix $\mathbf{X}$, what would be the vectors $\mathbf{u}$ and $\mathbf{v}$ that best represent $\mathbf{X}$ so that $\mathbf{X} \approx \mathbf{u} \mathbf{v}^{\textsf{T}}$? This is, in essence, what you do with principal component analysis (PCA).

Here, *best* has the same meaning as for [least squares regression](../supervised/linear.md): PCA chooses the $\mathbf{u}$ and $\mathbf{v}$ that minimize the sum of squared residuals between $\mathbf{X}$ and its approximation,

```{math}
\mathcal{L}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{N} \sum_{j=1}^{M} \left( X_{ij} - u_i v_j \right)^2 .
```

For the best approximation, the total sum of squares of $\mathbf{X}$ splits into the part that the approximation describes and the squared residuals that it leaves. For centered data, where the sum of squares measures the variance, minimizing the squared residuals is therefore the same as maximizing the variance that the component describes. The two are just different descriptions of the same principal component. Unlike in regression, no variable plays the role of a dependent variable $y$. Regression measures the residuals vertically, along $y$ only, whereas PCA measures them perpendicular to the component.


### A scenario where this might be useful

In PCA, we often deal with a matrix that encapsulates multiple sources of variation. Imagine a scenario where we’re studying gene expression levels across several samples. Here, the gene expression levels might be influenced by two key factors:

1. **Gene-Specific Effects**: These are inherent to each gene, such as its baseline expression level or potential for expression. For example, some genes may generally be expressed at high levels across all conditions, while others have lower baseline expression.
  
2. **Sample-Specific Effects**: These capture the influence of each sample (environmental condition, patient, disease state) on gene expression. For instance, one condition might increase expression levels across many genes, while another has a weaker or even suppressive effect.

If the expression matrix is the outer product of these two effects, $\mathbf{X} = \mathbf{u}\mathbf{v}^{\textsf{T}}$, then $\mathbf{u}$ holds the gene-specific effects and $\mathbf{v}$ the sample-specific effects.

### A toy example

Let us make this concrete with 4 genes and 4 samples. Let the gene-specific effects be $\mathbf{u} = [2, 1, 0.5, 3]^{\textsf{T}}$ and the sample-specific effects $\mathbf{v} = [1, 0.8, 1.2, 0.5]^{\textsf{T}}$. Each entry of the outer product is $X_{ij} = u_i v_j$:

$$
\mathbf{X} = \begin{bmatrix} 2 \\ 1 \\ 0.5 \\ 3 \end{bmatrix} \begin{bmatrix} 1 & 0.8 & 1.2 & 0.5 \end{bmatrix} =
\begin{bmatrix}
2 & 1.6 & 2.4 & 1 \\
1 & 0.8 & 1.2 & 0.5 \\
0.5 & 0.4 & 0.6 & 0.25 \\
3 & 2.4 & 3.6 & 1.5
\end{bmatrix}
$$

In practice we only observe $\mathbf{X}$, and PCA has to find the underlying factors. This toy matrix was built from a single pair of vectors, so it has rank 1: the first principal component reproduces $\mathbf{X}$ exactly, and all further components are zero. Real data are never exactly of rank 1, and the later components then describe the variation that the first one leaves unexplained.

### Scaling the components

The magnitude of a component can be moved freely between $\mathbf{u}$ and $\mathbf{v}$: if we scale $\mathbf{u}$ by a factor $\alpha$ and $\mathbf{v}$ by $1/\alpha$, the product $\mathbf{u}\mathbf{v}^{\textsf{T}}$ stays the same. To make the decomposition unique, we scale both vectors to unit length,

$$
\| \mathbf{u} \| = 1, \quad \| \mathbf{v} \| = 1,
$$

and collect the whole magnitude in a scalar $S$:

$$
\mathbf{X} \approx S\,\mathbf{u}\mathbf{v}^{\textsf{T}}.
$$

$\mathbf{u}$ (one entry per gene) and $\mathbf{v}$ (one entry per sample) then only give directions, while $S$ measures the strength of the component. The variance that the component explains is proportional to $S^2$.

## More than one principal component

- Once the first component is removed from $\mathbf{X}$, the residuals $\mathbf{X} - S_1\mathbf{u}^{(1)} \mathbf{v}^{(1)\textsf{T}}$ can in turn be decomposed. The vectors $\mathbf{u}^{(2)}$ and $\mathbf{v}^{(2)}$ that best describe this residual matrix, again in the sense of the smallest sum of squared residuals, form the second principal component, while the original pair forms the first.

- In this manner, we can derive as many principal components as there are rows or columns (whichever is smaller) in $\mathbf{X}$. For plotting, we usually look at the first two or three components; for further analysis, such as the clustering of single cells later in this part of the book, we often keep more.

## An illustration of PCA

|           | Sample $1$ | Sample $2$ | Sample $3$ |       | Sample $M$|       | Gene specific<br>$\mathbf{u}^{(1)}$ | Gene specific<br>$\mathbf{u}^{(2)}$ |
|-----------|----------|------------|-----------|----------|----------|-------|--------------------|----------------|
| Gene $1$  | $X_{11}$ | $X_{12}$   | $X_{13}$  | $\ldots$ | $X_{1M}$ |       | $u^{(1)}_1$ | $u^{(2)}_1$ |
| Gene $2$  | $X_{21}$ | $X_{22}$   | $X_{23}$  | $\ldots$ | $X_{2M}$ |       | $u^{(1)}_2$ | $u^{(2)}_2$ |
| Gene $3$  | $X_{31}$ | $X_{32}$   | $X_{33}$  | $\ldots$ | $X_{3M}$ |       | $u^{(1)}_3$ | $u^{(2)}_3$ |
| $\vdots$ | $\vdots$ | $\vdots$  | $\vdots$ | $\ddots$ | $\vdots$ |       | $\vdots$  | $\vdots$  |
| Gene $N$  | $X_{N1}$ | $X_{N2}$   | $X_{N3}$  | $\ldots$ | $X_{NM}$ |       | $u^{(1)}_N$ | $u^{(2)}_N$ |
|           |          |            |           |          |          |       |             |         |
| Sample specific $\mathbf{v}^{(1)\textsf{T}}$ | $v^{(1)}_1$ | $v^{(1)}_2$ | $v^{(1)}_3$ | $\ldots$ | $v^{(1)}_M$ |  | $S_1$ |  |
| Sample specific $\mathbf{v}^{(2)\textsf{T}}$ | $v^{(2)}_1$ | $v^{(2)}_2$ | $v^{(2)}_3$ | $\ldots$ | $v^{(2)}_M$ |  |        | $S_2$ |


## Principal Components

PCA aims to identify the directions, or **principal components**, in the data that describe the most variance. These directions are essentially the new axes into which the original data is [projected](https://en.wikipedia.org/wiki/Projection_(linear_algebra)), and they help us gain insight into the underlying structure and patterns. The principal components are ordered such that the first principal component describes the largest possible variance, while each subsequent component describes as much of the remaining variance as possible, subject to being orthogonal to the preceding components.

The gene-specific vector $\mathbf{u}^{(i)}$ and the sample-specific vector $\mathbf{v}^{(i)}$ come in pairs. Together, the first few pairs give the best low-dimensional description of $\mathbf{X}$, in the sense that they minimize the sum of squared residuals. The two members of a pair are linked by projection:

- Projecting $\mathbf{X}$ on $\mathbf{v}^{(1)}$ collapses each row (gene) of $\mathbf{X}$ onto a single number, and gives $\mathbf{X}\mathbf{v}^{(1)} = S_1\mathbf{u}^{(1)}$.
- Projecting $\mathbf{X}^{\textsf{T}}$ on $\mathbf{u}^{(1)}$ collapses each column (sample) of $\mathbf{X}$ onto a single number, and gives $\mathbf{X}^{\textsf{T}}\mathbf{u}^{(1)} = S_1\mathbf{v}^{(1)}$.

So $\mathbf{u}^{(1)}$ is, up to the scaling factor $S_1$, the projection of $\mathbf{X}$ on $\mathbf{v}^{(1)}$, and vice versa. The entries of $\mathbf{v}$ are used as sample-specific **scores**, which show where each sample lies along a component, and the entries of $\mathbf{u}$ as gene-specific **loadings**, which show which genes drive the component.

## Dimensionality Reduction and Explained Variance

Keeping only the first few components is a form of **dimensionality reduction**: each sample is described by its scores on a handful of components instead of by thousands of gene expression values. The share of the total variance that a component describes is called its **explained variance**. It decreases from the first component onwards, and a plot of the explained variance per component, a *scree plot*, shows how many components are needed to capture most of the variation.

Geometrically, PCA is an affine transformation: it subtracts the mean (a translation) and rotates the data onto new axes. Keeping only the first few axes then projects the data onto a lower-dimensional subspace. Because this projection is linear, it can flatten the data but cannot unfold a curved structure.

## Singular Value Decomposition (SVD)

Principal components are calculated with the **singular value decomposition (SVD)**. Before the decomposition, the data are *centered*: the mean of each variable (for example, each gene's mean expression across the samples) is subtracted, so that the components describe the variation around the mean rather than the mean itself. The SVD then decomposes the centered data matrix $\mathbf{X}$ as

```{math}
\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^{\textsf{T}}
```

where

- the columns of $\mathbf{U}$ are the gene-specific vectors $\mathbf{u}^{(i)}$,
- the columns of $\mathbf{V}$ are the sample-specific vectors $\mathbf{v}^{(i)}$,
- $\mathbf{S}$ is a diagonal matrix that holds the singular values $S_i$, the magnitude of each principal component pair.

Since the variance described by a component is proportional to $S_i^2$, the fraction of the variance explained by component $i$ is $R^2_i=\frac{S^2_i}{\sum_j S^2_j}$. Without centering, the first component would mostly describe the mean, and this ratio would be a fraction of the sum of squares rather than of the variance.

## An example of reconstruction with PCA, Eigenfaces

```{code-cell}
:tags: [hide-cell]

# When the page runs in the browser (JupyterLite), we let Python download files through the browser
import sys
if sys.platform == "emscripten":
    import micropip
    await micropip.install("pyodide-http")
    import pyodide_http
    pyodide_http.patch_all()
```

```{code-cell}
import matplotlib.pyplot as plt
from numpy.random import RandomState

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

rng = RandomState(0)


# Load the Olivetti faces dataset
faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape

# Global centering (focus on one feature, centering all samples)
faces_centered = faces - faces.mean(axis=0)

# Local centering (focus on one sample, centering all features)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)

# Define a base function to plot the gallery of faces
n_row, n_col = 5, 4
n_components = n_row * n_col
image_shape = (64, 64)

def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()

# Create a PCA model and fit it to the centered faces
print("Fitting PCA model to faces dataset...")
pca = PCA(n_components=n_components, random_state=rng)
pca.fit(faces_centered)

# Extract the eigenfaces (principal components)
eigenfaces = pca.components_

# Plot the first n_components eigenfaces
plot_gallery("Eigenfaces (Principal Components)", eigenfaces[:n_components])

# Project the original faces onto the PCA components to get their low-dimensional representation
faces_pca_projection = pca.transform(faces_centered)

# Reconstruct the faces using the PCA projection
faces_reconstructed = pca.inverse_transform(faces_pca_projection)

# Plot a gallery of original and reconstructed faces for comparison
plot_gallery("Original Centered Faces", faces_centered[:4*5],5,4)
plot_gallery("Reconstructed Faces", faces_reconstructed[:4*5],5,4)

print("Explained variance ratio of components: ", pca.explained_variance_ratio_)

# Plot how much variance is explained by each component
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Each Principal Component')
plt.show()
```

In this example, we reconstruct facial images from a low-dimensional representation. The Olivetti faces dataset contains 400 photos of 40 people, each of 64 × 64 = 4,096 pixels. Each photo is a sample and each pixel a variable. PCA needs every pixel to have mean zero across the photos. scikit-learn's `PCA` subtracts these means itself, so the explicit global centering in the code is not needed for the PCA, but it makes the plotted faces show deviations from the average face. The local centering subtracts each photo's own mean, which removes differences in overall brightness between the photos.

The principal components of this dataset are called eigenfaces. Each eigenface is a pattern over all 4,096 pixels that captures a direction of large variation between the faces (e.g., eyes, nose, or general facial structure). An eigenface plays the role of the gene-specific vector $\mathbf{u}$: it holds one value per pixel. The matching photo-specific scores, which `pca.transform` returns, play the role of the sample-specific vector $\mathbf{v}$: they say how much of each eigenface is present in each photo. Note that scikit-learn scales only the eigenfaces to unit length, and puts the magnitude of each component in the scores, so the scores of component $i$ equal $S_i\mathbf{v}^{(i)}$ rather than $\mathbf{v}^{(i)}$.

Projecting a face onto the 20 eigenfaces turns its 4,096 pixel values into 20 scores. The reconstruction (`inverse_transform`) reverses the process by adding up the eigenfaces, weighted by the face's scores. The reconstructed faces are blurred but still recognizable, although each of them is described by 20 numbers instead of 4,096. The last plot shows why they are not perfect: the first eigenface explains 18% of the variance and the 20 eigenfaces together about 70%, so the remaining 30% of the variation, the finer detail, is lost.

## An example from microbial ecology: bacteria in the Baltic Sea

The Baltic Sea is one of the largest bodies of brackish water on Earth. Rivers bring in fresh water, while salt water from the North Sea only enters through the narrow straits around Denmark. The salinity therefore falls from above 30 in the Skagerrak to about 2 in the northern Bothnian Bay (salinity is measured in practical salinity units, which roughly correspond to grams of salt per kilogram of water). The water also changes strongly with the seasons, from ice-cold winters to summer temperatures around 20 °C.

[Latz et al. (2024)](https://doi.org/10.1038/s41597-023-02825-5) sampled surface water every month between January 2019 and February 2020, at 22 stations along this gradient, as part of the Swedish national marine monitoring program. They identified the bacteria in each sample by sequencing a part of the bacterial 16S rRNA gene. Identical reads are grouped into **amplicon sequence variants (ASVs)**, which roughly correspond to bacterial species or strains. Each sample is thus described by the number of reads of each ASV, and the salinity, water temperature and sampling date of each sample were recorded as well. The data are available from the [SciLifeLab Data Repository](https://doi.org/10.17044/scilifelab.20751373) under a CC BY 4.0 license.

We download the data, and keep the samples from the monthly sampling (the dataset also contains samples used to test the protocol) that have at least 5,000 reads.

```{code-cell}
import os
import sys
import zipfile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_COLAB = 'google.colab' in sys.modules
data_dir = "data/" if IN_COLAB else "../data/"
os.makedirs(data_dir, exist_ok=True)

# The data of Latz et al. (2024), from the SciLifeLab Data Repository
files = {"baltic_metadata.txt": "https://ndownloader.figshare.com/files/41414001",
         "baltic_metabarcoding_data.zip": "https://ndownloader.figshare.com/files/41414010"}
for name, url in files.items():
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        urlretrieve(url, path)

# Information on each sample, such as its salinity, temperature and sampling date. The second line holds the units
meta = pd.read_csv(os.path.join(data_dir, "baltic_metadata.txt"), sep="\t", skiprows=[1], encoding="latin-1",
                   index_col="sample_alias/library_name")
# Read counts of each ASV (rows) in each sample (columns), and the taxonomy of each ASV
with zipfile.ZipFile(os.path.join(data_dir, "baltic_metabarcoding_data.zip")) as archive:
    counts = pd.read_csv(archive.open("metabarcoding_data/16S/seqtab_16S.txt"), sep="\t")
    taxa = pd.read_csv(archive.open("metabarcoding_data/16S/taxa_16S.txt"), sep="\t")

# Keep the 16S samples of the monthly sampling
meta = meta[meta["target gene"].str.contains("16S", na=False) & meta["test_type"].isna()]
counts = counts[meta.index]
# Remove samples with fewer than 5,000 reads
counts = counts.loc[:, counts.sum() >= 5000]
meta = meta.loc[counts.columns]
print(f"{counts.shape[1]} samples from {meta['station_name'].nunique()} stations, and {counts.shape[0]} ASVs")
```

The total number of reads differs a lot between the samples, so we first divide the counts of each sample by its total, which gives the relative abundance of each ASV. Most of the ASVs are rare, and we keep only the 1,000 most abundant ones, which together account for about 90% of the reads. The relative abundances span several orders of magnitude, and without a transformation, a few dominant ASVs would decide the whole analysis. We therefore take the logarithm of the relative abundances, after adding a small pseudocount of $10^{-5}$ to avoid taking the logarithm of zero.

The table is arranged like the genes $\times$ samples matrices earlier in this chapter: each ASV is a row and each sample a column, so the ASVs play the role of the genes. We center each ASV by subtracting its mean over the samples, and decompose the matrix with SVD, $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^{\textsf{T}}$. The columns of $\mathbf{U}$ are then the ASV-specific vectors $\mathbf{u}^{(i)}$, the rows of $\mathbf{V}^{\textsf{T}}$ are the sample-specific vectors $\mathbf{v}^{(i)}$, and both have unit length.

```{code-cell}
# Relative abundances of the 1,000 most abundant ASVs, on a logarithmic scale
rel = counts / counts.sum()
top = rel.mean(axis=1).sort_values(ascending=False).index[:1000]
print(f"The 1,000 most abundant ASVs account for {rel.loc[top].sum().mean():.0%} of the reads")
X = np.log10(rel.loc[top] + 1e-5)

# Center each ASV, and decompose the ASVs x samples matrix
X = X.sub(X.mean(axis=1), axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
R2 = S**2 / np.sum(S**2)

# Variables measured in each sample, which we compare to the scores below
salinity = meta["salinity_average"]
latitude = meta["geographic location (latitude)"]
temperature = meta["temperature_water_CTD_0-1m"]
date = pd.to_datetime(meta["collection date"])

# The sign of a principal component is arbitrary, as flipping both u and v leaves their product unchanged.
# We choose the signs so that the scores of the first three components, the rows of Vt,
# increase with salinity, latitude and water temperature, respectively
for i, values in enumerate([salinity, latitude, temperature]):
    measured = values.notna().to_numpy()
    if np.corrcoef(Vt[i, measured], values[measured])[0, 1] < 0:
        U[:, i] *= -1
        Vt[i] *= -1

print("Fraction of the variance explained by the first components:", R2[:3].round(3))

# Score plot of the first two components, colored by salinity
plt.figure(figsize=(8, 6))
plt.scatter(Vt[0], Vt[1], c=salinity, cmap='viridis', s=15)
plt.colorbar(label='Salinity')
plt.xlabel(f'PC1 ({R2[0]:.0%} of the variance)')
plt.ylabel(f'PC2 ({R2[1]:.0%} of the variance)')
plt.title('Bacterial communities of the Baltic Sea')
plt.show()
```

The first component separates the samples into two groups. The salty samples from the Kattegat and the Skagerrak lie to the right, and the brackish samples from the Baltic Sea itself, with salinities below 10, to the left. The second component spreads out the brackish samples. To see what the components capture, we plot the scores of each sample against the variables that were measured along with it.

```{code-cell}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, values, i, label in [(axes[0], salinity, 0, 'Salinity'), (axes[1], latitude, 1, 'Latitude (°N)')]:
    measured = values.notna().to_numpy()
    r = np.corrcoef(Vt[i, measured], values[measured])[0, 1]
    ax.scatter(values, Vt[i], s=15)
    ax.set_xlabel(label)
    ax.set_ylabel(f'PC{i + 1}')
    ax.set_title(f'PC{i + 1} against {label.split()[0].lower()}, r = {r:.2f}')

# The third component against the sampling date, colored by water temperature
measured = temperature.notna().to_numpy()
r = np.corrcoef(Vt[2, measured], temperature[measured])[0, 1]
points = axes[2].scatter(date, Vt[2], c=temperature, cmap='coolwarm', s=15)
fig.colorbar(points, ax=axes[2], label='Water temperature (°C)')
axes[2].set_xlabel('Sampling date')
axes[2].set_ylabel('PC3')
axes[2].set_title(f'PC3 over the year, r(temperature) = {r:.2f}')
axes[2].tick_params(axis='x', labelrotation=45)
plt.tight_layout()
plt.show()
```

Each of the first three components reflects an environmental gradient:

1. **PC1 follows salinity.** It is the largest source of variation, as the bacterial communities of the salty Kattegat and Skagerrak differ strongly from those of the brackish Baltic Sea.
2. **PC2 follows latitude.** Within the brackish Baltic Sea, it separates the northern Bothnian Sea and Bothnian Bay, where rivers supply fresh water rich in humic substances, from the southern basins.
3. **PC3 follows the time of year.** Its scores are high in summer and low in winter, in all parts of the sea, and closely follow the water temperature.

PCA was never told the salinity, location or date of the samples. It found these gradients from the composition of the bacterial communities alone, and we only compared the components with the measured variables afterwards. This is typical for unsupervised learning: the method finds the major patterns in the data, and it is up to us to interpret them.

The loadings, the entries of the ASV-specific vectors $\mathbf{u}^{(i)}$, tell us which ASVs drive a component. Below, we list the ASVs with the largest positive and negative loadings on the first component, together with their taxonomy.

```{code-cell}
loadings = pd.DataFrame(U[:, :3], index=top, columns=['PC1', 'PC2', 'PC3'])
loadings = loadings.join(taxa[['Class', 'Order', 'Genus']])
print("ASVs most abundant in salty water (largest positive loadings on PC1):")
print(loadings.sort_values('PC1', ascending=False).head(5)[['PC1', 'Class', 'Order', 'Genus']].round(3).to_string())
print("\nASVs most abundant in brackish water (largest negative loadings on PC1):")
print(loadings.sort_values('PC1').head(5)[['PC1', 'Class', 'Order', 'Genus']].round(3).to_string())
```

The ASVs with large positive loadings belong to typical marine bacteria, such as *Pelagibacter* of the SAR11 clade, the most abundant bacteria in the oceans, and *Planktomarina*. The ASVs with large negative loadings belong to the Acidimicrobiia and Nanopelagicales, groups that are common in fresh and brackish water. The list also contains an ASV of the SAR11 genus IMCC9063, which shows that different lineages of SAR11 prefer salty and brackish water.

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Calculate the outer product $\mathbf{u}\mathbf{v}^{\textsf{T}}$ of two small vectors by hand, and explain how PCA does the reverse by finding the vectors $\mathbf{u}$ and $\mathbf{v}$ for which $\mathbf{X} \approx \mathbf{u}\mathbf{v}^{\textsf{T}}$, in the sense that they minimize the sum of squared residuals. Explain why this is the same as maximizing the variance that the component describes, and how the residuals of PCA differ from those of regression.
2. Interpret $\mathbf{u}$ as gene-specific effects and $\mathbf{v}$ as sample-specific effects of a genes $\times$ samples matrix. Explain that $\mathbf{u}^{(1)}$ is, up to a scaling factor, the projection of $\mathbf{X}$ on $\mathbf{v}^{(1)}$, and vice versa.
3. Explain why the magnitude of a component can be moved freely between $\mathbf{u}$ and $\mathbf{v}$, and how scaling both to unit length, $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$, puts the whole magnitude in the singular value $S$.
4. Describe how the second and later principal components are found by decomposing the residual $\mathbf{X} - S_1\mathbf{u}^{(1)}\mathbf{v}^{(1)\textsf{T}}$, and state how many components a matrix can have at most.
5. Explain why the data are centered before PCA. Write down the singular value decomposition $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^{\textsf{T}}$, and identify which matrix holds the gene-specific effects, the sample-specific effects and the singular values. Calculate the fraction of variance explained by component $i$, $R^2_i = S^2_i / \sum_j S^2_j$, from a set of singular values.
6. Explain how PCA reduces the dimensionality of data, and interpret a scree plot of the explained variance per component to judge how many components to keep.
7. Explain how a sample is reconstructed approximately as a weighted sum of a few principal components, as when the eigenfaces example represents each 4,096-pixel face by only 20 numbers.
:::
