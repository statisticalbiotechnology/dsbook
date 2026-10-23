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

- Once the first component is removed from $\mathbf{X}$, the residuals $\mathbf{X} - S_1\mathbf{u}^{(1)} \mathbf{v}^{(1)\textsf{T}}$ can in turn be decomposed. The vectors $\mathbf{u}^{(2)}$ and $\mathbf{v}^{(2)}$ that best describe this residual matrix form the second principal component, while the original pair forms the first.

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

The principal components of this dataset are called eigenfaces. Each eigenface is a pattern over all 4,096 pixels that captures a direction of large variation between the faces (e.g., eyes, nose, or general facial structure). An eigenface plays the role of the gene-specific vector $\mathbf{u}$: it holds one value per pixel. The matching photo-specific scores play the role of the sample-specific vector $\mathbf{v}$: they say how much of each eigenface is present in each photo.

Projecting a face onto the 20 eigenfaces turns its 4,096 pixel values into 20 scores. The reconstruction (`inverse_transform`) reverses the process by adding up the eigenfaces, weighted by the face's scores. The reconstructed faces are blurred but still recognizable, although each of them is described by 20 numbers instead of 4,096. The last plot shows why they are not perfect: the first eigenface explains 18% of the variance and the 20 eigenfaces together about 70%, so the remaining 30% of the variation, the finer detail, is lost.

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Calculate the outer product $\mathbf{u}\mathbf{v}^{\textsf{T}}$ of two small vectors by hand, and explain how PCA does the reverse by finding the vectors $\mathbf{u}$ and $\mathbf{v}$ for which $\mathbf{X} \approx \mathbf{u}\mathbf{v}^{\textsf{T}}$.
2. Interpret $\mathbf{u}$ as gene-specific effects and $\mathbf{v}$ as sample-specific effects of a genes $\times$ samples matrix. Explain that $\mathbf{u}^{(1)}$ is, up to a scaling factor, the projection of $\mathbf{X}$ on $\mathbf{v}^{(1)}$, and vice versa.
3. Explain why the magnitude of a component can be moved freely between $\mathbf{u}$ and $\mathbf{v}$, and how scaling both to unit length, $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$, puts the whole magnitude in the singular value $S$.
4. Describe how the second and later principal components are found by decomposing the residual $\mathbf{X} - S_1\mathbf{u}^{(1)}\mathbf{v}^{(1)\textsf{T}}$, and state how many components a matrix can have at most.
5. Explain why the data are centered before PCA. Write down the singular value decomposition $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^{\textsf{T}}$, and identify which matrix holds the gene-specific effects, the sample-specific effects and the singular values. Calculate the fraction of variance explained by component $i$, $R^2_i = S^2_i / \sum_j S^2_j$, from a set of singular values.
6. Explain how PCA reduces the dimensionality of data, and interpret a scree plot of the explained variance per component to judge how many components to keep.
7. Explain how a sample is reconstructed approximately as a weighted sum of a few principal components, as when the eigenfaces example represents each 4,096-pixel face by only 20 numbers.
:::
