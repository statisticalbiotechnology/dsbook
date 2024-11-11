---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a powerful unsupervised learning method used for finding the axes of maximum variation in your data and projecting the data into new coordinate systems. 

## Decomposition of a matrix

- From linear algebra, we know that we can multiply two vectors into a matrix. $$ \mathbf{X} = \mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^{\textsf{T}} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix} \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix} = \begin{bmatrix} u_1 v_1 & u_1 v_2 & u_1 v_3 \\ u_2 v_1 & u_2 v_2 & u_2 v_3 \\ u_3 v_1 & u_3 v_2 & u_3 v_3 \\ u_4 v_1 & u_4 v_2 & u_4 v_3 \end{bmatrix}. $$

- What if we could do the opposite? I.e., given a matrix $\mathbf{X}$, what would be the vectors $\mathbf{u}$ and $\mathbf{v}$ that best represent $\mathbf{X}$ so that $\mathbf{X} \approx \mathbf{u} \mathbf{v}^{\textsf{T}}$? This is, in essence, what you do with principal component analysis (PCA).


### A scenario where this might be useful

In PCA, we often deal with a matrix that encapsulates multiple sources of variation. Imagine a scenario where we’re studying gene expression levels across several samples. Here, the gene expression levels might be influenced by two key factors:

1. **Gene-Specific Effects**: These are inherent to each gene, such as its baseline expression level or potential for expression. For example, some genes may generally be expressed at high levels across all conditions, while others have lower baseline expression.
  
2. **Sample-Specific Effects**: These capture the influence of each sample (environmental condition, patient, disease state) on gene expression. For instance, one condition might increase expression levels across many genes, while another has a weaker or even suppressive effect.

By setting up the gene expression matrix $X$ as a product of two vectors:
```{maths}
X = u \cdot v^T
```

we express the matrix as an outer product of these two effects:
- $u$ represents the gene-specific effects.
- $v$ represents the sample-specific effects.

When we perform PCA on $ X $, the primary principal component should capture the main pattern driven by these combined effects, showing a dominant structure that reflects how genes and samples contribute to variation together. Secondary components may capture additional variations or deviations, if any, from this main trend.

### Fictive Example

Now, let’s apply this theory in a concrete, 4x4 example with 4 samples and 4 genes.

#### Setup
Let:
- $u$ be a vector representing the gene-specific effects: $ u = [2, 1, 0.5, 3] $.
- $v$ be a vector representing the sample-specific effects: $ v = [1, 0.8, 1.2, 0.5] $.

To build our matrix $ X $, we calculate the outer product $ X = u \cdot v^T $, where each entry $ X_{ij} = u_i \cdot v_j $.

#### Calculations
1. **Gene-Specific Effects** $ u = [2, 1, 0.5, 3] $
2. **Sample-Specific Effects** $ v = [1, 0.8, 1.2, 0.5] $

Using the outer product:

$$
X = \begin{bmatrix} 2 \\ 1 \\ 0.5 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 1 & 0.8 & 1.2 & 0.5 \end{bmatrix} = 
\begin{bmatrix} 
2 & 1.6 & 2.4 & 1 \\
1 & 0.8 & 1.2 & 0.5 \\
0.5 & 0.4 & 0.6 & 0.25 \\
3 & 2.4 & 3.6 & 1.5 
\end{bmatrix}
$$

### Scaling the components

When decomposing a matrix with PCA, we aim to find directions that capture the main sources of variation. For a matrix $ X $, where we assume a factorization as an outer product of two vectors $ u $ and $ v $, the decomposition can be represented as:

$$
X \approx S \cdot u \cdot v^T
$$

where $ S $ is a scalar that captures the magnitude, while $ u $ and $ v $ provide the directions.

In PCA, the directions $ u $ and $ v $ indicate the axes along which most of the data's variance lies. However, the magnitude of the variation (how much variance is explained) can technically be distributed between $ u $ and $ v $. This is because if we scale $ u $ by a factor $ \alpha $ and scale $ v $ by $ 1 / \alpha $, the product $ u \cdot v^T $ remains the same, allowing flexibility in how we allocate the magnitude.

To standardize this decomposition, it is common to select $ u $ and $ v $ to have a norm of 1. This normalization ensures that $ u $ and $ v $ only represent directions, while the entire magnitude of the variation is captured by the scalar $ s $. Thus, $ s $ becomes a single value that encapsulates the strength of the variation along the principal component, while $ u $ and $ v $ are pure directional vectors, each constrained to unit length:

$$
\| u \| = 1, \quad \| v \| = 1
$$

This approach makes the interpretation of the PCA decomposition more consistent, as $ s $ represents the variance captured by this component, and $ u $ and $ v $ describe the specific directions of variation in row and column space, respectively.

## More principal components to your PCA

- Once you remove the principal components from a matrix $\mathbf{X}$, the remaining residues, i.e., $\mathbf{X - S_1u^{(1)} v^{(1)T}}$, might in turn be decomposed into vectors. We can calculate the vectors $\mathbf{u^{(2)}}$ and $\mathbf{v^{(2)}}$ that best describe the matrix $\mathbf{X - S_1u^{(1)} v^{(1)T}}$. These are called the second principal components, while the original ones are called the first principal components.

- In this manner, we can derive as many principal components as there are rows or columns (whichever is smaller) in $\mathbf{X}$. In most applications, we settle for two such components.

## An illustration of PCA

|           |          |            |           |          |          |       | $\mathbf{u}^{(1)}$ | $\mathbf{u}^{(2)}$ |
|-----------|----------|------------|-----------|----------|----------|-------|--------------------|----------------|
| Gene $1$  | $X_{11}$ | $X_{12}$   | $X_{13}$  | $\ldots$ | $X_{1M}$ |       | $u^{(1)}_1$ | $u^{(2)}_1$ |
| Gene $2$  | $X_{21}$ | $X_{22}$   | $X_{23}$  | $\ldots$ | $X_{2M}$ |       | $u^{(1)}_2$ | $u^{(2)}_2$ |
| Gene $3$  | $X_{31}$ | $X_{32}$   | $X_{33}$  | $\ldots$ | $X_{3M}$ |       | $u^{(1)}_3$ | $u^{(2)}_3$ |
| $\vdots$ | $\vdots$ | $\vdots$  | $\vdots$ | $\ddots$ | $\vdots$ |       | $\vdots$  | $\vdots$  |
| Gene $N$  | $X_{N1}$ | $X_{N2}$   | $X_{N3}$  | $\ldots$ | $X_{NM}$ |       | $u^{(1)}_N$ | $u^{(2)}_N$ |
|           |          |            |           |          |          |       |             |         |
| Eigengene $\mathbf{v}^{T(1)}$ | $v^{(1)}_1$ | $v^{(1)}_2$ | $v^{(1)}_3$ | $\ldots$ | $v^{(1)}_M$ |  | $S_1$ |  |
| Eigengene $\mathbf{v}^{T(2)}$ | $v^{(2)}_1$ | $v^{(2)}_2$ | $v^{(2)}_3$ | $\ldots$ | $v^{(2)}_M$ |  |        | $S_2$ |


## Principal Components

PCA aims to identify the directions, or **principal components**, in the data that describe the most variance. These directions are essentially the new axes into which the original data is projected, and they help us gain insight into the underlying structure and patterns. The principal components are ordered such that the first principal component describes the largest possible variance, while each subsequent component describes as much of the remaining variance as possible, subject to being orthogonal to the preceding components.

One interesting property of principal components is that the components generated from a data matrix $X$ are equivalent to the projections of the principal components derived from the transposed matrix $X^T$. This means that we can often examine **principal component pairs**, which are generated from both rows and columns of the matrix. Each pair of principal components minimizes the squared sum of residuals, providing an optimal lower-dimensional representation of the data.

## Eigensamples and Eigengenes

When working with data matrices where rows represent genes and columns represent samples, PCA provides meaningful biological interpretations through **eigensamples** and **eigengenes**:

- **Eigensample**: This is a linear combination of genes that captures the maximum variation in the dataset. It can be thought of as the "most typical sample" in the data, behaving as a representative of the dominant patterns among samples.

- **Eigengene**: Conversely, this is a linear combination of samples that captures the most variance from the perspective of genes. It behaves like the "most typical gene" in the dataset, summarizing the dominant variation across different genes.

These eigensamples and eigengenes provide insights into the major biological signals in the dataset and are frequently used for data visualization, clustering, and downstream analyses.

## Dimensionality Reduction and Explained Variance

One of the key advantages of PCA is **dimensionality reduction**. By focusing only on the principal components that describe most of the variance in the data, PCA allows us to reduce the number of random variables under consideration. This is particularly useful when dealing with high-dimensional datasets, such as those encountered in genomics and transcriptomics, where the number of features (e.g., genes) can be overwhelming.

The contribution of each principal component to the overall variance is called the **explained variance**. The first principal component typically explains the most variance, followed by the second, and so on. By examining the explained variance, we can determine how many principal components are needed to capture most of the important variation in the data. For instance, if the first few principal components explain a large proportion of the total variance, then the dimensionality of the dataset can be substantially reduced without significant loss of information.

### PCA as Dimensionality Reduction

Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance.

Here is an example of using PCA as a dimensionality reduction transform:

```{code-cell} python
import numpy as np

# Define the column vector and row vector
column_vector = np.array([[1], [-2], [3]])  # 3x1 column vector
row_vector = np.array([[2, -1, 2]])         # 1x3 row vector

# Calculate the matrix as the product of the column vector and the row vector
matrix = np.dot(column_vector, row_vector)

# Create an augmented matrix with space (filled with NaNs) between the matrix and the column vector
space_column = np.full((matrix.shape[0], 1), np.nan)
augmented_matrix_right = np.hstack([matrix, space_column, column_vector])

# Create an augmented matrix with space (filled with NaNs) between the matrix and the row vector
space_row = np.full((1, matrix.shape[1]), np.nan)
augmented_matrix_below = np.vstack([matrix, space_row, row_vector])

# Print results in the same printout
print("Augmented matrix with column vector on the right and row vector below:\n")
for row in augmented_matrix_right:
    for value in row:
        if np.isnan(value):
            print(" ", end="\t")
        else:
            print(f"{int(value)}", end="\t")
    print()
print()
for column in range(row_vector.shape[1]):
    print(f"{int(row_vector[0][column])}", end="\t")
```

The light points are the original data, while the dark points are the projected version. This makes clear what a PCA dimensionality reduction means: the information along the least important principal axis or axes is removed, leaving only the component(s) of the data with the highest variance. The fraction of variance that is cut out (proportional to the spread of points about the line formed in this figure) is roughly a measure of how much "information" is discarded in this reduction of dimensionality.

This reduced-dimension dataset is in some senses "good enough" to encode the most important relationships between the points: despite reducing the dimension of the data by 50%, the overall relationship between the data points is mostly preserved.

## Singular Value Decomposition (SVD)

The calculation of principal components can be efficiently performed using **singular value decomposition (SVD)**, a fundamental linear algebra technique. Given a data matrix $X$, SVD decomposes it as follows:

\[
X = USV^T
\]

Here:

- $U$ is a matrix formed from the **eigensamples**.
- $V^T$ is a matrix formed from the **eigengenes**.
- $S$ is a diagonal matrix that carries the magnitude (singular values) of each principal component pair.

SVD provides a robust way to perform PCA, allowing for the identification of eigensamples and eigengenes while efficiently capturing the major patterns in the data. Since $S$ contains the singular values, it directly relates to the explained variance of the principal components, with larger singular values corresponding to more significant components.

## Affine Transformation and Interpretation

PCA is an **affine transformation**, meaning that it can involve translation, rotation, and uniform scaling of the original data. Importantly, PCA maintains the relationships between points, straight lines, and planes, which allows for a meaningful geometric interpretation of the results. By transforming the data into a new set of axes aligned with the directions of maximum variance, PCA enables us to discover the key structural relationships while preserving important properties.

## Applications of PCA

In data science, PCA is widely applied for visualization, clustering, and feature extraction. In biotechnology, PCA is often used to:

- Identify major sources of variation in gene expression data.
- Visualize sample relationships in a reduced-dimensional space.
- Detect batch effects and outliers in high-throughput experiments.

By applying PCA, researchers can distill complex, high-dimensional data into a manageable form, making it easier to interpret and analyze the underlying biological phenomena.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Original data (5 individuals with height in cm and height in inches)
data = np.array([[6, 5, -5, 0, 0],
                 [2, 2, -1, 0, 0]])

# Fit PCA
pca = PCA(n_components=2)
pca.fit(data.T)

# Transform data to principal components
transformed_data = pca.transform(data.T)

# Prepare for plotting
fig, ax = plt.subplots()

# Plot the original data points
ax.scatter(data[0, :], data[1, :], color='blue', label='Original Data', s=30)
for i in range(data.shape[1]):
    ax.text(data[0, i], data[1, i], f'Person {i+1}', fontsize=9, color='blue')

# Plot the principal components
origin = [0, 0]  # Origin point for arrows
pc1 = pca.components_[0] * pca.explained_variance_ratio_[0]
pc2 = pca.components_[1] * pca.explained_variance_ratio_[1]
ax.quiver(*origin, *pc1, scale=3, color='red', angles='xy', scale_units='xy', label='PC 1')
ax.quiver(*origin, *pc2, scale=3, color='green', angles='xy', scale_units='xy', label='PC 2')

# Set labels and title
ax.set_xlabel('Height in cm')
ax.set_ylabel('Height in inches')
ax.set_title('PCA Visualization with Principal Components')
ax.axhline(y=0, color='black', lw=0.5)
ax.axvline(x=0, color='black', lw=0.5)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_aspect('equal')
ax.legend()

# Show plot
plt.show()
```
