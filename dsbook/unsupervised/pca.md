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

Principal Component Analysis (PCA) is a powerful unsupervised learning method used for finding the axes of maximum variation in your data and projecting the data into new coordinate systems. In this chapter, we'll explore how PCA works, its mathematical foundations.


## Decomposition of a matrix

- From linear algebra, we know that we can multiply two vectors into a matrix. $$ \mathbf{X} = \mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^{\textsf{T}} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix} \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix} = \begin{bmatrix} u_1 v_1 & u_1 v_2 & u_1 v_3 \\ u_2 v_1 & u_2 v_2 & u_2 v_3 \\ u_3 v_1 & u_3 v_2 & u_3 v_3 \\ u_4 v_1 & u_4 v_2 & u_4 v_3 \end{bmatrix}. $$

- What if we could do the opposite? I.e., given a matrix $\mathbf{X}$, what would be the vectors $\mathbf{u}$ and $\mathbf{v}$ that best represent $\mathbf{X}$ so that $\mathbf{X} \approx \mathbf{u} \mathbf{v}^{\textsf{T}}$? This is, in essence, what you do with principal component analysis (PCA).


## More principal components to your PCA

- Once you remove the principal components from a matrix $\mathbf{X}$, the remaining residues, i.e., $\mathbf{X - u^{(1)} v^{(1)T}}$, might in turn be decomposed into vectors. We can calculate the vectors $\mathbf{u^{(2)}}$ and $\mathbf{v^{(2)}}$ that best describe the matrix $\mathbf{X - u^{(1)} v^{(1)T}}$. These are called the second principal components, while the original ones are called the first principal components.

- In this manner, we can derive as many principal components as there are rows or columns (whichever is smaller) in $\mathbf{X}$. In most applications, we settle for two such components.

## An illustration of PCA

|           |          | Sample     |           |          |       |       | $\mathbf{u}^{(1)}$ |         |
|-----------|----------|------------|-----------|----------|-------|-------|----------------|---------|
| Gene $1$  | $X_{11}$ | $X_{12}$   | $X_{13}$  | $\ldots$ | $X_{1M}$ |       | $u^{(1)}_1$ |         |
| Gene $2$  | $X_{21}$ | $X_{22}$   | $X_{23}$  | $\ldots$ | $X_{2M}$ |       | $u^{(1)}_2$ |         |
| Gene $3$  | $X_{31}$ | $X_{32}$   | $X_{33}$  | $\ldots$ | $X_{3M}$ |       | $u^{(1)}_3$ |         |
| $\vdots$ | $\vdots$ | $\vdots$  | $\vdots$ | $\ddots$ | $\vdots$ |       | $\vdots$  |         |
| Gene $N$  | $X_{N1}$ | $X_{N2}$   | $X_{N3}$  | $\ldots$ | $X_{NM}$ |       | $u^{(1)}_N$ |         |
|           |          |            |           |          |       |       |                |         |
| Eigengene $\mathbf{v}^{T(1)}$ | $v^{(1)}_1$ | $v^{(1)}_2$ | $v^{(1)}_3$ | $\ldots$ | $v^{(1)}_M$ |  | $S_1$  |

more...

|           |          |            |           |          |          |       | $\mathbf{u}^{(1)}$ | $\mathbf{u}^{(2)}$ |
|-----------|----------|------------|-----------|----------|----------|-------|--------------------|----------------|
| Gene $1$  | $X_{11}$ | $X_{12}$   | $X_{13}$  | $\ldots$ | $X_{1M}$ |       | $u^{(1)}_1$ | $u^{(2)}_1$ |
| Gene $2$  | $X_{21}$ | $X_{22}$   | $X_{23}$  | $\ldots$ | $X_{2M}$ |       | $u^{(1)}_2$ | $u^{(2)}_2$ |
| Gene $3$  | $X_{31}$ | $X_{32}$   | $X_{33}$  | $\ldots$ | $X_{3M}$ |       | $u^{(1)}_3$ | $u^{(2)}_3$ |
| $\vdots$ | $\vdots$ | $\vdots$  | $\vdots$ | $\ddots$ | $\vdots$ |       | $\vdots$  | $\vdots$  |
| Gene $N$  | $X_{N1}$ | $X_{N2}$   | $X_{N3}$  | $\ldots$ | $X_{NM}$ |       | $u^{(1)}_N$ | $u^{(2)}_N$ |
|           |          |            |           |          |          |       |             |         |
| Eigengene $\mathbf{v}^{T(1)}$ | $v^{(1)}_1$ | $v^{(1)}_2$ | $v^{(1)}_3$ | $\ldots$ | $v^{(1)}_M$ |  | $S_1$ |
| Eigengene $\mathbf{v}^{T(2)}$ | $v^{(2)}_1$ | $v^{(2)}_2$ | $v^{(2)}_3$ | $\ldots$ | $v^{(2)}_M$ |  |      $S_2$ |



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
