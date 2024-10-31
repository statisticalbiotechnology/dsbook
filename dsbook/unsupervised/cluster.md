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
# Unsupervised Machine Learning

## Introduction

Unsupervised machine learning aims to learn patterns from data without predefined labels. Specifically, the goal is to learn a function $f(x)$ from the dataset $D = \{\mathbf{x}_i\}$ by optimizing an objective function $g(D, f)$, or by simply partitioning the dataset $D$. This chapter provides an overview of clustering methods, which are a core part of unsupervised machine learning.

## Clustering

Clustering is a technique used to partition data into distinct groups, where data points in the same group share similar characteristics. Clustering can be broadly categorized into **hard clustering** and **soft (fuzzy) clustering**.

### Hard Clustering
In hard clustering, each data point is assigned to a single cluster. This means that each data point belongs exclusively to one group.

### Soft (Fuzzy) Clustering
In soft clustering, a data point can belong to multiple clusters with different probabilities. This allows for a more nuanced assignment, where data points can have partial membership across clusters.

## k-Means Clustering

One of the most common clustering algorithms is **k-Means**. It is a hard clustering algorithm that aims to partition the dataset into $k$ clusters by iteratively refining the cluster assignments.

### Algorithm Overview
- **Cluster Centers**: The cluster center $\mathbf{m}_i$ is defined as the arithmetic mean of all the data points belonging to the cluster.
- **Distance Metric**: Typically, the distance metric used is Euclidean distance, defined as $||\mathbf{x}_p - \mathbf{m}_i||^2$.
- **Cluster Assignment**: Each data point is assigned to the nearest cluster center:
  
  $$S_i^{(t)} = \{\mathbf{x}_p: ||\mathbf{x}_p - \mathbf{m}_i^{(t)}||^2 \le ||\mathbf{x}_p - \mathbf{m}_j^{(t)}||^2 \; \forall j \}$$
  
- **Recalculation of Cluster Centers**: The cluster centers are recalculated based on the new assignments:
  
  $$\mathbf{m}_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{\mathbf{x}_j \in S_i^{(t)}} \mathbf{x}_j$$

These steps are repeated until the cluster centers converge.

### Illustrations of k-Means

To better understand how the k-Means algorithm works, let's consider the following visualization steps:

```{code-cell} ipython3
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50)
plt.show()

# Function to perform one iteration of the k-Means EM step
def plot_kmeans_step(X, centers, step_title):
    labels = pairwise_distances_argmin(X, centers)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='plasma')
    sns.scatterplot(x=centers[:, 0], y=centers[:, 1], color='black', s=200, alpha=0.5)
    plt.title(step_title)
    plt.show()
    return labels

# Step 1: Random initialization of cluster centers
rng = np.random.RandomState(1)
i = rng.permutation(X.shape[0])[:5]
centers = X[i]
plot_kmeans_step(X, centers, "Initial Random Cluster Centers")

# Step 2: First E-Step (assign points to the nearest cluster center)
labels = plot_kmeans_step(X, centers, "First E-Step: Assign Points to Nearest Cluster")

# Step 3: First M-Step (recalculate cluster centers)
centers = np.array([X[labels == i].mean(0) for i in range(5)])
plot_kmeans_step(X, centers, "First M-Step: Recalculate Cluster Centers")

# Step 4: Second E-Step (assign points to the nearest cluster center)
labels = plot_kmeans_step(X, centers, "Second E-Step: Assign Points to Nearest Cluster")

# Step 5: Second M-Step (recalculate cluster centers)
centers = np.array([X[labels == i].mean(0) for i in range(5)])
plot_kmeans_step(X, centers, "Second M-Step: Recalculate Cluster Centers")

# Step 6: Second E-Step (assign points to the nearest cluster center)
labels = plot_kmeans_step(X, centers, "Third E-Step: Assign Points to Nearest Cluster")

# Step 7: Second M-Step (recalculate cluster centers)
centers = np.array([X[labels == i].mean(0) for i in range(5)])
plot_kmeans_step(X, centers, "Third M-Step: Recalculate Cluster Centers")
```

The algorithm automatically assigns the points to clusters, and we can see that it closely matches what we would expect by visual inspection.

### Drawbacks of k-Means

The k-Means algorithm is simple and effective, but it has some drawbacks:

1. **Initialization Sensitivity**: The algorithm's result can be highly dependent on the initial choice of cluster centers. Poor initialization may lead to suboptimal clustering, as k-Means may converge to a local minimum. For example, using a different random seed can lead to different cluster assignments.
2. **Predefined Number of Clusters**: k-Means requires specifying the number of clusters beforehand. For example, if we set the number of clusters to three instead of five, the algorithm will still proceed, but the result may not be meaningful.

```{code-cell}ipython3
from sklearn.cluster import KMeans

labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='plasma');
plt.show()
```

3. **Linear Cluster Boundaries**: The k-Means algorithm assumes that clusters are spherical and separated by linear boundaries. It struggles with complex geometries. Consider the following dataset with two crescent-shaped clusters:

```{code-cell}ipython3
from sklearn.datasets import make_moons

X, y = make_moons(200, noise=.05, random_state=42)
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='plasma');
plt.show()
```

4. **Differences in size**: K-Means assumes that the cluster sizes are fairly similar for all clusters

```{code-cell}ipython3
# Parameters for the blobs
n_samples = [600, 100, 100]  
centers = [(0, 0), (3, 3), (-3, 3)]  # Center coordinates
cluster_std = [2., 0.5, 0.5]  # Standard deviations for each blob

# Generate blobs
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=1)

labels = KMeans(3, random_state=0).fit_predict(X)
# Define markers for the original labels
markers = ['o', 's', 'D']  # Circle, square, and diamond markers for each blob

# Plot with original labels using different marker types
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=[i]*len(X[y == i]), 
                label=f"Cluster {i+1} (Original)", marker=markers[i], edgecolor='k', s=50, cmap='plasma')

# Overlay KMeans labels as color-coded dots without specific marker shapes
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='plasma', alpha=0.4)
plt.title("Generated Blobs with Original Labels and KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
```

## Gaussian Mixture Models (GMM)

**Gaussian Mixture Models (GMMs)** provide a probabilistic approach to clustering and are an example of soft clustering. GMMs assume that the data is generated from a mixture of several Gaussian distributions, each representing a cluster.

### Illustrations of GMM

To understand GMM better, let's consider the following visualizations:

1. **Data Generation**: Generate a dataset with distinct clusters.

```{code-cell}ipython3
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # flip axes for better plotting
plt.scatter(X[:, 0], X[:, 1], s=40)
plt.title("Generated Data with Four Clusters")
plt.show()
```

2. **k-Means Limitations**: Apply k-Means and visualize its limitations.

```{code-cell}ipython3
from sklearn.cluster import KMeans

kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='plasma');
plt.title("k-Means Clustering with Limitations")
plt.show()
```

The k-Means algorithm assigns hard cluster labels, with no intrinsic measure of uncertainty or probabilistic membership.

3. **GMM Clustering**: Apply GMM and visualize the probabilistic assignment of clusters.

```{code-cell}ipython3
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='plasma');
plt.title("Gaussian Mixture Model Clustering")
plt.show()
```

Unlike k-Means, GMM provides a soft clustering where each point is assigned a probability of belonging to each cluster.

4. **Probabilistic Cluster Assignment**: Visualize the uncertainty in cluster assignment.

```{code-cell}ipython3
probs = gmm.predict_proba(X)
size = 50 * probs.max(1) ** 2  # emphasize differences in certainty
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='plasma');
plt.title("GMM Probabilistic Cluster Assignment")
plt.show()
```

Points near the cluster boundaries have lower certainty, reflected in smaller marker sizes.

5. **Flexible Cluster Shapes**: GMM can model elliptical clusters, unlike k-Means, which assumes spherical clusters.

```{code-cell}ipython3
from matplotlib.patches import Ellipse
import numpy as np

def draw_ellipse(position, covariance, ax=None, **kwargs):
  """Draw an ellipse with a given position and covariance"""
  ax = ax or plt.gca()
  if covariance.shape == (2, 2):
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
  else:
    angle = 0
    width, height = 2 * np.sqrt(covariance)
  for nsig in range(1, 4):
    ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  if label:
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='plasma', zorder=2)
  else:
    ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
  ax.axis('equal')
  w_factor = 0.2 / gmm.weights_.max()
  for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X)
plt.title("GMM with Elliptical Cluster Boundaries")
plt.show()
```

GMM is able to model more complex, elliptical cluster boundaries, addressing one of the main limitations of k-Means.

## Expectation-Maximization (EM) Algorithm
The **Expectation-Maximization (EM)** algorithm is a statistical technique used for finding maximum likelihood estimates in models with latent variables, such as GMMs. The EM algorithm consists of two main steps:

- **Expectation (E) Step**: This step calculates the expected value of the latent variables given the current parameter estimates and the data.
- **Maximization (M) Step**: In this step, the parameters are updated by maximizing the expected likelihood found in the E step.

The E and M steps are repeated until the algorithm converges, usually when the change in the log-likelihood is below a certain threshold.

## Multivariate Normal Distribution

A **multivariate normal distribution**, also known as a **multinormal distribution**, is a generalization of the one-dimensional normal distribution to multiple dimensions.

### Definition
A multivariate normal distribution is characterized by a vector of mean values ($\boldsymbol{\mu}$) and a covariance matrix ($\boldsymbol{\Sigma}$). It describes the joint distribution of a set of random variables, each of which has a univariate normal distribution, but with the potential for covariance between them, allowing for dependencies. Mathematically, if $\mathbf{X}$ is a random vector $(X_1, X_2, \dots, X_n)$ following a multivariate normal distribution, its probability density function is given by:

$$\mathcal{N}( \mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^k |\boldsymbol{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where:
- $\mathbf{x}$ is a real $k$-dimensional vector,
- $\boldsymbol{\mu}$ is the mean vector,
- $\boldsymbol{\Sigma}$ is the $k \times k$ covariance matrix,
- $k$ is the number of dimensions (variables),
- $|\boldsymbol{\Sigma}|$ is the determinant of the covariance matrix.

The multivariate normal distribution is fundamental to Gaussian Mixture Models and provides a natural way to model the clusters in high-dimensional data.

## Comparison between k-Means and GMM

The following table highlights the similarities and differences between the k-Means and GMM algorithms in terms of their iteration steps:

| Step           | k-Means                                                                                      | GMM                                                                                                     |
|----------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Initialization** | Select $K$ cluster centers $(\mathbf{m}_1^{(1)}, \ldots, \mathbf{m}_K^{(1)})$               | $K$ components with means $\mu_k$, covariance $\Sigma_k$, and mixing coefficients $P_k$                |
| **E-Step**     | Allocate data points to clusters:                                                           | Update the probability that component $k$ generated data point $\mathbf{x}_n$:                          |
|                | $S_i^{(t)} = \{\mathbf{x}_p: ||\mathbf{x}_p - \mathbf{m}_i^{(t)}||^2 \le ||\mathbf{x}_p - \mathbf{m}_j^{(t)}||^2 \; \forall j \}$ | $\gamma_{nk} = \frac{P_k \mathcal{N}(\mathbf{x}_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K P_j \mathcal{N}(\mathbf{x}_n | \mu_j, \Sigma_j)}$ |
| **M-Step**     | Re-estimate cluster centers:                                                                | Calculate estimated number of cluster members $N_k$, means $\mu_k$, covariance $\Sigma_k$, and mixing coefficients $P_k$: |
|                | $\mathbf{m}_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{\mathbf{x}_j \in S_i^{(t)}} \mathbf{x}_j$ | $N_k = \sum_{n=1}^N \gamma_{nk}$, $\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} \mathbf{x}_n$, $\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (\mathbf{x}_n - \mu_k^{\text{new}})(\mathbf{x}_n - \mu_k^{\text{new}})^T$, $P_k^{\text{new}} = \frac{N_k}{N}$ |
| **Stopping Criterion** | Stop when there are no changes in cluster assignments.                                   | Stop when the log-likelihood does not increase significantly:                                           |
|                |                                                                                             | $\ln \Pr(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}, \mathbf{P}) = \sum_{n=1}^N \ln \left( \sum_{k=1}^K P_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$ |
