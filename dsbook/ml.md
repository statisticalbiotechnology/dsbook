# Introduction to Machine Learning

Machine learning (ML) is a branch of artificial intelligence (AI) that enables computers to learn from data without being explicitly programmed. In biotechnology, ML is applied to analyze large datasets, find patterns, and make predictions. Below, we introduce different types of ML approaches with examples relevant to biotechnology. During the course we will cover these concepts more in detail.

## Supervised Learning

In supervised learning, we are given a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i$ represents feature vectors (such as gene expressions or protein levels), and $y_i$ is the target, which could be a class label (e.g., disease vs. no disease) or a continuous value in regression tasks (e.g., protein concentration). The objective is to learn a function $f(\mathbf{x})$ that approximates the mapping from $\mathbf{x}_i$ to $y_i$, enabling accurate predictions on new, unseen data. This is frequently done by minimization of a loss function. Examples of supervised learning include classification models such as support vector machines and regression models like linear regression.

## Unsupervised Learning

In unsupervised learning, we are given a dataset $\{\mathbf{x}_i\}_{i=1}^N$, where no labels $y_i$ are provided. The goal is to find underlying structure in the data, such as grouping similar data points into clusters or reducing the dimensionality of the data for visualization and analysis. A typical example is clustering, where algorithms like k-means and hierarchical clustering are applied to categorize data into meaningful groups based solely on $\mathbf{x}_i$.

## Semi-Supervised Learning

Semi-supervised learning combines labeled and unlabeled data. We are provided with a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^M$ where $M < N$, and a larger set of unlabeled data $\{\mathbf{x}_i\}_{i=M+1}^N$. The goal is to improve the prediction of the function $f(\mathbf{x})$ by using the structure of the unlabeled data to assist in training the model, particularly when labeled data is scarce but unlabeled data is abundant.

## Machine Supervised Learning

In machine-supervised learning, systems are often trained or corrected by other automated systems rather than human supervisors. A popular choice in this context is the use of **autoencoders**, where for a given set $\{\mathbf{x}_i\}_{i=1}^N$, the system learns an encoder $e$ and a decoder $d$, trained such that the reconstruction approximates $\mathbf{x}_i \approx d(e(\mathbf{x}_i))$. Autoencoders are particularly useful for dimensionality reduction and feature extraction, often applied in biotechnology to reduce complex data (e.g., genomic or proteomic data) into more manageable representations.

## Generative vs. Discriminative Models

- **Generative Models:** Generative models aim to capture the joint probability distribution $P(\mathbf{x}, y)$ of the data. These models can generate new samples $\mathbf{x}$ given $y$ or vice versa, making them useful for simulating new data points similar to those in the training set. A notable example is the Variational Autoencoder (VAE), which is used to model high-dimensional data distributions, making it useful for generating realistic biological data such as gene expressions or protein structures. In biotechnology, generative models can be applied to simulate potential experimental outcomes or explore synthetic biology designs.

- **Discriminative Models:** Discriminative models focus on modeling the conditional probability $P(y|\mathbf{x})$, directly learning the decision boundary between different classes in the data. These models are concerned with how to best separate or distinguish between different categories or predict a continuous target based on features. Examples of discriminative models include logistic regression for classification and linear regression for continuous prediction. These models are commonly used in biological applications, such as classifying tumor subtypes or predicting gene expression levels based on input features.
