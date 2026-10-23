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
---

# Introduction to Machine Learning

Machine learning (ML) is a branch of artificial intelligence (AI) that enables computers to learn from data without being explicitly programmed. In biotechnology, ML is applied to analyze large datasets, find patterns, and make predictions. Below, we introduce different types of ML approaches with examples relevant to biotechnology. Later chapters cover these concepts in more detail.

## Supervised Learning

In supervised learning, we are given a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i$ represents feature vectors (such as gene expressions or protein levels), and $y_i$ is the target, which could be a class label (e.g., disease vs. no disease) or a continuous value in regression tasks (e.g., protein concentration). The objective is to learn a function $f(\mathbf{x})$ that approximates the mapping from $\mathbf{x}_i$ to $y_i$, enabling accurate predictions on new, unseen data. This is frequently done by minimization of a loss function. Examples of supervised learning include classification models such as support vector machines and regression models like linear regression.

## Unsupervised Learning

In unsupervised learning, we are given a dataset $\{\mathbf{x}_i\}_{i=1}^N$, where no labels $y_i$ are provided. The goal is to find underlying structure in the data, such as grouping similar data points into clusters or reducing the dimensionality of the data for visualization and analysis. A typical example is clustering, where algorithms like k-means and Gaussian mixture models are applied to categorize data into meaningful groups based solely on $\mathbf{x}_i$.

### Self-Supervised Learning

Self-supervised learning is an important special case of unsupervised learning. No labels are given, but the supervisory signal is generated automatically from the input data itself rather than from human-provided labels, for example by hiding part of each input, such as some residues of a protein sequence, and training the model to predict the hidden part from the rest.

## Semi-Supervised Learning

Semi-supervised learning combines labeled and unlabeled data. We are provided with a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^M$, and a (typically larger) set of unlabeled data $\{\mathbf{x}_i\}_{i=M+1}^N$. The goal is to improve the prediction of the function $f(\mathbf{x})$ by using the structure of the unlabeled data to assist in training the model, particularly when labeled data is scarce but unlabeled data is abundant.

## Generative vs. Discriminative Models

- **Generative Models:** Generative models aim to capture the distribution of the data itself: the joint distribution $P(\mathbf{x}, y)$ when labels are modeled as well, or simply $P(\mathbf{x})$ in the unlabeled case. Because they model how the data were generated, they can be used to draw new samples resembling those in the training set. An example is the Gaussian mixture model, which we will meet in the chapter on clustering; it models $P(\mathbf{x})$ as a weighted sum of normal distributions. In biotechnology, generative models can be applied to simulate potential experimental outcomes or explore synthetic biology designs.

- **Discriminative Models:** Discriminative models focus on modeling the conditional probability $P(y|\mathbf{x})$, directly learning the decision boundary between different classes in the data. Examples of discriminative models include logistic regression for classification and linear regression for continuous prediction. These models are commonly used in biological applications, such as classifying tumor subtypes or predicting gene expression levels based on input features.

:::{dropdown} What you should know after reading this chapter
You should be able to:
1. Contrast supervised, unsupervised and semi-supervised learning in terms of the data each one is given (labeled pairs $(\mathbf{x}_i, y_i)$, only feature vectors $\mathbf{x}_i$, or a mix of both) and what each one aims to learn, and explain why self-supervised learning is a special case of unsupervised learning.
2. Identify whether a supervised problem is classification or regression from its target $y_i$, which is either a class label, such as disease vs. no disease, or a continuous value, such as a protein concentration.
3. Contrast generative models, which model $P(\mathbf{x}, y)$ or $P(\mathbf{x})$ and can generate new samples, with discriminative models, which model $P(y|\mathbf{x})$ directly.
:::
