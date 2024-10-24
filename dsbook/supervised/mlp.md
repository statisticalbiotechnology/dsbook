---
file_format: mystnb
kernelspec:
  name: python3
---
# Multi-Layer Perceptrons (MLPs)

A **Multi-Layer Perceptron (MLP)** is a class of feedforward artificial neural network. An MLP consists of at least three layers: an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next, and the network is trained using backpropagation to adjust the weights based on the error.

MLPs are widely used in supervised learning tasks such as classification and regression. They can model complex, non-linear relationships between input and output by stacking multiple layers of neurons.

## Components of an MLP

1. **Input Layer**: The input data is fed into this layer. The number of neurons in this layer corresponds to the number of features in the input data.
2. **Hidden Layers**: These are the intermediate layers where neurons apply weights to the input and pass them through an activation function (e.g., ReLU, Sigmoid). Adding more hidden layers allows the network to capture more complex patterns in the data.
3. **Output Layer**: This layer produces the final output of the network. In classification tasks, it typically uses a softmax or sigmoid function to output probabilities.

## Training an MLP

MLPs are trained using **backpropagation** and **gradient descent**. The process involves the following steps:

1. **Forward Pass**: Input data passes through the network, and predictions are generated.
2. **Loss Calculation**: A loss function (e.g., mean squared error for regression or cross-entropy for classification) is used to measure the error between the predictions and true labels.
3. **Backward Pass**: Gradients of the loss with respect to the weights are computed, and the weights are updated using optimization techniques like stochastic gradient descent (SGD).

## Example: Training an MLP on Artificial Data

In this example, we'll generate artificial data from two non-linearly separable distributions and train an MLP to classify them into two classes. We'll use the **`MLPClassifier`** from `sklearn` to simplify the process.

```{code-cell} ipython3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

sns.set_style("whitegrid")

# Generate non-linearly separable data
np.random.seed(0)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Predict the test set
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", s=60, edgecolor='k')
plt.title("MLP Decision Boundary on Non-Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

## Explanation

1. **Artificial Data Generation**: We generate data from two non-linearly separable distributions using the `make_moons` function, which creates two interleaving half-moon shapes representing the two classes.
2. **Training the MLP**: We split the data into training and test sets and train an MLP with two hidden layers (one with 10 neurons and one with 5 neurons).
3. **Evaluation**: After training, we evaluate the accuracy of the model on the test data and visualize the decision boundary of the MLP using a contour plot.

## Tuning the MLP

- **Hidden Layers**: The number and size of hidden layers affect the capacity of the MLP. More layers and neurons allow the model to capture more complex relationships but may lead to overfitting if not properly regularized.
- **Activation Function**: Common activation functions include ReLU, sigmoid, and tanh. The choice of activation function affects how the model learns non-linear patterns.
- **Learning Rate**: Adjusting the learning rate controls how quickly the model updates weights during training. A learning rate that's too high can lead to instability, while one that's too low can result in slow convergence.

## Conclusion

MLPs are powerful models for supervised learning tasks, particularly when dealing with non-linear data. They can model complex relationships by adding more hidden layers and neurons, but care must be taken to prevent overfitting. By training an MLP on artificial data, we've demonstrated how this model can learn to classify data points into two classes.

In the next chapter, we'll explore **regularization techniques** and how they can be used to improve the generalization of neural networks.

