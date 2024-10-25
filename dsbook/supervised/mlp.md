---
file_format: mystnb
kernelspec:
  name: python3
---
# Multi-Layer Perceptrons (MLPs)

A **Multi-Layer Perceptron (MLP)** is a class of feedforward artificial neural network. An MLP consists of at least three layers: an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next, and the network is trained using backpropagation to adjust the weights based on the error.

MLPs are widely used in supervised learning tasks such as classification and regression. They can model complex, non-linear relationships between input and output by stacking multiple layers of neurons.

## Artificial Neuron

An **artificial neuron** is a fundamental building block of neural networks. It is inspired by the biological neuron and functions as a mathematical model that takes multiple inputs, processes them, and produces an output. The artifical neuron makes a linear combination of its input that is forwarded to a non-linear activation function. This can be expressed as:

$$ f(\mathbf{x};\mathbf{w}) = g(\sum_{i=1}^n w_i x_i + b) $$

This expression can be broken down into these two steps:

**Linear combination**: Each input is multiplied by a weight, and then all the weighted inputs are summed together. Additionally, a bias term is added to the sum. Mathematically, this can be represented as:

$$ z = \sum_{i=1}^n w_i x_i + b $$
   
Where:
   - $x_i$ are the inputs
   - $w_i$ are the weights
   - $b$ is the bias term

2. **Activation Function**: The weighted sum is passed through an **activation function**, $g(x)$, to introduce non-linearity. Common activation functions include:

   - **Sigmoid**: Maps the input to a value between 0 and 1.

   - **ReLU (Rectified Linear Unit)**: Sets all negative values to zero and passes positive values unchanged.

   - **Tanh**: Maps the input to a value between -1 and 1.

## Components of an MLP

The output of the activation function becomes the output of the neuron, which can then be used as input to other neurons.

```{figure} ./img/mlp.svg
:name: fig_mlp

An example MLP. Each layer of the MLP consists of a set of artificial neurons, that receive input from the previous layer, process it, and pass the output to the next layer.  
```

1. **Input Layer**: The input layer consists of neurons that receive the raw data. Each neuron in the input layer represents a feature of the input data.

2. **Hidden Layers**: Between the input and output layers are one or more **hidden layers**. Neurons in the hidden layers receive input from the previous layer, process it, and pass the output to the next layer. Adding more hidden layers allows the network to learn more complex relationships in the data.

3. **Output Layer**: The output layer produces the final prediction or classification. The number of neurons in this layer depends on the type of task (e.g., a single neuron for binary classification or multiple neurons for multi-class classification).

Neurons in each layer are **fully connected** to neurons in the subsequent layer, meaning each neuron receives input from all neurons in the previous layer. The strength of these connections is determined by the **weights**, which are adjusted during training to minimize the error between the predicted output and the actual output.

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
