---
file_format: mystnb
kernelspec:
  name: python3
---

# Validation

## The Importance of Independent Validation Sets in Machine Learning

In machine learning, the use of independent validation sets is crucial for evaluating model performance and avoiding one of the most common pitfalls: overfitting. Overfitting occurs when a model is trained to perform exceptionally well on the training data but fails to generalize to new, unseen data. This typically happens when the model is excessively complex for the given task or when it memorizes the noise and peculiarities of the training dataset instead of learning the underlying patterns.

An independent validation set helps to detect overfitting by providing a dataset that is separate from the training set, allowing for an unbiased evaluation of the model’s performance. To better understand the concept of overfitting, let’s consider an illustrative Python example using synthetic data.

### Example of Overfitting

Below, we create a small dataset of 10 data points, and we compare the performance of two polynomial regression models—a simple linear model and an overfitted higher-order polynomial model.

```{code-cell}ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating synthetic data
np.random.seed(42)
x = np.linspace(-3, 3, 10).reshape(-1, 1)
y = 0.5 * x**2 + np.random.normal(0, 0.5, x.shape)

# Creating a linear regression model and a higher-order polynomial regression model
linear_model = LinearRegression()
quad_model = LinearRegression()
poly_model = LinearRegression()
quadratic_features = PolynomialFeatures(degree=2)
polynomial_features = PolynomialFeatures(degree=8)

# Fitting the models
linear_model.fit(x, y)
x_quad = quadratic_features.fit_transform(x)
quad_model.fit(x_quad, y)
x_poly = polynomial_features.fit_transform(x)
poly_model.fit(x_poly, y)

# Predicting using the fitted models
x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_linear_pred = linear_model.predict(x_test)
y_quad_pred = quad_model.predict(quadratic_features.transform(x_test))
y_poly_pred = poly_model.predict(polynomial_features.transform(x_test))

# Plotting the data
plt.scatter(x, y, color='black', label='Data')
plt.plot(x_test, y_linear_pred, label='Linear Model (Underfitting)', color='blue')
plt.plot(x_test, y_quad_pred, label='Quadratic Model', color='green')
plt.plot(x_test, y_poly_pred, label='High-Degree Model (Overfitting)', color='red')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

In the above code, we create a simple quadratic dataset, and fit both a linear regression model, a second degree polynomial model and an eighth-degree polynomial model. 
The linear model apears to miss an important trend in the data and is hence underfitting to the data.
The eighth-degree model fits the training data almost perfectly, capturing all of the data points and the noise, but it is unlikely to perform well on new data—an example of overfitting.

### Cross Validation to the Rescue

To evaluate how well a model will perform on new, unseen data, it’s important to use validation techniques such as cross validation. Cross validation is a strategy that partitions the available data into multiple subsets, or folds, allowing the model to be trained on one subset and validated on another. This ensures that each data point is eventually used both for training and for validation. The method allows us to assess how well a model generalizes to unseen data, providing an estimate of the expected accuracy on new samples -- stil using all available data.

In **k-fold cross validation**, the dataset is split into `k` equally sized folds. The model is trained `k` times, each time using a different fold as the validation set and the remaining folds for training. The average of the validation metrics across all folds gives a more robust estimate of the model's performance compared to using a single validation set.

### Illustration of Three-Fold Cross Validation

Imagine we have a dataset with **six data points**, represented as `A`, `B`, `C`, `D`, `E`, `F`. In three-fold cross validation, we can split this dataset into three subsets, each containing **two data points**. For example:

- **Fold 1**: Training on `[C, D, E, F]`, Validation on `[A, B]`
- **Fold 2**: Training on `[A, B, E, F]`, Validation on `[C, D]`
- **Fold 3**: Training on `[A, B, C, D]`, Validation on `[E, F]`

Each fold is used for validation once, and the model is trained on the remaining data. This way, every data point contributes to both training and evaluation, reducing the risk of overfitting and improving the robustness of the performance estimate.

Here is a code example illustrating three-fold cross validation with six data points:

```{code-cell}ipython3
from sklearn.model_selection import KFold

# Example dataset with six data points
x_example = np.array([[1], [2], [3], [4], [5], [6]])
y_example = np.array([1, 4, 9, 16, 25, 36])

# Setting up three-fold cross validation
kf_example = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf_example.split(x_example)):
    print(f"Fold {fold + 1}: Training on {x_example[train_index].flatten()}, Validation on {x_example[test_index].flatten()}")
```

This code demonstrates how the dataset is split into three folds, with each fold being used for validation once while the remaining data points are used for training. This helps ensure that all data points are used for both training and validation, providing a comprehensive evaluation of the model's performance.

```{mermaid}
flowchart TB
    subgraph Dataset [Dataset]
      A["    Fold 1    "]
      B["    Fold 2    "]
      C["    Fold 3    "]
    end

    subgraph Mod1 [Model 1]
        T1[Train Model] --> M1[[Model 1]]
        M1 --> V1[Validate Model on unseen data]
    end
    subgraph Mod2 [Model 2]
        T2[Train Model] --> M2[[Model 2]]
        M2 --> V2[Validate Model on unseen data]
    end
    subgraph Mod3 [Model 3]
        T3[Train Model] --> M3[[Model 3]]
        M3 --> V3[Validate Model on unseen data]
    end

    %% Fold 1
    A --> T1
    B --> T1
    C -. Test .-> V1

    %% Fold 2
    A --> T2
    B -. Test .-> V2
    C --> T2

    %% Fold 3
    A -. Test .-> V3
    B --> T3
    C --> T3
```

### Cross Validation to detect overfitting

In the following code, we demonstrate how to use **scikit-learn's KFold cross validation** with three-fold cross validation for a simple linear regression model:

```{code-cell}ipython3
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

# Creating a pipeline for polynomial regression with degree 2 and 8
pipeline_2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
pipeline_8 = make_pipeline(PolynomialFeatures(degree=8), LinearRegression())

# Setting up three-fold cross validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
scores_2, scores_8 = [], []

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Fit and test second degree polynomia
    pipeline_2.fit(x_train, y_train)
    y_pred = pipeline_2.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    scores_2.append(mse)
    # Fit and test eith degree polynomia
    pipeline_8.fit(x_train, y_train)
    y_pred = pipeline_8.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    scores_8.append(mse)

# Calculating and printing the mean score
print(f"Mean cross-validated MSE, for 2nd degree: {np.mean(scores_2):.2f} and for 8th degree: {np.mean(scores_8):.2f}")
```

In this code, we use `KFold` to split the dataset into three folds. The `for` loop iterates through each fold, training the model on the training set and evaluating it on the test set. The mean squared error (MSE) is calculated for each fold and averaged to give a more robust measure of model performance. By splitting the data into different training and validation sets multiple times, cross validation helps to detect overfitting and provides a more reliable measure of model generalizability.
