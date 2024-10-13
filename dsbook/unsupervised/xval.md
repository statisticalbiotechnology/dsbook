**Cross Validation in Machine Learning**
---
file_format: mystnb
kernelspec:
  name: python3
---


```{code-cell} python
:hidden:
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Data for illustration
n_samples = 15
k_folds = 3
fold_size = n_samples // k_folds

# Colors for each fold
colors = ['#ff9999','#66b3ff','#99ff99']

# Create the rectangles for folds
for i in range(k_folds):
    # Training data (in white)
    for j in range(k_folds):
        if j != i:
            rect = patches.Rectangle((j * fold_size, k_folds - i - 1), fold_size, 1, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)

    # Validation fold
    rect = patches.Rectangle((i * fold_size, k_folds - i - 1), fold_size, 1, linewidth=1, edgecolor='black', facecolor=colors[i])
    ax.add_patch(rect)

# Adding labels and styling
ax.set_yticks(np.arange(k_folds) + 0.5)
ax.set_yticklabels([f'Fold {i+1}' for i in range(k_folds)])
ax.set_xticks(np.arange(n_samples))
ax.set_xticklabels([f'S{i+1}' for i in range(n_samples)], rotation=90)
ax.set_xlim(0, n_samples)
ax.set_ylim(0, k_folds)
ax.invert_yaxis()
ax.set_xlabel('Data Samples')
ax.set_title('3-Fold Cross Validation')

plt.tight_layout()
plt.savefig("3_fold_cross_validation.svg", format='svg')
plt.show()
```

```{figure} 3_fold_cross_validation.svg
---
name: 3-fold-cross-validation
---
Illustration of 3-Fold Cross Validation
```

Cross validation is an essential technique in machine learning that ensures the reliability of a model's performance. It allows us to assess how well a model generalizes to unseen data, providing an estimate of the expected accuracy on new samples. This is critical because machine learning models are often at risk of overfitting—that is, learning the peculiarities of the training dataset rather than the underlying pattern. Cross validation mitigates this by repeatedly testing the model on different segments of the data, giving a more realistic picture of its ability to generalize.

The most common form of cross validation is *k-fold cross validation*. In k-fold cross validation, the data is split into *k* equally sized folds, or subsets. The model is trained on *k – 1* of these folds and tested on the remaining one, and this process is repeated *k* times—each time using a different fold as the test set. The final evaluation metric, such as accuracy or mean squared error, is then computed as the average across all *k* trials. This approach ensures that every observation in the dataset is used both for training and for validation, minimizing the chances of bias.

Another popular approach is *leave-one-out cross validation (LOOCV)*. In LOOCV, the number of folds is equal to the number of samples in the dataset, meaning that each sample is used once as a validation set while the remaining samples are used for training. LOOCV is computationally expensive for large datasets but provides a very thorough validation, particularly useful when data is scarce.

Choosing the right form of cross validation depends on the nature of the problem and the available computational resources. A higher value for *k* can lead to a more accurate estimate of performance but comes at the cost of increased computation. Typically, values like *k = 5* or *k = 10* offer a good balance between accuracy and efficiency.

Cross validation is especially useful when fine-tuning hyperparameters or selecting between different models. By using a consistent cross validation strategy, one can confidently compare models and select the one with the best generalization capabilities. This makes cross validation a fundamental building block in the pipeline of designing, training, and evaluating robust machine learning models.