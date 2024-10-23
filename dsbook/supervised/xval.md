---
file_format: mystnb
kernelspec:
  name: python3
---

# Cross Validation

Cross validation is an essential technique in machine learning that ensures the reliability of a model's performance. It allows us to assess how well a model generalizes to unseen data, providing an estimate of the expected accuracy on new samples. This is critical because machine learning models are often at risk of overfitting—that is, learning the peculiarities of the training dataset rather than the underlying pattern. Cross validation mitigates this by repeatedly testing the model on different segments of the data, giving a more realistic picture of its ability to generalize.

The most common form of cross validation is *k-fold cross validation*. In k-fold cross validation, the data is split into *k* equally sized folds, or subsets. The model is trained on *k – 1* of these folds and tested on the remaining one, and this process is repeated *k* times—each time using a different fold as the test set. The final evaluation metric, such as accuracy or mean squared error, is then computed as the average across all *k* trials. This approach ensures that every observation in the dataset is used both for training and for validation, minimizing the chances of bias.

```{code-cell} python
:caption: Illustration of 3-fold cross validation
:label: 3-fold-xval
:tags: [hide-input]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Data for illustration
n_samples = 12
k_folds = 3
fold_size = n_samples // k_folds

# Create the illustration similar to the attached diagram
for i in range(k_folds):
    # Training subsets (in green)
    for j in range(k_folds):
        if j != i:
            rect = patches.Rectangle((j * 1.5, k_folds - i - 1), 1, 1, linewidth=1, edgecolor='black', facecolor='#66b3ff')
            ax.add_patch(rect)
            ax.text(j * 1.5 + 0.5, k_folds - i - 0.5, f'Train {j + 1}', ha='center', va='center', fontsize=10, color='black')
    
    # Validation subset (in orange)
    rect = patches.Rectangle((i * 1.5, k_folds - i - 1), 1, 1, linewidth=1, edgecolor='black', facecolor='#ff9999')
    ax.add_patch(rect)
    ax.text(i * 1.5 + 0.5, k_folds - i - 0.5, f'Validate {i + 1}', ha='center', va='center', fontsize=10, color='black')

# Adding labels and styling
ax.set_xticks(np.arange(0, n_samples * 1.5, 1.5))
ax.set_xticklabels([f'S{i+1}' for i in range(n_samples)], rotation=90)
ax.set_xlim(-0.5, n_samples * 1.5 - 0.5)
ax.set_ylim(-0.5, k_folds)
ax.invert_yaxis()
ax.set_xlabel('Data Samples')
ax.set_title('3-Fold Cross Validation with Separate Models for Each Fold')

plt.tight_layout()
plt.show()
```

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


Another popular approach is *leave-one-out cross validation (LOOCV)*. In LOOCV, the number of folds is equal to the number of samples in the dataset, meaning that each sample is used once as a validation set while the remaining samples are used for training. LOOCV is computationally expensive for large datasets but provides a very thorough validation, particularly useful when data is scarce.

Choosing the right form of cross validation depends on the nature of the problem and the available computational resources. A higher value for *k* can lead to a more accurate estimate of performance but comes at the cost of increased computation. Typically, values like *k = 5* or *k = 10* offer a good balance between accuracy and efficiency.

Cross validation is especially useful when fine-tuning hyperparameters or selecting between different models. By using a consistent cross validation strategy, one can confidently compare models and select the one with the best generalization capabilities. This makes cross validation a fundamental building block in the pipeline of designing, training, and evaluating robust machine learning models.