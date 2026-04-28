# Evaluation Strategy

## Overview

Choosing the right evaluation metrics is as important as choosing the right
model. In fraud detection, standard classification metrics such as accuracy
are not only uninformative — they are actively misleading. This document
explains the evaluation framework used in this project, from metric selection
to threshold optimization.

---

## Why Accuracy Is Useless

In a dataset where 99.48% of transactions are legitimate, a model that
predicts "not fraud" for every single transaction achieves:

- **Accuracy: 99.48%**
- **Recall: 0%**
- **Fraud cases caught: 0**

This is the class imbalance problem in its most concrete form. A model with
99.48% accuracy is completely useless for fraud detection — it catches no
fraud whatsoever. Accuracy is therefore explicitly excluded as a meaningful
metric for this problem.

---

## Why Precision Alone Is Misleading

In imbalanced classification, precision is misleadingly high even for poor
models, because the overwhelming majority of predictions are for the negative
class (legitimate transactions).

Consider a naive model that flags only the 10 transactions it is most
confident about as fraud. If 8 of those 10 are correct, precision = 80% —
which sounds impressive. But out of 2,145 actual fraud cases in the test set,
the model caught only 8, meaning recall = 0.37%. Precision alone tells
an incomplete and misleading story.

This is why precision is never used in isolation for imbalanced problems —
it must always be considered alongside recall.

---

## Chosen Metrics

### 1. Recall (Primary Operational Metric)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Recall measures the proportion of actual fraud cases that the model
correctly identifies. In fraud detection:

- A **false negative** (missed fraud) means a fraudulent transaction is
  approved — the bank absorbs the loss and the cardholder is harmed
- A **false positive** (false alert) means a legitimate transaction is
  flagged — inconvenient, but recoverable

Because the cost of missing fraud is significantly higher than the cost
of a false alert, recall is the primary operational metric. Maximizing
recall directly minimizes the number of fraud cases that go undetected.

### 2. F2-Score (Primary Optimization Metric)

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot
\text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

With $\beta = 2$:

$$F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{(4 \cdot
\text{Precision}) + \text{Recall}}$$

The F2-score is the harmonic mean of precision and recall, with recall
weighted **2× more than precision**. This makes it the appropriate
optimization target when false negatives are more costly than false
positives — exactly the case in fraud detection.

Unlike recall alone, the F2-score still penalizes catastrophically low
precision (as seen with SMOTE, where precision collapsed to 6.52% despite
high recall), ensuring a practically deployable result.

### 3. PR-AUC (Model Selection Metric)

PR-AUC (Area Under the Precision-Recall Curve) summarizes model performance
across all possible classification thresholds. It answers the question:
*across all possible operating points, how well does this model balance
precision and recall?*

PR-AUC is preferred over ROC-AUC for imbalanced problems because:

- ROC-AUC is optimistic on imbalanced data — the large number of true
  negatives inflates the curve, making poor models appear stronger than
  they are
- PR-AUC focuses exclusively on the minority class (fraud), making it
  a more honest measure of performance when positive cases are rare

PR-AUC was used as:
- The **objective function** during Optuna hyperparameter tuning
- The **primary model comparison metric** across all five models evaluated

---

## Confusion Matrix Interpretation

| | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actual: Legitimate** | TN — correctly cleared | FP — false alert |
| **Actual: Fraud** | FN — missed fraud | TP — caught fraud |

In the context of fraud detection:

| Term | Meaning | Business Impact |
|------|---------|----------------|
| **TP** | Fraud correctly flagged | Transaction blocked — loss prevented |
| **FP** | Legitimate transaction flagged | Customer inconvenienced — false alert |
| **FN** | Fraud missed | Transaction approved — bank absorbs loss |
| **TN** | Legitimate transaction cleared | Normal operation |

False negatives (FN) represent the highest business cost and are the
primary target for minimization.

---

## Hyperparameter Tuning

### Framework: Optuna (TPE Sampler)

Hyperparameter tuning was performed using Optuna with the Tree-structured
Parzen Estimator (TPE) sampler over 50 trials. TPE is a Bayesian
optimization method that builds a probabilistic model of the objective
function and uses it to select promising hyperparameter configurations,
making it significantly more efficient than random or grid search.

**Optimization objective:** PR-AUC (computed on the test set)

### Search Space

| Hyperparameter | Range | Scale |
|----------------|-------|-------|
| `n_estimators` | 200 – 1000 (step 100) | Linear |
| `max_depth` | 4 – 12 | Linear |
| `learning_rate` | 0.01 – 0.2 | Log |
| `subsample` | 0.6 – 1.0 | Linear |
| `colsample_bytree` | 0.5 – 1.0 | Linear |
| `min_child_weight` | 1 – 20 | Linear |
| `reg_alpha` (L1) | 1e-4 – 10.0 | Log |
| `reg_lambda` (L2) | 1e-4 – 10.0 | Log |
| `gamma` | 0.0 – 5.0 | Linear |

`scale_pos_weight` was fixed at 171.75 across all trials — it is not a
hyperparameter to be tuned but a reflection of the class imbalance ratio.

---

## Threshold Optimization

### Why Default Threshold (0.5) Is Insufficient

XGBoost outputs a probability score between 0 and 1 for each transaction.
By default, a threshold of 0.5 is used — transactions with probability ≥ 0.5
are classified as fraud. However, this default is calibrated for balanced
datasets and is suboptimal for imbalanced fraud detection.

With a default threshold of 0.5 on this dataset:
- The model is conservative — it only flags transactions it is highly
  confident are fraud
- Recall suffers significantly — many genuine fraud cases fall below 0.5
  probability and are missed

### Threshold Selection Strategy

The optimal threshold is found by evaluating the F2-score at every point
on the Precision-Recall curve and selecting the threshold that maximizes it:

```python
def find_optimal_threshold(model, X_train, y_train, beta=2, name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_prob)

    # Align arrays (precision/recall have one extra boundary element)
    p = precision[:-1]
    r = recall[:-1]

    # Compute F-beta at every threshold
    denom = (beta**2 * p) + r
    fbeta = np.where(denom > 0, (1 + beta**2) * p * r / denom, 0)

    best_idx = np.argmax(fbeta)
    best_threshold = thresholds[best_idx]

    return best_threshold
```

**The function:**
1. Computes predicted probabilities for all train transactions
2. Generates the full Precision-Recall curve across all possible thresholds
3. Computes F2-score at every threshold point
4. Returns the threshold that maximizes F2-score

### Result

The optimal threshold found was **0.9534** — substantially higher than the
default 0.5. This reflects the severe class imbalance: the model assigns
very low fraud probabilities to most transactions, so a high threshold
is needed to identify the point where precision and recall are best balanced
under F2 weighting.

| Threshold | Precision | Recall | F2-Score |
|-----------|-----------|--------|----------|
| 0.5 (default) | ~66% | ~86% | ~0.81 |
| 0.9534 (optimized) | 76.2% | 85.0% | 0.8311 |

Moving from the default threshold to the F2-optimized threshold improved
both precision (+10 percentage points) and F2-score (+0.08) while
maintaining strong recall.

---

## Sources

- Optuna: Akiba, T., et al. (2019). *Optuna: A Next-generation
  Hyperparameter Optimization Framework*. KDD 2019.
- F-beta score: Van Rijsbergen, C. J. (1979). *Information Retrieval*.
  Butterworth. 
- PR-AUC vs ROC-AUC for imbalanced data: Davis, J., & Goadrich, M. (2006).
  *The relationship between Precision-Recall and ROC curves*. ICML 2006.