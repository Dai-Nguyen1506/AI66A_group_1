# Imbalance Handling

## Overview

The dataset contains a severe class imbalance — approximately 0.58% of
training transactions are fraudulent. This means for every 1 fraud case,
there are roughly 172 legitimate transactions. Without explicit handling,
a model trained on this data will be heavily biased toward predicting
the majority class (legitimate), resulting in high accuracy but near-zero
recall on fraud cases.

Three strategies were evaluated:
1. `scale_pos_weight` — cost-sensitive learning
2. SMOTE — synthetic minority oversampling
3. Random undersampling — majority class reduction

---

## Strategy 1: scale_pos_weight (Selected)

### What It Is

`scale_pos_weight` is a parameter native to XGBoost that directly adjusts
the loss function to penalize misclassification of the minority class more
heavily. It instructs the model to treat each fraud case as if it were
worth `scale_pos_weight` times more than a legitimate transaction during
training.

### Calculation

```python
def class_weights(y_train):
    fraud = y_train.sum()
    non_fraud = len(y_train) - fraud

    scale_pos_weight = (non_fraud / fraud).round(2)
    print("scale_pos_weight:", scale_pos_weight)

    return scale_pos_weight
```

$$\text{scale pos weight} = \frac{n_{negatives}}{n_{positives}} =
\frac{1{,}289{,}169}{7{,}506} \approx 171.75$$

This value tells XGBoost to penalize a missed fraud case 171.75× more
than a missed legitimate transaction, directly reflecting the class
imbalance ratio in the training set.

### Why It Works Well Here

- No modification to the training data — the model trains on the real
  distribution of transactions
- No synthetic samples — fraud signal remains genuine
- Directly integrated into the XGBoost objective function — the model
  learns the imbalanced problem directly rather than being presented
  with an artificially balanced version of it
- Computationally free — no additional preprocessing step required

---

## Strategy 2: SMOTE (Not Selected)

### What It Is

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
fraud samples by interpolating between existing fraud cases in feature
space, increasing the minority class size until a target ratio is reached.

### Implementation

```python
def smote_oversampling(X_train, y_train):
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
```

With `sampling_strategy=0.2`, the minority class was oversampled to 20%
of the majority class size:

| Class | Before SMOTE | After SMOTE |
|-------|--------------|-------------|
| Legitimate (0) | 1,289,169 | 1,289,169 |
| Fraud (1) | 7,506 | 257,833 |

### Results

| Metric | Value |
|--------|-------|
| Precision | 6.52% |
| Recall | 96.04% |
| F2-score | 0.2564 |
| PR-AUC | 0.8378 |
| ROC-AUC | 0.9918 |

Confusion Matrix:
|                | Predicted Non-Fraud | Predicted Fraud |
|----------------|---------------------|-----------------|
| Actual Non-Fraud | 524,038           | 29,536          |
| Actual Fraud     | 85                | 2,060           |

### Why It Was Not Selected

Despite achieving the highest recall (96.04%), SMOTE produced catastrophic
precision — only 6.52% of flagged transactions were actually fraud, meaning
29,536 legitimate transactions were incorrectly flagged. This is operationally
unacceptable in a real fraud detection system where false alerts erode
customer trust and create investigation overhead.

The F2-score of 0.2564 — despite recall weighting recall 2× over precision —
confirms that the precision collapse was severe enough to make SMOTE the
worst-performing strategy overall.

### Theoretical Explanation

SMOTE interpolates between existing minority class samples in feature space.
In tabular fraud detection, this interpolation is problematic for several
reasons:

- **Synthetic samples are not real fraud**: Interpolating between two fraud
  transactions does not guarantee the synthetic point represents a realistic
  fraud pattern — it may fall in a region of feature space that is
  indistinguishable from legitimate transactions
- **Decision boundary distortion**: By flooding the training set with
  synthetic fraud cases, SMOTE pushes the decision boundary aggressively
  toward the legitimate class, causing the model to flag large portions
  of legitimate transactions as fraud
- **Feature interaction assumptions**: SMOTE treats each feature
  independently during interpolation, ignoring complex feature interactions
  that define real fraud patterns

---

## Strategy 3: Random Undersampling (Not Selected)

### What It Is

Random undersampling randomly removes majority class samples (legitimate
transactions) from the training set until a target minority-to-majority
ratio is reached.

### Implementation

```python
def undersampling(X_train, y_train, strategy=0.1):
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res
```

With `sampling_strategy=0.1`, the majority class was reduced until fraud
represented 10% of the majority class size:

| Class | Before Undersampling | After Undersampling |
|-------|---------------------|---------------------|
| Legitimate (0) | 1,289,169 | 75,060 |
| Fraud (1) | 7,506 | 7,506 |

### Results

| Metric | Value |
|--------|-------|
| Precision | 27.68% |
| Recall | 95.34% |
| F2-score | 0.6403 |
| PR-AUC | 0.8856 |
| ROC-AUC | 0.9971 |

Confusion Matrix:
|                | Predicted Non-Fraud | Predicted Fraud |
|----------------|---------------------|-----------------|
| Actual Non-Fraud | 548,230           | 5,344           |
| Actual Fraud     | 100               | 2,045           |

### Why It Was Not Selected

Undersampling performed considerably better than SMOTE in precision
(27.68% vs 6.52%) and F2-score (0.6403 vs 0.2564), but still fell short
of `scale_pos_weight` across all key metrics. The false alert rate remained
high at 5,344 incorrectly flagged legitimate transactions.

### Theoretical Explanation

Random undersampling discards the majority of legitimate transaction data —
in this case, over 1.2 million real transactions are removed from training.
This introduces several problems:

- **Information loss**: Legitimate transactions contain important boundary
  information that helps the model distinguish edge cases. Discarding them
  degrades the model's ability to correctly clear borderline transactions
- **Non-representative training distribution**: The model is trained on
  a highly artificial ratio (10:1) that does not reflect the real-world
  distribution (172:1), potentially causing miscalibrated probability
  estimates at inference time
- **Random removal**: Important legitimate transaction patterns may be
  accidentally removed, further degrading the model's understanding of
  the majority class

---

## Comparison Summary

| Strategy | Precision | Recall | F2-Score | PR-AUC | False Alerts |
|----------|-----------|--------|----------|--------|--------------|
| scale_pos_weight | **65.6%** | 86.2% | **0.8115** | **0.8871** | **969** |
| Undersampling | 27.68% | 95.34% | 0.6403 | 0.8856 | 5,344 |
| SMOTE | 6.52% | **96.04%** | 0.2564 | 0.8378 | 29,536 |

`scale_pos_weight` achieved the best balance across all metrics —
highest PR-AUC, highest F2-score, and by far the lowest false alert rate —
while maintaining strong recall. It was selected as the imbalance handling
strategy for the final model.

---

## Sources

- SMOTE: Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority
  Over-sampling Technique*. Journal of Artificial Intelligence Research.
  — Please verify citation details before publishing
- XGBoost scale_pos_weight: Chen, T., & Guestrin, C. (2016). *XGBoost:
  A Scalable Tree Boosting System*. KDD 2016.
  — Please verify citation details before publishing