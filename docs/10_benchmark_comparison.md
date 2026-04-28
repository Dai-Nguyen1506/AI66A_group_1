# Benchmark Comparison

## Overview

This document compares the final model from this project against two
highly-voted Kaggle notebooks that tackle credit card fraud detection.
The comparison is intended to contextualize our results within the broader
community of approaches on this problem.

---

## [ ! ] Important Note on Dataset Differences

The two reference notebooks use a **different dataset** from this project:

| | This Project | Reference Notebooks |
|---|---|---|
| Dataset | Simulated US transactions (Kaggle — Kartik Shenoy) | European cardholders, 2013 (Kaggle — ULB) |
| Features | Raw interpretable features (`amt`, `category`, etc.) | PCA-transformed anonymous features (`V1`–`V28`) |
| Samples | ~1.85M transactions | ~284K transactions |
| Fraud rate | ~0.38% | ~0.17% |

Because the datasets are fundamentally different — different size, different
fraud rate, different feature space — this is **not a direct apples-to-apples
comparison**. The purpose is to compare modeling approaches and final
performance at a high level, not to make absolute claims about superiority.

---

## Reference Notebooks

### Notebook 1 — Rutecki (2021)
*"Best Techniques and Metrics for Imbalanced Dataset"*
- Evaluated multiple imbalance handling strategies on Random Forest
- Strategies tested: No resampling, Random Oversampling, SMOTE,
  SMOTE + Tomek, Class weights

### Notebook 2 — Kim (2021)
*"CreditCard Fraud — Balance is Key (feat. PyCaret)"*
- Final model: CatBoost
- Used PyCaret AutoML framework for model selection and tuning

---

## Metric Note: PR-AUC Not Available

Neither reference notebook reports PR-AUC (Area Under the
Precision-Recall Curve). This is a significant limitation for comparison
purposes.

PR-AUC is the most informative metric for imbalanced fraud detection
because:
- It focuses exclusively on the minority class (fraud), ignoring the
  overwhelming number of true negatives
- It summarizes model performance across **all possible thresholds**,
  rather than at a single operating point
- Accuracy and F1-score — commonly reported in both notebooks — are
  misleading on imbalanced data (a model predicting all legitimate
  achieves near-perfect accuracy while catching zero fraud)

The comparisons below are therefore limited to Recall, Precision, and
F2-score — the latter computed from their reported Recall and Precision
using:

$$F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{(4 \cdot
\text{Precision}) + \text{Recall}}$$

---

## Notebook 1 Comparison — Rutecki (Random Forest)

### Their Results

| # | Strategy | Recall | Precision | F1 | F2 (computed) |
|---|----------|--------|-----------|-----|---------------|
| 2 | SMOTE Oversampling | 0.8521 | 0.2542 | 0.3916 | 0.5550 |
| 1 | Random Oversampling | 0.8380 | 0.2371 | 0.3696 | 0.5392 |
| 4 | Class weights | 0.8239 | 0.3145 | 0.4553 | 0.5810 |
| 0 | No resampling | 0.7676 | 0.9646 | 0.8549 | 0.8001 |
| 3 | SMOTE + Tomek | 0.7606 | 0.9231 | 0.8340 | 0.7899 |

Their best F2 result is **0.8001** (No resampling strategy).

### Our Result vs Their Best

| | Rutecki — Best (No resampling) | This Project |
|---|---|---|
| Model | Random Forest | XGBoost (tuned) |
| Imbalance Handling | None | scale_pos_weight |
| Recall | 0.7676 | **0.8500** |
| Precision | 0.9646 | 0.7620 |
| F2-Score | 0.8001 | **0.8311** |
| PR-AUC | Not reported | **0.8975** |

### Key Findings

- Our model achieves **higher recall (+0.0824)** and **higher F2-score
  (+0.031)** than their best configuration
- Their best result uses no imbalance handling at all — a finding
  consistent with our own experiments showing that naive resampling
  (SMOTE, random oversampling) degrades F2-score substantially
- Their precision (0.9646) is higher than ours (0.7620), but this
  comes at the cost of significantly lower recall — a trade-off that
  is unfavorable in fraud detection where missing fraud is more costly
  than false alerts
- The use of Random Forest vs tuned XGBoost likely accounts for a
  meaningful portion of the performance gap, as XGBoost's sequential
  boosting and direct imbalance weighting (`scale_pos_weight`) are
  better suited to fraud detection than Random Forest

---

## Notebook 2 Comparison — Kim (CatBoost)

### Their Result

| Model | Recall | Precision | Accuracy | F2 (computed) |
|-------|--------|-----------|----------|---------------|
| CatBoost (PyCaret) | 0.8028 | 0.9881 | 0.8966 | 0.8310 |

### Our Result vs Theirs

| | Kim — CatBoost | This Project |
|---|---|---|
| Model | CatBoost (AutoML) | XGBoost (tuned) |
| Imbalance Handling | Not specified | scale_pos_weight |
| Recall | 0.8028 | **0.8500** |
| Precision | **0.9881** | 0.7620 |
| F2-Score | 0.8310 | **0.8311** |
| PR-AUC | Not reported | **0.8975** |

### Key Findings

- Our model achieves **higher recall (+0.0472)** with a virtually
  identical F2-score (+0.0001) — suggesting both models reach a
  similar overall balance point, but our model catches more fraud
- Their precision (0.9881) is substantially higher than ours (0.7620),
  indicating their model is more conservative — it flags fewer
  transactions as fraud but is more often correct when it does
- In a fraud detection context, our model's higher recall is preferable
  — catching 85.0% of fraud vs 80.28% means approximately 47 more
  fraud cases caught per 10,000 fraud attempts, at the cost of
  additional false alerts
- Their use of PyCaret (AutoML) simplifies the modeling pipeline but
  limits control over threshold tuning and imbalance handling — both
  of which were critical to our final performance

---

## Overall Comparison

| | Rutecki Best | Kim CatBoost | This Project |
|---|---|---|---|
| Model | Random Forest | CatBoost | XGBoost |
| Dataset | ULB (284K) | ULB (284K) | Simulated US (1.85M) |
| Recall | 0.7676 | 0.8028 | **0.8500** |
| Precision | 0.9646 | **0.9881** | 0.7620 |
| F2-Score | 0.8001 | 0.8310 | **0.8311** |
| PR-AUC | Not reported | Not reported | **0.8975** |
| Threshold Tuned | No | No | Yes (F2-optimized) |
| Imbalance Handling | None (best) | Not specified | scale_pos_weight |

---

## Summary

Across both reference notebooks, our model achieves:
- **Highest recall** — catching the most fraud cases of any compared model
- **Highest or equal F2-score** — best overall balance between catching
  fraud and minimizing false alerts under fraud-appropriate weighting
- **Only model with reported PR-AUC** — the most honest metric for
  imbalanced fraud detection, at 0.8975

The trade-off is lower precision — our model generates more false alerts
than either reference. This is a deliberate design choice driven by
threshold optimization for F2-score, which prioritizes recall over
precision. In a real deployment, the threshold can be adjusted to
increase precision at the cost of recall depending on the business
cost ratio.

It is worth noting that the reference notebooks use a smaller, different
dataset with PCA-transformed features, making direct performance
comparison inherently limited. The more meaningful comparison is in
**methodology** — our use of out-of-fold target encoding, explicit
threshold tuning, and `scale_pos_weight` produced strong results that
hold up against community benchmarks on a significantly larger and more
complex dataset.

---

## Sources

- Rutecki, M. (2023). *Best Techniques and Metrics for Imbalanced Dataset*.
  Kaggle Notebook.
  [www.kaggle.com/code/marcinrutecki/best-techniques-and-metrics-for-imbalanced-dataset/notebook](https://www.kaggle.com/code/marcinrutecki/best-techniques-and-metrics-for-imbalanced-dataset/notebook)

- Kim, O. (2022). *CreditCard Fraud — Balance is Key (feat. PyCaret)*.
  Kaggle Notebook.
  [www.kaggle.com/code/ohseokkim/creditcard-fraud-balance-is-key-feat-pycaret](https://www.kaggle.com/code/ohseokkim/creditcard-fraud-balance-is-key-feat-pycaret)