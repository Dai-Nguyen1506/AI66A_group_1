# Modeling

## Overview

Five models were evaluated across two paradigms — tree-based ensemble methods
and neural networks. Each model was selected for a specific reason, and all
were evaluated under the same conditions: trained on the same set of features,
with class imbalance handled via `scale_pos_weight`, and evaluated using
PR-AUC, F2-score, and recall on the held-out test set.

The winning model was selected based on the highest combined performance
across recall, F2-score, and PR-AUC.

---

## Models Evaluated

### 1. Logistic Regression (Baseline)

**Why it was tried:**
Logistic Regression serves as the baseline model. In fraud detection, a
baseline is essential — it establishes the minimum performance threshold
that any more complex model must meaningfully exceed to justify its
additional complexity and computational cost.

**Strengths:**
- Fast to train and interpret
- Provides calibrated probability outputs
- Works reasonably well when fraud signal is linearly separable

**Weaknesses:**
- Assumes linear decision boundaries — fraud patterns are rarely linear
- Struggles with feature interactions (e.g., high `amt` at night from
  a high-risk merchant is more suspicious than any single feature alone)
- Sensitive to feature scaling

**Performance:**
Logistic Regression produced the weakest results across all metrics,
confirming that fraud detection in this dataset requires a non-linear model.
PR-AUC was substantially lower than all tree-based models.

---

### 2. Random Forest

**Why it was tried:**
Random Forest is a strong general-purpose ensemble method that handles
non-linear relationships and feature interactions naturally. In fraud
detection, it is a common first step beyond logistic regression due to
its robustness and interpretability via feature importance.

**Strengths:**
- Handles non-linear relationships and feature interactions well
- Robust to outliers (important given skewed `amt` distribution)
- Built-in feature importance via Gini impurity
- Less prone to overfitting than a single decision tree

**Weaknesses:**
- Slower to train than gradient boosting on large datasets
- Less effective than gradient boosting at handling class imbalance
- Feature importance can be misleading for correlated features

**Performance:**
Random Forest outperformed Logistic Regression but fell short of gradient
boosting models. Recall was notably low — the model was conservative in
flagging fraud, resulting in many missed cases.

---

### 3. XGBoost

**Why it was tried:**
XGBoost (Extreme Gradient Boosting) is one of the most widely used models
in structured/tabular fraud detection tasks. It builds trees sequentially,
with each tree correcting the errors of the previous one. Its
`scale_pos_weight` parameter allows direct handling of class imbalance
without resampling.

**Strengths:**
- Sequential boosting corrects previous errors — highly effective on
  imbalanced data when combined with `scale_pos_weight`
- Handles missing values natively
- Regularization (L1/L2) reduces overfitting
- Highly tunable — responds well to hyperparameter optimization
- `scale_pos_weight` directly penalizes the model for missing fraud cases

**Weaknesses:**
- Slower to train than LightGBM on large datasets
- More hyperparameters to tune than Random Forest

**Performance:**
XGBoost achieved the highest PR-AUC, recall, and F2-score across all models
and was selected as the final model.

**Tuning Strategy:**
XGBoost was tuned using Optuna (Tree-structured Parzen Estimator sampler)
over 50 trials, optimizing directly for PR-AUC. Threshold selection was
performed separately after tuning, optimizing for F2-score. Full tuning
details are documented in `evaluation_strategy.md`.

---

### 4. LightGBM

**Why it was tried:**
LightGBM is a gradient boosting framework designed for speed and efficiency
on large datasets. It uses a histogram-based algorithm and leaf-wise tree
growth, making it significantly faster than XGBoost on datasets with
millions of rows.

**Strengths:**
- Faster training than XGBoost on large datasets
- Handles high-cardinality categorical features natively
- Lower memory footprint

**Weaknesses:**
- Leaf-wise growth can lead to overfitting on smaller datasets
- Slightly less effective than XGBoost on this specific dataset and
  feature set

**Performance:**
LightGBM produced competitive but slightly lower PR-AUC and recall compared
to XGBoost. The performance gap was consistent across multiple runs,
suggesting it is a property of the model-data interaction rather than
random variance.

---

### 5. MLP — Multilayer Perceptron (PyTorch)

**Why it was tried:**
Neural networks represent a fundamentally different modeling paradigm from
tree-based methods. While gradient boosting builds an ensemble of shallow
decision trees, an MLP learns hierarchical representations through
successive layers of non-linear transformations. It was included to test
whether deep learning could find patterns in the tabular feature space
that tree-based models missed.

**Architecture:**
A fully connected feedforward network trained with:
- Binary cross-entropy loss with class weighting
- Adam optimizer
- Dropout regularization
- Early stopping based on validation PR-AUC

**Strengths:**
- Theoretically capable of learning arbitrary complex functions
- Can model high-order feature interactions implicitly
- Scales well with more data

**Weaknesses (on this dataset):**
- Tabular data with hand-crafted features does not provide the same
  advantage to deep learning as raw sequential or spatial data would
- Tree-based models handle the sharp, discontinuous decision boundaries
  common in fraud detection more naturally than smooth neural activations
- Requires more careful tuning (learning rate, dropout, batch size,
  architecture depth) compared to XGBoost
- No native equivalent of `scale_pos_weight` — class imbalance handling
  requires manual loss weighting

**Performance:**
The MLP achieved a PR-AUC of approximately 0.831, compared to XGBoost's
0.898 — a gap of ~0.067. This result is consistent with the general finding
in the machine learning literature that gradient boosting tends to
outperform neural networks on structured tabular data, particularly when
the feature set is relatively small and hand-engineered rather than raw.

The MLP would likely be more competitive if the feature set included raw
sequential transaction history (e.g., per-card transaction sequences) or
graph-based features, where deep learning architectures such as LSTMs or
Graph Neural Networks have a natural advantage.

---

## Model Comparison Summary

| Model | PR-AUC | Recall | F2-Score | Notes |
|-------|--------|--------|----------|-------|
| Logistic Regression | Low | Medium | Low | Linear baseline |
| Random Forest | High | Medium | High | Conservative — low recall |
| LightGBM | High | Highest | Medium | Slightly below XGBoost |
| XGBoost | High | High | Highest | Selected as final model |
| MLP (PyTorch) | High | Medium | Medium | Deep learning baseline |

*Exact metric values for all models are reported in `results.md`.*

---

## Sources

- XGBoost: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree
  Boosting System*. KDD 2016.
- LightGBM: Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree*. NeurIPS 2017.
- General finding on tabular data: Grinsztajn, L., Oyallon, E., & Varoquaux,
  G. (2022). *Why tree-based models still outperform deep learning on tabular
  data*. NeurIPS 2022.