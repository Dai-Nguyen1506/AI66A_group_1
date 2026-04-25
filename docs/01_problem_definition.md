# Problem Definition

## Overview

Credit card fraud is a significant financial threat to banks, payment processors,
and cardholders. According to the Nilson Report, global card fraud losses reached
$33.41 billion in 2024, with projections continuing to rise as digital transactions
grow. For financial institutions, every undetected fraudulent transaction represents
a direct monetary loss, reputational damage, and erosion of customer trust.

---

## Problem Statement

Given a set of credit card transactions, the goal is to build a binary classification
model that distinguishes fraudulent transactions (is_fraud = 1) from legitimate ones
(is_fraud = 0).

This is a supervised learning problem where:
- Input: transaction-level features (amount, time, location, merchant, cardholder info)
- Output: binary label — fraud (1) or not fraud (0)

---

## Why Fraud Detection Is Hard

### 1. Severe Class Imbalance (Primary Challenge)

In real-world transaction data, fraud is rare. In this dataset, only ~0.52% of
transactions are fraudulent — roughly 1 in every 192 transactions. This extreme
imbalance creates several problems:

- A naive model that predicts "not fraud" for every transaction achieves ~99.5%
  accuracy, yet catches zero fraud cases
- Standard training objectives (e.g., cross-entropy loss) are dominated by the
  majority class, causing the model to ignore the minority class
- Evaluation metrics like accuracy become meaningless

Addressing class imbalance is therefore not optional — it is central to the
entire modeling strategy.

### 2. Evolving Fraud Patterns (Secondary Challenge)

Fraud is not a static phenomenon. Fraudsters continuously adapt their behavior
to evade detection systems:

- New attack vectors emerge over time (e.g., card-not-present fraud, synthetic
  identity fraud)
- Patterns that indicate fraud today may appear legitimate tomorrow
- A model trained on historical data will degrade in performance over time
  without retraining or online learning mechanisms (concept drift)

This means that even a high-performing model has a limited shelf life in
production without monitoring and periodic retraining.

### 3. High Similarity Between Classes

Many fraudulent transactions are deliberately designed to mimic legitimate
behavior — small amounts, familiar merchant categories, normal hours. This
makes the decision boundary inherently noisy, and some fraud cases are
fundamentally indistinguishable from legitimate transactions given only
tabular features.

---

## Objectives

The primary objective is to maximize the detection of fraudulent transactions
(recall), while maintaining an acceptable false alert rate (precision).

Because the cost of missing fraud (false negative) is significantly higher than
the cost of a false alert (false positive), the model is optimized for:

- **Recall**: Proportion of actual fraud cases correctly identified
- **F2-score**: Weighted harmonic mean of precision and recall, where recall
  is weighted 2× more than precision — appropriate when false negatives are
  more costly than false positives
- **PR-AUC**: Area under the Precision-Recall curve — the standard evaluation
  metric for imbalanced classification problems

Accuracy is explicitly excluded as a meaningful metric for this problem.

## Sources
[Nilson Report (2026)](https://finance.yahoo.com/news/global-card-fraud-losses-33-170800960.html)