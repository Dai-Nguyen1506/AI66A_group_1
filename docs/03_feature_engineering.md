# Feature Engineering

## Overview

The raw dataset contains transactional, geographic, and cardholder features.
Raw features alone are insufficient for effective fraud detection — several
high-signal features must be derived from the existing columns. This document
describes all engineered features, the rationale behind each, and the feature
selection process used to arrive at the final 9 features.

---

## Engineered Features

### 1. Time-Based Features

Extracted from `trans_date_trans_time`:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `hour` | Hour of transaction (0–23) | Fraud tends to cluster at unusual hours |
| `day_of_week` | Day of the week (0–6) | Weekly spending patterns differ |
| `month` | Month of the year (1–12) | Seasonal fraud patterns |
| `is_weekend` | Binary flag — 1 if weekend | Weekend transactions may behave differently |
| `is_night` | Binary flag — 1 if hour is between 22:00–06:00 | Night transactions carry higher fraud risk |

`trans_date_trans_time` is dropped after extraction. `unix_time` is retained
as a raw numeric feature but ultimately dropped during feature selection.

---

### 2. Amount-Based Features

Derived from `amt`:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `amt_log` | Log-transformed transaction amount | Reduces right skew |
| `amt_zscore` | Z-score normalized transaction amount | Normalizes scale |

Note: Both `amt_log` and `amt_zscore` were dropped during feature selection.
Since raw `amt` alone carries strong fraud signal (fraud transactions tend to
have significantly higher amounts), the transformed variants added no
additional predictive value.

---

### 3. Age Feature

Derived from `dob`:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `age` | Cardholder age in years at time of transaction | Age may correlate with fraud vulnerability |

`dob` is dropped after `age` is computed.

---

### 4. Geographic Distance Feature

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `distance_km` | Haversine distance between cardholder location (`lat`, `long`) and merchant location (`merch_lat`, `merch_long`) | Large distances may indicate suspicious activity |

Computed using the Haversine formula:

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
```

`distance_km` was dropped during feature selection — permutation importance
showed near-zero contribution, indicating that frequency and fraud rate
features captured location-related signals more effectively.

---

### 5. Frequency Encoding

Transaction frequency per category, merchant, and city:

| Feature | Description |
|---------|-------------|
| `category_freq` | Number of transactions per category in the training set |
| `merchant_freq` | Number of transactions per merchant in the training set |
| `city_freq` | Number of transactions per city in the training set |

Frequency encoding captures how common a given category, merchant, or city
is in the dataset. Rare merchants or categories may signal unusual activity.

`city_freq` was dropped during feature selection — permutation importance
showed it was harming model performance.

---

### 6. Target Encoding (Fraud Rate)

Fraud rate per category, merchant, and city, computed using
out-of-fold estimation to prevent target leakage:

| Feature | Description |
|---------|-------------|
| `category_fraud_rate` | Historical fraud rate for each transaction category |
| `merchant_fraud_rate` | Historical fraud rate for each merchant |
| `city_fraud_rate` | Historical fraud rate for each city |

#### Implementation

Target encoding is applied using StratifiedKFold (k=5) with Laplace smoothing
(alpha=10) to handle rare categories:

$$\text{smoothed\_rate} = \frac{n \cdot \bar{y}_{group} + \alpha \cdot \bar{y}_{global}}{n + \alpha}$$

Where:
- $n$ = number of transactions in the group
- $\bar{y}_{group}$ = mean fraud rate within the group
- $\bar{y}_{global}$ = global mean fraud rate across the training set
- $\alpha$ = smoothing factor (10)

```python
def target_encode(train, test, col, target='is_fraud', n_splits=5, alpha=10):
    train_encoded = np.zeros(len(train))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train, train[target]):
        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]

        global_mean = X_tr[target].mean()
        stats = X_tr.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)

        train_encoded[val_idx] = X_val[col].map(smooth)

    global_mean = train[target].mean()
    train[col + '_fraud_rate'] = np.where(np.isnan(train_encoded), global_mean, train_encoded)

    stats = train.groupby(col)[target].agg(['mean', 'count'])
    smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)

    test[col + '_fraud_rate'] = test[col].map(smooth).fillna(global_mean)

    return train, test
```

**Leakage prevention:**
- Encoding is computed on training folds only, then applied to the
  validation fold within each split
- Test set encoding uses statistics computed from the full training set
- Categories unseen in training are filled with the global mean

#### [ ! ] Important Finding: `city_fraud_rate` Was Dropped

Despite being computed with the same leakage-prevention methodology as
`category_fraud_rate` and `merchant_fraud_rate`, `city_fraud_rate` was found
to **severely harm model performance**.

Permutation importance showed `city_fraud_rate` as the most damaging feature
in the entire feature set (importance score of approximately −0.36), far worse
than any other feature. Dropping it resulted in approximately 30–40 point
improvement in both recall and F2-score.

The exact cause is uncertain — possible explanations include:

- **Residual data leakage**: Despite out-of-fold encoding, city-level
  aggregation may have introduced subtle leakage due to the high cardinality
  and uneven distribution of cities
- **Noise amplification**: City-level fraud rates may be too noisy to be
  predictive (many cities have very few transactions, making their fraud
  rates unreliable even with smoothing)
- **Collinearity with harmful signal**: `city_fraud_rate` may correlate with
  legitimate regional patterns in a way that confuses the model

Regardless of the cause, the empirical evidence was clear and consistent:
removing `city_fraud_rate` substantially improved model performance.

---

## Feature Selection

Feature selection was performed using two complementary methods:

### SHAP Values
SHAP (SHapley Additive exPlanations) was used to measure the contribution
of each feature to individual predictions, providing both direction and
magnitude of influence.

### Permutation Importance
Permutation importance was used to measure the drop in PR-AUC when each
feature's values are randomly shuffled. Features with negative permutation
importance actively harm the model and are strong candidates for removal.

Features were removed if they showed:
- Near-zero or negative permutation importance
- Negligible SHAP contribution
- Redundancy with a stronger correlated feature (e.g., `amt_log`, `amt_zscore`
  vs raw `amt`)

---

## Final Feature Set

After feature selection, 9 features were retained:

| Feature | Type | Origin |
|---------|------|--------|
| `amt` | Numeric | Raw |
| `hour` | Numeric | Engineered from `trans_date_trans_time` |
| `age` | Numeric | Engineered from `dob` |
| `city_pop` | Numeric | Raw |
| `gender` | Categorical | Raw |
| `category_freq` | Numeric | Frequency encoding |
| `merchant_freq` | Numeric | Frequency encoding |
| `category_fraud_rate` | Numeric | Target encoding |
| `merchant_fraud_rate` | Numeric | Target encoding |

---

## Sources

- Haversine formula: standard spherical geometry, no external citation required
- SHAP methodology: Lundberg, S. M., & Lee, S. I. (2017). *A unified approach
  to interpreting model predictions*. NeurIPS 2017.
  — Please verify citation details before publishing
- Laplace smoothing for target encoding: standard technique, widely documented
  in the feature engineering literature