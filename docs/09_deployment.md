# Deployment

## Overview

This document describes how the final model is saved, how inference is
performed, and what a production deployment would look like beyond the
scope of this project.

---

## Model Artifacts

Two files are saved to the `models/` directory:

### 1. Model File — `xgb_fraud_final.pkl`

The trained XGBoost model is serialized using `joblib`:

```python
joblib.dump(xgb_tuned, '../models/xgb_fraud_final.pkl')
```

`joblib` is preferred over `pickle` for scikit-learn compatible objects
as it handles large NumPy arrays more efficiently.

### 2. Metadata File — `model_metadata.json`

A JSON file storing everything needed to reproduce inference correctly:

```json
{
  "features": [
    "amt", "category_fraud_rate", "category_freq", "city_pop",
    "gender", "hour", "age", "merchant_fraud_rate", "merchant_freq"
  ],
  "threshold": 0.9534,
  "metrics": {
    "pr_auc": 0.8975,
    "roc_auc": 0.9978,
    "precision": 0.762,
    "recall": 0.850,
    "f2": 0.8311
  },
  "model_params": {
    "n_estimators": 300,
    "max_depth": 10,
    "learning_rate": 0.024,
    "subsample": 0.97,
    "colsample_bytree": 0.94,
    "min_child_weight": 15,
    "reg_alpha": 0.2,
    "reg_lambda": 0.0007,
    "gamma": 4.8,
    "scale_pos_weight": 171.75
  }
}
```

Storing metadata alongside the model ensures that:
- The correct feature list and order is always used at inference time
- The decision threshold is versioned with the model — not hardcoded
  separately in application code
- Model performance metrics are traceable to the saved artifact

---

## Inference Pipeline

### Prerequisites

Before calling the inference function, incoming transactions must already
have the 9 model features computed. This means the following preprocessing
steps must be applied upstream:

| Step | Features Produced | Notes |
|------|------------------|-------|
| Clean feature names | All columns | Replace special characters with underscores |
| Extract hour from timestamp | `hour` | From `trans_date_trans_time` |
| Compute age from date of birth | `age` | At time of transaction |
| Lookup category frequency | `category_freq` | From training set statistics |
| Lookup merchant frequency | `merchant_freq` | From training set statistics |
| Lookup category fraud rate | `category_fraud_rate` | From training set encoding |
| Lookup merchant fraud rate | `merchant_fraud_rate` | From training set encoding |
| Label encode categorical features | `gender` | Must use encoder fit on training data |
| Pass through raw features | `amt`, `city_pop` | No transformation needed |

Training set statistics (frequency counts and fraud rate encodings) must
be saved and loaded at inference time — the model cannot compute these
from a single incoming transaction.

### Inference Function

```python
def predict_fraud(transactions: pd.DataFrame,
                  model, features: list, threshold: float) -> pd.DataFrame:
    """
    Parameters
    ----------
    transactions : DataFrame with at minimum the 9 model features
    model        : fitted XGBClassifier
    features     : ordered feature list from metadata
    threshold    : decision threshold from metadata

    Returns
    -------
    DataFrame with columns: fraud_probability, is_fraud_predicted
    """
    X = transactions[features]
    probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        'fraud_probability'  : probs,
        'is_fraud_predicted' : (probs >= threshold).astype(int),
    }, index=transactions.index)
```

The function:
1. Selects the 9 model features in the correct order from the input DataFrame
2. Computes fraud probability scores using the trained model
3. Applies the decision threshold (0.9534) to produce binary predictions
4. Returns both the raw probability and the binary prediction for each
   transaction

### Usage Example

```python
# Load model and metadata
model = joblib.load('../models/xgb_fraud_final.pkl')

with open('../models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

features  = metadata['features']
threshold = metadata['threshold']

# Score transactions
result = predict_fraud(transactions, model, features, threshold)
```

Output format:

| fraud_probability | is_fraud_predicted |
|------------------|-------------------|
| 0.0023 | 0 |
| 0.9812 | 1 |
| 0.1547 | 0 |

---

## Limitations of Current Setup

The current implementation is suitable for offline batch scoring and
portfolio demonstration. It is not production-ready as-is due to the
following limitations:

| Limitation | Description |
|------------|-------------|
| No API layer | The model is invoked directly from Python — no REST endpoint |
| No real-time preprocessing | Feature engineering must be done manually upstream |
| Static encodings | Frequency and fraud rate encodings are frozen at training time — they do not update as new transactions arrive |
| No monitoring | There is no mechanism to detect model drift or degradation over time |
| No retraining pipeline | The model must be manually retrained when performance degrades |

---

## Path to Production

To deploy this model in a real fraud detection system, the following
components would need to be added:

### 1. REST API
Wrap the inference function in a REST API (e.g., FastAPI or Flask) to
allow real-time transaction scoring. The API would accept incoming
transaction features as a request payload and return the fraud probability
and binary prediction. This enables the model to be called from any
application — payment processors, banking systems, or monitoring dashboards —
without requiring direct Python access.

### 2. Feature Store
Training set statistics (frequency encodings, fraud rate encodings) should
be stored in a feature store (e.g., Feast, Tecton) that:
- Serves precomputed features at low latency for real-time inference
- Updates feature values as new transactions arrive
- Ensures consistency between training and serving feature distributions

### 3. Model Registry
Use a model registry (e.g., MLflow) to:
- Version and track model artifacts
- Store evaluation metrics alongside each model version
- Enable rollback to a previous model version if performance degrades

### 4. Monitoring
Deploy monitoring to detect:
- **Data drift** — incoming transaction distributions shifting away from
  training distribution
- **Concept drift** — fraud patterns evolving over time, degrading model
  recall
- **Performance degradation** — PR-AUC or recall dropping below acceptable
  thresholds

### 5. Retraining Pipeline
Schedule periodic retraining (e.g., monthly) on a sliding window of recent
transactions to keep the model current with evolving fraud patterns.

---

## Sources

- joblib documentation: [https://joblib.readthedocs.io](https://joblib.readthedocs.io)
- FastAPI documentation: [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- MLflow documentation: [https://mlflow.org](https://mlflow.org)