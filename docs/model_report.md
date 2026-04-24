# 1. Executive Summary
- Fraud detection model for credit card transactions.
- Dataset: 1.85M transactions, 0.38% fraud rate.
- Best model: XGBoost — catches 85% of fraud with 76% precision.
- False alert rate: 0.10% of legitimate transactions.

# 2. Data & Features
Dataset
- Train: 1,296,675 transactions
- Test:    555,719 transactions
- Fraud rate: ~0.38% (heavily imbalanced)

Features selected (9 final)
┌─────────────────────────┬───────────────────────────────────────────┐
│ Feature                 │ Description                               │
├─────────────────────────┼───────────────────────────────────────────┤
│ amt                     │ Transaction amount                        │
│ age                     │ Cardholder age                            │
│ hour                    │ Hour of transaction                       │
│ city_pop                │ Population of cardholder city             │
│ gender                  │ Cardholder gender                         │
│ category_fraud_rate     │ Target-encoded fraud rate by category     │
│ merchant_fraud_rate     │ Target-encoded fraud rate by merchant     │
│ category_freq           │ Transaction frequency by category         │
│ merchant_freq           │ Transaction frequency by merchant         │
└─────────────────────────┴───────────────────────────────────────────┘

Feature engineering notes
- Target encoding used out-of-fold (StratifiedKFold, k=5) to prevent leakage
- Smoothing (alpha=10) applied to handle rare categories
- Distancecy features engineered but subsumed by frequency features

# 3. Modelling Approach
Class imbalance handling
- scale_pos_weight = 171.75 (ratio of negatives to positives)
- Evaluated SMOTE and undersampling — scale_pos_weight outperformed both

Models evaluated
1. Logistic Regression   (baseline)
2. Random Forest
3. XGBoost               ← winner
4. LightGBM
5. MLP (PyTorch)

Hyperparameter tuning
- Optuna (TPE sampler, 50 trials) on XGBoost
- Optimisation metric: PR-AUC (appropriate for imbalanced data)
- Threshold tuning: F2-score maximisation (recall weighted 2× precision)

# 4. Results
Final model: Tuned XGBoost
Threshold:   0.9534 (F2-optimised)

Metric          Value
──────────────────────
PR-AUC          0.8975
ROC-AUC         0.9978
Precision       76.2%
Recall          85.0%
F2-score        0.8311
──────────────────────
TP              1,824   fraud correctly caught
FP                570   false alerts
FN                321   missed fraud
TN            553,004   correctly cleared
──────────────────────
False alert rate  0.10% of legitimate transactions
Fraud caught      85.0% of all fraud cases

# 5. Key Findings
1. Amount + fraud rate features dominate
   → amt, category_fraud_rate, merchant_fraud_rate account for ~65%
     of model importance (SHAP)

2. Tree models outperform neural networks on this feature set
   → MLP PR-AUC: 0.831 vs XGBoost: 0.898
   → Deep learning requires sequential/graph data to compete here

3. Threshold matters more than model architecture
   → Moving threshold from 0.5 → 0.95 changed recall from 29% → 85%
   → Business cost ratio should drive threshold choice

4. Diminishing returns after feature engineering
   → All gradient boosting models converged to ~0.89-0.90 PR-AUC
   → Remaining 15% missed fraud likely indistinguishable from
     legitimate transactions given available features

# 6. Limitations & Future Work
Limitations
- Static fraud rates — concept drift will degrade performance over time
- No graph features (shared devices, IPs, merchant networks)
- Model not calibrated for production (threshold=0.95 not intuitive)

Future work (expected impact)
┌──────────────────────────────────┬─────────────────┐
│ Improvement                      │ Expected gain   │
├──────────────────────────────────┼─────────────────┤
│ Per-card velocity features       │ +2-4% PR-AUC    │
│ Graph neural network             │ +3-6% PR-AUC    │
│ Online learning / concept drift  │ Maintain recall │
│ Probability calibration          │ Better UX       │
└──────────────────────────────────┴─────────────────┘

# 7. Notebook Structure
01_eda.ipynb                → data exploration, class distribution
02_data_preprocessing.ipynb → feature engineering, encoding, handle class imbalance
03_model_training.ipynb     → baseline models, feature selection
04_model_optimization.ipynb → Optuna tuning, threshold search
05_deeplearning.ipynb       → MLP experiment
06_final_model.ipynb        → save model, inference function
