# ⚙️ Configuration Guide

Hướng dẫn cấu hình hyperparameters và settings cho dự án.

---

## 📁 File: `configs/config.yaml`

File này chứa tất cả configuration cho training, model, và evaluation.

---

## 🔧 Các phần cấu hình

### 1. Data Configuration

```yaml
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  target_column: "is_fraud"  # Tên cột target
```

**Giải thích:**
- `raw_path`: Thư mục chứa dữ liệu gốc
- `processed_path`: Thư mục lưu dữ liệu đã xử lý
- `target_column`: Tên cột cần dự đoán (fraud/normal)

---

### 2. Model Configuration

```yaml
model:
  name: "fraud_detection_rf_v1"
  type: "random_forest"  # Options: "logistic", "random_forest", "xgboost"
  save_path: "models"
  
  params:
    n_estimators: 100     # Số lượng trees
    max_depth: 10         # Độ sâu tối đa của tree
    min_samples_split: 5  # Số samples tối thiểu để split node
    class_weight: "balanced"  # Auto-balance classes
    random_state: 42
    n_jobs: -1  # Dùng tất cả CPU cores
```

**Giải thích params:**

| Parameter | Ý nghĩa | Typical Values |
|-----------|---------|----------------|
| `n_estimators` | Số lượng decision trees | 50-200 |
| `max_depth` | Độ sâu max của tree | 5-20 |
| `min_samples_split` | Min samples để split node | 2-10 |
| `min_samples_leaf` | Min samples ở leaf node | 1-5 |
| `class_weight` | Cân bằng classes | `balanced` hoặc `None` |
| `random_state` | Seed cho reproducibility | 42 (convention) |

**Để thay đổi model:**

```yaml
# Logistic Regression
model:
  type: "logistic"
  params:
    max_iter: 1000
    solver: 'lbfgs'

# XGBoost
model:
  type: "xgboost"
  params:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
```

---

### 3. Training Configuration

```yaml
training:
  test_size: 0.2        # 20% cho test set
  validation_size: 0.1  # 10% cho validation
  random_state: 42
  
  cv_folds: 5  # Số folds cho cross-validation
  
  sampling_strategy: "SMOTE"  # Xử lý imbalanced data
```

**Sampling strategies:**

- **`SMOTE`:** Tạo synthetic samples cho minority class
- **`undersampling`:** Giảm majority class
- **`none`:** Không xử lý imbalance

---

### 4. Feature Engineering

```yaml
features:
  create_time_features: true
  create_rolling_features: false
  rolling_windows: [3, 5, 10]
  
  scaling_method: "standard"  # "standard", "minmax", "robust"
```

**Scaling methods:**

- **`standard`:** (x - mean) / std → Mean=0, Std=1
- **`minmax`:** (x - min) / (max - min) → Range [0, 1]
- **`robust`:** Dùng median, robust với outliers

---

### 5. Evaluation Configuration

```yaml
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc"
  
  classification_threshold: 0.5
  
  save_confusion_matrix: true
  save_roc_curve: true
  save_feature_importance: true
```

---

## 🎯 Ví dụ Configurations

### Config 1: Quick Training (Fast)

```yaml
model:
  type: "logistic"
  params:
    max_iter: 500

training:
  test_size: 0.3  # Ít training data
```

**Ưu điểm:** Nhanh (~1 phút)
**Nhược điểm:** Accuracy thấp hơn

---

### Config 2: High Performance (Slow)

```yaml
model:
  type: "random_forest"
  params:
    n_estimators: 200  # Nhiều trees
    max_depth: 15
    min_samples_split: 2

training:
  test_size: 0.2
  sampling_strategy: "SMOTE"  # Xử lý imbalance
```

**Ưu điểm:** Accuracy cao
**Nhược điểm:** Chậm (~10 phút)

---

### Config 3: Balanced (Recommended)

```yaml
model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    class_weight: "balanced"

training:
  test_size: 0.2
  sampling_strategy: "SMOTE"
```

**Ưu điểm:** Cân bằng giữa speed và accuracy
**Training time:** ~5 phút

---

## 🔄 Hyperparameter Tuning

### Manual Tuning

Thử các giá trị khác nhau:

```yaml
# Iteration 1
model:
  params:
    n_estimators: 50
# → F1-Score: 0.75

# Iteration 2
model:
  params:
    n_estimators: 100
# → F1-Score: 0.82 ✅

# Iteration 3
model:
  params:
    n_estimators: 200
# → F1-Score: 0.83 (không cải thiện nhiều, training chậm hơn)
```

**Kết luận:** Dùng `n_estimators=100`

---

### Grid Search (Advanced)

```python
# Trong notebook hoặc script
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

---

## 📊 Monitoring & Logging

```yaml
logging:
  log_dir: "logs"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

Logs được lưu trong `logs/training.log`

---

## ✅ Best Practices

1. **Start simple:** Thử Logistic Regression trước
2. **Baseline first:** Train model đơn giản, sau đó improve
3. **Track experiments:** Ghi lại mỗi config và kết quả
4. **Small data first:** Test với subset nhỏ trước
5. **Cross-validation:** Luôn dùng CV để tránh overfitting

---

## 🎯 Recommended Configs theo Use Case

### Use Case 1: Research/Exploration
```yaml
model: {type: "logistic"}
training: {test_size: 0.3}
```
→ Nhanh, để khám phá data

### Use Case 2: Production Deployment
```yaml
model: {type: "random_forest", params: {n_estimators: 100, class_weight: "balanced"}}
training: {test_size: 0.2, sampling_strategy: "SMOTE"}
```
→ Balanced performance

### Use Case 3: Competition/Maximum Accuracy
```yaml
model: {type: "xgboost", params: {n_estimators: 200, learning_rate: 0.05}}
training: {cv_folds: 10, sampling_strategy: "SMOTE"}
```
→ Chậm nhưng accurate nhất

---

Xem thêm: [TUTORIAL.md](TUTORIAL.md) | [README.md](README.md)
