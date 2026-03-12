# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu với dự án Credit Card Fraud Detection.

---

## ⚡ Setup nhanh (5 phút)

### 1. Clone & Install
```bash
# Clone repo
git clone https://github.com/your-username/AI66A_group_1.git
cd AI66A_group_1

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Tải Dataset

**Cách 1: Tự động (Khuyến nghị) ✨**
```bash
# Windows
setup_data.bat

# Linux/Mac
bash setup_data.sh

# Hoặc chạy trực tiếp script Python
python src/data/download.py
```

**Cách 2: Thủ công**
```bash
# Tải từ Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection
# Đặt file fraudTrain.csv và fraudTest.csv vào data/raw/
```

### 3. Chạy EDA (Exploratory Data Analysis)
```bash
jupyter notebook notebooks/01_eda_exploration.ipynb
```

### 4. Training Model
```bash
python scripts/train.py
```

### 5. Evaluation
```bash
python scripts/evaluate.py
```

### 6. Prediction
```bash
python scripts/predict.py
```

---

## 📂 Cấu trúc thư mục quan trọng

```
├── data/raw/              ← Đặt creditcard.csv vào đây
├── notebooks/             ← Jupyter notebooks để EDA
├── src/                   ← Source code (modules)
├── scripts/               ← Scripts để chạy (train, predict, evaluate)
├── configs/config.yaml    ← Cấu hình hyperparameters
└── requirements.txt       ← Dependencies
```

---

## 🔧 Customize Configuration

Chỉnh sửa `configs/config.yaml` để thay đổi:

```yaml
model:
  type: "random_forest"  # Thay thành "logistic", "xgboost"
  params:
    n_estimators: 100    # Số lượng trees
    max_depth: 10        # Độ sâu tối đa

training:
  test_size: 0.2         # Tỉ lệ test set
  sampling_strategy: "SMOTE"  # Xử lý imbalanced data
```

---

## 📊 Workflow chuẩn

```
1. EDA
   ↓
2. Feature Engineering  
   ↓
3. Train Model
   ↓
4. Evaluate
   ↓
5. Predict
```

---

## 💡 Tips

### Run tests
```bash
pytest tests/ -v
```

### View logs
```bash
# Logs được lưu trong logs/training.log
cat logs/training.log
```

### Save model với tên khác
Chỉnh sửa trong `configs/config.yaml`:
```yaml
model:
  name: "fraud_detection_xgb_v2"
```

---

## ❓ Troubleshooting

### Lỗi: Module not found
```bash
# Đảm bảo đang ở root directory của project
cd AI66A_group_1

# Thử install lại dependencies
pip install -r requirements.txt
```

### Lỗi: File not found (creditcard.csv)
```bash
# Đảm bảo file data đã được đặt đúng chỗ:
# data/raw/creditcard.csv
```

### Model training chậm
```bash
# Giảm số lượng trees trong config.yaml
n_estimators: 50  # thay vì 100
```

---

## 📚 Resources

- [Full Documentation](README.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

Bắt đầu thôi! 🎉
