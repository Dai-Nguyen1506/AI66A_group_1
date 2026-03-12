# 📚 HƯỚNG DẪN CHI TIẾT - Credit Card Fraud Detection

**AI66A - Group 1**

> Hướng dẫn từng bước chi tiết để mọi người đều hiểu và có thể chạy được dự án Machine Learning phát hiện gian lận thẻ tín dụng.

---

## 📖 MỤC LỤC

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
3. [Cài đặt từng bước](#3-cài-đặt-từng-bước)
4. [Hiểu về cấu trúc dự án](#4-hiểu-về-cấu-trúc-dự-án)
5. [Chạy dự án từng bước](#5-chạy-dự-án-từng-bước)
6. [Giải thích code chi tiết](#6-giải-thích-code-chi-tiết)
7. [Troubleshooting](#7-troubleshooting)
8. [FAQs](#8-faqs)

---

## 1. GIỚI THIỆU DỰ ÁN

### 🎯 Dự án này làm gì?

Dự án sử dụng Machine Learning để **tự động phát hiện giao dịch thẻ tín dụng gian lận**.

**Ví dụ thực tế:**
- Bạn có 10,000 giao dịch thẻ tín dụng
- Trong đó có 50 giao dịch là gian lận (0.5%)
- Model sẽ học cách nhận biết pattern của giao dịch gian lận
- Khi có giao dịch mới, model sẽ dự đoán: "Gian lận" hay "Bình thường"

### 🔍 Tại sao quan trọng?

- **Ngân hàng:** Giảm thiệt hại do gian lận (hàng tỷ đô mỗi năm)
- **Khách hàng:** Bảo vệ tài khoản, tránh mất tiền
- **Học ML:** Bài toán thực tế với dữ liệu imbalanced

### 📊 Dataset

Chúng ta sử dụng dataset từ Kaggle:
- **Link:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Số lượng:** ~100,000 - 1,000,000 giao dịch
- **Features:** Thông tin giao dịch (amount, time, merchant, category, v.v.)
- **Target:** `is_fraud` (0 = Normal, 1 = Fraud)

---

## 2. CHUẨN BỊ MÔI TRƯỜNG

### 💻 Yêu cầu hệ thống

- **OS:** Windows, macOS, hoặc Linux
- **Python:** 3.8 hoặc mới hơn
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk:** ~500MB cho code và libraries

### ✅ Kiểm tra Python

Mở terminal/command prompt và chạy:

```bash
python --version
```

**Kết quả mong đợi:**
```
Python 3.9.x (hoặc 3.8+)
```

**Nếu không có Python:**
1. Download từ: https://www.python.org/downloads/
2. Cài đặt (nhớ check "Add Python to PATH")
3. Restart terminal và kiểm tra lại

### 📦 Cài đặt Git (Optional)

Nếu muốn clone từ GitHub:

```bash
git --version
```

Download tại: https://git-scm.com/downloads

---

## 3. CÀI ĐẶT TỪNG BƯỚC

### Bước 1: Tải dự án

**Cách 1: Download ZIP**
1. Vào GitHub repository
2. Click "Code" → "Download ZIP"
3. Giải nén vào thư mục bạn muốn

**Cách 2: Clone với Git**
```bash
git clone https://github.com/your-username/AI66A_group_1.git
cd AI66A_group_1
```

### Bước 2: Tạo Virtual Environment (Khuyến nghị)

**Tại sao cần virtual environment?**
- Tránh conflict giữa các dự án
- Quản lý dependencies dễ dàng
- Không ảnh hưởng đến Python system

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Kiểm tra đã activate chưa:**
- Terminal sẽ hiện `(venv)` ở đầu dòng
- Ví dụ: `(venv) C:\Users\YOGA\...\AI66A_group_1>`

### Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Giải thích:**
- `requirements.txt` chứa list tất cả libraries cần thiết
- `pip install` sẽ tự động download và cài đặt

**Thời gian:** 2-5 phút (tùy internet)

**Kiểm tra cài đặt thành công:**
```bash
pip list
```

Bạn sẽ thấy: pandas, numpy, scikit-learn, kagglehub, v.v.

### Bước 4: Cài đặt Kaggle CLI (Để tải dataset)

```bash
pip install kagglehub
```

**Cấu hình Kaggle API:**

1. Đăng nhập vào Kaggle: https://www.kaggle.com
2. Vào "Account" → "API" → Click "Create New API Token"
3. File `kaggle.json` sẽ được download
4. Đặt file vào:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

5. Phân quyền (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 4. HIỂU VỀ CẤU TRÚC DỰ ÁN

### 📂 Cấu trúc thư mục

```
AI66A_group_1/
│
├── 📂 data/                    # Dữ liệu
│   ├── raw/                    # Data gốc (tải từ Kaggle)
│   └── processed/              # Data đã xử lý
│
├── 📂 notebooks/               # Jupyter Notebooks
│   ├── 01_eda_exploration.ipynb      # Khám phá dữ liệu
│   ├── 02_feature_engineering.ipynb  # Tạo features mới
│   └── 03_model_training.ipynb       # Training models
│
├── 📂 src/                     # Source code (Module hóa)
│   ├── data/                   # Xử lý dữ liệu
│   ├── features/               # Feature engineering
│   ├── models/                 # Training & evaluation
│   ├── utils/                  # Utilities
│   └── visualization/          # Vẽ biểu đồ
│
├── 📂 scripts/                 # Scripts chạy nhanh
│   ├── train.py                # Chạy training
│   ├── predict.py              # Chạy prediction
│   └── evaluate.py             # Đánh giá model
│
├── 📂 models/                  # Lưu trained models
├── 📂 reports/                 # Báo cáo, biểu đồ
├── 📂 tests/                   # Unit tests
└── 📂 configs/                 # Cấu hình
    └── config.yaml             # Hyperparameters
```

### 🔑 File quan trọng

| File | Mục đích |
|------|----------|
| `scripts/train.py` | Chạy toàn bộ pipeline training |
| `configs/config.yaml` | Cấu hình hyperparameters |
| `requirements.txt` | Danh sách libraries |
| `README.md` | Documentation tổng quan |
| `TUTORIAL.md` | File này - hướng dẫn chi tiết |

---

## 5. CHẠY DỰ ÁN TỪNG BƯỚC

### 🚀 Phương pháp 1: Chạy Script (Nhanh nhất)

#### Bước 1: Training Model

```bash
python scripts/train.py
```

**Quá trình sẽ thực hiện:**

1. ✅ Load dữ liệu từ Kaggle (tự động download)
2. ✅ Làm sạch dữ liệu (xóa duplicates, missing values)
3. ✅ Tạo features mới
4. ✅ Chia train/test set
5. ✅ Training model (Random Forest)
6. ✅ Đánh giá model
7. ✅ Lưu model vào `models/`

**Thời gian:** 5-15 phút (tùy dữ liệu và CPU)

**Output:**
```
==============================================================
  CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE
  AI66A - Group 1
==============================================================

[1/8] LOADING DATA
📥 Downloading dataset 'kartik2112/fraud-detection' from Kaggle...
✓ Loaded 1,000,000 rows from Kaggle

...

[8/8] EVALUATING MODEL
==============================================================
Accuracy:  0.9950
Precision: 0.8523
Recall:    0.7821
F1-Score:  0.8157
ROC-AUC:   0.9845
==============================================================

✅ TRAINING COMPLETED SUCCESSFULLY!
```

#### Bước 2: Đánh giá Model

```bash
python scripts/evaluate.py
```

#### Bước 3: Dự đoán mới

```bash
python scripts/predict.py
```

### 📊 Phương pháp 2: Chạy Notebooks (Chi tiết hơn)

#### Bước 1: Mở Jupyter Notebook

```bash
jupyter notebook
```

Browser sẽ tự động mở (http://localhost:8888)

#### Bước 2: Chạy từng notebook

**Notebook 1: EDA (Exploratory Data Analysis)**
```
notebooks/01_eda_exploration.ipynb
```

**Làm gì:**
- Load dữ liệu từ Kaggle
- Khám phá cấu trúc dữ liệu
- Phân tích missing values, duplicates
- Visualize phân bổ fraud/normal
- Tìm insights từ data

**Cách chạy:**
1. Click vào notebook
2. Click "Cell" → "Run All"
3. Hoặc nhấn `Shift + Enter` từng cell

**Notebook 2: Feature Engineering**
```
notebooks/02_feature_engineering.ipynb
```

**Làm gì:**
- Tạo features mới từ time, amount
- Scaling/normalization
- Feature selection

**Notebook 3: Model Training**
```
notebooks/03_model_training.ipynb
```

**Làm gì:**
- Thử nhiều models (Logistic, Random Forest, XGBoost)
- So sánh performance
- Hyperparameter tuning
- Chọn model tốt nhất

---

## 6. GIẢI THÍCH CODE CHI TIẾT

### 📝 Hiểu file `train.py`

```python
# 1. Import libraries
import pandas as pd
from src.data.loader import load_from_kagglehub

# 2. Load data
df = load_from_kagglehub("kartik2112/fraud-detection")
```

**Giải thích:**
- `pandas`: Library xử lý data dạng bảng
- `load_from_kagglehub`: Function tự viết để download từ Kaggle
- `df`: DataFrame chứa toàn bộ dữ liệu

```python
# 3. Preprocessing
from src.data.preprocessing import remove_duplicates, handle_missing_values

df = remove_duplicates(df)              # Xóa dòng trùng lặp
df = handle_missing_values(df, 'drop')  # Xóa dòng có missing
```

**Giải thích:**
- **Duplicates:** Dữ liệu trùng lặp gây overfitting
- **Missing values:** Dữ liệu thiếu phải xử lý (xóa hoặc fill)

```python
# 4. Split features và target
X = df.drop('is_fraud', axis=1)  # Features (tất cả cột trừ target)
y = df['is_fraud']               # Target (0 hoặc 1)
```

**Giải thích:**
- **X:** Input cho model (amount, time, merchant, v.v.)
- **y:** Output mà model cần dự đoán (fraud hay không)

```python
# 5. Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Giải thích:**
- Train 80% dữ liệu, test 20%
- `random_state=42`: Để kết quả reproducible

```python
# 6. Training model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

**Giải thích:**
- **Random Forest:** Thuật toán ensemble (nhiều decision trees)
- `n_estimators=100`: Số lượng trees
- `fit()`: Quá trình training

```python
# 7. Prediction
y_pred = model.predict(X_test)
```

**Giải thích:**
- Dự đoán trên test set
- `y_pred`: Array chứa predictions (0 hoặc 1)

```python
# 8. Evaluation
from sklearn.metrics import accuracy_score, precision_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
```

**Giải thích metrics:**

- **Accuracy:** Tỉ lệ dự đoán đúng tổng thể
  - Formula: (TP + TN) / Total
  - Ví dụ: 99.5% → 995/1000 predictions đúng

- **Precision:** Trong những cái dự đoán là Fraud, bao nhiêu % thực sự là Fraud
  - Formula: TP / (TP + FP)
  - Ví dụ: 85% → 85/100 dự đoán fraud là đúng

- **Recall:** Trong tất cả Fraud thực tế, model bắt được bao nhiêu %
  - Formula: TP / (TP + FN)
  - Ví dụ: 75% → Bắt được 75/100 fraud

- **F1-Score:** Trung bình hài hòa của Precision và Recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)

**Confusion Matrix:**
```
                Predicted
                0       1
Actual    0    TN      FP
          1    FN      TP
```

- **TP (True Positive):** Dự đoán Fraud, thực tế Fraud ✅
- **TN (True Negative):** Dự đoán Normal, thực tế Normal ✅
- **FP (False Positive):** Dự đoán Fraud, thực tế Normal ❌ (False alarm)
- **FN (False Negative):** Dự đoán Normal, thực tế Fraud ❌ (Nguy hiểm!)

---

## 7. TROUBLESHOOTING

### ❌ Lỗi: "Module not found"

**Nguyên nhân:** Chưa cài đặt library hoặc chưa activate venv

**Giải pháp:**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Cài lại dependencies
pip install -r requirements.txt
```

### ❌ Lỗi: "kagglehub authentication failed"

**Nguyên nhân:** Chưa setup Kaggle API token

**Giải pháp:**
1. Tải `kaggle.json` từ Kaggle account
2. Đặt vào `~/.kaggle/kaggle.json` (Linux/Mac) hoặc `C:\Users\<User>\.kaggle\kaggle.json` (Windows)
3. Restart terminal

### ❌ Lỗi: "No such file or directory"

**Nguyên nhân:** Đang ở sai thư mục

**Giải pháp:**
```bash
# Kiểm tra thư mục hiện tại
pwd  # Mac/Linux
cd  # Windows

# Di chuyển đến project root
cd path/to/AI66A_group_1
```

### ❌ Lỗi: "Memory Error" / "Killed"

**Nguyên nhân:** RAM không đủ

**Giải pháp:**
- Giảm dataset size (sample nhỏ hơn)
- Giảm `n_estimators` trong config
- Dùng máy có RAM lớn hơn

### ❌ Model training quá lâu

**Giải pháp:**
1. Giảm số lượng data:
   ```python
   df = df.sample(n=100000)  # Chỉ lấy 100k rows
   ```

2. Giảm hyperparameters trong `config.yaml`:
   ```yaml
   model:
     params:
       n_estimators: 50  # Giảm từ 100 xuống 50
       max_depth: 5      # Giảm depth
   ```

3. Dùng model đơn giản hơn:
   ```yaml
   model:
     type: "logistic"  # Thay vì "random_forest"
   ```

---

## 8. FAQs

### ❓ Tôi không biết Python, có chạy được không?

**Trả lời:** Được! Chỉ cần làm theo từng bước trong tutorial này.

**Khuyến nghị:**
- Học Python cơ bản: https://www.w3schools.com/python/
- Học Pandas: https://pandas.pydata.org/docs/getting_started/intro_tutorials/

### ❓ Tôi muốn thay đổi hyperparameters?

**Trả lời:** Chỉnh sửa file `configs/config.yaml`

```yaml
model:
  type: "random_forest"  # Hoặc "logistic", "xgboost"
  params:
    n_estimators: 200    # Tăng số trees
    max_depth: 15        # Tăng depth
    min_samples_split: 10
```

### ❓ Tôi muốn dùng dataset khác?

**Trả lời:** Có 2 cách:

**Cách 1:** Đặt file CSV vào `data/raw/`
```python
# Trong train.py
df = pd.read_csv('data/raw/your_data.csv')
```

**Cách 2:** Dùng dataset khác từ Kaggle
```python
df = load_from_kagglehub("username/dataset-name")
```

### ❓ Làm sao biết model tốt hay xấu?

**Trả lời:** Xem các metrics:

**Tốt:**
- Precision > 80%
- Recall > 70%
- F1-Score > 75%
- ROC-AUC > 0.90

**Xấu:**
- Accuracy cao (>99%) nhưng Recall thấp (<50%)
  → Model chỉ dự đoán "Normal" toàn bộ!

**Cải thiện:**
- Thử model khác (XGBoost, LightGBM)
- Feature engineering tốt hơn
- Xử lý imbalanced data (SMOTE)
- Hyperparameter tuning

### ❓ Tôi muốn deploy model lên web?

**Trả lời:** Có thể dùng Flask/FastAPI:

```python
from flask import Flask, request
import joblib

app = Flask(__name__)
model = joblib.load('models/fraud_detection_rf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return {'is_fraud': int(prediction[0])}

if __name__ == '__main__':
    app.run()
```

Xem thêm: https://flask.palletsprojects.com/

---

## 📚 TÀI LIỆU THAM KHẢO

### 📖 Official Documentation
- **Pandas:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Matplotlib:** https://matplotlib.org/stable/contents.html

### 🎓 Học Machine Learning
- **Coursera - Andrew Ng:** https://www.coursera.org/learn/machine-learning
- **Google ML Crash Course:** https://developers.google.com/machine-learning/crash-course
- **Kaggle Learn:** https://www.kaggle.com/learn

### 📊 Fraud Detection Papers
- Credit Card Fraud Detection: A Realistic Modeling
- Handling Imbalanced Datasets in Machine Learning
- Ensemble Methods for Fraud Detection

---

## 🎯 KẾT LUẬN

Giờ bạn đã hiểu:
- ✅ Dự án làm gì
- ✅ Cách setup môi trường
- ✅ Cách chạy code
- ✅ Code hoạt động như thế nào
- ✅ Cách debug khi có lỗi

**Next Steps:**
1. Chạy thử `python scripts/train.py`
2. Khám phá notebooks
3. Thử điều chỉnh hyperparameters
4. Thử models khác nhau
5. Deploy model lên web!

**Chúc bạn thành công! 🚀**

---

## 📞 LIÊN HỆ & HỖ TRỢ

- **GitHub Issues:** [Link to issues]
- **Email:** your-email@example.com
- **Team:** AI66A - Group 1

**Nếu gặp vấn đề, đừng ngại hỏi!**

---

**Created with ❤️ by AI66A Group 1**
