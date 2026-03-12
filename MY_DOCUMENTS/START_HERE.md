# 🌟 HƯỚNG DẪN BẮT ĐẦU NHANH - 5 PHÚT

> **Dành cho người mới:** Chưa biết Python? Chưa biết ML? Không sao! Làm theo đây!

---

## 🎯 Mục tiêu

Chạy được model Machine Learning phát hiện gian lận thẻ tín dụng trong **5 phút**!

---

## ⚡ 3 BƯỚC ĐƠN GIẢN

### Bước 1: Cài đặt (2 phút)

**1.1 Tải Python** (Nếu chưa có)
- Link: https://www.python.org/downloads/
- Tải và cài đặt (nhớ check "Add to PATH")

**1.2 Tải dự án**
```bash
# Download ZIP từ GitHub hoặc:
git clone https://github.com/your-username/AI66A_group_1.git
cd AI66A_group_1
```

**1.3 Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

⏰ **Thời gian:** ~2 phút

---

### Bước 2: Setup Kaggle (1 phút)

**Để tải dataset, cần Kaggle API token:**

1. Vào: https://www.kaggle.com → Đăng nhập
2. Click **Account** (góc phải)
3. Scroll xuống **API** → Click **"Create New API Token"**
4. File `kaggle.json` sẽ download
5. Đặt vào:
   - **Windows:** `C:\Users\<TênBạn>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

💡 **Tip:** Tạo thư mục `.kaggle` nếu chưa có.

⏰ **Thời gian:** ~1 phút

---

### Bước 3: Chạy! (2 phút)

```bash
python scripts/train.py
```

**Chờ kết quả:**
```
==============================================================
  CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE
==============================================================

[1/8] LOADING DATA
📥 Downloading from Kaggle...
✓ Loaded 1,000,000 rows from Kaggle

[2/8] PREPROCESSING DATA
✓ Data cleaned: 1,000,000 → 998,500 rows

[3/8] IDENTIFYING TARGET COLUMN
✓ Target column found: 'is_fraud'

[4/8] ANALYZING CLASS DISTRIBUTION
Class distribution:
  Class 0: 990,000 (99.05%)
  Class 1: 9,500 (0.95%)

[5/8] FEATURE ENGINEERING
✓ Selected 22 numeric features
✓ Scaler saved to models/scaler.pkl

[6/8] SPLITTING TRAIN/TEST
✓ Data split: Train=798,800, Test=199,700

[7/8] TRAINING MODEL
Training random_forest model...
✓ Training completed!
✓ Model saved to models/fraud_detection_rf.pkl

[8/8] EVALUATING MODEL
==============================================================
Accuracy:  0.9950
Precision: 0.8523
Recall:    0.7821
F1-Score:  0.8157
ROC-AUC:   0.9845
==============================================================

✅ TRAINING COMPLETED SUCCESSFULLY!

📁 Outputs:
   ├─ Model: models/fraud_detection_rf.pkl
   ├─ Scaler: models/scaler.pkl
   ├─ Processed data: data/processed/
   └─ Reports: reports/

🚀 Next step: Run 'python scripts/evaluate.py'
==============================================================
```

⏰ **Thời gian:** ~2-5 phút (tùy máy)

---

## ✅ XONG! Giờ bạn đã có:

- ✅ Model đã train: `models/fraud_detection_rf.pkl`
- ✅ Confusion Matrix: `reports/confusion_matrix.png`
- ✅ ROC Curve: `reports/roc_curve.png`
- ✅ Processed data: `data/processed/`

---

## 🎓 Hiểu code (Optional)

### Model làm gì?

**Input:** Thông tin giao dịch (số tiền, thời gian, merchant, v.v.)
```python
transaction = {
    'amt': 100.50,
    'unix_time': 1234567890,
    'merchant': 'Amazon',
    ...
}
```

**Output:** Dự đoán Fraud (1) hay Normal (0)
```python
prediction = model.predict([transaction])
# → 0 (Normal) hoặc 1 (Fraud)
```

### Metrics nghĩa là gì?

- **Accuracy (99.5%):** Tổng thể đúng bao nhiêu %
- **Precision (85.2%):** Khi dự đoán Fraud, đúng 85.2%
- **Recall (78.2%):** Bắt được 78.2% tất cả Fraud
- **F1-Score (81.6%):** Trung bình hài hòa Precision/Recall
- **ROC-AUC (98.5%):** Khả năng phân biệt tốt/xấu

**Càng cao càng tốt!**

---

## 📚 Muốn học thêm?

1. **Chi tiết đầy đủ:** [TUTORIAL.md](TUTORIAL.md) - Hướng dẫn từng bước
2. **Notebooks:** `notebooks/01_eda_exploration.ipynb` - Khám phá dữ liệu
3. **Cấu trúc:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Giải thích thư mục
4. **Full docs:** [README.md](README.md) - Documentation đầy đủ

---

## ❓ Gặp lỗi?

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Kaggle authentication failed"
- Check file `kaggle.json` đã đúng vị trí chưa
- Xem lại Bước 2

### "Memory Error"
- Giảm data: Mở `train.py`, thêm: `df = df.sample(n=100000)`
- Hoặc giảm `n_estimators` trong `config.yaml`

**Xem thêm:** [TUTORIAL.md](TUTORIAL.md) - Phần Troubleshooting

---

## 🚀 Next Steps

1. ✅ Thử prediction: `python scripts/predict.py`
2. ✅ Xem evaluation: `python scripts/evaluate.py`
3. ✅ Chạy notebooks để học chi tiết
4. ✅ Thử models khác: Sửa `config.yaml` → `type: "xgboost"`
5. ✅ Deploy lên web!

---

## 📞 Cần giúp?

- **GitHub Issues:** [Link]
- **Email:** your-email@example.com
- **Tutorial chi tiết:** [TUTORIAL.md](TUTORIAL.md)

---

**Chúc bạn thành công! 🎉**

---

**AI66A - Group 1** | [Full README](README.md) | [Tutorial](TUTORIAL.md)
