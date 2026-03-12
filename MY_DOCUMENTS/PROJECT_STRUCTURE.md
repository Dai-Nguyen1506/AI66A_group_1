# Cấu trúc dự án Machine Learning

## 📁 Tổng quan cấu trúc thư mục

```
AI66A_group_1/
│
├── data/                      # Dữ liệu
│   ├── raw/                   # Dữ liệu thô chưa xử lý
│   ├── processed/             # Dữ liệu đã xử lý
│   └── external/              # Dữ liệu từ nguồn bên ngoài
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/           # EDA và phân tích khám phá
│   └── experiments/           # Thử nghiệm models
│
├── src/                       # Source code chính
│   ├── data/                  # Scripts xử lý dữ liệu
│   ├── features/              # Feature engineering
│   ├── models/                # Định nghĩa models
│   ├── utils/                 # Utility functions
│   └── visualization/         # Visualizations
│
├── models/                    # Trained models (*.pkl, *.h5, *.pt)
│
├── tests/                     # Unit tests
│
├── configs/                   # Configuration files
│   └── config.yaml            # File config chính
│
├── scripts/                   # Scripts chạy training/prediction
│   ├── train.py               # Training pipeline
│   └── predict.py             # Prediction pipeline
│
├── docs/                      # Documentation
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # Project README

```

## 📖 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Workflow chuẩn

1. **Thu thập dữ liệu**: Đặt vào `data/raw/`
2. **Khám phá dữ liệu**: Sử dụng notebooks trong `notebooks/exploratory/`
3. **Xử lý dữ liệu**: Viết code trong `src/data/`, output vào `data/processed/`
4. **Feature engineering**: Code trong `src/features/`
5. **Training model**: Viết model trong `src/models/`, chạy `scripts/train.py`
6. **Lưu model**: Trained models vào `models/`
7. **Testing**: Viết tests trong `tests/`
8. **Prediction**: Sử dụng `scripts/predict.py`

## 🎯 Best Practices

- Luôn version control code, không commit data và models lớn
- Sử dụng config files cho hyperparameters
- Viết docstrings cho functions
- Viết unit tests cho các functions quan trọng
- Đặt tên files và functions rõ ràng, dễ hiểu
