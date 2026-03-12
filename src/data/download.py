"""
Data Download Module
Chức năng: Tải dữ liệu từ Kaggle và lưu vào data/raw
Sử dụng: Chạy script này lần đầu tiên sau khi cài đặt dependencies

Usage:
    python src/data/download.py
    
Hoặc:
    python -m src.data.download
"""

import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional


def save_data_local(df: pd.DataFrame, filepath: str) -> None:
    """
    Lưu dữ liệu vào file CSV
    
    Args:
        df (pd.DataFrame): DataFrame cần lưu
        filepath (str): Đường dẫn lưu file
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Lưu file
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df):,} rows to {filepath}")


def load_from_kagglehub(dataset_name: str = "kartik2112/fraud-detection",
                        file_path: str = "",
                        save_to: Optional[str] = None) -> pd.DataFrame:
    """
    Load dữ liệu từ Kaggle sử dụng kagglehub
    
    Args:
        dataset_name (str): Tên dataset trên Kaggle (format: username/dataset)
        file_path (str): Đường dẫn file trong dataset (để trống nếu chỉ có 1 file)
        save_to (str, optional): Đường dẫn để lưu file CSV sau khi download
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu
        
    Example:
        >>> df = load_from_kagglehub("kartik2112/fraud-detection", "fraudTrain.csv")
    """
    try:
        import kagglehub
        
        print(f"📥 Downloading dataset '{dataset_name}' from Kaggle...")
        
        # Download dataset và lấy đường dẫn thư mục
        dataset_path = kagglehub.dataset_download(dataset_name)
        
        # Tìm file CSV trong thư mục download
        if file_path:
            full_path = os.path.join(dataset_path, file_path)
        else:
            # Tự động tìm file CSV đầu tiên
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("Không tìm thấy file CSV trong dataset")
            full_path = os.path.join(dataset_path, csv_files[0])
        
        print(f"📖 Reading file: {os.path.basename(full_path)}")
        
        # Thử đọc với nhiều encoding khác nhau
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(full_path, encoding=encoding)
                print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Không thể đọc file với các encoding thông dụng")
        
        # Lưu vào thư mục local nếu được chỉ định
        if save_to:
            save_data_local(df, save_to)
        
        return df
        
    except ImportError:
        print("❌ kagglehub chưa được cài đặt!")
        print("Cài đặt bằng lệnh: pip install kagglehub")
        raise
    except Exception as e:
        print(f"❌ Lỗi khi load từ Kaggle: {e}")
        raise


def download_default_dataset():
    """
    Tự động download dataset mặc định cho project
    (Credit Card Fraud Detection Dataset từ Kaggle)
    
    Dataset sẽ được lưu vào thư mục data/raw/
    """
    print("=" * 70)
    print("🚀 CREDIT CARD FRAUD DETECTION - DATA DOWNLOADER")
    print("=" * 70)
    print("📦 Dataset: kartik2112/fraud-detection")
    print("🌐 Source: Kaggle")
    print()
    
    # Xác định đường dẫn lưu trữ
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Save location: {raw_data_dir}")
    print()
    
    # Download training data
    train_file = raw_data_dir / "fraudTrain.csv"
    test_file = raw_data_dir / "fraudTest.csv"
    
    try:
        # Download fraudTrain.csv
        if not train_file.exists():
            print("📥 Downloading TRAINING dataset...")
            df_train = load_from_kagglehub(
                dataset_name="kartik2112/fraud-detection",
                file_path="fraudTrain.csv",
                save_to=str(train_file)
            )
            print(f"✅ Training data saved to: {train_file}")
            print(f"   Shape: {df_train.shape}")
        else:
            print(f"⏭️  Training data already exists: {train_file}")
        
        print()
        
        # Download fraudTest.csv
        if not test_file.exists():
            print("📥 Downloading TEST dataset...")
            df_test = load_from_kagglehub(
                dataset_name="kartik2112/fraud-detection",
                file_path="fraudTest.csv",
                save_to=str(test_file)
            )
            print(f"✅ Test data saved to: {test_file}")
            print(f"   Shape: {df_test.shape}")
        else:
            print(f"⏭️  Test data already exists: {test_file}")
        
        print()
        print("=" * 70)
        print("✨ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📂 Data location: {raw_data_dir}")
        print(f"   • fraudTrain.csv")
        print(f"   • fraudTest.csv")
        print()
        print("📋 Next steps:")
        print("   1. Load data trong scripts: from src.data.loader import load_fraud_detection_data")
        print("   2. Chạy EDA notebook: notebooks/01_eda_exploration.ipynb")
        print("   3. Hoặc bắt đầu training: python scripts/train.py")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ ERROR: {e}")
        print("=" * 70)
        print()
        print("💡 Troubleshooting:")
        print("   1. Cài đặt kagglehub: pip install kagglehub")
        print("   2. Kiểm tra Kaggle API credentials")
        print("   3. Kiểm tra kết nối internet")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    """
    Script khởi chạy để download dữ liệu lần đầu tiên
    
    Chạy sau khi cài đặt dependencies:
        pip install -r requirements.txt
        
    Sau đó chạy:
        python src/data/download.py
    
    Hoặc từ thư mục gốc:
        python -m src.data.download
    """
    download_default_dataset()