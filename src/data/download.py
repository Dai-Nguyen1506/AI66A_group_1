import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional

# Thêm đường dẫn gốc để có thể import khi chạy script trực tiếp
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.data.loader import save_processed_data
else:
    from .loader import save_processed_data

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
        
        # Optionally save to local
        if save_to:
            save_processed_data(df, save_to)
        
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
    (Credit Card Fraud Detection Dataset)
    """
    print("=" * 60)
    print("🚀 CREDIT CARD FRAUD DETECTION - DATA DOWNLOADER")
    print("=" * 60)
    print()
    
    # Xác định đường dẫn lưu trữ
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
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
        print("=" * 60)
        print("✨ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📂 Data location: {raw_data_dir}")
        print()
        print("Next steps:")
        print("  1. Run EDA notebook: notebooks/01_eda_exploration.ipynb")
        print("  2. Or start training: python scripts/train.py")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        print()
        print("💡 Troubleshooting:")
        print("  1. Make sure you have Kaggle API configured")
        print("  2. Run: pip install kagglehub")
        print("  3. Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    """
    Chạy script này để tự động download dữ liệu:
    
    Usage:
        python src/data/download.py
    
    Or from project root:
        python -m src.data.download
    """
    download_default_dataset()