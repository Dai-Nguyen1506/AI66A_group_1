"""
Data Loader Module
Chức năng: Load dữ liệu từ file CSV nếu đã tồn tại
Lưu ý: Nếu chưa có dữ liệu, chạy lệnh download trước: python src/data/download.py
"""

import pandas as pd
import os
from typing import Optional


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load dữ liệu từ file CSV
    
    Args:
        filepath (str): Đường dẫn tới file CSV
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File không tồn tại: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns from {filepath}")
    return df


def load_fraud_detection_data(data_dir: str = 'data/raw',
                              filename: Optional[str] = None) -> pd.DataFrame:
    """
    Load dữ liệu fraud detection từ thư mục local
    
    Args:
        data_dir (str): Thư mục chứa dữ liệu (mặc định: 'data/raw')
        filename (str, optional): Tên file cụ thể. Nếu None, sẽ load file CSV đầu tiên
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy file hoặc thư mục
    """
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(data_dir):
        print("=" * 70)
        print("❌ LỖI: Thư mục dữ liệu không tồn tại!")
        print("=" * 70)
        print(f"📂 Thư mục: {data_dir}")
        print()
        print("💡 Vui lòng chạy lệnh download để tải dữ liệu:")
        print("   python src/data/download.py")
        print()
        print("   Hoặc từ thư mục gốc:")
        print("   python -m src.data.download")
        print("=" * 70)
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")
    
    # Nếu chỉ định file cụ thể
    if filename:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print("=" * 70)
            print(f"❌ LỖI: File '{filename}' không tồn tại trong {data_dir}")
            print("=" * 70)
            print()
            print("💡 Vui lòng chạy lệnh download để tải dữ liệu:")
            print("   python src/data/download.py")
            print("=" * 70)
            raise FileNotFoundError(f"File không tồn tại: {filepath}")
        
        print(f"📁 Loading from: {filename}")
        return load_csv(filepath)
    
    # Tìm file CSV trong thư mục
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("=" * 70)
        print(f"❌ LỖI: Không tìm thấy file CSV trong thư mục {data_dir}")
        print("=" * 70)
        print()
        print("💡 Vui lòng chạy lệnh download để tải dữ liệu:")
        print("   python src/data/download.py")
        print()
        print("   Hoặc từ thư mục gốc:")
        print("   python -m src.data.download")
        print("=" * 70)
        raise FileNotFoundError(f"Không có file CSV trong {data_dir}")
    
    # Load file CSV đầu tiên
    filepath = os.path.join(data_dir, csv_files[0])
    print(f"📁 Loading from local: {csv_files[0]}")
    return load_csv(filepath)


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Lưu dữ liệu đã xử lý
    
    Args:
        df (pd.DataFrame): DataFrame cần lưu
        filepath (str): Đường dẫn lưu file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df):,} rows to {filepath}")


def get_data_info(df: pd.DataFrame) -> None:
    """
    In thông tin tổng quan về dataset
    
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print("=" * 60)
