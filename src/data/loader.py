"""
Data Loader Module
Chức năng: Load dữ liệu từ CSV, Kaggle, database, hoặc các nguồn khác
"""

import pandas as pd
import os
from typing import Optional, Union


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

def load_fraud_detection_data(source: str = 'local',
                              data_dir: str = 'data/raw',
                              dataset_name: str = "kartik2112/fraud-detection") -> pd.DataFrame:
    """
    Load dữ liệu fraud detection từ nhiều nguồn
    
    Args:
        source (str): Nguồn dữ liệu - 'local', 'kaggle', hoặc 'kagglehub'
        data_dir (str): Thư mục chứa dữ liệu local
        dataset_name (str): Tên dataset trên Kaggle
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu
    """
    if source == 'local':
        # Thử tìm file CSV trong thư mục
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"⚠ Không tìm thấy file CSV trong {data_dir}")
            print("💡 Thử load từ Kaggle bằng cách đổi source='kagglehub'")
            raise FileNotFoundError(f"Không có file CSV trong {data_dir}")
        
        filepath = os.path.join(data_dir, csv_files[0])
        print(f"📁 Loading from local: {csv_files[0]}")
        return load_csv(filepath)
    
    elif source in ['kaggle', 'kagglehub']:
        save_path = os.path.join(data_dir, 'fraud_data.csv')
        return load_from_kagglehub(dataset_name, save_to=save_path)
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'local' or 'kagglehub'")


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


# def get_data_info(df: pd.DataFrame) -> None:
#     """
#     In thông tin tổng quan về dataset
    
#     Args:
#         df (pd.DataFrame): DataFrame cần kiểm tra
#     """
#     print("=" * 60)
#     print("DATASET INFORMATION")
#     print("=" * 60)
#     print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
#     print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
#     print(f"\nColumn types:")
#     print(df.dtypes.value_counts())
#     print(f"\nMissing values: {df.isnull().sum().sum()}")
#     print("=" * 60)
