"""
Data Preprocessing Module
Chức năng: Làm sạch dữ liệu, xử lý missing values, outliers
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra missing values trong dataset
    
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra
        
    Returns:
        pd.DataFrame: Thống kê missing values
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percent': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    return missing[missing['missing_count'] > 0]


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Xử lý missing values
    
    Args:
        df (pd.DataFrame): DataFrame cần xử lý
        strategy (str): 'drop', 'mean', 'median', 'mode'
        
    Returns:
        pd.DataFrame: DataFrame đã xử lý
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    print(f"✓ Handled missing values using '{strategy}' strategy")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các dòng duplicate
    
    Args:
        df (pd.DataFrame): DataFrame cần xử lý
        
    Returns:
        pd.DataFrame: DataFrame đã loại bỏ duplicates
    """
    original_len = len(df)
    df_clean = df.drop_duplicates()
    removed = original_len - len(df_clean)
    
    if removed > 0:
        print(f"✓ Removed {removed} duplicate rows")
    
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Phát hiện outliers sử dụng IQR method
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Tên cột cần kiểm tra
        threshold (float): IQR threshold (default: 1.5)
        
    Returns:
        pd.Series: Boolean mask của outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    print(f"✓ Found {outliers.sum()} outliers in column '{column}'")
    
    return outliers
