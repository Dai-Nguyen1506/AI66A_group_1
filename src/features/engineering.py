"""
Feature Engineering Module
Chức năng: Tạo các features mới từ dữ liệu gốc
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple


def create_time_features(df: pd.DataFrame, time_column: str = 'Time') -> pd.DataFrame:
    """
    Tạo features từ cột thời gian
    
    Args:
        df (pd.DataFrame): DataFrame
        time_column (str): Tên cột thời gian
        
    Returns:
        pd.DataFrame: DataFrame với time features mới
    """
    df_new = df.copy()
    
    # Giả sử Time là số giây từ thời điểm bắt đầu
    df_new['hour'] = (df_new[time_column] // 3600) % 24
    df_new['day'] = df_new[time_column] // (3600 * 24)
    
    # Is night time (23:00 - 6:00)
    df_new['is_night'] = ((df_new['hour'] >= 23) | (df_new['hour'] <= 6)).astype(int)
    
    print(f"✓ Created time features: hour, day, is_night")
    return df_new


def create_rolling_features(df: pd.DataFrame, 
                           column: str, 
                           windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Tạo rolling window features
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Tên cột để tính rolling
        windows (List[int]): Danh sách window sizes
        
    Returns:
        pd.DataFrame: DataFrame với rolling features
    """
    df_new = df.copy()
    
    for window in windows:
        df_new[f'{column}_rolling_mean_{window}'] = df_new[column].rolling(window).mean()
        df_new[f'{column}_rolling_std_{window}'] = df_new[column].rolling(window).std()
    
    # Fill NaN với giá trị gốc
    df_new = df_new.fillna(df_new[column])
    
    print(f"✓ Created rolling features for '{column}' with windows {windows}")
    return df_new


def scale_features(X: pd.DataFrame, 
                   method: str = 'standard',
                   fit_scaler: bool = True) -> Tuple[pd.DataFrame, object]:
    """
    Chuẩn hóa features
    
    Args:
        X (pd.DataFrame): Features cần scale
        method (str): 'standard' hoặc 'minmax'
        fit_scaler (bool): Fit scaler mới hoặc dùng scaler có sẵn
        
    Returns:
        Tuple[pd.DataFrame, object]: (Scaled features, Scaler object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    print(f"✓ Scaled features using '{method}' scaler")
    
    return X_scaled_df, scaler


def create_interaction_features(df: pd.DataFrame, 
                                col1: str, 
                                col2: str) -> pd.DataFrame:
    """
    Tạo interaction features giữa 2 cột
    
    Args:
        df (pd.DataFrame): DataFrame
        col1 (str): Cột 1
        col2 (str): Cột 2
        
    Returns:
        pd.DataFrame: DataFrame với interaction feature
    """
    df_new = df.copy()
    df_new[f'{col1}_x_{col2}'] = df_new[col1] * df_new[col2]
    df_new[f'{col1}_div_{col2}'] = df_new[col1] / (df_new[col2] + 1e-5)
    
    print(f"✓ Created interaction features between '{col1}' and '{col2}'")
    return df_new
