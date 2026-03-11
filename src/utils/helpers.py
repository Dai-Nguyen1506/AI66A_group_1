"""
Utility functions cho dự án
"""

import os
import json
import pickle
import joblib
from typing import Any


def ensure_dir(directory: str) -> None:
    """
    Tạo thư mục nếu chưa tồn tại
    
    Args:
        directory (str): Đường dẫn thư mục
    """
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory ensured: {directory}")


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Lưu object dưới dạng pickle
    
    Args:
        obj: Object cần lưu
        filepath (str): Đường dẫn file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object từ pickle file
    
    Args:
        filepath (str): Đường dẫn file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"✓ Loaded pickle from {filepath}")
    return obj


def save_json(data: dict, filepath: str, indent: int = 2) -> None:
    """
    Lưu dictionary dưới dạng JSON
    
    Args:
        data (dict): Dictionary cần lưu
        filepath (str): Đường dẫn file
        indent (int): Số space indent
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load dictionary từ JSON file
    
    Args:
        filepath (str): Đường dẫn file
        
    Returns:
        Dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded JSON from {filepath}")
    return data
