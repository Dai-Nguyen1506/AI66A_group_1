"""
Model Trainer Module
Chức năng: Training các models cho fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
from typing import Tuple, Dict, Any


class FraudDetectionTrainer:
    """
    Class để train models cho Credit Card Fraud Detection
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize trainer
        
        Args:
            model_type (str): Loại model ('logistic', 'random_forest', 'xgboost')
            **kwargs: Các tham số cho model
        """
        self.model_type = model_type
        self.model = None
        self.model_params = kwargs
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Khởi tạo model theo loại được chọn"""
        if self.model_type == 'logistic':
            # Default params cho Logistic Regression
            default_params = {'max_iter': 1000}
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)
            
        elif self.model_type == 'random_forest':
            # Default params cho Random Forest
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✓ Initialized {self.model_type} model")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        print("✓ Training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model chưa được train!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model chưa được train!")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Lưu model
        
        Args:
            filepath (str): Đường dẫn lưu model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load model đã lưu
        
        Args:
            filepath (str): Đường dẫn tới model
            
        Returns:
            Model đã load
        """
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


def split_data(X: pd.DataFrame, 
               y: pd.Series, 
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple:
    """
    Chia dữ liệu train/test
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        test_size (float): Tỉ lệ test set
        random_state (int): Random seed
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test
