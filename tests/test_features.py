"""
Unit tests cho feature engineering module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.engineering import (
    create_time_features,
    scale_features
)


class TestFeatureEngineering(unittest.TestCase):
    """Test cases cho feature engineering functions"""
    
    def setUp(self):
        """Tạo sample data cho testing"""
        self.df = pd.DataFrame({
            'Time': [3600, 7200, 86400],  # 1h, 2h, 1 day
            'Amount': [100, 200, 300]
        })
    
    def test_create_time_features(self):
        """Test create_time_features function"""
        df_new = create_time_features(self.df, time_column='Time')
        
        # Kiểm tra có tạo ra các cột mới
        self.assertIn('hour', df_new.columns)
        self.assertIn('day', df_new.columns)
        self.assertIn('is_night', df_new.columns)
        
        # Kiểm tra giá trị
        self.assertEqual(df_new.loc[0, 'hour'], 1)  # 3600s = 1h
        self.assertEqual(df_new.loc[1, 'hour'], 2)  # 7200s = 2h
        self.assertEqual(df_new.loc[2, 'day'], 1)   # 86400s = 1 day
    
    def test_scale_features(self):
        """Test scale_features function"""
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        X_scaled, scaler = scale_features(X, method='standard')
        
        # Kiểm tra shape giữ nguyên
        self.assertEqual(X_scaled.shape, X.shape)
        
        # Kiểm tra mean gần 0, std gần 1 (standard scaling)
        self.assertAlmostEqual(X_scaled['A'].mean(), 0, places=10)
        self.assertAlmostEqual(X_scaled['A'].std(), 1, places=10)


if __name__ == '__main__':
    unittest.main()
