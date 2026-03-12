"""
Unit tests cho data preprocessing module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.preprocessing import (
    check_missing_values, 
    handle_missing_values,
    remove_duplicates
)


class TestPreprocessing(unittest.TestCase):
    """Test cases cho preprocessing functions"""
    
    def setUp(self):
        """Tạo sample data cho testing"""
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 1, 2, 2, 3]
        })
    
    def test_check_missing_values(self):
        """Test check_missing_values function"""
        missing = check_missing_values(self.df)
        self.assertEqual(len(missing), 1)  # Chỉ có 1 cột có missing
        self.assertEqual(missing.iloc[0]['column'], 'A')
    
    def test_handle_missing_values_drop(self):
        """Test handle_missing_values với strategy='drop'"""
        df_clean = handle_missing_values(self.df, strategy='drop')
        self.assertEqual(len(df_clean), 4)  # 1 row bị drop
        self.assertEqual(df_clean.isnull().sum().sum(), 0)  # Không còn missing
    
    def test_handle_missing_values_mean(self):
        """Test handle_missing_values với strategy='mean'"""
        df_clean = handle_missing_values(self.df, strategy='mean')
        self.assertEqual(len(df_clean), 5)  # Không row nào bị drop
        self.assertEqual(df_clean.isnull().sum().sum(), 0)  # Không còn missing
        # Mean của A (không tính NaN) = (1+2+4+5)/4 = 3
        self.assertAlmostEqual(df_clean.loc[2, 'A'], 3.0)
    
    def test_remove_duplicates(self):
        """Test remove_duplicates function"""
        df_with_dup = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [1, 1, 2, 3]
        })
        df_clean = remove_duplicates(df_with_dup)
        self.assertEqual(len(df_clean), 3)  # 1 duplicate removed


if __name__ == '__main__':
    unittest.main()
