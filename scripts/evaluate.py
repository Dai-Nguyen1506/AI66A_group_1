"""
Evaluation Script
Chạy evaluation cho trained model
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
from src.data.loader import load_csv
from src.models.evaluator import ModelEvaluator


def main():
    """Main evaluation function"""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION")
    print("="*60)
    
    # TODO: Load test data
    # X_test = pd.read_csv('data/processed/X_test.csv')
    # y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # TODO: Load trained model
    # model = joblib.load('models/random_forest_v1.pkl')
    
    # TODO: Make predictions
    # y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)
    
    # TODO: Evaluate
    # evaluator = ModelEvaluator()
    # evaluator.print_evaluation_report(y_test, y_pred, y_pred_proba)
    
    # TODO: Plot confusion matrix
    # evaluator.plot_confusion_matrix(y_test, y_pred, 
    #                                save_path='reports/confusion_matrix.png')
    
    # TODO: Plot ROC curve
    # evaluator.plot_roc_curve(y_test, y_pred_proba,
    #                         save_path='reports/roc_curve.png')
    
    print("\n✓ Evaluation completed!")
    print("Check 'reports/' folder for visualizations")


if __name__ == "__main__":
    main()
