"""
Model Evaluator Module
Chức năng: Đánh giá performance của models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Class để đánh giá model performance
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Tính toán các metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Dict[str, float]: Dictionary chứa các metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def print_evaluation_report(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> None:
        """
        In báo cáo đánh giá chi tiết
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities (optional)
        """
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Basic metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # ROC-AUC (nếu có probability)
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Classification report
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
        
        # Confusion matrix
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\n[TN  FP]")
        print("[FN  TP]")
        print("=" * 60)
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             save_path: str = None) -> None:
        """
        Vẽ confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            save_path (str): Đường dẫn lưu plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, 
                      y_pred_proba: np.ndarray,
                      save_path: str = None) -> None:
        """
        Vẽ ROC curve
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            save_path (str): Đường dẫn lưu plot (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.show()
