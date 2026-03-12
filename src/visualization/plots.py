"""
Visualization Module
Chức năng: Tạo các visualizations cho data và model results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def plot_class_distribution(y: pd.Series, save_path: str = None) -> None:
    """
    Vẽ phân bổ các classes
    
    Args:
        y (pd.Series): Labels
        save_path (str): Đường dẫn lưu plot (optional)
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    y.value_counts().plot(kind='bar', ax=ax[0], color=['#3498db', '#e74c3c'])
    ax[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)
    
    # Percentage plot
    y.value_counts(normalize=True).plot(kind='bar', ax=ax[1], color=['#3498db', '#e74c3c'])
    ax[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Percentage')
    ax[1].set_xticklabels(['Normal', 'Fraud'], rotation=0)
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, 
                               features: List[str],
                               target: str = 'Class',
                               save_path: str = None) -> None:
    """
    Vẽ phân bổ của features theo class
    
    Args:
        df (pd.DataFrame): DataFrame
        features (List[str]): Danh sách features cần plot
        target (str): Tên cột target
        save_path (str): Đường dẫn lưu plot (optional)
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        for class_val in df[target].unique():
            data = df[df[target] == class_val][feature]
            axes[idx].hist(data, bins=50, alpha=0.6, 
                          label=f'Class {class_val}')
        
        axes[idx].set_title(f'Distribution of {feature}', fontweight='bold')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature distributions plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, 
                            figsize: tuple = (12, 10),
                            save_path: str = None) -> None:
    """
    Vẽ correlation matrix
    
    Args:
        df (pd.DataFrame): DataFrame
        figsize (tuple): Kích thước figure
        save_path (str): Đường dẫn lưu plot (optional)
    """
    corr = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, 
                           feature_names: List[str],
                           top_n: int = 20,
                           save_path: str = None) -> None:
    """
    Vẽ feature importance từ tree-based models
    
    Args:
        model: Trained model (phải có attribute feature_importances_)
        feature_names (List[str]): Tên các features
        top_n (int): Số lượng top features hiển thị
        save_path (str): Đường dẫn lưu plot (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠ Model không có feature_importances_ attribute")
        return
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances['importance'], color='steelblue')
    plt.yticks(range(len(importances)), importances['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.show()
