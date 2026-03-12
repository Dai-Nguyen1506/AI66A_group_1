# Module __init__.py
__version__ = "1.0.0"

# Import main classes for easier access
from .data.loader import load_csv, load_fraud_detection_data
from .models.trainer import FraudDetectionTrainer
from .models.evaluator import ModelEvaluator

__all__ = [
    'load_csv',
    'load_fraud_detection_data',
    'FraudDetectionTrainer',
    'ModelEvaluator'
]
