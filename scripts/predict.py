"""
Prediction Script
Chạy prediction với trained model
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
import yaml


def load_config(config_path: str = 'configs/config.yaml'):
    """Load configuration từ YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def predict_single_transaction(model, scaler, transaction_data: dict):
    """
    Dự đoán cho 1 transaction
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        transaction_data: Dictionary chứa features của transaction
        
    Returns:
        prediction, probability
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return prediction, probability


def main():
    """Main prediction function"""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - PREDICTION")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Load model
    print("\n[1/3] Loading model...")
    model_path = config['model']['save_path'] + f"/{config['model']['name']}.pkl"
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model không tìm thấy tại {model_path}")
        print("Vui lòng chạy train.py trước!")
        return
    
    # Load scaler
    print("\n[2/3] Loading scaler...")
    scaler_path = 'models/scaler.pkl'
    try:
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        print(f"⚠ Scaler không tìm thấy. Bỏ qua scaling.")
        scaler = None
    
    # Predict
    print("\n[3/3] Making predictions...")
    
    # Example: Predict cho dữ liệu test
    # TODO: Load dữ liệu mới cần predict
    # X_new = pd.read_csv('data/new_transactions.csv')
    
    # Example prediction cho 1 transaction
    sample_transaction = {
        'Time': 3600,
        'V1': -1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        # ... thêm các features khác
        'Amount': 149.62
    }
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    print("⚠ This is a demo. Please provide real transaction data.")
    
    # prediction, probability = predict_single_transaction(model, scaler, sample_transaction)
    # print(f"\nPrediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
    # print(f"Fraud Probability: {probability[1]:.4f}")
    # print(f"Normal Probability: {probability[0]:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
