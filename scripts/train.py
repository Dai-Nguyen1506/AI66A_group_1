"""
Training Script
Chạy pipeline để train model Credit Card Fraud Detection
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import yaml
from src.data.loader import load_from_kagglehub, save_processed_data, get_data_info
from src.data.preprocessing import handle_missing_values, remove_duplicates
from src.features.engineering import create_time_features, scale_features
from src.models.trainer import FraudDetectionTrainer, split_data
from src.models.evaluator import ModelEvaluator
from src.visualization.plots import plot_class_distribution
import joblib


def load_config(config_path: str = 'configs/config.yaml'):
    """Load configuration từ YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    print("="*70)
    print("  CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("  AI66A - Group 1")
    print("="*70)
    
    # Load config
    try:
        config = load_config()
        print("\n✅ Configuration loaded")
    except FileNotFoundError:
        print("\n❌ Config file not found. Using default config.")
        config = {
            'data': {'target_column': 'is_fraud'},
            'training': {'test_size': 0.2},
            'model': {
                'type': 'random_forest',
                'name': 'fraud_detection_rf',
                'save_path': 'models',
                'params': {}
            }
        }
    
    # 1. Load dữ liệu
    print("\n" + "="*70)
    print("[1/8] LOADING DATA")
    print("="*70)
    
    try:
        # Thử load từ local trước
        if os.path.exists('data/raw/fraud_data.csv'):
            print("📁 Loading from local file...")
            df = pd.read_csv('data/raw/fraud_data.csv')
            print(f"✅ Loaded {len(df):,} rows from local file")
        else:
            # Download từ Kaggle
            print("📥 Downloading from Kaggle...")
            df = load_from_kagglehub(
                dataset_name="kartik2112/fraud-detection",
                file_path="fraudTrain.csv",
                save_to='data/raw/fraud_data.csv'
            )
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print("\n💡 Hướng dẫn:")
        print("   1. Cài đặt kagglehub: pip install kagglehub")
        print("   2. Hoặc tải dataset thủ công và đặt vào data/raw/")
        print("   3. Xem TUTORIAL.md để biết chi tiết")
        return
    
    # Show data info
    get_data_info(df)
    
    # 2. Preprocessing
    print("\n" + "="*70)
    print("[2/8] PREPROCESSING DATA")
    print("="*70)
    
    original_len = len(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy='drop')
    print(f"✅ Data cleaned: {original_len:,} → {len(df):,} rows")
    
    # 3. Identify target column
    print("\n" + "="*70)
    print("[3/8] IDENTIFYING TARGET COLUMN")
    print("="*70)
    
    # Tìm target column (is_fraud, Class, isFraud, etc.)
    fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or col == 'Class']
    
    if fraud_cols:
        target_col = fraud_cols[0]
        print(f"✅ Target column found: '{target_col}'")
    else:
        print("❌ Target column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # 4. Analyze class distribution
    print("\n" + "="*70)
    print("[4/8] ANALYZING CLASS DISTRIBUTION")
    print("="*70)
    
    class_dist = df[target_col].value_counts()
    print(f"\nClass distribution:")
    for cls, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
    
    # Save class distribution plot
    try:
        plot_class_distribution(df[target_col], save_path='reports/class_distribution.png')
    except:
        pass
    
    # 5. Feature Engineering
    print("\n" + "="*70)
    print("[5/8] FEATURE ENGINEERING")
    print("="*70)
    
    # Tách features và target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Chỉ giữ numeric features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X = X[numeric_cols]
    print(f"✅ Selected {len(numeric_cols)} numeric features")
    
    # Scale features
    print("Scaling features...")
    X_scaled, scaler = scale_features(X, method='standard')
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")
    
    # 6. Split train/test
    print("\n" + "="*70)
    print("[6/8] SPLITTING TRAIN/TEST")
    print("="*70)
    
    test_size = config['training'].get('test_size', 0.2)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y,
        test_size=test_size,
        random_state=42
    )
    
    # 7. Train model
    print("\n" + "="*70)
    print("[7/8] TRAINING MODEL")
    print("="*70)
    
    model_type = config['model'].get('type', 'random_forest')
    model_params = config['model'].get('params', {})
    
    trainer = FraudDetectionTrainer(
        model_type=model_type,
        **model_params
    )
    
    print(f"Training {model_type} model...")
    trainer.train(X_train, y_train)
    
    # Save model
    model_name = config['model'].get('name', 'fraud_detection_model')
    model_path = f"models/{model_name}.pkl"
    trainer.save_model(model_path)
    
    # Save processed data
    save_processed_data(pd.DataFrame(X_train, columns=X.columns), 'data/processed/X_train.csv')
    save_processed_data(pd.DataFrame(X_test, columns=X.columns), 'data/processed/X_test.csv')
    save_processed_data(pd.DataFrame(y_train, columns=[target_col]), 'data/processed/y_train.csv')
    save_processed_data(pd.DataFrame(y_test, columns=[target_col]), 'data/processed/y_test.csv')
    
    # 8. Evaluation
    print("\n" + "="*70)
    print("[8/8] EVALUATING MODEL")
    print("="*70)
    
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    evaluator = ModelEvaluator()
    evaluator.print_evaluation_report(y_test, y_pred, y_pred_proba)
    
    # Save evaluation plots
    try:
        evaluator.plot_confusion_matrix(y_test, y_pred, 
                                       save_path='reports/confusion_matrix.png')
        evaluator.plot_roc_curve(y_test, y_pred_proba,
                                save_path='reports/roc_curve.png')
    except:
        print("⚠️ Could not save evaluation plots")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Outputs:")
    print(f"   ├─ Model: {model_path}")
    print(f"   ├─ Scaler: {scaler_path}")
    print(f"   ├─ Processed data: data/processed/")
    print(f"   └─ Reports: reports/")
    print("\n🚀 Next step: Run 'python scripts/evaluate.py' or 'python scripts/predict.py'")
    print("="*70)


if __name__ == "__main__":
    main()
