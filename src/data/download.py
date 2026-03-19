import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional

def save_data_local(df: pd.DataFrame, filepath: str) -> None:
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save file
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df):,} rows to {filepath}")

def load_from_kagglehub(dataset_name: str = "kartik2112/fraud-detection",
                        file_path: str = "",
                        save_to: Optional[str] = None) -> pd.DataFrame:
    try:
        import kagglehub
        print(f"📥 Downloading dataset '{dataset_name}' from Kaggle...")
        # Download dataset and get directory path
        dataset_path = kagglehub.dataset_download(dataset_name)
        # Find CSV file
        if file_path:
            full_path = os.path.join(dataset_path, file_path)
        else:
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV file found in dataset")
            full_path = os.path.join(dataset_path, csv_files[0])
        
        print(f"📖 Reading file: {os.path.basename(full_path)}")
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(full_path, encoding=encoding)
                print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Unable to read file with common encodings")
        # Save locally if specified
        if save_to:
            save_data_local(df, save_to)
        
        return df
    
    except ImportError:
        print("❌ kagglehub is not installed!")
        print("Install it using: pip install kagglehub")
        raise
    except Exception as e:
        print(f"❌ Error loading from Kaggle: {e}")
        raise

def download_default_dataset():
    print("=" * 70)
    print("🚀 CREDIT CARD FRAUD DETECTION - DATA DOWNLOADER")
    print("=" * 70)
    print("📦 Dataset: kartik2112/fraud-detection")
    print("🌐 Source: Kaggle")
    print()
    # Define save path
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Save location: {raw_data_dir}")
    print()
    
    # File paths
    train_file = raw_data_dir / "fraudTrain.csv"
    test_file = raw_data_dir / "fraudTest.csv"
    try:
        # Download training data
        if not train_file.exists():
            print("📥 Downloading TRAINING dataset...")
            df_train = load_from_kagglehub(
                dataset_name="kartik2112/fraud-detection",
                file_path="fraudTrain.csv",
                save_to=str(train_file)
            )
            print(f"✅ Training data saved to: {train_file}")
            print(f"   Shape: {df_train.shape}")
        else:
            print(f"⏭️ Training data already exists: {train_file}")
        print()
        # Download test data
        if not test_file.exists():
            print("📥 Downloading TEST dataset...")
            df_test = load_from_kagglehub(
                dataset_name="kartik2112/fraud-detection",
                file_path="fraudTest.csv",
                save_to=str(test_file)
            )
            print(f"✅ Test data saved to: {test_file}")
            print(f"   Shape: {df_test.shape}")
        else:
            print(f"⏭️ Test data already exists: {test_file}")
        
        print()
        print("=" * 70)
        print("✨ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📂 Data location: {raw_data_dir}")
        print("   • fraudTrain.csv")
        print("   • fraudTest.csv")
        print()
        print("📋 Next steps:")
        print("   1. Load data: from src.data.loader import load_fraud_detection_data")
        print("   2. Run EDA notebook: notebooks/01_eda_exploration.ipynb")
        print("   3. Or start training: python scripts/train.py")
        print("=" * 70)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ ERROR: {e}")
        print("=" * 70)
        print()
        print("💡 Troubleshooting:")
        print("   1. Install kagglehub: pip install kagglehub")
        print("   2. Check Kaggle API credentials")
        print("   3. Check internet connection")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    download_default_dataset()