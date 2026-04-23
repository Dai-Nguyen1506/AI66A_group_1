import os
import pandas as pd
from pathlib import Path

def get_data_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print("=" * 60)

def load_csv(file_path: str, nrows: int | None = None) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1]
    full_path = base_path / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"❌ File not found: {full_path}")
    print(f"👉 Loading: {file_path}")
    # ===== READ CSV =====
    df = pd.read_csv(full_path)
    if nrows is not None:
        df = df.head(nrows)
        
    get_data_info(df)

    return df


def save_file(df, file_path):
    save_dir = "../data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_path)
    df.to_csv(save_path, index=False)
    print(f"File saved as: {save_path}")
