import pandas as pd
import numpy as np
import os
from pathlib import Path

CLEANED_DIR = Path("data_stocks/cleaned")
OUTPUT_DIR = Path("data/processed")
TICKERS = ["WMT", "TGT", "COST"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add financial indicators required by analysis.py.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # Moving averages
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()

    # Volatility (30-day rolling std of daily return)
    df["Volatility"] = df["Daily_Return"].rolling(window=30).std()

    # Outlier flag: daily return > 3 std deviations
    mean_ret = df["Daily_Return"].mean()
    std_ret = df["Daily_Return"].std()
    df["Outlier_Flag"] = (
        (df["Daily_Return"] - mean_ret).abs() > 3 * std_ret
    ).astype(int)

    return df


def prepare_processed_data():
    """
    Read cleaned stock CSVs, add features, save to data/processed/
    for use by analysis.py (Member A).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        input_path = CLEANED_DIR / f"{ticker}_cleaned.csv"

        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        print(f"\nProcessing: {ticker}")
        df = pd.read_csv(input_path)
        df["Date"] = pd.to_datetime(df["Date"])

        df = add_features(df)

        output_path = OUTPUT_DIR / f"{ticker}_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"Shape: {df.shape} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    prepare_processed_data()