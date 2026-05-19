import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Project Directories (Targeting data_stocks/)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

CLEANED_DIR = BASE_DIR / "data_stocks" / "cleaned"
OUTPUT_DIR  = BASE_DIR / "data_stocks" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Input Files
# ─────────────────────────────────────────────
FILES = {
    "COST": CLEANED_DIR / "COST_cleaned.csv",
    "TGT":  CLEANED_DIR / "TGT_cleaned.csv",
    "WMT":  CLEANED_DIR / "WMT_cleaned.csv"
}

# ─────────────────────────────────────────────
# Feature Engineering Function
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute financial indicators:
    - Daily Return
    - MA7
    - MA30
    - Volatility
    """
    logger.info(f"Starting feature engineering for {ticker}")

    # Normalize date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalize numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove missing Close values
    df = df.dropna(subset=["Close"])

    # Feature 1: Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    # Feature 2: 7-day Moving Average
    # Sử a dụng phép gán biến tường minh thay vì gán trực tiếp để tránh cảnh báo sao chép dữ liệu
    df["MA7"] = df["Close"].rolling(window=7).mean()

    # Feature 3: 30-day Moving Average
    df["MA30"] = df["Close"].rolling(window=30).mean()

    # Feature 4: Volatility
    df["Volatility"] = df["Daily_Return"].rolling(window=30).std()

    # Optional: Outlier Detection
    df["Outlier_Flag"] = np.where(df["Daily_Return"].abs() > 0.10, 1, 0)

    # Add ticker column if missing
    if "Ticker" not in df.columns:
        df["Ticker"] = ticker

    logger.info(f"Completed feature engineering for {ticker}")
    logger.info(f"Final shape: {df.shape}")

    return df

# ─────────────────────────────────────────────
# Save Processed Data
# ─────────────────────────────────────────────
def save_processed_data(df: pd.DataFrame, ticker: str):
    output_path = OUTPUT_DIR / f"{ticker}_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed file → {output_path}")

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run_pipeline():
    logger.info("=" * 60)
    logger.info("PHASE 3 — FEATURE ENGINEERING STARTED")
    logger.info("=" * 60)

    combined_data = []

    for ticker, filepath in FILES.items():
        logger.info(f"\nProcessing: {ticker}")

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows")

            df_features = engineer_features(df, ticker)
            save_processed_data(df_features, ticker)
            combined_data.append(df_features)

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    # Save Combined Dataset
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_output = OUTPUT_DIR / "all_assets_features.csv"
        combined_df.to_csv(combined_output, index=False)
        logger.info(f"Saved combined dataset → {combined_output}")

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()