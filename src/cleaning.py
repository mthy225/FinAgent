import pandas as pd
import os


def clean_news_data(filepath: str) -> pd.DataFrame:
    """
    Clean raw news data.

    Args:
        filepath: Path to raw CSV file

    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)

    print(f"Before cleaning: {df.shape[0]} rows")

    # Remove duplicate articles
    df = df.drop_duplicates(subset=["title"])

    # Remove rows with missing values
    df = df.dropna(subset=["title", "date"])

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    print(f"After cleaning: {df.shape[0]} rows")

    return df


def clean_stock_data(filepath: str) -> pd.DataFrame:
    """
    Clean raw stock price data.

    Args:
        filepath: Path to raw stock CSV file

    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)

    print(f"Before cleaning: {df.shape[0]} rows")

    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"])

    # Remove rows with missing close price
    df = df.dropna(subset=["Close"])

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"After cleaning: {df.shape[0]} rows")

    return df


if __name__ == "__main__":

    # =========================================================
    # CLEAN NEWS DATA
    # =========================================================

    news_tickers = ["walmart", "target", "costco"]

    for ticker in news_tickers:

        # UPDATED: data -> data_news
        raw_path = f"data_news/raw/news_{ticker}.csv"

        if not os.path.exists(raw_path):
            print(f"File not found: {raw_path}")
            continue

        print(f"\nCleaning news data: {ticker}")

        df_clean = clean_news_data(raw_path)

        # UPDATED: data/cleaned -> data_news/cleaned
        os.makedirs("data_news/cleaned", exist_ok=True)

        output_path = f"data_news/cleaned/news_{ticker}_cleaned.csv"

        df_clean.to_csv(output_path, index=False)

        print(f"Saved cleaned file to: {output_path}")

    # =========================================================
    # CLEAN STOCK DATA
    # =========================================================

    stock_tickers = ["WMT", "TGT", "COST"]

    for ticker in stock_tickers:

        raw_path = f"data_stocks/raw/stock_prices_{ticker}.csv"

        if not os.path.exists(raw_path):
            print(f"File not found: {raw_path}")
            continue

        print(f"\nCleaning stock data: {ticker}")

        df_clean = clean_stock_data(raw_path)

        os.makedirs("data_stocks/cleaned", exist_ok=True)

        output_path = f"data_stocks/cleaned/{ticker}_cleaned.csv"

        df_clean.to_csv(output_path, index=False)

        print(f"Saved cleaned file to: {output_path}")