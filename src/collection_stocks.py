import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Logging Setup ──────────────────────────────────────────────────────────────
RAW_DIR = Path("data_stocks/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

log_path = RAW_DIR / "collection_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS = ["WMT", "TGT", "COST"]
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")  # 3 years
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ── Helper: retry wrapper ──────────────────────────────────────────────────────
def fetch_with_retry(func, *args, retries: int = MAX_RETRIES, delay: int = RETRY_DELAY, **kwargs):
    """
    Calls func(*args, **kwargs) up to `retries` times.
    Sleeps `delay` seconds between attempts.
    Returns the result or raises the last exception.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"All {retries} attempts failed.") from last_exc


# ── Stock Price Collection ─────────────────────────────────────────────────────
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker via yfinance.

    Parameters
    ----------
    ticker : str   e.g. "WMT"
    start  : str   "YYYY-MM-DD"
    end    : str   "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Ticker
    Raises RuntimeError if download fails or returns empty.
    """
    logger.info(f"Fetching {ticker} | {start} → {end}")

    def _download():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # adjusts for splits & dividends automatically
            progress=False,
        )
        if df.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
        return df

    df = fetch_with_retry(_download)

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns that exist
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df["Ticker"] = ticker

    # Basic raw validation
    n_rows = len(df)
    n_nulls = df[["Close"]].isna().sum().sum()
    logger.info(f"  ✓ {ticker}: {n_rows} rows | {n_nulls} null Close values")

    return df


def collect_all_stocks(tickers: list, start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Fetch stock data for all tickers. Returns dict: {ticker: DataFrame}.
    Logs per-ticker success/failure without crashing the whole run.
    """
    results = {}
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, start, end)
            results[ticker] = df
        except Exception as exc:
            logger.error(f"  ✗ FAILED {ticker}: {exc}")
            logger.debug(traceback.format_exc())
    return results


# ── Save Utilities ─────────────────────────────────────────────────────────────
def save_individual(stock_dict: dict, output_dir: Path) -> None:
    """Save each ticker's DataFrame to its own CSV."""
    for ticker, df in stock_dict.items():
        path = output_dir / f"stock_prices_{ticker}.csv"
        df.to_csv(path)
        logger.info(f"Saved → {path}")


def save_combined(stock_dict: dict, output_dir: Path) -> pd.DataFrame:
    """
    Concatenate all tickers into one CSV.
    Returns the combined DataFrame.
    """
    if not stock_dict:
        logger.warning("No data to combine.")
        return pd.DataFrame()

    combined = pd.concat(stock_dict.values(), axis=0)
    combined.sort_index(inplace=True)

    path = output_dir / "stock_prices_combined.csv"
    combined.to_csv(path)
    logger.info(f"Saved combined → {path}")
    return combined


# ── Summary Report ─────────────────────────────────────────────────────────────
def print_summary(stock_dict: dict) -> None:
    """Print a quick summary table to console and log."""
    if not stock_dict:
        logger.warning("No data collected.")
        return

    logger.info("\n" + "=" * 55)
    logger.info(f"{'Ticker':<8} {'Rows':>6} {'Start':>12} {'End':>12} {'Null Close':>10}")
    logger.info("-" * 55)
    for ticker, df in stock_dict.items():
        null_close = int(df["Close"].isna().sum())
        logger.info(
            f"{ticker:<8} {len(df):>6} "
            f"{str(df.index.min().date()):>12} "
            f"{str(df.index.max().date()):>12} "
            f"{null_close:>10}"
        )
    logger.info("=" * 55)


# ── Main Entrypoint ────────────────────────────────────────────────────────────
def run_collection(
    tickers: list = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Full collection pipeline.

    Returns
    -------
    dict: {ticker: cleaned raw DataFrame}
    """
    logger.info("=" * 55)
    logger.info("FinAgent — Data Collection Module Starting")
    logger.info(f"Tickers : {tickers}")
    logger.info(f"Period  : {start} → {end}")
    logger.info(f"Output  : {output_dir.resolve()}")
    logger.info("=" * 55)

    stock_dict = collect_all_stocks(tickers, start, end)

    if not stock_dict:
        logger.error("Collection failed for ALL tickers. Aborting.")
        return {}

    save_individual(stock_dict, output_dir)
    save_combined(stock_dict, output_dir)
    print_summary(stock_dict)

    logger.info("Collection complete.\n")
    return stock_dict


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_collection()
