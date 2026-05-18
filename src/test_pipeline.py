import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src/ to path so we can import cleaning and processing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cleaning import clean_stock_data, clean_news_data
from processing import engineer_features


# ==============================================================================
# HELPERS — Mock data generation
# ==============================================================================

def make_stock_df(n=50, include_nulls=False, include_dupes=False):
    """Generate mock stock DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    close_prices = 100 + np.cumsum(np.random.randn(n))

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open":   close_prices - 1,
        "High":   close_prices + 2,
        "Low":    close_prices - 2,
        "Close":  close_prices,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
        "Ticker": "WMT",
    })

    if include_nulls:
        df.loc[5, "Close"] = None
        df.loc[10, "Close"] = None

    if include_dupes:
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)

    return df


def make_news_df(include_nulls=False, include_dupes=False):
    """Generate mock news DataFrame."""
    df = pd.DataFrame({
        "title": [f"News headline {i}" for i in range(10)],
        "date":  pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
        "source": ["Reuters"] * 10,
    })

    if include_nulls:
        df.loc[2, "title"] = None
        df.loc[4, "date"] = None

    if include_dupes:
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)

    return df


def save_temp_csv(df, path):
    """Save DataFrame to temporary CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ==============================================================================
# TEST CLASS 1 — cleaning.py (Member B)
# ==============================================================================

class TestCleanStockData(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "data/test_tmp/stock_test.csv"

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_removes_duplicate_dates(self):
        df = make_stock_df(include_dupes=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertEqual(result["Date"].duplicated().sum(), 0)

    def test_removes_null_close(self):
        df = make_stock_df(include_nulls=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertEqual(result["Close"].isna().sum(), 0)

    def test_date_column_is_datetime(self):
        df = make_stock_df()
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["Date"]))

    def test_sorted_by_date(self):
        df = make_stock_df()
        df = df.sample(frac=1).reset_index(drop=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertTrue(result["Date"].is_monotonic_increasing)

    def test_returns_dataframe(self):
        df = make_stock_df()
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)


class TestCleanNewsData(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "data/test_tmp/news_test.csv"

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_removes_duplicate_titles(self):
        df = make_news_df(include_dupes=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertEqual(result["title"].duplicated().sum(), 0)

    def test_removes_null_title_and_date(self):
        df = make_news_df(include_nulls=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertEqual(result["title"].isna().sum(), 0)
        self.assertEqual(result["date"].isna().sum(), 0)

    def test_date_column_is_datetime(self):
        df = make_news_df()
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))


# ==============================================================================
# TEST CLASS 2 — processing.py (Member C)
# ==============================================================================

class TestEngineerFeatures(unittest.TestCase):

    def setUp(self):
        self.df = make_stock_df(n=60)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.ticker = "WMT"

    def test_daily_return_column_exists(self):
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("Daily_Return", result.columns)

    def test_ma7_column_exists(self):
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("MA7", result.columns)

    def test_ma30_column_exists(self):
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("MA30", result.columns)

    def test_volatility_column_exists(self):
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("Volatility", result.columns)

    def test_daily_return_calculation(self):
        result = engineer_features(self.df.copy(), self.ticker)
        expected = self.df["Close"].pct_change()

        pd.testing.assert_series_equal(
            result["Daily_Return"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_ma7_calculation(self):
        result = engineer_features(self.df.copy(), self.ticker)
        expected = self.df["Close"].rolling(window=7).mean()

        pd.testing.assert_series_equal(
            result["MA7"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_ma30_calculation(self):
        result = engineer_features(self.df.copy(), self.ticker)
        expected = self.df["Close"].rolling(window=30).mean()

        pd.testing.assert_series_equal(
            result["MA30"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_volatility_is_non_negative(self):
        result = engineer_features(self.df.copy(), self.ticker)
        vol = result["Volatility"].dropna()
        self.assertTrue((vol >= 0).all())

    def test_outlier_flag_is_binary(self):
        result = engineer_features(self.df.copy(), self.ticker)
        unique_vals = set(result["Outlier_Flag"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_outlier_flag_triggers_at_10_percent(self):
        result = engineer_features(self.df.copy(), self.ticker)
        flagged = result[result["Outlier_Flag"] == 1]["Daily_Return"].dropna()
        for val in flagged:
            self.assertGreater(abs(val), 0.10)

    def test_ticker_column_added(self):
        df_no_ticker = self.df.copy().drop(columns=["Ticker"])
        result = engineer_features(df_no_ticker, self.ticker)
        self.assertIn("Ticker", result.columns)
        self.assertTrue((result["Ticker"] == self.ticker).all())

    def test_no_rows_lost_unexpectedly(self):
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertGreaterEqual(len(result), len(self.df) - 1)


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FinAgent — Unit Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
