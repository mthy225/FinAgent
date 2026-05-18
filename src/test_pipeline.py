"""
test_pipeline.py
================
Phase 3 - Unit Tests (Lead: Member A - Hân)

Tests for:
- cleaning.py  (Member B)
- processing.py (Member C)

Usage:
    python test_pipeline.py
"""

import unittest
import pandas as pd
import numpy as np
from io import StringIO
import sys
import os

# ── Add src/ to path so we can import cleaning and processing ─────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cleaning import clean_stock_data, clean_news_data
from processing import engineer_features


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — tạo dữ liệu giả để test, không cần file CSV thật
# ══════════════════════════════════════════════════════════════════════════════

def make_stock_df(n=50, include_nulls=False, include_dupes=False):
    """Tạo DataFrame cổ phiếu giả."""
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
    """Tạo DataFrame tin tức giả."""
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
    """Lưu DataFrame ra file CSV tạm."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1 — cleaning.py (Member B)
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanStockData(unittest.TestCase):

    def setUp(self):
        """Tạo file CSV tạm trước mỗi test."""
        self.tmp_path = "data/test_tmp/stock_test.csv"

    def tearDown(self):
        """Xóa file tạm sau mỗi test."""
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_removes_duplicate_dates(self):
        """clean_stock_data phải xóa các dòng có Date trùng lặp."""
        df = make_stock_df(include_dupes=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertEqual(
            result["Date"].duplicated().sum(), 0,
            "Vẫn còn duplicate Date sau khi clean!"
        )

    def test_removes_null_close(self):
        """clean_stock_data phải xóa các dòng Close bị null."""
        df = make_stock_df(include_nulls=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertEqual(
            result["Close"].isna().sum(), 0,
            "Vẫn còn null Close sau khi clean!"
        )

    def test_date_column_is_datetime(self):
        """Cột Date phải được convert sang datetime."""
        df = make_stock_df()
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["Date"]),
            "Cột Date không phải kiểu datetime!"
        )

    def test_sorted_by_date(self):
        """DataFrame phải được sắp xếp tăng dần theo Date."""
        df = make_stock_df()
        df = df.sample(frac=1).reset_index(drop=True)  # xáo trộn ngẫu nhiên
        save_temp_csv(df, self.tmp_path)

        result = clean_stock_data(self.tmp_path)

        self.assertTrue(
            result["Date"].is_monotonic_increasing,
            "Date không được sắp xếp tăng dần!"
        )

    def test_returns_dataframe(self):
        """Hàm phải trả về DataFrame, không phải None."""
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
        """clean_news_data phải xóa tin tức trùng title."""
        df = make_news_df(include_dupes=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertEqual(
            result["title"].duplicated().sum(), 0,
            "Vẫn còn duplicate title sau khi clean!"
        )

    def test_removes_null_title_and_date(self):
        """Phải xóa dòng có title hoặc date bị null."""
        df = make_news_df(include_nulls=True)
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertEqual(result["title"].isna().sum(), 0)
        self.assertEqual(result["date"].isna().sum(), 0)

    def test_date_column_is_datetime(self):
        """Cột date phải là kiểu datetime."""
        df = make_news_df()
        save_temp_csv(df, self.tmp_path)

        result = clean_news_data(self.tmp_path)

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["date"]),
            "Cột date không phải kiểu datetime!"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2 — processing.py (Member C)
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineerFeatures(unittest.TestCase):

    def setUp(self):
        """Tạo DataFrame sạch để test engineer_features."""
        self.df = make_stock_df(n=60)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.ticker = "WMT"

    def test_daily_return_column_exists(self):
        """Phải có cột Daily_Return."""
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("Daily_Return", result.columns)

    def test_ma7_column_exists(self):
        """Phải có cột MA7."""
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("MA7", result.columns)

    def test_ma30_column_exists(self):
        """Phải có cột MA30."""
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("MA30", result.columns)

    def test_volatility_column_exists(self):
        """Phải có cột Volatility."""
        result = engineer_features(self.df.copy(), self.ticker)
        self.assertIn("Volatility", result.columns)

    def test_daily_return_calculation(self):
        """Daily_Return phải = (Close[t] - Close[t-1]) / Close[t-1]."""
        result = engineer_features(self.df.copy(), self.ticker)

        # Tính tay để so sánh
        expected = self.df["Close"].pct_change()

        # Bỏ NaN đầu tiên, so sánh phần còn lại
        pd.testing.assert_series_equal(
            result["Daily_Return"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_ma7_calculation(self):
        """MA7 phải là rolling mean 7 ngày của Close."""
        result = engineer_features(self.df.copy(), self.ticker)
        expected = self.df["Close"].rolling(window=7).mean()

        pd.testing.assert_series_equal(
            result["MA7"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_ma30_calculation(self):
        """MA30 phải là rolling mean 30 ngày của Close."""
        result = engineer_features(self.df.copy(), self.ticker)
        expected = self.df["Close"].rolling(window=30).mean()

        pd.testing.assert_series_equal(
            result["MA30"].dropna().reset_index(drop=True),
            expected.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_volatility_is_non_negative(self):
        """Volatility (std) không được âm."""
        result = engineer_features(self.df.copy(), self.ticker)
        vol = result["Volatility"].dropna()
        self.assertTrue(
            (vol >= 0).all(),
            "Có giá trị Volatility âm!"
        )

    def test_outlier_flag_is_binary(self):
        """Outlier_Flag chỉ được chứa 0 hoặc 1."""
        result = engineer_features(self.df.copy(), self.ticker)
        unique_vals = set(result["Outlier_Flag"].unique())
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            f"Outlier_Flag có giá trị lạ: {unique_vals}"
        )

    def test_outlier_flag_triggers_at_10_percent(self):
        """Outlier_Flag = 1 khi |Daily_Return| > 10%."""
        result = engineer_features(self.df.copy(), self.ticker)

        flagged = result[result["Outlier_Flag"] == 1]["Daily_Return"].dropna()
        for val in flagged:
            self.assertGreater(
                abs(val), 0.10,
                f"Outlier_Flag = 1 nhưng Daily_Return = {val:.4f} không vượt 10%!"
            )

    def test_ticker_column_added(self):
        """Phải có cột Ticker với đúng giá trị."""
        df_no_ticker = self.df.copy().drop(columns=["Ticker"])
        result = engineer_features(df_no_ticker, self.ticker)
        self.assertIn("Ticker", result.columns)
        self.assertTrue((result["Ticker"] == self.ticker).all())

    def test_no_rows_lost_unexpectedly(self):
        """Số dòng sau feature engineering không được ít hơn input quá nhiều."""
        result = engineer_features(self.df.copy(), self.ticker)
        # Chấp nhận mất tối đa 1 dòng (NaN đầu tiên của pct_change)
        self.assertGreaterEqual(len(result), len(self.df) - 1)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FinAgent — Unit Test Suite (Member A)")
    print("=" * 60)
    unittest.main(verbosity=2)