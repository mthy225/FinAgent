"""
visualization.py
================
Phase 4 - Visualization Module

Produces 4 required chart types from processed feature data:
    1. Price Trend + Volume Overlay  (price_volume_{ticker}.png)
    2. Correlation Heatmap           (correlation_heatmap.png)
    3. Daily Returns Distribution    (returns_distribution.png)
    4. Bollinger Bands               (bollinger_bands_{ticker}.png)

Also exports a structured JSON summary consumed by the AI Analysis Module (Member A).

Usage:
    python visualization.py

Input:
    data_stocks/processed/{TICKER}_features.csv   (output of processing.py)

Output:
    reports/figures/price_volume_{ticker}.png
    reports/figures/correlation_heatmap.png
    reports/figures/returns_distribution.png
    reports/figures/bollinger_bands_{ticker}.png
    reports/structured_summary.json              ← for LLM prompt (Member B)
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data_stocks" / "processed" 
FIG_DIR    = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS = ["WMT", "TGT", "COST"]

TICKER_COLORS = {
    "WMT": "#0071CE",   # Walmart blue
    "TGT": "#CC0000",   # Target red
    "COST": "#005DAA",  # Costco blue
}

STYLE = "seaborn-v0_8-whitegrid"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ── Data Loader ────────────────────────────────────────────────────────────────
def load_processed_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load processed CSV files for all tickers. Skips missing files."""
    data = {}
    for ticker in tickers:
        path = PROC_DIR / f"{ticker}_features.csv"
        if not path.exists():
            logger.warning(f"File not found, skipping: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        data[ticker] = df
        logger.info(f"Loaded {ticker}: {len(df)} rows  ({df['Date'].min().date()} → {df['Date'].max().date()})")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Price Trend + Volume Overlay
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_volume(ticker: str, df: pd.DataFrame) -> Path:
    """
    Two-panel chart:
      Top   : Close price with MA7 and MA30 overlaid.
      Bottom: Trading volume as bar chart.

    Returns the saved file path.
    """
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle(f"{ticker} — Price Trend & Volume", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    ax_price  = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)

    color = TICKER_COLORS.get(ticker, "#333333")

    # ── Price panel ────────────────────────────────────────────────────────────
    ax_price.plot(df["Date"], df["Close"], color=color, linewidth=1.5,
                  label="Close Price", alpha=0.9)
    ax_price.plot(df["Date"], df["MA7"],  color="orange",  linewidth=1.2,
                  linestyle="--", label="MA7",  alpha=0.85)
    ax_price.plot(df["Date"], df["MA30"], color="green",   linewidth=1.2,
                  linestyle="--", label="MA30", alpha=0.85)

    # Highlight outlier dates (|Daily_Return| > 10%)
    if "Outlier_Flag" in df.columns:
        outliers = df[df["Outlier_Flag"] == 1]
        if not outliers.empty:
            ax_price.scatter(outliers["Date"], outliers["Close"],
                             color="red", s=40, zorder=5, label="Outlier (>10%)")

    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(loc="upper left")
    ax_price.tick_params(labelbottom=False)

    # ── Volume panel ──────────────────────────────────────────────────────────
    ax_volume.bar(df["Date"], df["Volume"] / 1e6,
                  color=color, alpha=0.5, width=1.5)
    ax_volume.set_ylabel("Volume (M)")
    ax_volume.set_xlabel("Date")
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_volume.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=30, ha="right")

    path = FIG_DIR / f"price_volume_{ticker}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart 1 saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(data: dict[str, pd.DataFrame]) -> Path:
    """
    Pearson correlation heatmap of Close prices across all tickers.
    Also includes Daily_Return correlations in a second panel.

    Returns the saved file path.
    """
    # ── Build aligned close-price and return matrices ─────────────────────────
    close_frames  = {}
    return_frames = {}

    for ticker, df in data.items():
        close_frames[ticker]  = df.set_index("Date")["Close"]
        return_frames[ticker] = df.set_index("Date")["Daily_Return"]

    close_df  = pd.DataFrame(close_frames).dropna()
    return_df = pd.DataFrame(return_frames).dropna()

    corr_close  = close_df.corr()
    corr_return = return_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation Heatmap — WMT · TGT · COST", fontsize=15, fontweight="bold")

    kw = dict(annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
              linewidths=0.5, annot_kws={"size": 12})

    sns.heatmap(corr_close,  ax=axes[0], **kw, cbar=True)
    axes[0].set_title("Close Price Correlation")

    sns.heatmap(corr_return, ax=axes[1], **kw, cbar=True)
    axes[1].set_title("Daily Return Correlation")

    plt.tight_layout()
    path = FIG_DIR / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart 2 saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Daily Returns Distribution (Histogram + KDE)
# ══════════════════════════════════════════════════════════════════════════════

def plot_returns_distribution(data: dict[str, pd.DataFrame]) -> Path:
    """
    Overlaid histogram + KDE of daily returns for all tickers.
    Normal distribution reference curve included per ticker.

    Returns the saved file path.
    """
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), sharey=False)
    if len(data) == 1:
        axes = [axes]

    fig.suptitle("Daily Returns Distribution", fontsize=15, fontweight="bold")

    for ax, (ticker, df) in zip(axes, data.items()):
        returns = df["Daily_Return"].dropna()
        color   = TICKER_COLORS.get(ticker, "#555555")

        # Histogram
        ax.hist(returns, bins=60, color=color, alpha=0.45, density=True, label="Empirical")

        # KDE overlay via seaborn
        sns.kdeplot(returns, ax=ax, color=color, linewidth=2, label="KDE")

        # Normal reference
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 300)
        ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                "k--", linewidth=1.2, label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")

        # Vertical lines for mean and ±1σ
        ax.axvline(mu,          color="black",  linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(mu + sigma,  color="grey",   linestyle=":",  linewidth=0.8, alpha=0.7)
        ax.axvline(mu - sigma,  color="grey",   linestyle=":",  linewidth=0.8, alpha=0.7)

        ax.set_title(f"{ticker} Daily Returns")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "returns_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart 3 saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Bollinger Bands
# ══════════════════════════════════════════════════════════════════════════════

def plot_bollinger_bands(ticker: str, df: pd.DataFrame, window: int = 20) -> Path:
    """
    Classic Bollinger Band chart:
      - Middle band  : MA{window}
      - Upper/Lower  : MA ± 2 × rolling std
      - Shaded band between upper and lower
      - Volatility subplot below

    Returns the saved file path.
    """
    df = df.copy()
    df["BB_Mid"]   = df["Close"].rolling(window).mean()
    df["BB_Std"]   = df["Close"].rolling(window).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df = df.dropna(subset=["BB_Mid"])

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle(f"{ticker} — Bollinger Bands (Window={window}) & Volatility",
                 fontsize=16, fontweight="bold")

    gs    = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    ax_bb = fig.add_subplot(gs[0])
    ax_vl = fig.add_subplot(gs[1], sharex=ax_bb)

    color = TICKER_COLORS.get(ticker, "#333333")

    # ── Bollinger Band panel ──────────────────────────────────────────────────
    ax_bb.plot(df["Date"], df["Close"],    color=color,   linewidth=1.4, label="Close", alpha=0.9)
    ax_bb.plot(df["Date"], df["BB_Mid"],   color="navy",  linewidth=1.2, linestyle="--", label=f"MA{window}")
    ax_bb.plot(df["Date"], df["BB_Upper"], color="tomato",linewidth=0.9, linestyle="--", label="Upper Band (+2σ)")
    ax_bb.plot(df["Date"], df["BB_Lower"], color="tomato",linewidth=0.9, linestyle="--", label="Lower Band (−2σ)")
    ax_bb.fill_between(df["Date"], df["BB_Upper"], df["BB_Lower"],
                        color="tomato", alpha=0.08, label="Band Width")

    ax_bb.set_ylabel("Price (USD)")
    ax_bb.legend(loc="upper left")
    ax_bb.tick_params(labelbottom=False)

    # ── Volatility panel ─────────────────────────────────────────────────────
    ax_vl.plot(df["Date"], df["Volatility"] * 100, color="purple", linewidth=1.2)
    ax_vl.fill_between(df["Date"], df["Volatility"] * 100, alpha=0.2, color="purple")
    ax_vl.set_ylabel("Volatility (%)")
    ax_vl.set_xlabel("Date")
    ax_vl.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_vl.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_vl.xaxis.get_majorticklabels(), rotation=30, ha="right")

    path = FIG_DIR / f"bollinger_bands_{ticker}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart 4 saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED SUMMARY EXPORT  (for Member A/B — LLM prompt context)
# ══════════════════════════════════════════════════════════════════════════════

def build_structured_summary(data: dict[str, pd.DataFrame]) -> dict:
    """
    Build a structured dict with key statistics per ticker.
    Exported to reports/structured_summary.json for use in LLM prompts.

    Fields per ticker:
        period_start / period_end
        last_close, 30d_change_pct, ytd_change_pct
        avg_daily_return, daily_return_std (annualised volatility)
        max_drawdown_pct
        avg_volume_30d
        bollinger_position  : where last close sits relative to BB
        recent_outlier_dates
        moving_average_signal : "bullish" | "bearish" | "neutral"
    """
    summary = {}

    for ticker, df in data.items():
        df = df.copy().dropna(subset=["Close"])

        # ── Basic stats ───────────────────────────────────────────────────────
        last_close   = round(float(df["Close"].iloc[-1]), 2)
        close_30d    = float(df["Close"].iloc[-31]) if len(df) > 31 else float(df["Close"].iloc[0])
        change_30d   = round((last_close - close_30d) / close_30d * 100, 2)

        # YTD change
        current_year = df["Date"].iloc[-1].year
        ytd_df       = df[df["Date"].dt.year == current_year]
        ytd_start    = float(ytd_df["Close"].iloc[0]) if not ytd_df.empty else float(df["Close"].iloc[0])
        ytd_change   = round((last_close - ytd_start) / ytd_start * 100, 2)

        # ── Return metrics ────────────────────────────────────────────────────
        returns         = df["Daily_Return"].dropna()
        avg_return      = round(float(returns.mean()) * 100, 4)       # %
        ann_volatility  = round(float(returns.std()) * np.sqrt(252) * 100, 2)  # annualised %

        # ── Max drawdown ──────────────────────────────────────────────────────
        rolling_max = df["Close"].cummax()
        drawdown    = (df["Close"] - rolling_max) / rolling_max
        max_drawdown = round(float(drawdown.min()) * 100, 2)

        # ── Volume ────────────────────────────────────────────────────────────
        avg_vol_30d = int(df["Volume"].tail(30).mean())

        # ── Bollinger position ────────────────────────────────────────────────
        win = 20
        if len(df) >= win:
            bb_mid   = df["Close"].rolling(win).mean().iloc[-1]
            bb_std   = df["Close"].rolling(win).std().iloc[-1]
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            if last_close > bb_upper:
                bb_pos = "above_upper_band (potentially overbought)"
            elif last_close < bb_lower:
                bb_pos = "below_lower_band (potentially oversold)"
            else:
                pct = round((last_close - bb_lower) / (bb_upper - bb_lower) * 100, 1)
                bb_pos = f"within_band ({pct}% from lower to upper)"
        else:
            bb_pos = "insufficient_data"

        # ── MA signal ────────────────────────────────────────────────────────
        ma7_last  = df["MA7"].iloc[-1]  if "MA7"  in df.columns else None
        ma30_last = df["MA30"].iloc[-1] if "MA30" in df.columns else None

        if ma7_last and ma30_last:
            if ma7_last > ma30_last:
                ma_signal = "bullish (MA7 > MA30)"
            elif ma7_last < ma30_last:
                ma_signal = "bearish (MA7 < MA30)"
            else:
                ma_signal = "neutral"
        else:
            ma_signal = "unavailable"

        # ── Outliers ─────────────────────────────────────────────────────────
        if "Outlier_Flag" in df.columns:
            outlier_dates = (
                df[df["Outlier_Flag"] == 1]["Date"]
                .dt.strftime("%Y-%m-%d")
                .tail(5)
                .tolist()
            )
        else:
            outlier_dates = []

        summary[ticker] = {
            "period_start":          str(df["Date"].min().date()),
            "period_end":            str(df["Date"].max().date()),
            "last_close_usd":        last_close,
            "change_30d_pct":        change_30d,
            "ytd_change_pct":        ytd_change,
            "avg_daily_return_pct":  avg_return,
            "annualised_volatility_pct": ann_volatility,
            "max_drawdown_pct":      max_drawdown,
            "avg_volume_30d":        avg_vol_30d,
            "bollinger_position":    bb_pos,
            "moving_average_signal": ma_signal,
            "recent_outlier_dates":  outlier_dates,
        }

    return summary


def save_structured_summary(summary: dict) -> Path:
    """Write the structured summary JSON used by Member A's LLM module."""
    path = REPORT_DIR / "structured_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Structured summary saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_visualization(tickers: list[str] = TICKERS) -> None:
    """
    Full visualization pipeline.

    Steps
    -----
    1. Load processed feature data
    2. Chart 1 — Price + Volume         (per ticker)
    3. Chart 2 — Correlation Heatmap    (all tickers combined)
    4. Chart 3 — Returns Distribution   (all tickers combined)
    5. Chart 4 — Bollinger Bands        (per ticker)
    6. Export structured summary JSON   (for LLM module)
    """
    logger.info("=" * 60)
    logger.info("FinAgent — Visualization Module Starting")
    logger.info(f"Tickers : {tickers}")
    logger.info(f"Figures : {FIG_DIR.resolve()}")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_processed_data(tickers)

    if not data:
        logger.error("No processed data found. Run processing.py first.")
        return

    saved_files: list[Path] = []

    # ── Chart 1 — Price + Volume (per ticker) ─────────────────────────────────
    logger.info("\n── Chart 1: Price Trend + Volume ──")
    for ticker, df in data.items():
        try:
            p = plot_price_volume(ticker, df)
            saved_files.append(p)
        except Exception as e:
            logger.error(f"Chart 1 failed for {ticker}: {e}")

    # ── Chart 2 — Correlation Heatmap ─────────────────────────────────────────
    logger.info("\n── Chart 2: Correlation Heatmap ──")
    if len(data) >= 2:
        try:
            p = plot_correlation_heatmap(data)
            saved_files.append(p)
        except Exception as e:
            logger.error(f"Chart 2 failed: {e}")
    else:
        logger.warning("Need at least 2 tickers for heatmap — skipping.")

    # ── Chart 3 — Returns Distribution ────────────────────────────────────────
    logger.info("\n── Chart 3: Returns Distribution ──")
    try:
        p = plot_returns_distribution(data)
        saved_files.append(p)
    except Exception as e:
        logger.error(f"Chart 3 failed: {e}")

    # ── Chart 4 — Bollinger Bands (per ticker) ────────────────────────────────
    logger.info("\n── Chart 4: Bollinger Bands ──")
    for ticker, df in data.items():
        try:
            p = plot_bollinger_bands(ticker, df)
            saved_files.append(p)
        except Exception as e:
            logger.error(f"Chart 4 failed for {ticker}: {e}")

    # ── Structured Summary JSON ───────────────────────────────────────────────
    logger.info("\n── Structured Summary for LLM ──")
    try:
        summary = build_structured_summary(data)
        save_structured_summary(summary)
    except Exception as e:
        logger.error(f"Structured summary failed: {e}")

    # ── Final Report ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"Visualization complete. {len(saved_files)} figures saved.")
    for p in saved_files:
        logger.info(f"  → {p}")
    logger.info("=" * 60)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_visualization()
