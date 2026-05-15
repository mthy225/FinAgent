"""
analysis.py
===========
Phase 4 - AI Analysis Module (Lead: Member A - Hân)

Uses Groq (primary) + Gemini (fallback) to generate:
1. Trend summary for each asset
2. Anomaly/notable events detection
3. Risk commentary based on volatility
4. Comparison between assets

Usage:
    python analysis.py

Output:
    data/analysis/ai_analysis_report.txt
    data/analysis/ai_analysis_report.json
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# ── Load API keys từ .env ──────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Logging ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "analysis_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS     = ["WMT", "TGT", "COST"]
PROCESSED_DIR = Path("data/processed")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_processed_data(ticker: str) -> pd.DataFrame:
    """Load processed CSV for a ticker."""
    path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA SUMMARIZER — tạo context ngắn gọn cho LLM
# ══════════════════════════════════════════════════════════════════════════════

def summarize_ticker(df: pd.DataFrame, ticker: str) -> dict:
    """
    Tóm tắt số liệu quan trọng của 1 ticker để đưa vào prompt.
    Chỉ lấy 30 ngày gần nhất để tránh prompt quá dài.
    """
    recent = df.tail(30).copy()
    # --- Investment metrics for decision making ---
    sharpe = recent["Daily_Return"].mean() / recent["Daily_Return"].std() if recent["Daily_Return"].std() != 0 else 0
    max_drawdown = ((recent["Close"] / recent["Close"].cummax()) - 1).min()
    positive_days = (recent["Daily_Return"] > 0).sum()
    negative_days = (recent["Daily_Return"] < 0).sum()

    latest_close     = round(float(recent["Close"].iloc[-1]), 2)
    close_30d_ago    = round(float(recent["Close"].iloc[0]), 2)
    pct_change_30d   = round((latest_close - close_30d_ago) / close_30d_ago * 100, 2)

    avg_daily_return = round(float(recent["Daily_Return"].mean()) * 100, 4)
    avg_volatility   = round(float(recent["Volatility"].mean()) * 100, 4)
    latest_ma7       = round(float(recent["MA7"].iloc[-1]), 2)
    latest_ma30      = round(float(recent["MA30"].iloc[-1]), 2)

    # Đếm outlier trong 30 ngày gần nhất
    outlier_count = int(recent["Outlier_Flag"].sum()) if "Outlier_Flag" in recent.columns else 0

    # Ngày có return cao nhất và thấp nhất
    max_return_row = recent.loc[recent["Daily_Return"].idxmax()]
    min_return_row = recent.loc[recent["Daily_Return"].idxmin()]

    return {
        "ticker":           ticker,
        "latest_close":     latest_close,
        "close_30d_ago":    close_30d_ago,
        "pct_change_30d":   pct_change_30d,
        "avg_daily_return": avg_daily_return,
        "avg_volatility":   avg_volatility,
        "ma7":              latest_ma7,
        "ma30":             latest_ma30,
        "outlier_count":    outlier_count,
        "best_day":         str(max_return_row["Date"].date()),
        "best_day_return":  round(float(max_return_row["Daily_Return"]) * 100, 2),
        "worst_day":        str(min_return_row["Date"].date()),
        "worst_day_return": round(float(min_return_row["Daily_Return"]) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown_percent": round(float(max_drawdown) * 100, 2),
        "positive_days": int(positive_days),
        "negative_days": int(negative_days),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(summaries: list[dict], structured_data) -> str:
    """
    Xây dựng prompt có cấu trúc JSON để LLM phân tích.
    Prompt engineering: cung cấp số liệu cụ thể, yêu cầu output rõ ràng.
    """
    data_block = json.dumps(summaries, indent=2)

    prompt = f"""You are a professional financial analyst specializing in retail sector stocks.
You have been given 30-day performance data for three major US retail companies: Walmart (WMT), Target (TGT), and Costco (COST).

## INPUT DATA (last 30 trading days):
{data_block}

## YOUR TASK:
Based ONLY on the data provided above, write a structured financial analysis report with these 5 sections:

### 1. TREND SUMMARY
For each ticker (WMT, TGT, COST), write 2-3 sentences describing:
- Current price trend (bullish/bearish/sideways) based on price change and MA7 vs MA30
- Recent 30-day performance with specific numbers from the data

### 2. NOTABLE EVENTS & ANOMALIES
- Identify any unusual trading days (outlier_count > 0)
- Mention specific dates with the best/worst single-day returns
- Flag any tickers with abnormally high volatility

### 3. RISK COMMENTARY
- Compare volatility levels across the 3 tickers
- Identify which ticker carries the most/least risk
- Provide a brief risk assessment for a short-term investor (1 month horizon)

### 4. COMPARATIVE ANALYSIS
- Compare the 30-day performance of all 3 tickers
- Which ticker performed best/worst and by how much?
- What does the MA7 vs MA30 relationship tell us about momentum?

### 5. PORTFOLIO DECISION GUIDANCE (MOST IMPORTANT)

Based on all numerical data above:

Assume an investor has $10,000 for a 1-month investment horizon.

1. Recommend how to allocate weight (%) across WMT, TGT, COST.
2. Justify using:
    - sharpe_ratio
    - volatility
    - max_drawdown_percent
    - price trend (MA7 vs MA30)
3. Clearly state:
    - Which ticker is best for stability
    - Which ticker is best for growth
    - Which ticker is most risky

This section must be quantitative, actionable, and data-referenced.

## RULES:
- Reference specific numbers from the data in every section
- Be concise: max 150 words per section
- Do NOT fabricate data not present in the input
- Use professional financial language
"""

    prompt += f"""

### 6. INSIGHTS FROM VISUALIZATION (Chart-derived data)

The following structured data was extracted from charts:

{json.dumps(structured_data, indent=2)}

Use this to support:
- trend summary
- anomaly explanation
- risk commentary
- comparison
- portfolio allocation decision
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def call_groq(prompt: str) -> str:
    """Gọi Groq API (primary)."""
    logger.info("Calling Groq API...")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def call_gemini(prompt: str) -> str:
    """Gọi Gemini API (fallback)."""
    logger.info("Calling Gemini API (fallback)...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def call_llm(prompt: str) -> tuple[str, str]:
    """
    Gọi Groq trước, nếu lỗi thì fallback sang Gemini.
    Returns: (analysis_text, model_used)
    """
    try:
        result = call_groq(prompt)
        logger.info("✓ Groq API success")
        return result, "groq/llama-3.3-70b-versatile"
    except Exception as e:
        logger.warning(f"Groq failed: {e} → switching to Gemini")
        try:
            result = call_gemini(prompt)
            logger.info("✓ Gemini API success")
            return result, "gemini/gemini-1.5-flash"
        except Exception as e2:
            raise RuntimeError(f"Both APIs failed. Groq: {e} | Gemini: {e2}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(analysis: str, summaries: list, model_used: str) -> None:
    """Lưu báo cáo ra cả .txt và .json."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── TXT report ──
    txt_path = OUTPUT_DIR / "ai_analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"FinAgent AI Analysis Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Model: {model_used}\n")
        f.write("=" * 60 + "\n\n")
        f.write(analysis)
    logger.info(f"Saved TXT → {txt_path}")

    # ── JSON report ──
    json_path = OUTPUT_DIR / "ai_analysis_report.json"
    output = {
        "generated_at": timestamp,
        "model_used":   model_used,
        "input_data":   summaries,
        "analysis":     analysis,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(tickers: list = TICKERS) -> str:
    """
    Full AI analysis pipeline.
    Returns the analysis text.
    """
    logger.info("=" * 60)
    logger.info("FinAgent — AI Analysis Module Starting")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 60)

    # Step 1: Load & summarize data
    summaries = []
    for ticker in tickers:
        try:
            df = load_processed_data(ticker)
            summary = summarize_ticker(df, ticker)
            summaries.append(summary)
            logger.info(f"✓ Loaded {ticker}: close={summary['latest_close']}, 30d change={summary['pct_change_30d']}%")
        except Exception as e:
            logger.error(f"✗ Failed to load {ticker}: {e}")

    if not summaries:
        raise RuntimeError("No data loaded. Cannot proceed.")

    # Step 2: Build prompt
    # Load structured chart-derived data for section 6 of the prompt
    with open("reports/structured_summary.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    prompt = build_prompt(summaries, structured_data)
    logger.info(f"Prompt built ({len(prompt)} chars)")

    # Step 3: Call LLM
    analysis, model_used = call_llm(prompt)
    logger.info(f"Analysis generated ({len(analysis)} chars)")

    # Step 4: Save
    save_report(analysis, summaries, model_used)

    logger.info("AI Analysis complete!\n")

    # Print to console
    print("\n" + "=" * 60)
    print("AI ANALYSIS REPORT")
    print("=" * 60)
    print(analysis)
    print("=" * 60)

    return analysis


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_analysis()