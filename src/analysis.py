"""
analysis.py
===========
Phase 4 - AI Portfolio Advisor Module (Lead: Member A - Hân)

Level: Investment Intelligence — reads stock BEHAVIOR, not just metrics.

Uses Groq (primary) + Gemini (fallback).

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
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# ── Load API keys ──────────────────────────────────────────────────────────────
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

TICKERS       = ["WMT", "TGT", "COST"]
PROCESSED_DIR = Path("data/processed")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_processed_data(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def summarize_ticker(df: pd.DataFrame, ticker: str) -> dict:
    recent_30 = df.tail(30).copy()
    recent_7  = df.tail(7).copy()

    def pct_chg(n):
        if len(df) < n + 1:
            return None
        return round((df["Close"].iloc[-1] - df["Close"].iloc[-n]) / df["Close"].iloc[-n] * 100, 2)

    # Momentum
    ret_7d  = pct_chg(7)
    ret_30d = pct_chg(30)
    ret_90d = pct_chg(90)

    ma7  = round(float(df["MA7"].iloc[-1]), 2)
    ma30 = round(float(df["MA30"].iloc[-1]), 2)
    ma_gap_pct = round((ma7 - ma30) / ma30 * 100, 3) if ma30 else 0

    x = np.arange(len(recent_30))
    y = recent_30["Close"].values
    slope = round(float(np.polyfit(x, y, 1)[0]), 4)

    # === RISK METRICS FOR VaR (NEW) ===
    daily_returns = df["Close"].pct_change().dropna()

    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    # 95% one-day VaR (parametric)
    var_95 = avg_daily_return - (1.645 * std_daily_return)

    # 1-month VaR estimate using sqrt time rule (22 trading days)
    monthly_var_95 = var_95 * (22 ** 0.5)

    # Deep Risk
    max_drawdown = round(float(((df["Close"] / df["Close"].cummax()) - 1).min()) * 100, 2)

    consec = cur = 0
    for r in df["Daily_Return"].dropna():
        if r < 0:
            cur += 1
            consec = max(consec, cur)
        else:
            cur = 0

    vol_recent = recent_30.tail(15)["Daily_Return"].std() * 100
    vol_prev   = recent_30.head(15)["Daily_Return"].std() * 100
    vol_trend  = "decreasing" if vol_recent < vol_prev else "increasing"
    vol_change = round(vol_recent - vol_prev, 4)
    avg_vol    = round(float(recent_30["Volatility"].mean()) * 100, 4)

    # Consistency
    ret_30 = recent_30["Daily_Return"].dropna()
    pos_days = int((ret_30 > 0).sum())
    neg_days = int((ret_30 < 0).sum())
    win_rate = round(pos_days / (pos_days + neg_days) * 100, 1)
    std_ret  = round(float(ret_30.std()) * 100, 4)

    # Sharpe & close
    sharpe = round(float(ret_30.mean() / ret_30.std()), 4) if ret_30.std() != 0 else 0
    latest_close = round(float(df["Close"].iloc[-1]), 2)

    # Events
    outlier_count = int(recent_30["Outlier_Flag"].sum()) if "Outlier_Flag" in recent_30.columns else 0
    max_row = recent_30.loc[recent_30["Daily_Return"].idxmax()]
    min_row = recent_30.loc[recent_30["Daily_Return"].idxmin()]

    # ── Phase classification (Python-side, không để AI tự suy) ───────────────
    phase, phase_reason = classify_phase(
        ret_7d, ret_30d, ret_90d,
        ma_gap_pct, slope,
        max_drawdown, avg_vol, vol_trend,
        win_rate, sharpe
    )

    # ── Risk level ────────────────────────────────────────────────────────────
    risk_level = classify_risk(avg_vol, max_drawdown, sharpe, vol_trend)

    # ── Portfolio role ────────────────────────────────────────────────────────
    role, role_reason = classify_role(
        ret_90d, ma_gap_pct, avg_vol,
        max_drawdown, win_rate, sharpe, slope
    )

    return {
        "ticker":        ticker,
        "latest_close":  latest_close,
        # Momentum
        "return_7d_pct":   ret_7d,
        "return_30d_pct":  ret_30d,
        "return_90d_pct":  ret_90d,
        "ma7":             ma7,
        "ma30":            ma30,
        "ma_gap_pct":      ma_gap_pct,
        "price_slope_30d": slope,
        "avg_daily_return": round(avg_daily_return, 6),
        "std_daily_return": round(std_daily_return, 6),
        "monthly_var_95": round(monthly_var_95, 6),
        # Deep Risk
        "max_drawdown_pct":          max_drawdown,
        "max_consecutive_neg_days":  consec,
        "volatility_trend":          vol_trend,
        "volatility_change_pct":     vol_change,
        "avg_volatility_30d_pct":    avg_vol,
        # Consistency
        "positive_days_30d": pos_days,
        "negative_days_30d": neg_days,
        "win_rate_pct":      win_rate,
        "std_daily_return":  std_ret,
        # Risk-adjusted
        "sharpe_ratio": sharpe,
        # Events
        "outlier_count":        outlier_count,
        "best_day":             str(max_row["Date"].date()),
        "best_day_return_pct":  round(float(max_row["Daily_Return"]) * 100, 2),
        "worst_day":            str(min_row["Date"].date()),
        "worst_day_return_pct": round(float(min_row["Daily_Return"]) * 100, 2),
        # Pre-classified (Python-side)
        "stock_phase":   phase,
        "phase_reason":  phase_reason,
        "risk_level":    risk_level,
        "portfolio_role": role,
        "role_reason":   role_reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON-SIDE CLASSIFIERS
# (tính trước để AI không bị trống Section 6)
# ══════════════════════════════════════════════════════════════════════════════

import math

def compute_monthly_var(avg_daily_return_pct, std_daily_return_pct, investment=10000):
    mu = avg_daily_return_pct / 100
    sigma = std_daily_return_pct / 100
    var_day = mu - 1.645 * sigma
    var_month = var_day * math.sqrt(22)
    monthly_loss = var_month * investment
    return round(monthly_loss, 2)

def classify_phase(ret_7d, ret_30d, ret_90d, ma_gap_pct, slope,
                   max_drawdown, avg_vol, vol_trend, win_rate, sharpe) -> tuple[str, str]:
    """
    Đọc phase cổ phiếu từ tổ hợp metrics — không đọc từng cái riêng lẻ.

    Logic:
    - Post-crash recovery: 90d cao + drawdown rất sâu + ma_gap lớn (institutional re-entry)
    - Defensive accumulation: drawdown thấp + vol giảm + winrate cao + return đều
    - Mature/stable trend: slope tốt + vol thấp + return đều nhưng không explosive
    - Breakout momentum: cả 7d/30d/90d đều tăng đều + ma_gap lớn + slope dốc
    """
    # Post-crash recovery: 90d rất cao + drawdown sâu (> -35%) + ma_gap lớn
    if ret_90d and ret_90d > 25 and max_drawdown < -35 and ma_gap_pct > 2:
        return (
            "Post-Crash Recovery",
            f"90d return {ret_90d}% is high but max drawdown is {max_drawdown}% — "
            f"this stock collapsed historically and is now recovering. "
            f"MA gap {ma_gap_pct}% signals institutional re-entry, not organic growth."
        )

    # Defensive accumulation: vol giảm + winrate cao + drawdown nhỏ + return ổn định
    if vol_trend == "decreasing" and win_rate >= 60 and max_drawdown > -25 and sharpe > 0.1:
        return (
            "Defensive Accumulation",
            f"Volatility is {vol_trend}, win rate is {win_rate}%, "
            f"max drawdown only {max_drawdown}% — institutional money is quietly accumulating. "
            f"This stock moves slowly but reliably upward."
        )

    # Mature stable trend: slope tốt + vol thấp + return không explosive
    if avg_vol < 1.3 and slope > 0 and ret_30d and ret_30d < 8:
        return (
            "Mature Stable Trend",
            f"Low volatility ({avg_vol}%), positive slope ({slope}), "
            f"but modest 30d return ({ret_30d}%) — this is a steady compounder. "
            f"Reliable but unlikely to deliver explosive gains."
        )

    # Breakout momentum: 7d/30d/90d tất cả tăng đều + slope tốt
    if ret_7d and ret_30d and ret_90d and ret_7d > 0 and ret_30d > 5 and ma_gap_pct > 1.5:
        return (
            "Breakout Momentum",
            f"Returns across 7d ({ret_7d}%), 30d ({ret_30d}%), 90d ({ret_90d}%) "
            f"are consistently positive. MA gap {ma_gap_pct}% confirms short-term strength. "
            f"Momentum is broad-based and accelerating."
        )

    return (
        "Consolidation / Unclear Trend",
        f"Mixed signals: 30d return {ret_30d}%, ma_gap {ma_gap_pct}%, "
        f"avg volatility {avg_vol}%. No clear directional conviction yet."
    )


def classify_risk(avg_vol: float, max_drawdown: float,
                  sharpe: float, vol_trend: str) -> str:
    """
    Risk level dựa trên HIỆN TẠI, không chỉ quá khứ.
    vol_trend giảm + sharpe tốt = risk hiện tại thấp hơn drawdown lịch sử cho thấy.
    """
    # Nếu vol đang giảm + sharpe tốt → risk hiện tại thấp hơn quá khứ
    if vol_trend == "decreasing" and sharpe > 0.15 and avg_vol < 1.5:
        return "Medium (improving — past drawdown overstates current risk)"
    if avg_vol > 1.6 or max_drawdown < -40:
        return "High"
    if avg_vol < 1.2 and max_drawdown > -25:
        return "Low"
    return "Medium"


def classify_role(ret_90d, ma_gap_pct, avg_vol, max_drawdown,
                  win_rate, sharpe, slope) -> tuple[str, str]:
    """
    Portfolio role dựa trên tổ hợp metrics.
    """
    # Growth Driver: momentum mạnh + return cao + chấp nhận risk cao hơn
    if ret_90d and ret_90d > 25 and ma_gap_pct > 2.5:
        return (
            "Growth Driver",
            f"Highest 90d return ({ret_90d}%) and strongest MA gap ({ma_gap_pct}%) "
            f"make this the portfolio's return engine — but accept higher volatility."
        )

    # Defensive Hedge: drawdown thấp + vol thấp + win rate cao
    if max_drawdown > -25 and avg_vol < 1.6 and win_rate >= 63:
        return (
            "Defensive Hedge",
            f"Low max drawdown ({max_drawdown}%), win rate {win_rate}%, "
            f"and declining volatility make this the portfolio's shock absorber."
        )

    # Balanced Core: không extreme ở chiều nào
    return (
        "Balanced Core",
        f"Moderate on all dimensions — slope {slope}, avg vol {avg_vol}%, "
        f"sharpe {sharpe} — makes this the stable anchor of the portfolio."
    )


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlations(dfs: dict) -> dict:
    returns = pd.DataFrame({
        t: df.set_index("Date")["Daily_Return"]
        for t, df in dfs.items()
    }).dropna()
    corr = returns.corr().round(4)
    result = {}
    tickers = list(dfs.keys())
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            result[f"{t1}_vs_{t2}"] = float(corr.loc[t1, t2])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT — Investment Intelligence Level
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(summaries: list, correlations: dict, rankings_block: str) -> str:
    data_block = json.dumps(summaries, indent=2)
    corr_block = json.dumps(correlations, indent=2)

    # Extract pre-classified info for Section 6
    roles_block = "\n".join([
        f"- {s['ticker']}: Phase = {s['stock_phase']} | Role = {s['portfolio_role']} | Risk = {s['risk_level']}"
        for s in summaries
    ])

    prompt = f"""You are a senior investment advisor at a hedge fund. Your job is NOT to narrate numbers.
Your job is to READ STOCK BEHAVIOR from numbers and tell the investor what is REALLY happening.

The difference:
- Data narration: "TGT has high 90d return and high MA gap"
- Investment intelligence: "TGT is in a post-crash recovery — it collapsed 49.78% historically and institutions are now re-entering. This is explosive but emotionally hard to hold."

ALWAYS interpret the COMBINATION of metrics together, never one metric in isolation.

## STOCK DATA (pre-analyzed by Python):
{data_block}

## CORRELATION MATRIX:
{corr_block}

## PRE-CLASSIFIED PHASES AND ROLES (use these as anchors, expand with your interpretation):
{roles_block}

---

Write a 7-section AI Portfolio Advisory Report. Write like a human expert, not like a data summary tool.

---

## CRITICAL NUMERIC READING RULES (MUST FOLLOW)

When comparing ANY metric between tickers (price_slope_30d, return, volatility, MA gap, drawdown):

1. You MUST explicitly sort the numbers from highest to lowest before writing.
2. You MUST show the sorted order in the sentence.
3. You are NOT allowed to describe momentum, strength, or phase without referencing this sorted order.
4. NEVER infer strength from narrative (returns, stories, recovery). ONLY use the numeric ranking.

Example format:
"Based on price_slope_30d, the correct order is:
COST (0.5187) > TGT (0.4715) > WMT (0.2835)"

If this rule is violated, the analysis is considered incorrect.

### SECTION 1: STOCK PHASE DIAGNOSIS — "What is each stock actually doing right now?"

For each ticker, read its BEHAVIOR by combining multiple metrics:
- What phase is it in? (Use the pre-classified phase as anchor, then EXPLAIN why with specific numbers)
- What does the COMBINATION of ret_7d + ret_30d + ret_90d + ma_gap_pct + slope tell you?
  * Is momentum accelerating or fading? (compare 7d vs 30d vs 90d)
  * Is the MA gap widening (building momentum) or narrowing (losing steam)?
- Critically: is the high return STRUCTURAL growth or a RECOVERY BOUNCE from past crash?
  * Hint: cross-reference max_drawdown_pct — if drawdown was very deep, high 90d return may just be mean reversion
- Write 1 vivid, insight-driven paragraph per ticker (like the examples below)

Example of the level required:
> "TGT is not simply strong — it is in a post-collapse recovery phase. The 32.11% 90-day return looks impressive until you see the -49.78% max drawdown. Institutions are re-entering after a historic collapse. This momentum is explosive but fragile — emotionally hard to hold, and could reverse sharply if sentiment shifts."

> "WMT shows a completely different character. Moderate returns, high win rate, and shrinking volatility signal defensive accumulation. This is where institutional capital hides during uncertainty — slow, boring, but remarkably reliable."

> "COST is the definition of a mature compounder. Low volatility, consistent slope, but weaker MA gap means it delivers steady returns without drama. You won't get rich quick, but you'll sleep well."

---

SECTION 2 — RISK READING (STRICT INSTRUCTION)

You MUST NOT estimate 1-month loss using "worst day × 22".

You MUST use the provided metric: monthly_var_95.

Formula already applied in data:
monthly_var_95 = (avg_daily_return - 1.645 * std_daily_return) * sqrt(22)

Interpretation rule:
Estimated 1-month worst-case loss = monthly_var_95 × investment_amount.

This is the ONLY correct risk estimate to use.
Explain risk using:
- std_daily_return
- max_drawdown_pct
- monthly_var_95

Do not invent any other loss formula.

---

## PRECOMPUTED RANKINGS (YOU MUST FOLLOW THIS ORDER, DO NOT RE-SORT):
{rankings_block}

### SECTION 3: CONSISTENCY READING — "Steady grower or lucky gambler?"

IMPORTANT RULE: Low std is NOT always good. A stock with low std AND low return is a dead stock.
A GREAT consistent stock has: high win_rate + reasonable return + low std (consistency ON TOP OF growth).

For each ticker:
- Read the COMBINATION of win_rate_pct + std_daily_return + return_30d together
- Is this a consistent GROWER or a consistent STAGNATOR?
- Which ticker gives you the best "sleep quality" as an investor? (high win rate + low std)
- Which ticker is a high-variance gamble? (high return potential but wide daily swings)
- Rank: most consistent grower → least consistent
From a pure consistency standpoint (std_daily_return + win), the correct order is:
---

### SECTION 4: CORRELATION — "Are you actually diversifying or just doubling down?"

For each pair, interpret the correlation number meaningfully:
- WMT vs TGT ({correlations.get('WMT_vs_TGT', 'N/A')}): what does this mean practically?
- WMT vs COST ({correlations.get('WMT_vs_COST', 'N/A')}): what does this mean practically?
- TGT vs COST ({correlations.get('TGT_vs_COST', 'N/A')}): what does this mean practically?

Then answer:
- If you hold all 3, are you truly diversified or are some positions redundant?
- Which pair gives the MOST diversification benefit (lowest correlation)?
- Portfolio construction implication: does the correlation pattern suggest concentrating or spreading?

---

### SECTION 5: MACRO SENSITIVITY — "Who survives when things get tough?"

Connect the stock behavior to the macro environment:
- These are retail stocks — they are sensitive to inflation (CPI), consumer spending, and interest rates
- Based on volatility pattern and drawdown history, which ticker is most DEFENSIVE when macro deteriorates?
- Which ticker is most CYCLICAL — moves sharply with consumer sentiment?
- If CPI rises → which ticker holds up best and why?
- If consumer spending weakens → which ticker gets hit hardest and why?
- Assign macro label: Defensive / Balanced / Cyclical — with specific number justification

---

### SECTION 6: PORTFOLIO ROLE TABLE

Using the pre-classified roles (anchors provided), fill and EXPAND this table:

| Ticker | Phase | Portfolio Role | Risk Level | Key Insight (1 sentence, cite 2 numbers) |
|--------|-------|----------------|------------|------------------------------------------|
| WMT    | [phase] | [role] | [risk] | [insight with numbers] |
| TGT    | [phase] | [role] | [risk] | [insight with numbers] |
| COST   | [phase] | [role] | [risk] | [insight with numbers] |

Then write 2-3 sentences explaining how these 3 roles COMPLEMENT each other in a portfolio.

---

SECTION 7 — STRICT NUMERIC RULES (MANDATORY)

You are given for each ticker:
- return_30d (in %)
- monthly_var_95 (in decimal, e.g. -0.096069)

Bull case portfolio value MUST be calculated as:
$10,000 + sum(weight_i × capital × (return_30d_i / 100))

Bear case portfolio value MUST be calculated as:
$10,000 + SUM( weight_i × monthly_var_95_i × 10000 )

You MUST show the math.
You are NOT allowed to assume, estimate, or invent any percentage.

IMPORTANT:
return_30d_pct is in PERCENT form (e.g. 5.7 means 5.7%).
You MUST divide by 100 when multiplying with money.

### SECTION 7: ALLOCATION FOR 3 INVESTOR PROFILES ($10,000)

For each profile, give exact allocation with full reasoning rooted in the data:

**Profile A — Conservative**
- Priority: capital preservation, sleep at night
- Allocation: WMT ?% = $?, TGT ?% = $?, COST ?% = $? (must sum to $10,000)
- Key metrics driving this: max_drawdown_pct, win_rate_pct, volatility_trend, sharpe_ratio
- Bull case (trend continues 30 more days): expected portfolio value = $?
- Bear case (max drawdown repeats): expected portfolio value = $?
- One-line verdict: why THIS allocation for THIS profile

**Profile B — Balanced**
- Priority: growth + stability, moderate risk
- Allocation: WMT ?% = $?, TGT ?% = $?, COST ?% = $?
- Key metrics: return_30d, sharpe_ratio, correlation (use actual numbers)
- Bull case: expected portfolio value = $?
- Bear case: expected portfolio value = $?
- One-line verdict

**Profile C — Aggressive**
- Priority: maximum return, willing to accept drawdowns
- Allocation: WMT ?% = $?, TGT ?% = $?, COST ?% = $?
- Key metrics: return_7d, return_90d, ma_gap_pct, stock_phase
- Bull case: expected portfolio value = $?
- Bear case: expected portfolio value = $?
- One-line verdict

For Bull/Bear case projections, you MUST derive numbers from:
- return_30d for bull case
- monthly_var_95 for bear case

End Section 7 with: "The single most important risk each profile must accept."

---

## RULES — NON-NEGOTIABLE:
1. NEVER write a sentence that only describes a number without interpreting what it MEANS for the investor
2. NEVER rank without saying WHY the ranking exists in plain English
3. ALWAYS cross-reference at least 2 metrics when making any claim
4. Section 6 table MUST be fully filled — no empty cells, no "?" left
5. All dollar amounts in Section 7 must be calculated, not estimated vaguely
6. Write like a human expert who cares about the investor's money, not like a report generator
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def call_groq(prompt: str) -> str:
    logger.info("Calling Groq API...")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=6000,
    )
    return response.choices[0].message.content


def call_gemini(prompt: str) -> str:
    logger.info("Calling Gemini API (fallback)...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def call_llm(prompt: str) -> tuple[str, str]:
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
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_report(analysis: str, summaries: list, correlations: dict, model_used: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    txt_path = OUTPUT_DIR / "ai_analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FinAgent AI Portfolio Advisor Report\n")
        f.write(f"Generated : {timestamp}\n")
        f.write(f"Model     : {model_used}\n")
        f.write("=" * 60 + "\n\n")
        f.write(analysis)
    logger.info(f"Saved TXT → {txt_path}")

    json_path = OUTPUT_DIR / "ai_analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": timestamp,
            "model_used":   model_used,
            "input_data":   summaries,
            "correlations": correlations,
            "analysis":     analysis,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(tickers: list = TICKERS) -> str:
    logger.info("=" * 60)
    logger.info("FinAgent — AI Portfolio Advisor (Investment Intelligence)")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 60)

    # Load
    dfs = {}
    for ticker in tickers:
        try:
            dfs[ticker] = load_processed_data(ticker)
            logger.info(f"✓ Loaded {ticker}")
        except Exception as e:
            logger.error(f"✗ {ticker}: {e}")

    if not dfs:
        raise RuntimeError("No data loaded.")

    # Summarize
    summaries = []
    for ticker, df in dfs.items():
        try:
            s = summarize_ticker(df, ticker)
            summaries.append(s)
            logger.info(
                f"✓ {ticker}: phase={s['stock_phase']} | "
                f"role={s['portfolio_role']} | risk={s['risk_level']}"
            )
        except Exception as e:
            logger.error(f"✗ summarize {ticker}: {e}")

    # =========================
    # Ranking helpers
    # =========================

    def rank_desc(metric):
        return sorted(
            summaries,
            key=lambda x: x.get(metric, float("-inf")),
            reverse=True
        )

    def rank_asc(metric):
        return sorted(
            summaries,
            key=lambda x: x.get(metric, float("inf"))
        )

    def portfolio_bear_case(allocation_dict):
        total = 10000
        loss = 0

        for ticker, weight in allocation_dict.items():
            stock = next(x for x in summaries if x["ticker"] == ticker)
            loss += total * weight * stock["monthly_var_95"]

        return round(total + loss, 2)

    win_rate_rank = " > ".join(
        [f"{x['ticker']} ({x['win_rate_pct']}%)" for x in rank_desc("win_rate_pct")]
    )

    std_rank = " < ".join(
        [f"{x['ticker']} ({x['std_daily_return']})" for x in rank_asc("std_daily_return")]
    )

    return_rank = " > ".join(
        [f"{x['ticker']} ({x['return_30d_pct']}%)" for x in rank_desc("return_30d_pct") if x.get("return_30d_pct") is not None]
    )

    rankings_block = f"""
    WIN RATE RANK (high to low):
    {win_rate_rank}

    STD DAILY RETURN RANK (low to high):
    {std_rank}

    30D RETURN RANK (high to low):
    {return_rank}
    """
    # Correlations
    correlations = {}
    try:
        correlations = compute_correlations(dfs)
        logger.info(f"✓ Correlations: {correlations}")
    except Exception as e:
        logger.warning(f"Correlation failed: {e}")

    # Prompt
    prompt = build_prompt(summaries, correlations, rankings_block)
    logger.info(f"Prompt: {len(prompt)} chars")

    # LLM
    analysis, model_used = call_llm(prompt)
    logger.info(f"Analysis: {len(analysis)} chars")

    # Save
    save_report(analysis, summaries, correlations, model_used)
    logger.info("Done!\n")

    print("\n" + "=" * 60)
    print("AI PORTFOLIO ADVISOR REPORT")
    print("=" * 60)
    print(analysis)
    print("=" * 60)

    return analysis


if __name__ == "__main__":
    run_analysis()