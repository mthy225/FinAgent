import pandas as pd
import json
import os


def format_stock_summary(ticker: str) -> dict:
    """
    Read cleaned stock CSV and return summary statistics for LLM prompt.
    """
    filepath = f"data_stocks/cleaned/{ticker}_cleaned.csv"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    return {
        "ticker": ticker,
        "period": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "latest_close": round(df["Close"].iloc[-1], 2),
        "highest_close": round(df["Close"].max(), 2),
        "lowest_close": round(df["Close"].min(), 2),
        "avg_close": round(df["Close"].mean(), 2),
        "price_change_pct": round(
            (df["Close"].iloc[-1] - df["Close"].iloc[0])
            / df["Close"].iloc[0] * 100, 2
        ),
        "recent_7_days": (
            df[["Date", "Close", "Volume"]]
            .tail(7)
            .assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        )
    }


def format_news_summary(ticker_name: str) -> list:
    """
    Read cleaned news CSV and return top headlines for LLM prompt.
    """
    filepath = f"data/cleaned/news_{ticker_name}_cleaned.csv"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    df = pd.read_csv(filepath)
    return df["title"].head(5).tolist()


def build_prompt_data() -> dict:
    """
    Combine stock and news data into structured format for LLM prompt.
    """
    stock_tickers = ["WMT", "TGT", "COST"]
    news_tickers = ["walmart", "target", "costco"]

    prompt_data = {
        "topic": "Retail & Inflation Analysis",
        "stocks": {},
        "news": {}
    }

    for ticker in stock_tickers:
        prompt_data["stocks"][ticker] = format_stock_summary(ticker)

    for ticker_name in news_tickers:
        prompt_data["news"][ticker_name] = format_news_summary(ticker_name)

    return prompt_data


if __name__ == "__main__":
    print("Building structured data for LLM prompt...")
    data = build_prompt_data()

    os.makedirs("data", exist_ok=True)
    output_path = "data/prompt_input.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print("\nPreview:")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:800])