import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_news(query: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch financial news from NewsAPI.

    Args:
        query: Search keyword (e.g. "Walmart inflation")
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame containing news articles
    """
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] != "ok":
        print(f"API Error: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()

    articles = data["articles"]

    df = pd.DataFrame([{
        "date": a["publishedAt"][:10],
        "title": a["title"],
        "source": a["source"]["name"],
        "url": a["url"]
    } for a in articles])

    return df


if __name__ == "__main__":
    tickers = ["Walmart", "Target", "Costco"]
    
    for ticker in tickers:
        filename = f"data/raw/news_{ticker.lower()}.csv"
        
        # Skip if file already exists
        if os.path.exists(filename):
            print(f"Data already exists for {ticker}, skipping API call.")
            continue
        
        print(f"\nFetching news for: {ticker}")
        df = get_news(
            query=f"{ticker} inflation retail",
            from_date="2026-04-16",
            to_date="2026-05-16"
        )
        
        if not df.empty:
            os.makedirs("data/raw", exist_ok=True)
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} articles to {filename}")
        else:
            print(f"No data available for {ticker}")
