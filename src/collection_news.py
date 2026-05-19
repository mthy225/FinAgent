import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_news(query: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch financial news from NewsAPI.
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

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Handle API errors safely
        if data.get("status") != "ok":
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        articles = data.get("articles", [])

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "date": article["publishedAt"][:10],
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"]
            }
            for article in articles
        ])

        return df

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":

    tickers = ["Walmart", "Target", "Costco"]

    for ticker in tickers:
        
        # UPDATED:'data_news/raw'
        filename = f"data_news/raw/news_{ticker.lower()}.csv"

        old_df = pd.DataFrame()

        # NewsAPI free plan only supports the last 30 days
        max_allowed_past_date = (
            datetime.today() - timedelta(days=29)
        ).strftime("%Y-%m-%d")

        # ── Load existing file if available ───────────────────────────────
        if os.path.exists(filename):

            old_df = pd.read_csv(filename)

            if not old_df.empty:

                latest_date = pd.to_datetime(old_df["date"]).max()

                calculated_start = (
                    latest_date + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")

                # Prevent requesting dates older than NewsAPI allows
                from_date = max(calculated_start, max_allowed_past_date)

                print(
                    f"\n[Existing file found] {ticker}: "
                    f"Current data until {latest_date.strftime('%Y-%m-%d')}"
                )

                print(
                    f"-> Fetching new articles starting from: {from_date}"
                )

            else:
                from_date = max_allowed_past_date

                print(
                    f"\n[Empty file] {ticker}: "
                    f"Fetching from earliest allowed date: {from_date}"
                )

        else:

            from_date = max_allowed_past_date

            print(
                f"\n[No existing file] {ticker}: "
                f"Creating new dataset from {from_date}"
            )

        to_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        # ── Fetch new articles ───────────────────────────────────────────
        new_df = get_news(
            query=f"{ticker} inflation retail",
            from_date=from_date,
            to_date=to_date
        )

        # ── Merge and save ───────────────────────────────────────────────
        if not new_df.empty:

            # Merge with old data if available
            if not old_df.empty:

                combined_df = pd.concat(
                    [old_df, new_df],
                    ignore_index=True
                )

                # Remove duplicate articles
                combined_df.drop_duplicates(
                    subset=["title"],
                    inplace=True
                )

            else:
                combined_df = new_df

            # UPDATED: Creates 'data_news/raw' directory
            os.makedirs("data_news/raw", exist_ok=True)

            combined_df.to_csv(filename, index=False)

            print(
                f"Successfully updated {filename} "
                f"with {len(combined_df)} total articles."
            )

        else:

            print(
                f"No new articles found for {ticker} "
                f"from {from_date} to {to_date}."
            )