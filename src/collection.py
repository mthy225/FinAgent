import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_news(query: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Lấy tin tức tài chính từ NewsAPI.
    
    Args:
        query: Từ khóa tìm kiếm (VD: "Walmart inflation")
        from_date: Ngày bắt đầu (YYYY-MM-DD)
        to_date: Ngày kết thúc (YYYY-MM-DD)
    
    Returns:
        DataFrame chứa các bài báo
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
        print(f"Lỗi API: {data.get('message', 'Unknown error')}")
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
        print(f"\nLấy tin tức cho: {ticker}")
        df = get_news(
            query=f"{ticker} inflation retail",
            from_date="2026-04-06",
            to_date="2026-05-06"
        )
        
        if not df.empty:
            os.makedirs("data/raw", exist_ok=True)
            filename = f"data/raw/news_{ticker.lower()}.csv"
            df.to_csv(filename, index=False)
            print(f"Đã lưu {len(df)} bài báo vào {filename}")
        else:
            print(f"Không có dữ liệu cho {ticker}")