"""
data_collector.py
Fetches all stock data from yfinance and NewsAPI.
"""

import os
import time
import datetime
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Disable SSL verification warnings (corporate proxy / self-signed cert environments)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Shared requests session: SSL disabled + retry on rate-limit / transient errors
_SESSION = requests.Session()
_SESSION.verify = False

# Retry up to 5 times with exponential backoff on 429 / 5xx responses
_RETRY = Retry(
    total=5,
    backoff_factor=2,          # waits 2, 4, 8, 16, 32 s between retries
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD", "OPTIONS"],
    raise_on_status=False,
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

# Mimic a real browser to avoid Yahoo Finance bot-detection
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def _yf_ticker(ticker: str) -> yf.Ticker:
    """Return a yfinance Ticker, retrying up to 3 times on rate-limit errors."""
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker, session=_SESSION)
            # Force a lightweight call to confirm the ticker resolves
            _ = t.fast_info
            return t
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e) or "Rate" in str(e):
                wait = (attempt + 1) * 5
                time.sleep(wait)
            else:
                raise
    # Final attempt without retry guard
    return yf.Ticker(ticker, session=_SESSION)


def get_stock_data(ticker: str) -> dict:
    """Fetch comprehensive stock data from yfinance."""
    t = _yf_ticker(ticker)

    # Price history – 1 year
    history = t.history(period="1y")
    if history.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol and try again.")

    time.sleep(0.5)  # brief pause between burst calls

    # Company info
    info = t.info or {}
    time.sleep(0.5)

    # Financials
    try:
        financials = t.financials
        time.sleep(0.3)
    except Exception:
        financials = pd.DataFrame()

    try:
        balance_sheet = t.balance_sheet
        time.sleep(0.3)
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cashflow = t.cashflow
        time.sleep(0.3)
    except Exception:
        cashflow = pd.DataFrame()

    # Analyst data
    try:
        recommendations = t.recommendations
        time.sleep(0.3)
    except Exception:
        recommendations = pd.DataFrame()

    try:
        analyst_targets = t.analyst_price_targets
        time.sleep(0.3)
    except Exception:
        analyst_targets = {}

    # Earnings dates
    try:
        earnings_dates = t.earnings_dates
        time.sleep(0.3)
    except Exception:
        earnings_dates = pd.DataFrame()

    # yfinance built-in news
    try:
        yf_news = t.news or []
    except Exception:
        yf_news = []

    return {
        "ticker": ticker.upper(),
        "history": history,
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "recommendations": recommendations,
        "analyst_targets": analyst_targets,
        "earnings_dates": earnings_dates,
        "yf_news": yf_news,
    }


def get_news_articles(ticker: str, company_name: str) -> list[dict]:
    """Fetch news articles from NewsAPI for the last 12 months."""
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        return []

    from_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    query = f"{ticker} OR \"{company_name}\"" if company_name else ticker

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = _SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except Exception:
        return []


def get_stocktwits_sentiment(ticker: str) -> dict:
    """Fetch recent StockTwits messages for a ticker (no API key required)."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        response = _SESSION.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        messages = data.get("messages", [])
        return {"messages": messages, "symbol": ticker}
    except Exception:
        return {"messages": [], "symbol": ticker}


def get_analyst_summary(data: dict) -> dict:
    """Extract a clean analyst consensus from recommendations data."""
    recs = data.get("recommendations")
    if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
        return {"buy": 0, "hold": 0, "sell": 0, "strong_buy": 0, "strong_sell": 0}

    # yfinance recommendations schema varies; handle both formats
    if isinstance(recs, pd.DataFrame):
        if "To Grade" in recs.columns:
            grades = recs["To Grade"].str.lower()
        elif "period" in recs.columns:
            # Newer yfinance returns period-based summary
            latest = recs.iloc[0] if not recs.empty else {}
            return {
                "strong_buy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strong_sell": int(latest.get("strongSell", 0)),
            }
        else:
            return {"buy": 0, "hold": 0, "sell": 0, "strong_buy": 0, "strong_sell": 0}

        counts = {"buy": 0, "hold": 0, "sell": 0, "strong_buy": 0, "strong_sell": 0}
        for g in grades:
            if "strong buy" in g:
                counts["strong_buy"] += 1
            elif "buy" in g or "outperform" in g or "overweight" in g:
                counts["buy"] += 1
            elif "strong sell" in g or "underperform" in g or "underweight" in g:
                counts["strong_sell"] += 1
            elif "sell" in g:
                counts["sell"] += 1
            else:
                counts["hold"] += 1
        return counts

    return {"buy": 0, "hold": 0, "sell": 0, "strong_buy": 0, "strong_sell": 0}


def build_financials_csv(data: dict) -> pd.DataFrame:
    """Combine key financial tables into a single DataFrame for CSV export."""
    frames = []
    for label, df in [
        ("Income Statement", data.get("financials")),
        ("Balance Sheet", data.get("balance_sheet")),
        ("Cash Flow", data.get("cashflow")),
    ]:
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df_copy = df.copy()
            df_copy.insert(0, "Statement", label)
            frames.append(df_copy)

    if frames:
        return pd.concat(frames)
    return pd.DataFrame()
