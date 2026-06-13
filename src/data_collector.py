"""
data_collector.py
Fetches all stock data from yfinance and NewsAPI.
"""

import os
import time
import datetime
import urllib3
import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Load .env FIRST so CORPORATE_PROXY is available before _SESSION is built
load_dotenv()

# On corporate proxy (CORPORATE_PROXY=true): use a custom requests.Session with
# SSL verification disabled. On Streamlit Cloud / any standard env: pass session=None
# so yfinance uses its built-in curl_cffi (TLS fingerprinting) which bypasses
# Yahoo Finance bot-detection without needing a custom session.
_USE_PROXY_SESSION = os.getenv("CORPORATE_PROXY", "false").lower() == "true"

if _USE_PROXY_SESSION:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _RETRY = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    _ADAPTER = HTTPAdapter(max_retries=_RETRY)
    _SESSION = requests.Session()
    _SESSION.verify = False
    _SESSION.mount("https://", _ADAPTER)
    _SESSION.mount("http://", _ADAPTER)
    _SESSION.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
else:
    # Let yfinance use its internal curl_cffi session (handles bot-detection natively)
    _SESSION = None

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def _make_ticker(ticker: str) -> yf.Ticker:
    """Create a yfinance Ticker, using custom session only when on corporate proxy."""
    if _SESSION is not None:
        return yf.Ticker(ticker, session=_SESSION)
    return yf.Ticker(ticker)


def _fetch_history_direct(ticker: str) -> pd.DataFrame:
    """
    Fetch 1-year OHLCV data via Yahoo Finance chart API directly.
    This bypasses yfinance session management and works reliably on cloud.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "10y", "interval": "1d", "events": "history", "includePrePost": "false"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com",
    }
    try:
        verify = False if _USE_PROXY_SESSION else True
        r = requests.get(url, params=params, headers=headers, timeout=20, verify=verify)
        data = r.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
        df = pd.DataFrame({
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": adjclose,
            "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None))
        df.index.name = "Date"
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def _fetch_history(ticker: str) -> tuple:
    """
    Fetch 1-year price history. Returns (history_df, ticker_obj).
    Tries direct HTTP API first, then yf.download(), then t.history().
    """
    t = _make_ticker(ticker)

    # --- Attempt 1: Direct Yahoo Finance API (most reliable on cloud) ---
    hist = _fetch_history_direct(ticker)
    if not hist.empty:
        return hist, t

    # --- Attempt 2: yf.download() ---
    try:
        hist = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            return hist, t
    except Exception:
        pass

    # --- Attempt 3: t.history() ---
    for attempt in range(3):
        try:
            hist = t.history(period="10y")
            if not hist.empty:
                return hist, t
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                time.sleep((attempt + 1) * 5)
            elif "SSL" in err or "certificate" in err:
                time.sleep(2)
            else:
                break
        time.sleep(1)

    return pd.DataFrame(), t


def get_stock_data(ticker: str) -> dict:
    """Fetch comprehensive stock data from yfinance."""
    history, t = _fetch_history(ticker)

    if history.empty:
        raise ValueError(
            f"No price data returned for '{ticker}'. "
            "This may be a network/proxy issue or an invalid symbol. "
            "Try again in a moment."
        )

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
        upgrades_downgrades = t.upgrades_downgrades
        time.sleep(0.3)
    except Exception:
        upgrades_downgrades = pd.DataFrame()

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

    try:
        dividends = t.dividends
    except Exception:
        dividends = pd.Series(dtype=float)

    return {
        "ticker": ticker.upper(),
        "history": history,
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "recommendations": recommendations,
        "upgrades_downgrades": upgrades_downgrades,
        "analyst_targets": analyst_targets,
        "earnings_dates": earnings_dates,
        "yf_news": yf_news,
        "dividends": dividends,
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
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    try:
        if _SESSION is not None:
            response = _SESSION.get(url, timeout=10)
        else:
            response = requests.get(url, headers=headers, timeout=10)
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


def get_insider_trades(ticker: str) -> list[dict]:
    """
    Fetch recent insider trades via yfinance (insider_transactions).
    Returns list of dicts with insider name, title, type, shares, price, value, date.
    """
    try:
        t = _make_ticker(ticker)
        df = t.insider_transactions
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return []

        df = df.reset_index(drop=True)
        trades = []
        for _, row in df.iterrows():
            # normalise column names (yfinance may vary by version)
            def _g(keys):
                for k in keys:
                    for col in df.columns:
                        if col.lower().replace(" ", "") == k.lower().replace(" ", ""):
                            v = row.get(col)
                            if v is not None and str(v) not in ("nan", "None", ""):
                                return v
                return ""

            shares = _g(["shares", "sharesTraded"])
            price  = _g(["value", "startPrice", "price"])
            txn_type = str(_g(["transaction", "transactionType", "type"]))
            date_val = _g(["startDate", "date", "transactionDate"])
            insider  = str(_g(["insider", "insiderName", "name"]))
            title    = str(_g(["position", "title", "officerTitle"]))
            url      = str(_g(["url", "filingUrl"]))

            try:
                shares_f = float(shares)
            except Exception:
                shares_f = None
            try:
                price_f = float(price)
            except Exception:
                price_f = None

            value = round(shares_f * price_f) if shares_f and price_f else None

            date_str = ""
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            elif date_val:
                date_str = str(date_val)[:10]

            trades.append({
                "insider":  insider,
                "title":    title,
                "type":     txn_type,
                "shares":   str(int(shares_f)) if shares_f else str(shares),
                "price":    f"{price_f:.2f}" if price_f else str(price),
                "value":    value,
                "date":     date_str,
                "security": "Common Stock",
                "url":      url,
            })

        trades.sort(key=lambda x: x.get("date", ""), reverse=True)
        return trades[:25]
    except Exception:
        return []
