"""
sentiment.py
Scores news articles and social posts using VADER sentiment analysis.
Fetches Reddit posts via public JSON API (no credentials required).
"""

import os
import re
import datetime
import urllib3
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Disable SSL verification warnings for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

analyzer = SentimentIntensityAnalyzer()

SUBREDDITS = ["stocks", "investing", "wallstreetbets", "StockMarket", "options"]


def score_text(text: str) -> dict:
    """Return VADER compound score and label for a text string."""
    if not text:
        return {"compound": 0.0, "label": "Neutral"}
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"compound": round(compound, 4), "label": label}


def analyze_news(articles: list[dict]) -> list[dict]:
    """Score a list of NewsAPI articles and return enriched list."""
    scored = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        combined = f"{title}. {description}"
        sentiment = score_text(combined)
        scored.append({
            "title": title,
            "source": article.get("source", {}).get("name", ""),
            "url": article.get("url", ""),
            "published_at": article.get("publishedAt", ""),
            "sentiment_score": sentiment["compound"],
            "sentiment_label": sentiment["label"],
            "description": description,
        })
    # Sort by date descending
    scored.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return scored


def analyze_yf_news(yf_news: list) -> list[dict]:
    """Score yfinance news items (fallback when NewsAPI is not configured).

    Handles both old format (providerPublishTime, title, publisher, link)
    and new yfinance 0.2.50+ format (content.title, content.pubDate, etc.).
    """
    scored = []
    for item in yf_news:
        # ── New format: data nested under 'content' key ──
        content = item.get("content", {})
        if content:
            title = content.get("title", "")
            source = content.get("provider", {}).get("displayName", "")
            url = (content.get("canonicalUrl") or {}).get("url", "") or \
                  (content.get("clickThroughUrl") or {}).get("url", "")
            raw_desc = content.get("summary", "") or content.get("description", "") or ""
            description = re.sub(r"<[^>]+>", "", raw_desc)
            pub_raw = content.get("pubDate", "")
            pub_date = pub_raw[:19] + "Z" if pub_raw else ""
        else:
            # ── Old format: flat dict ──
            title = item.get("title", "")
            source = item.get("publisher", "")
            url = item.get("link", "")
            description = ""
            ts = item.get("providerPublishTime", 0)
            try:
                pub_date = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else ""
            except Exception:
                pub_date = ""

        if not title:
            continue

        sentiment = score_text(f"{title}. {description}")
        scored.append({
            "title": title,
            "source": source,
            "url": url,
            "published_at": pub_date,
            "sentiment_score": sentiment["compound"],
            "sentiment_label": sentiment["label"],
            "description": description[:200] if description else "",
        })
    scored.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return scored


def get_reddit_posts(ticker: str, company_name: str = "") -> list[dict]:
    """Fetch Reddit posts using the public JSON API — no credentials required."""
    query = f"{ticker} {company_name}".strip()
    verify = False if os.getenv("CORPORATE_PROXY", "false").lower() == "true" else True
    headers = {"User-Agent": "StockAnalyzer/1.0 (public research tool)"}
    posts = []

    for subreddit_name in SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{subreddit_name}/search.json"
            params = {"q": query, "sort": "relevance", "t": "year", "limit": 10}
            r = requests.get(url, headers=headers, params=params,
                             timeout=10, verify=verify)
            if r.status_code != 200:
                continue
            children = r.json().get("data", {}).get("children", [])
            for child in children:
                d = child.get("data", {})
                title = d.get("title", "")
                if not title:
                    continue
                body = d.get("selftext", "")[:200]
                sentiment = score_text(f"{title} {body}")
                ts = d.get("created_utc", 0)
                created = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
                posts.append({
                    "subreddit": subreddit_name,
                    "title": title,
                    "score": d.get("score", 0),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "created_utc": created,
                    "num_comments": d.get("num_comments", 0),
                    "sentiment_score": sentiment["compound"],
                    "sentiment_label": sentiment["label"],
                })
        except Exception:
            continue

    posts.sort(key=lambda x: x.get("score", 0), reverse=True)
    return posts[:30]


def analyze_stocktwits(messages: list) -> list[dict]:
    """Score StockTwits messages."""
    scored = []
    for msg in messages:
        body = msg.get("body", "")
        sentiment = score_text(body)
        # StockTwits also provides its own sentiment sometimes
        st_sentiment = msg.get("entities", {}).get("sentiment", {})
        st_label = st_sentiment.get("basic", "") if st_sentiment else ""
        scored.append({
            "body": body[:280],
            "created_at": msg.get("created_at", ""),
            "username": msg.get("user", {}).get("username", ""),
            "sentiment_score": sentiment["compound"],
            "sentiment_label": st_label if st_label else sentiment["label"],
        })
    return scored


def aggregate_sentiment(
    news_scored: list[dict],
    reddit_posts: list[dict],
    stocktwits_scored: list[dict],
) -> dict:
    """Produce an overall sentiment summary across all sources."""
    def _avg(items, key="sentiment_score"):
        vals = [i[key] for i in items if key in i]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    news_avg = _avg(news_scored)
    reddit_avg = _avg(reddit_posts)
    st_avg = _avg(stocktwits_scored)

    sources = [s for s, v in [("news", news_avg), ("reddit", reddit_avg), ("stocktwits", st_avg)] if v != 0.0]
    all_vals = [v for v in [news_avg, reddit_avg, st_avg] if v != 0.0]
    overall = round(sum(all_vals) / len(all_vals), 4) if all_vals else 0.0

    if overall >= 0.05:
        overall_label = "Positive"
    elif overall <= -0.05:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "overall_score": overall,
        "overall_label": overall_label,
        "news_avg": news_avg,
        "reddit_avg": reddit_avg,
        "stocktwits_avg": st_avg,
        "news_count": len(news_scored),
        "reddit_count": len(reddit_posts),
        "stocktwits_count": len(stocktwits_scored),
        "active_sources": sources,
    }
