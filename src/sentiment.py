"""
sentiment.py
Scores news articles and social posts using VADER sentiment analysis.
Also fetches Reddit posts via PRAW.
"""

import os
import ssl
import datetime
import urllib3
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
from dotenv import load_dotenv

# Disable SSL verification warnings for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "StockAnalyzer/1.0")

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
            description = content.get("summary", "")
            pub_raw = content.get("pubDate", "")
            # pubDate is already ISO string e.g. "2024-06-12T10:00:00Z"
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
    """Fetch Reddit posts mentioning the ticker from key subreddits."""
    if not REDDIT_CLIENT_ID or REDDIT_CLIENT_ID == "your_reddit_client_id":
        return []

    try:
        # On corporate proxy, use a custom session with SSL verification disabled.
        # On Streamlit Cloud (CORPORATE_PROXY not set), use default PRAW session.
        use_proxy = os.getenv("CORPORATE_PROXY", "false").lower() == "true"
        if use_proxy:
            import prawcore
            _session = requests.Session()
            _session.verify = False
            requestor = prawcore.Requestor(REDDIT_USER_AGENT, session=_session)
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
                requestor=requestor,
            )
        else:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
        posts = []
        query = f"{ticker} {company_name}".strip()
        for subreddit_name in SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for submission in subreddit.search(query, limit=10, time_filter="year"):
                    sentiment = score_text(f"{submission.title} {submission.selftext[:200]}")
                    posts.append({
                        "subreddit": subreddit_name,
                        "title": submission.title,
                        "score": submission.score,
                        "url": f"https://reddit.com{submission.permalink}",
                        "created_utc": datetime.datetime.fromtimestamp(
                            submission.created_utc
                        ).strftime("%Y-%m-%d"),
                        "num_comments": submission.num_comments,
                        "sentiment_score": sentiment["compound"],
                        "sentiment_label": sentiment["label"],
                    })
            except Exception:
                continue
        posts.sort(key=lambda x: x.get("score", 0), reverse=True)
        return posts[:30]
    except Exception:
        return []


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
