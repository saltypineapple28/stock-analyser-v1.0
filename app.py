"""
app.py
Streamlit GUI for the Stock Analysis application.
Run with: streamlit run app.py
"""

import os
import sys
import datetime
import pandas as pd
import streamlit as st
from pathlib import Path

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collector import (
    get_stock_data, get_news_articles, get_stocktwits_sentiment,
    get_analyst_summary, build_financials_csv, get_insider_trades,
)
from technical import compute_indicators, derive_price_targets, get_technical_signal
from sentiment import (
    analyze_news, analyze_yf_news, get_reddit_posts,
    analyze_stocktwits, aggregate_sentiment,
)
from ai_summary import generate_ai_analysis
from charts import (
    price_chart, macd_chart, rsi_chart,
    sentiment_bar_chart, analyst_donut_chart,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockAnalyzr",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* hide sidebar toggle */
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    .main-header {font-size: 2.4rem; font-weight: 800; color: #1565C0; letter-spacing:-1px;}
    .sub-header  {font-size: 1rem; color: #9E9E9E; margin-top:-8px; margin-bottom:16px;}
    .metric-card {
        background: #F5F5F5; border-radius: 8px;
        padding: 12px 16px; margin: 4px 0;
    }
    .buy-tag  {color: #2E7D32; font-weight: 700; font-size: 1.1rem;}
    .sell-tag {color: #C62828; font-weight: 700; font-size: 1.1rem;}
    .hold-tag {color: #F57F17; font-weight: 700; font-size: 1.1rem;}
    .signal-bullish {color: #2E7D32; font-weight: 700;}
    .signal-bearish {color: #C62828; font-weight: 700;}
    .signal-neutral {color: #F57F17; font-weight: 700;}
    div[data-testid="column"] {min-width: 0 !important;}
    [data-testid="stMetric"] {border-bottom: none !important; box-shadow: none !important;}
    @media (max-width: 640px) {
        .main-header {font-size: 1.6rem !important;}
        .sub-header  {font-size: 0.82rem !important;}
    }
    .stButton>button {
        background-color: #1565C0; color: white;
        border-radius: 6px; border: none;
        padding: 0.55rem 2rem; font-weight: 700; font-size: 1rem;
    }
    .stButton>button:hover {background-color: #0D47A1;}
    .feature-box {
        border-radius: 10px;
        padding: 14px 16px; font-size: 0.85rem; line-height: 1.7;
        border-left: 3px solid #1565C0;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 StockAnalyzr</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered Stock Analysis · Price targets · Technical · Sentiment · Insider trades</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Input + About ─────────────────────────────────────────────────────────────
run_ai = run_reddit = run_news = True  # always run all analysis

inp_col, about_col = st.columns([2, 3], gap="large")

with inp_col:
    st.markdown("#### 🔎 Ticker Symbol")
    ticker_input = st.text_input(
        "ticker",
        placeholder="e.g. AAPL, MSFT, TSLA, AMD",
        label_visibility="collapsed",
    ).strip().upper()
    st.caption("Don't know the ticker? [Search on Yahoo Finance](https://finance.yahoo.com/lookup/) · [Search on Google Finance](https://www.google.com/finance/)")
    st.markdown("")
    analyze_btn = st.button("🔍 Analyze Stock", use_container_width=True)

with about_col:
    st.markdown("#### ℹ️ About")
    st.markdown("""
- Financial data, Technicals and News via **Yahoo Finance**
- Sentiment via **Reddit** (RSS) + **StockTwits**
- Insider trades via **SEC EDGAR** Form 4
- AI analysis via **GPT-4o**
""")
    st.markdown("<small style='color:#9E9E9E'>✍️ from Linnie with 😘 @okelaloli &nbsp;·&nbsp; v1.0 &nbsp;·&nbsp; Jun 2026</small>", unsafe_allow_html=True)

st.markdown("---")


if analyze_btn:
    if not ticker_input:
        st.error("Please enter a ticker symbol above.")
        st.stop()

    # ── Progress tracking ──────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Starting analysis…")
    status_box = st.empty()
    log_lines = []

    def update_status(msg: str, pct: int):
        log_lines.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
        status_box.markdown(
            "```\n" + "\n".join(log_lines[-12:]) + "\n```"
        )
        progress_bar.progress(pct, text=msg)

    # ── Step 1: Fetch stock data ───────────────────────────────────────────────
    update_status(f"Fetching market data for {ticker_input}…", 5)
    try:
        data = get_stock_data(ticker_input)
    except ValueError as e:
        st.error(str(e))
        progress_bar.empty()
        status_box.empty()
        st.stop()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        progress_bar.empty()
        status_box.empty()
        st.stop()

    info = data["info"]
    company_name = info.get("longName", ticker_input)
    update_status(f"✓ Data fetched for {company_name}", 12)

    # ── Step 2: Technical indicators ──────────────────────────────────────────
    update_status("Computing technical indicators…", 20)
    df = compute_indicators(data["history"])
    price_targets = derive_price_targets(df, data["analyst_targets"], info)
    technical_signal = get_technical_signal(df)
    update_status(f"✓ Technical analysis complete — Signal: {technical_signal['overall']}", 30)

    # ── Step 3: News ──────────────────────────────────────────────────────────
    news_scored = []
    if run_news:
        update_status("Fetching news articles…", 38)
        articles = get_news_articles(ticker_input, company_name)
        if articles:
            news_scored = analyze_news(articles)
            update_status(f"✓ {len(news_scored)} news articles fetched and scored", 45)
        else:
            update_status("NewsAPI not configured — using yfinance news…", 40)
            news_scored = analyze_yf_news(data["yf_news"])
            update_status(f"✓ {len(news_scored)} yfinance news items scored", 45)

    # ── Step 4: Reddit ────────────────────────────────────────────────────────
    reddit_posts = []
    if run_reddit:
        update_status("Fetching Reddit posts…", 50)
        reddit_posts = get_reddit_posts(ticker_input, company_name)
        update_status(f"✓ {len(reddit_posts)} Reddit posts retrieved", 57)

    # ── Step 5: StockTwits ────────────────────────────────────────────────────
    update_status("Fetching StockTwits…", 60)
    st_data = get_stocktwits_sentiment(ticker_input)
    stocktwits_scored = analyze_stocktwits(st_data.get("messages", []))
    update_status(f"✓ {len(stocktwits_scored)} StockTwits messages scored", 65)

    # ── Step 6: Aggregate sentiment ───────────────────────────────────────────
    sentiment_summary = aggregate_sentiment(news_scored, reddit_posts, stocktwits_scored)
    analyst_consensus = get_analyst_summary(data)
    update_status(f"✓ Sentiment: {sentiment_summary['overall_label']} ({sentiment_summary['overall_score']:.2f})", 70)

    # ── Step 7: AI summary ────────────────────────────────────────────────────
    headlines = [a["title"] for a in news_scored[:10]]
    if run_ai:
        update_status("Generating AI analysis (GPT-4o)…", 75)
    else:
        update_status("Generating rule-based summary…", 75)

    ai_text = generate_ai_analysis(
        ticker_input, info, price_targets, technical_signal,
        sentiment_summary, analyst_consensus, headlines,
    )
    update_status("✓ Analysis complete", 82)

    # ── Step 8: Charts ────────────────────────────────────────────────────────
    update_status("Rendering charts…", 85)
    fig_price = price_chart(df, ticker_input, price_targets)
    fig_macd = macd_chart(df, ticker_input)
    fig_rsi = rsi_chart(df, ticker_input)
    fig_sentiment = sentiment_bar_chart(sentiment_summary)
    fig_analyst = analyst_donut_chart(analyst_consensus)
    update_status("✓ Charts rendered", 90)

    # ── Step 9: CSV + Insider Trades ───────────────────────────────────────────────────────
    update_status("Preparing exports…", 90)
    financials_df = build_financials_csv(data)
    csv_bytes = financials_df.to_csv().encode("utf-8") if not financials_df.empty else b"No financial data available"

    update_status("Fetching insider trades (SEC EDGAR)…", 95)
    insider_trades = get_insider_trades(ticker_input)
    update_status(f"✓ {len(insider_trades)} insider transactions found", 97)

    update_status("✅ Analysis complete!", 100)
    progress_bar.empty()
    status_box.empty()

    # Save all results to session state so selectbox reruns don't wipe them
    st.session_state["results"] = {
        "ticker": ticker_input,
        "company_name": company_name,
        "info": info,
        "price_targets": price_targets,
        "df": df,
        "technical_signal": technical_signal,
        "sentiment_summary": sentiment_summary,
        "analyst_consensus": analyst_consensus,
        "recommendations": data.get("recommendations"),
        "upgrades_downgrades": data.get("upgrades_downgrades"),
        "ai_text": ai_text,
        "news_scored": news_scored,
        "reddit_posts": reddit_posts,
        "stocktwits_scored": stocktwits_scored,
        "insider_trades": insider_trades,
        "financials_df": financials_df,
        "csv_bytes": csv_bytes,
        "fig_price": fig_price,
        "fig_macd": fig_macd,
        "fig_rsi": fig_rsi,
        "fig_sentiment": fig_sentiment,
        "fig_analyst": fig_analyst,
    }

if "results" in st.session_state:
    _r = st.session_state["results"]
    ticker_input      = _r["ticker"]
    company_name      = _r["company_name"]
    info              = _r["info"]
    price_targets     = _r["price_targets"]
    _df               = _r.get("df")
    technical_signal  = _r["technical_signal"]
    sentiment_summary = _r["sentiment_summary"]
    analyst_consensus = _r["analyst_consensus"]
    _recommendations  = _r.get("recommendations")
    _upgrades         = _r.get("upgrades_downgrades")
    ai_text           = _r["ai_text"]
    news_scored       = _r["news_scored"]
    reddit_posts      = _r["reddit_posts"]
    stocktwits_scored = _r["stocktwits_scored"]
    insider_trades    = _r.get("insider_trades", [])
    financials_df     = _r["financials_df"]
    csv_bytes         = _r["csv_bytes"]
    fig_price         = _r["fig_price"]
    fig_macd          = _r["fig_macd"]
    fig_rsi           = _r["fig_rsi"]
    fig_sentiment     = _r["fig_sentiment"]
    fig_analyst       = _r["fig_analyst"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"## {company_name} ({ticker_input})")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    col_info2.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    col_info3.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"${price_targets.get('current_price', 'N/A')}")
    m2.metric("Technical",     technical_signal.get("overall", "N/A"))
    m3.metric("Sentiment",     sentiment_summary.get("overall_label", "N/A"))

    # ── Compact info bar (Mkt Cap, Volume, 52W, PE) ───────────────────────────
    def _compact_bar(items):
        """Render a horizontal compact info bar like exchange tickers."""
        cells = "".join(
            f"<td style='padding:10px 0;border:none;width:{100//len(items)}%'>"
            f"<div style='font-size:0.72rem;color:#9E9E9E;margin-bottom:3px'>{label}</div>"
            f"<div style='font-size:1.0rem;font-weight:600'>{val}</div></td>"
            for label, val in items
        )
        st.markdown(f"<div style='overflow-x:auto;-webkit-overflow-scrolling:touch'><table style='border-collapse:collapse;border:none;width:100%;table-layout:fixed'><tr>{cells}</tr></table></div>",
                    unsafe_allow_html=True)

    def _fmt_big(n):
        if n is None: return "N/A"
        try:
            n = float(n)
            if n >= 1e12: return f"${n/1e12:.2f}T"
            if n >= 1e9:  return f"${n/1e9:.2f}B"
            if n >= 1e6:  return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"
        except Exception: return "N/A"

    def _fmt_vol(n):
        if n is None: return "N/A"
        try:
            n = float(n)
            if n >= 1e9: return f"{n/1e9:.2f}B"
            if n >= 1e6: return f"{n/1e6:.2f}M"
            if n >= 1e3: return f"{n/1e3:.2f}K"
            return str(int(n))
        except Exception: return "N/A"

    _compact_bar([
        ("Market Cap",   _fmt_big(info.get("marketCap"))),
        ("Volume",    _fmt_vol(info.get("volume"))),
        ("52W High",  f"${info.get('fiftyTwoWeekHigh','N/A')}"),
        ("52W Low",   f"${info.get('fiftyTwoWeekLow','N/A')}"),
        ("PE Ratio",  round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A"),
        ("Div Yield", f"{round(info.get('dividendYield',0)*100,2)}%" if info.get("dividendYield") else "N/A"),
        ("Beta",      info.get("beta", "N/A")),
    ])

    # ── Helper functions ──────────────────────────────────────────────────────
    def _ma(days):
        try:
            close = _df["Close"].dropna()
            if len(close) >= days:
                return round(float(close.rolling(days).mean().iloc[-1]), 2)
        except Exception:
            pass
        return None

    def _bb():
        """Returns (lower, upper) Bollinger Band values (20-day, 2σ)."""
        try:
            close = _df["Close"].dropna()
            if len(close) >= 20:
                mid = close.rolling(20).mean()
                std = close.rolling(20).std()
                return round(float((mid - 2*std).iloc[-1]), 2), round(float((mid + 2*std).iloc[-1]), 2)
        except Exception:
            pass
        return None, None

    def _fib():
        """Returns Fibonacci retracement dict from 60-day swing high/low."""
        try:
            close = _df["Close"].dropna()
            window = close.iloc[-60:] if len(close) >= 60 else close
            high = float(window.max())
            low  = float(window.min())
            diff = high - low
            return {
                "23.6%": round(high - 0.236 * diff, 2),
                "38.2%": round(high - 0.382 * diff, 2),
                "50.0%": round(high - 0.500 * diff, 2),
                "61.8%": round(high - 0.618 * diff, 2),
                "Ext 127.2%": round(low + 1.272 * diff, 2),
                "Ext 161.8%": round(low + 1.618 * diff, 2),
            }
        except Exception:
            return {}

    def _fmt(v):
        return f"${v}" if v is not None else "N/A"

    bb_lower, bb_upper = _bb()
    fib = _fib()
    atr = price_targets.get("atr")
    atr_1x = round(price_targets["current_price"] - 1.5 * atr, 2) if atr and price_targets.get("current_price") else None
    atr_2x = round(price_targets["current_price"] - 2.0 * atr, 2) if atr and price_targets.get("current_price") else None

    def _price_table(emoji, title, color, groups):
        """Render a price target table with fixed column widths and group headers."""
        all_items = [(lbl, val) for _, items in groups for lbl, val in items]
        td_base = "width:110px;min-width:110px;padding:5px 10px;"

        gh = "".join(
            f"<th colspan='{len(items)}' style='font-size:0.65rem;font-weight:500;"
            f"color:#9E9E9E;text-align:left;padding:5px 10px 3px;"
            f"border-bottom:1px solid #333;border-right:2px solid #444'>{grp}</th>"
            for grp, items in groups
        )
        lr = "".join(
            f"<td style='{td_base}font-size:0.68rem;color:#9E9E9E'>{lbl}</td>"
            for lbl, _ in all_items
        )
        vr = "".join(
            f"<td style='{td_base}font-size:0.93rem;font-weight:600;padding-bottom:8px'>{val}</td>"
            for _, val in all_items
        )
        st.markdown(
            f"<div style='font-size:0.85rem;font-weight:700;margin:14px 0 4px'>"
            f"<span style='color:{color}'>{emoji}</span> {title}</div>"
            f"<div style='overflow-x:auto;-webkit-overflow-scrolling:touch'>"
            f"<table style='table-layout:fixed;border-collapse:collapse;"
            f"border:1px solid #333;margin-bottom:4px'>"
            f"<tr>{gh}</tr><tr>{lr}</tr><tr>{vr}</tr></table></div>",
            unsafe_allow_html=True,
        )

    _price_table("🟢", "Buy Zone", "#4CAF50", [
        ("Moving Average",  [("MA 5d", _fmt(_ma(5))), ("MA 10d", _fmt(_ma(10))), ("MA 30d", _fmt(_ma(30))), ("MA 60d", _fmt(_ma(60)))]),
        ("Bollinger Band",  [("Lower (2σ)", _fmt(bb_lower))]),
        ("Fibonacci Retracement", [("38.2%", _fmt(fib.get("38.2%"))), ("50.0%", _fmt(fib.get("50.0%"))), ("61.8%", _fmt(fib.get("61.8%")))]),
    ])

    _price_table("🔴", "Sell Target", "#F44336", [
        ("Moving Average",  [("MA 5d", _fmt(_ma(5))), ("MA 10d", _fmt(_ma(10))), ("MA 30d", _fmt(_ma(30))), ("MA 60d", _fmt(_ma(60)))]),
        ("Bollinger Band",  [("Upper (2σ)", _fmt(bb_upper))]),
        ("Fibonacci Extension", [("127.2%", _fmt(fib.get("Ext 127.2%"))), ("161.8%", _fmt(fib.get("Ext 161.8%")))]),
        ("Analyst",         [("Consensus Target", _fmt(price_targets.get("analyst_mean_target")))]),
    ])

    _price_table("🛑", "Cut-Loss (Stop)", "#FF5722", [
        ("Moving Average",  [("MA 5d", _fmt(_ma(5))), ("MA 10d", _fmt(_ma(10))), ("MA 30d", _fmt(_ma(30))), ("MA 60d", _fmt(_ma(60)))]),
        ("Bollinger Band",  [("Lower (2σ)", _fmt(bb_lower))]),
        ("ATR Stop",        [("1.5× ATR", _fmt(atr_1x)), ("2.0× ATR", _fmt(atr_2x))]),
    ])

    # ── Download buttons ──────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2 = st.columns([2, 8])

    fname_csv = f"{ticker_input}_financials_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    dl1.download_button(
        label="⬇ Download CSV Data",
        data=csv_bytes,
        file_name=fname_csv,
        mime="text/csv",
        use_container_width=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_price, tab_tech, tab_funds, tab_analyst, tab_ai, tab_news, tab_social = st.tabs([
        "📊 Price Chart",
        "📉 Technical",
        "🏦 Fundamentals",
        "🎯 Analysts",
        "🤖 AI Analysis",
        "📰 News",
        "💬 Social",
    ])

    # ── Tab: Price chart ──────────────────────────────────────────────────────
    with tab_price:
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Tab: Technical ────────────────────────────────────────────────────────
    with tab_tech:
        signal_class = f"signal-{technical_signal.get('overall','neutral').lower()}"
        st.markdown(f"**Overall Signal:** "
                    f"<span class='{signal_class}'>{technical_signal.get('overall','N/A')}</span>",
                    unsafe_allow_html=True)
        for detail in technical_signal.get("details", []):
            st.markdown(f"- {detail}")
        st.plotly_chart(fig_macd, use_container_width=True)
        st.markdown(
            "<span style='font-size:0.85rem;font-weight:600'>RSI (14)</span> "
            "<span title='Relative Strength Index (0–100): measures momentum. "
            "Above 70 = Overbought (may pull back). "
            "Below 30 = Oversold (may bounce). "
            "Around 50 = Neutral.' "
            "style='cursor:help;font-size:0.75rem;color:#9E9E9E;border:1px solid #9E9E9E;"
            "border-radius:50%;padding:0 4px;margin-left:4px'>ℹ</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        with st.expander("Raw indicator data (last 30 days)"):
            cols_to_show = [c for c in ["Close", "MA20", "MA50", "MA200",
                                         "RSI", "MACD", "MACD_signal",
                                         "BB_upper", "BB_lower", "ATR"]
                            if _df is not None and c in _df.columns]
            if _df is not None:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                _recent = _df[_df.index >= cutoff] if not _df.empty else _df
                st.dataframe(_recent[cols_to_show].sort_index(ascending=False).round(3), use_container_width=True)

    # ── Tab: Fundamentals ─────────────────────────────────────────────────────
    with tab_funds:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Valuation**")
            def _fmt(v): return str(round(v, 2)) if isinstance(v, (int, float)) else str(v) if v else "N/A"
            st.table(pd.DataFrame({
                "Metric": ["P/E (trailing)", "Forward P/E", "PEG Ratio",
                            "Price/Sales", "Price/Book", "EV/EBITDA"],
                "Value": [
                    _fmt(info.get("trailingPE")),
                    _fmt(info.get("forwardPE")),
                    _fmt(info.get("pegRatio")),
                    _fmt(info.get("priceToSalesTrailing12Months")),
                    _fmt(info.get("priceToBook")),
                    _fmt(info.get("enterpriseToEbitda")),
                ],
            }))
        with c2:
            st.markdown("**Profitability & Growth**")

            def pct(v):
                try:
                    return f"{float(v)*100:.1f}%" if v else "N/A"
                except Exception:
                    return "N/A"

            st.table(pd.DataFrame({
                "Metric": ["Revenue (TTM)", "Gross Margin", "Operating Margin",
                            "Profit Margin", "ROE", "ROA"],
                "Value": [
                    f"${info.get('totalRevenue', 'N/A'):,}" if info.get("totalRevenue") else "N/A",
                    pct(info.get("grossMargins")),
                    pct(info.get("operatingMargins")),
                    pct(info.get("profitMargins")),
                    pct(info.get("returnOnEquity")),
                    pct(info.get("returnOnAssets")),
                ],
            }))

        st.markdown("**Balance Sheet & Cash**")
        c3, c4 = st.columns(2)
        with c3:
            st.table(pd.DataFrame({
                "Metric": ["Debt/Equity", "Current Ratio", "Quick Ratio",
                            "Total Debt", "Total Cash"],
                "Value": [
                    _fmt(info.get("debtToEquity")),
                    _fmt(info.get("currentRatio")),
                    _fmt(info.get("quickRatio")),
                    f"${info.get('totalDebt'):,}" if info.get("totalDebt") else "N/A",
                    f"${info.get('totalCash'):,}" if info.get("totalCash") else "N/A",
                ],
            }))
        with c4:
            st.table(pd.DataFrame({
                "Metric": ["Free Cash Flow", "Operating Cash Flow",
                            "Market Cap", "EPS (TTM)", "Dividend Yield", "Beta"],
                "Value": [
                    f"${info.get('freeCashflow'):,}" if info.get("freeCashflow") else "N/A",
                    f"${info.get('operatingCashflow'):,}" if info.get("operatingCashflow") else "N/A",
                    f"${info.get('marketCap'):,}" if info.get("marketCap") else "N/A",
                    _fmt(info.get("trailingEps")),
                    pct(info.get("dividendYield")),
                    _fmt(info.get("beta")),
                ],
            }))

        if not financials_df.empty:
            with st.expander("Full Financial Statements"):
                st.dataframe(financials_df, use_container_width=True)

    # ── Tab: Analysts ─────────────────────────────────────────────────────────
    with tab_analyst:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                "<span style='font-size:1rem;font-weight:600'>Analyst Rating Distribution</span> "
                "<span title=\'Source: Yahoo Finance · Ratings are submitted by investment banks and "
                "brokerages (e.g. Goldman Sachs, Morgan Stanley, JPMorgan, Barclays). "
                "Each firm upgrades or downgrades a stock based on their own research. "
                "Strong Buy / Buy = bullish outlook. Hold = neutral. "
                "Sell / Strong Sell = bearish outlook. "
                "Counts reflect the most recent analyst consensus period.\' "
                "style=\'cursor:help;font-size:0.75rem;color:#9E9E9E;border:1px solid #9E9E9E;"
                "border-radius:50%;padding:0 4px;margin-left:4px\'>ℹ</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_analyst, use_container_width=True)
            # Individual firm ratings table from upgrades_downgrades
            _ud = _upgrades
            if _ud is not None and isinstance(_ud, pd.DataFrame) and not _ud.empty:
                _ud = _ud.reset_index()
                # normalise column names (yfinance may vary)
                _ud.columns = [c.strip() for c in _ud.columns]
                date_col  = next((c for c in _ud.columns if "date" in c.lower()), None)
                firm_col  = next((c for c in _ud.columns if "firm" in c.lower()), None)
                grade_col = next((c for c in _ud.columns if "tograde" in c.lower().replace(" ","") or c.lower() == "to grade"), None)
                if firm_col and grade_col:
                    disp_cols = [c for c in [date_col, firm_col, grade_col] if c]
                    _ud_show = _ud[disp_cols].copy()
                    if date_col:
                        _ud_show[date_col] = pd.to_datetime(_ud_show[date_col]).dt.strftime("%Y-%m-%d")
                        _ud_show = _ud_show.sort_values(date_col, ascending=False)
                    _ud_show.columns = [c.title() for c in _ud_show.columns]
                    st.dataframe(_ud_show.head(25), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(
                "<span style='font-size:1rem;font-weight:600'>Price Targets</span> "
                "<span title=\'Source: Yahoo Finance · Mean, Low, and High price targets are "
                "aggregated from Wall Street analysts who cover this stock. "
                "The Mean Target is the average of all analyst 12-month price forecasts. "
                "Low / High show the range of estimates. "
                "More analysts = higher confidence in the consensus.\' "
                "style=\'cursor:help;font-size:0.75rem;color:#9E9E9E;border:1px solid #9E9E9E;"
                "border-radius:50%;padding:0 4px;margin-left:4px\'>ℹ</span>",
                unsafe_allow_html=True,
            )
            st.metric("Analyst Mean Target",
                      f"${price_targets.get('analyst_mean_target', 'N/A')}")
            st.metric("Analyst Low", f"${price_targets.get('analyst_low_target', 'N/A')}")
            st.metric("Analyst High", f"${price_targets.get('analyst_high_target', 'N/A')}")
            st.markdown("---")
            st.markdown("**Recommended Price Levels**")
            st.metric("Sell / Target", f"${price_targets.get('sell_price', 'N/A')}")
            st.metric("Cut-Loss (Stop)", f"${price_targets.get('cut_loss_price', 'N/A')}",
                      help=f"Based on 2× ATR (${price_targets.get('atr', 'N/A')})")
            st.markdown("**Buy Zone (MA Support)**")
            st.metric("MA 5d",  _fmt(_ma(5)))
            st.metric("MA 14d", _fmt(_ma(14)))
            st.metric("MA 30d", _fmt(_ma(30)))
            st.metric("MA 60d", _fmt(_ma(60)))

    # ── Tab: AI Analysis ──────────────────────────────────────────────────────
    with tab_ai:
        st.markdown(ai_text)

    # ── Tab: News ─────────────────────────────────────────────────────────────
    with tab_news:
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.markdown(f"**{len(news_scored)} articles retrieved**")

        if news_scored:
            for article in news_scored[:20]:
                label = article.get("sentiment_label", "Neutral")
                score = article.get("sentiment_score", 0)
                color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                pub = article.get("published_at", "")[:10] or "Unknown date"
                src = article.get("source", "") or "Unknown source"
                title = article.get("title", "(No title)")
                url = article.get("url", "#") or "#"
                st.markdown(
                    f"**[{title}]({url})**  \n"
                    f"*{src} · {pub}*  "
                    f"&nbsp;&nbsp; :{color}[{label} ({score:+.2f})]"
                )
                desc = article.get("description", "")
                if desc:
                    st.markdown(f"> {desc[:250]}")
                st.markdown("---")
        else:
            st.info("No news articles available. Add a NewsAPI key to .env for full news coverage.")

        # ── Insider Trades (SEC Form 4) ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💼 Insider Trades (SEC Form 4)")
        if insider_trades:
            for trade in insider_trades:
                txn_type = trade.get("type", "")
                color = "green" if txn_type == "Purchase" else "red" if txn_type == "Sale" else "orange"
                shares = trade.get("shares", "N/A")
                price  = trade.get("price", "N/A")
                value  = trade.get("value")
                value_str = f" ≈ ${value:,.0f}" if value else ""
                try:
                    shares_fmt = f"{float(shares):,.0f}"
                except Exception:
                    shares_fmt = shares
                try:
                    price_fmt = f"${float(price):,.2f}"
                except Exception:
                    price_fmt = price
                st.markdown(
                    f"**[{trade.get('insider','Unknown')}]({trade.get('url','#')})** "
                    f"— {trade.get('title','')}  \n"
                    f":{color}[{txn_type}] &nbsp; {shares_fmt} shares @ {price_fmt}{value_str} "
                    f"— *{trade.get('date','')}*"
                )
                st.markdown("---")
        else:
            st.info("No insider trades found or SEC EDGAR unavailable.")

    # ── Tab: Social ───────────────────────────────────────────────────────────
    with tab_social:
        r_col, st_col = st.columns(2)

        with r_col:
            st.markdown("### Reddit")
            st.caption(f"Source: Reddit public RSS · Subreddits: stocks, investing, wallstreetbets · Sentiment scored via VADER")
            if reddit_posts:
                for post in reddit_posts[:10]:
                    label = post.get("sentiment_label", "Neutral")
                    color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                    st.markdown(
                        f"**[{post.get('title', '')[:80]}]({post.get('url', '#')})**  \n"
                        f"r/{post.get('subreddit','')} · {post.get('created_utc','')}  \n"
                        f":{color}[{label} ({post.get('sentiment_score',0):.2f})]"
                    )
                    st.markdown("---")
            else:
                st.info("No Reddit posts retrieved. Reddit may be blocked by your network/proxy — it works on Streamlit Cloud.")

        with st_col:
            st.markdown("### StockTwits")
            st.caption(f"Source: [stocktwits.com/${ticker_input}](https://stocktwits.com/symbol/{ticker_input}) · Sentiment scored via VADER")
            if stocktwits_scored:
                for msg in stocktwits_scored[:10]:
                    label = msg.get("sentiment_label", "Neutral")
                    color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                    username = msg.get('username','')
                    user_url = f"https://stocktwits.com/{username}" if username else "#"
                    st.markdown(
                        f"**[@{username}]({user_url})** · {msg.get('created_at','')[:10]}  \n"
                        f"{msg.get('body', '')}  \n"
                        f":{color}[{label}]"
                    )
                    st.markdown("---")
            else:
                st.info("No StockTwits messages retrieved for this ticker.")


else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.info("Enter a ticker symbol above and click **🔍 Analyze Stock** to get started.")
