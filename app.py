"""
app.py
Streamlit GUI for the Stock Analysis application.
Run with: streamlit run app.py
"""

import os
import sys
import datetime
import traceback
import pandas as pd
import streamlit as st
from pathlib import Path

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collector import (
    get_stock_data, get_news_articles, get_stocktwits_sentiment,
    get_analyst_summary, build_financials_csv,
)
from technical import compute_indicators, derive_price_targets, get_technical_signal
from sentiment import (
    analyze_news, analyze_yf_news, get_reddit_posts,
    analyze_stocktwits, aggregate_sentiment,
)
from ai_summary import generate_ai_analysis
from charts import (
    price_chart, macd_chart, rsi_chart,
    sentiment_bar_chart, analyst_donut_chart, figure_to_png_bytes,
)
from pdf_report import build_pdf_report

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1565C0;}
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
    div[data-testid="stProgress"] > div {background-color: #1565C0;}
    .stButton>button {
        background-color: #1565C0; color: white;
        border-radius: 6px; border: none;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stButton>button:hover {background-color: #0D47A1;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Analyser")
    st.markdown("---")

    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        placeholder="e.g. AAPL, MSFT, TSLA",
        help="Enter the stock ticker symbol (e.g. AAPL for Apple Inc.)",
    ).strip().upper()

    st.markdown("#### Analysis Options")
    run_ai = st.checkbox("AI Analysis (GPT-4o)", value=True,
                          help="Requires OpenAI API key in .env")
    run_reddit = st.checkbox("Reddit Sentiment", value=True,
                              help="Requires Reddit credentials in .env")
    run_news = st.checkbox("News Articles (NewsAPI)", value=True,
                            help="Requires NewsAPI key in .env. Falls back to yfinance news.")

    analyze_btn = st.button("🔍 Analyze Stock", use_container_width=True)

    st.markdown("---")
    st.markdown("#### About")
    st.markdown("""
This tool collects:
- **Price data** via yfinance
- **News** via NewsAPI + yfinance
- **Social** via Reddit + StockTwits
- **AI analysis** via GPT-4o
- **PDF + CSV** export
    """)
    st.markdown("---")
    st.markdown("<small>Configure API keys in `.env`</small>", unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Stock Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("Enter a ticker symbol in the sidebar and click **Analyze Stock** to begin.")

if analyze_btn:
    if not ticker_input:
        st.error("Please enter a ticker symbol in the sidebar.")
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

    # ── Step 9: CSV ───────────────────────────────────────────────────────────
    update_status("Preparing exports…", 93)
    financials_df = build_financials_csv(data)
    csv_bytes = financials_df.to_csv().encode("utf-8") if not financials_df.empty else b"No financial data available"

    # ── Step 10: PDF ──────────────────────────────────────────────────────────
    update_status("Building PDF report…", 96)
    try:
        chart_price_png = figure_to_png_bytes(fig_price, 900, 500)
        chart_macd_png = figure_to_png_bytes(fig_macd, 900, 300)
        chart_rsi_png = figure_to_png_bytes(fig_rsi, 900, 280)
        chart_sentiment_png = figure_to_png_bytes(fig_sentiment, 700, 300)
        chart_analyst_png = figure_to_png_bytes(fig_analyst, 600, 320)

        pdf_bytes = build_pdf_report(
            ticker=ticker_input, info=info, price_targets=price_targets,
            technical_signal=technical_signal, sentiment_summary=sentiment_summary,
            analyst_consensus=analyst_consensus, ai_analysis_text=ai_text,
            news_scored=news_scored, reddit_posts=reddit_posts,
            stocktwits_scored=stocktwits_scored,
            chart_price_png=chart_price_png, chart_macd_png=chart_macd_png,
            chart_rsi_png=chart_rsi_png, chart_sentiment_png=chart_sentiment_png,
            chart_analyst_png=chart_analyst_png,
        )
        pdf_ok = True
    except Exception as e:
        pdf_ok = False
        pdf_error = traceback.format_exc()

    update_status("✅ Analysis complete!", 100)
    progress_bar.empty()
    status_box.empty()

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS UI
    # ══════════════════════════════════════════════════════════════════════════

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"## {company_name} ({ticker_input})")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    col_info2.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    col_info3.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Current Price", f"${price_targets.get('current_price', 'N/A')}")
    m2.metric("Buy Zone", f"${price_targets.get('buy_price', 'N/A')}", delta="Entry")
    m3.metric("Sell Target", f"${price_targets.get('sell_price', 'N/A')}", delta="Exit",
              delta_color="normal")
    m4.metric("Cut-Loss", f"${price_targets.get('cut_loss_price', 'N/A')}", delta="Stop-loss",
              delta_color="inverse")
    m5.metric("Technical", technical_signal.get("overall", "N/A"))
    m6.metric("Sentiment", sentiment_summary.get("overall_label", "N/A"))

    # ── Download buttons ──────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2, dl3 = st.columns([2, 2, 6])

    if pdf_ok:
        fname_pdf = f"{ticker_input}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        dl1.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=fname_pdf,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        dl1.error("PDF generation failed")

    fname_csv = f"{ticker_input}_financials_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    dl2.download_button(
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
        st.plotly_chart(fig_rsi, use_container_width=True)

        with st.expander("Raw indicator data (last 30 rows)"):
            cols_to_show = [c for c in ["Close", "MA20", "MA50", "MA200",
                                         "RSI", "MACD", "MACD_signal",
                                         "BB_upper", "BB_lower", "ATR"]
                            if c in df.columns]
            st.dataframe(df[cols_to_show].tail(30).round(3), use_container_width=True)

    # ── Tab: Fundamentals ─────────────────────────────────────────────────────
    with tab_funds:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Valuation**")
            st.table(pd.DataFrame({
                "Metric": ["P/E (trailing)", "Forward P/E", "PEG Ratio",
                            "Price/Sales", "Price/Book", "EV/EBITDA"],
                "Value": [
                    info.get("trailingPE", "N/A"),
                    info.get("forwardPE", "N/A"),
                    info.get("pegRatio", "N/A"),
                    info.get("priceToSalesTrailing12Months", "N/A"),
                    info.get("priceToBook", "N/A"),
                    info.get("enterpriseToEbitda", "N/A"),
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
                    info.get("debtToEquity", "N/A"),
                    info.get("currentRatio", "N/A"),
                    info.get("quickRatio", "N/A"),
                    f"${info.get('totalDebt', 'N/A'):,}" if info.get("totalDebt") else "N/A",
                    f"${info.get('totalCash', 'N/A'):,}" if info.get("totalCash") else "N/A",
                ],
            }))
        with c4:
            st.table(pd.DataFrame({
                "Metric": ["Free Cash Flow", "Operating Cash Flow",
                            "Market Cap", "EPS (TTM)", "Dividend Yield", "Beta"],
                "Value": [
                    f"${info.get('freeCashflow', 'N/A'):,}" if info.get("freeCashflow") else "N/A",
                    f"${info.get('operatingCashflow', 'N/A'):,}" if info.get("operatingCashflow") else "N/A",
                    f"${info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "N/A",
                    info.get("trailingEps", "N/A"),
                    pct(info.get("dividendYield")),
                    info.get("beta", "N/A"),
                ],
            }))

        if not financials_df.empty:
            with st.expander("Full Financial Statements"):
                st.dataframe(financials_df, use_container_width=True)

    # ── Tab: Analysts ─────────────────────────────────────────────────────────
    with tab_analyst:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(fig_analyst, use_container_width=True)
        with c2:
            st.markdown("**Price Targets**")
            st.metric("Analyst Mean Target",
                      f"${price_targets.get('analyst_mean_target', 'N/A')}")
            st.metric("Analyst Low", f"${price_targets.get('analyst_low_target', 'N/A')}")
            st.metric("Analyst High", f"${price_targets.get('analyst_high_target', 'N/A')}")
            st.markdown("---")
            st.markdown("**Recommended Price Levels**")
            st.metric("Buy Zone", f"${price_targets.get('buy_price', 'N/A')}")
            st.metric("Sell / Target", f"${price_targets.get('sell_price', 'N/A')}")
            st.metric("Cut-Loss (Stop)", f"${price_targets.get('cut_loss_price', 'N/A')}",
                      help=f"Based on 2× ATR (${price_targets.get('atr', 'N/A')})")

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
                st.markdown(
                    f"**[{article.get('title', '')}]({article.get('url', '#')})**  \n"
                    f"*{article.get('source', '')} · {article.get('published_at', '')[:10]}*  "
                    f"&nbsp;&nbsp; :{color}[{label} ({score:.2f})]"
                )
                if article.get("description"):
                    st.markdown(f"> {article['description'][:200]}")
                st.markdown("---")
        else:
            st.info("No news articles available. Add a NewsAPI key to .env for full news coverage.")

    # ── Tab: Social ───────────────────────────────────────────────────────────
    with tab_social:
        r_col, st_col = st.columns(2)

        with r_col:
            st.markdown("### Reddit")
            if reddit_posts:
                for post in reddit_posts[:10]:
                    label = post.get("sentiment_label", "Neutral")
                    color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                    st.markdown(
                        f"**[{post.get('title', '')[:80]}]({post.get('url', '#')})**  \n"
                        f"r/{post.get('subreddit','')} · ↑{post.get('score',0)} · "
                        f"{post.get('num_comments',0)} comments · {post.get('created_utc','')}  \n"
                        f":{color}[{label} ({post.get('sentiment_score',0):.2f})]"
                    )
                    st.markdown("---")
            else:
                st.info("Reddit not configured. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env.")

        with st_col:
            st.markdown("### StockTwits")
            if stocktwits_scored:
                for msg in stocktwits_scored[:10]:
                    label = msg.get("sentiment_label", "Neutral")
                    color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                    st.markdown(
                        f"@{msg.get('username','')} · {msg.get('created_at','')[:10]}  \n"
                        f"{msg.get('body', '')}  \n"
                        f":{color}[{label}]"
                    )
                    st.markdown("---")
            else:
                st.info("No StockTwits messages retrieved for this ticker.")

    if not pdf_ok:
        with st.expander("PDF generation error details"):
            st.code(pdf_error)

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.info("**Step 1**\nEnter a ticker symbol in the sidebar (e.g. `AAPL`, `MSFT`, `TSLA`)")
    col_b.info("**Step 2**\nSelect your analysis options (AI, Reddit, News)")
    col_c.info("**Step 3**\nClick **Analyze Stock** and monitor live progress")

    st.markdown("---")
    st.markdown("### What you'll get")
    f1, f2, f3, f4 = st.columns(4)
    f1.markdown("📊 **Price Charts**\nCandlestick, MA, Bollinger Bands, volume")
    f2.markdown("📉 **Technicals**\nRSI, MACD, trend signals, price targets")
    f3.markdown("🤖 **AI Analysis**\nGPT-4o investment summary & recommendation")
    f4.markdown("📄 **PDF + CSV**\nDownloadable report and financial data")
