"""
charts.py
Generates all Plotly charts used in the Streamlit UI and PDF report.
"""

import io
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DatetimeIndex so kaleido can serialize it."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


CHART_COLORS = {
    "price": "#2196F3",
    "ma20": "#FF9800",
    "ma50": "#4CAF50",
    "ma200": "#F44336",
    "bb_upper": "#9E9E9E",
    "bb_lower": "#9E9E9E",
    "volume_up": "#26A69A",
    "volume_down": "#EF5350",
    "macd": "#2196F3",
    "signal": "#FF9800",
    "rsi": "#7B1FA2",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
}


def price_chart(df: pd.DataFrame, ticker: str, price_targets: dict = None) -> go.Figure:
    """Candlestick chart with MAs, Bollinger Bands, and price target lines."""
    df = _clean_df(df)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — Price & Indicators", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
    ), row=1, col=1)

    # Moving averages
    for col, color, name in [
        ("MA20", CHART_COLORS["ma20"], "MA 20"),
        ("MA50", CHART_COLORS["ma50"], "MA 50"),
        ("MA200", CHART_COLORS["ma200"], "MA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.2), opacity=0.8,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color=CHART_COLORS["bb_upper"], width=1, dash="dot"),
            opacity=0.5,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color=CHART_COLORS["bb_lower"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(158,158,158,0.08)", opacity=0.5,
        ), row=1, col=1)

    # Price target lines
    if price_targets:
        for label, key, color in [
            ("Sell Target", "sell_price", "#4CAF50"),
            ("Buy Zone", "buy_price", "#2196F3"),
            ("Cut-Loss", "cut_loss_price", "#F44336"),
        ]:
            val = price_targets.get(key)
            if val:
                fig.add_hline(
                    y=val, line_dash="dash", line_color=color,
                    annotation_text=f"{label}: ${val:.2f}",
                    annotation_position="right",
                    row=1, col=1,
                )

    # Volume bars
    colors = [
        CHART_COLORS["volume_up"] if close >= open_ else CHART_COLORS["volume_down"]
        for close, open_ in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        height=580, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def macd_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """MACD line + signal + histogram."""
    df = _clean_df(df)
    fig = go.Figure()

    if "MACD_hist" in df.columns:
        colors = [CHART_COLORS["positive"] if v >= 0 else CHART_COLORS["negative"]
                  for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="Histogram",
            marker_color=colors, opacity=0.7,
        ))

    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=CHART_COLORS["macd"], width=1.5),
        ))

    if "MACD_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color=CHART_COLORS["signal"], width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title=f"{ticker} — MACD",
        height=300, template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h"),
    )
    return fig


def rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """RSI with overbought/oversold zones."""
    df = _clean_df(df)
    fig = go.Figure()

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color=CHART_COLORS["rsi"], width=1.5),
        ))

    fig.add_hline(y=70, line_dash="dash", line_color="#F44336",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50",
                  annotation_text="Oversold (30)")
    fig.add_hrect(y0=70, y1=100, fillcolor="#F44336", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#4CAF50", opacity=0.05, line_width=0)

    fig.update_layout(
        title=f"{ticker} — RSI (14)",
        height=280, template="plotly_dark",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def sentiment_bar_chart(sentiment_summary: dict) -> go.Figure:
    """Bar chart comparing sentiment across sources."""
    sources = []
    scores = []
    bar_colors = []

    mapping = [
        ("News", "news_avg"),
        ("Reddit", "reddit_avg"),
        ("StockTwits", "stocktwits_avg"),
    ]
    for label, key in mapping:
        val = sentiment_summary.get(key, 0)
        if val != 0.0:
            sources.append(label)
            scores.append(val)
            bar_colors.append(
                CHART_COLORS["positive"] if val > 0.05
                else CHART_COLORS["negative"] if val < -0.05
                else CHART_COLORS["neutral"]
            )

    if not sources:
        sources = ["No data"]
        scores = [0]
        bar_colors = [CHART_COLORS["neutral"]]

    fig = go.Figure(go.Bar(
        x=sources, y=scores, marker_color=bar_colors,
        text=[f"{s:.3f}" for s in scores], textposition="outside",
    ))
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    fig.update_layout(
        title="Sentiment by Source (VADER Score)",
        height=300, template="plotly_dark",
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def analyst_donut_chart(analyst_consensus: dict) -> go.Figure:
    """Donut chart of analyst buy/hold/sell breakdown."""
    strong_buy = analyst_consensus.get("strong_buy", 0)
    buy = analyst_consensus.get("buy", 0)
    hold = analyst_consensus.get("hold", 0)
    sell = analyst_consensus.get("sell", 0)
    strong_sell = analyst_consensus.get("strong_sell", 0)

    labels = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    values = [strong_buy, buy, hold, sell, strong_sell]
    colors = ["#1B5E20", "#4CAF50", "#FFC107", "#F44336", "#B71C1C"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker_colors=colors,
        textinfo="label+value",
    ))
    fig.update_layout(
        title="Analyst Ratings Distribution",
        height=320, template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def figure_to_png_bytes(fig: go.Figure, width: int = 900, height: int = 500) -> bytes:
    """Convert a Plotly figure to PNG bytes for embedding in PDF.
    Falls back to a blank white image if kaleido is unavailable."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1.5)
    except Exception:
        # kaleido not available or failed — return a minimal blank PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_mpl, ax = plt.subplots(figsize=(width / 100, height / 100))
            ax.text(0.5, 0.5, "Chart unavailable\n(kaleido not installed)",
                    ha="center", va="center", fontsize=12, color="#999999",
                    transform=ax.transAxes)
            ax.axis("off")
            buf = io.BytesIO()
            fig_mpl.savefig(buf, format="png", bbox_inches="tight",
                            facecolor="#1e1e1e")
            plt.close(fig_mpl)
            return buf.getvalue()
        except Exception:
            # Absolute fallback: 1x1 white PNG
            return (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
                b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
                b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
                b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            )
