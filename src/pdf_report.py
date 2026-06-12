"""
pdf_report.py
Assembles a multi-section PDF stock analysis report using reportlab.
"""

import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Colour palette
BLUE = colors.HexColor("#1565C0")
DARK_BLUE = colors.HexColor("#0D47A1")
LIGHT_BLUE = colors.HexColor("#E3F2FD")
GREEN = colors.HexColor("#2E7D32")
RED = colors.HexColor("#C62828")
AMBER = colors.HexColor("#F57F17")
GREY = colors.HexColor("#424242")
LIGHT_GREY = colors.HexColor("#F5F5F5")
WHITE = colors.white
BLACK = colors.black


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=22, textColor=DARK_BLUE,
                                spaceAfter=6, fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontSize=11, textColor=GREY,
                                   spaceAfter=4, fontName="Helvetica"),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                              fontSize=14, textColor=DARK_BLUE,
                              spaceBefore=14, spaceAfter=4,
                              fontName="Helvetica-Bold"),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                              fontSize=11, textColor=BLUE,
                              spaceBefore=10, spaceAfter=3,
                              fontName="Helvetica-Bold"),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, textColor=GREY,
                               spaceAfter=4, leading=14,
                               fontName="Helvetica"),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontSize=9, textColor=GREY,
                                 spaceAfter=2, leading=13,
                                 leftIndent=12, fontName="Helvetica",
                                 bulletIndent=4),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=7.5, textColor=GREY,
                                spaceAfter=2, fontName="Helvetica"),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                  fontSize=8, textColor=GREY,
                                  spaceAfter=6, alignment=TA_CENTER,
                                  fontName="Helvetica-Oblique"),
        "price_big": ParagraphStyle("price_big", parent=base["Normal"],
                                    fontSize=18, textColor=GREEN,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER),
        "tag_buy": ParagraphStyle("tag_buy", parent=base["Normal"],
                                  fontSize=11, textColor=GREEN,
                                  fontName="Helvetica-Bold"),
        "tag_sell": ParagraphStyle("tag_sell", parent=base["Normal"],
                                   fontSize=11, textColor=RED,
                                   fontName="Helvetica-Bold"),
        "tag_hold": ParagraphStyle("tag_hold", parent=base["Normal"],
                                   fontSize=11, textColor=AMBER,
                                   fontName="Helvetica-Bold"),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
                                 fontSize=7, textColor=GREY,
                                 alignment=TA_CENTER, fontName="Helvetica"),
    }
    return custom


def _sentiment_color(label: str) -> colors.Color:
    l = label.lower()
    if "positive" in l:
        return GREEN
    if "negative" in l:
        return RED
    return AMBER


def _signal_color(signal: str) -> colors.Color:
    s = signal.lower()
    if "bullish" in s:
        return GREEN
    if "bearish" in s:
        return RED
    return AMBER


def _kv_table(rows: list[tuple], col_widths=None) -> Table:
    """Two-column key-value table."""
    st = _styles()
    data = [[Paragraph(f"<b>{k}</b>", st["small"]),
             Paragraph(str(v), st["small"])] for k, v in rows]
    tbl = Table(data, colWidths=col_widths or [6 * cm, 9 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND", (1, 0), (1, -1), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BDBDBD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _embed_chart(png_bytes, width_cm: float = 16, caption: str = "") -> list:
    """Wrap PNG bytes into a reportlab Image flowable. Skips if png_bytes is None."""
    if not png_bytes:
        return []
    st = _styles()
    buf = io.BytesIO(png_bytes)
    img = Image(buf, width=width_cm * cm, height=(width_cm * 0.56) * cm)
    items = [img]
    if caption:
        items.append(Paragraph(caption, st["caption"]))
    return items


def build_pdf_report(
    ticker: str,
    info: dict,
    price_targets: dict,
    technical_signal: dict,
    sentiment_summary: dict,
    analyst_consensus: dict,
    ai_analysis_text: str,
    news_scored: list[dict],
    reddit_posts: list[dict],
    stocktwits_scored: list[dict],
    chart_price_png: bytes,
    chart_macd_png: bytes,
    chart_rsi_png: bytes,
    chart_sentiment_png: bytes,
    chart_analyst_png: bytes,
) -> bytes:
    """Generate the full PDF report and return as bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )
    st = _styles()
    story = []
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    # ── COVER ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(f"Stock Analysis Report", st["title"]))
    name = info.get("longName", ticker)
    story.append(Paragraph(f"{name} ({ticker.upper()})", st["subtitle"]))
    story.append(Paragraph(
        f"{info.get('sector', '')}  ·  {info.get('industry', '')}  ·  {info.get('exchange', '')}",
        st["subtitle"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"<i>Generated: {now}</i>", st["small"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10))

    # Price targets highlight boxes (as a 3-column table)
    buy_p = f"${price_targets.get('buy_price', 'N/A')}"
    sell_p = f"${price_targets.get('sell_price', 'N/A')}"
    cut_p = f"${price_targets.get('cut_loss_price', 'N/A')}"

    highlights = Table(
        [[
            Paragraph(f"<b>BUY ZONE</b><br/><font size=16>{buy_p}</font>", st["body"]),
            Paragraph(f"<b>SELL TARGET</b><br/><font size=16>{sell_p}</font>", st["body"]),
            Paragraph(f"<b>CUT-LOSS</b><br/><font size=16>{cut_p}</font>", st["body"]),
        ]],
        colWidths=[5.5 * cm, 5.5 * cm, 5.5 * cm],
    )
    highlights.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#E3F2FD")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8F5E9")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#FFEBEE")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BLUE),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(highlights)
    story.append(Spacer(1, 0.4 * cm))

    # Technical signal + sentiment tag
    signal = technical_signal.get("overall", "Neutral")
    sentiment_label = sentiment_summary.get("overall_label", "Neutral")
    tags = Table(
        [[
            Paragraph(f"Technical Signal: <b>{signal}</b>", st["body"]),
            Paragraph(f"Market Sentiment: <b>{sentiment_label}</b>", st["body"]),
            Paragraph(
                f"Current Price: <b>${price_targets.get('current_price', 'N/A')}</b>",
                st["body"],
            ),
        ]],
        colWidths=[5.5 * cm, 5.5 * cm, 5.5 * cm],
    )
    tags.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BDBDBD")),
    ]))
    story.append(tags)
    story.append(PageBreak())

    # ── SECTION 1: COMPANY SNAPSHOT ────────────────────────────────────────────
    story.append(Paragraph("1. Company Snapshot", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))

    desc = info.get("longBusinessSummary", "No description available.")
    story.append(Paragraph(desc[:800], st["body"]))
    story.append(Spacer(1, 0.3 * cm))

    mc = info.get("marketCap")
    mc_str = f"${mc:,.0f}" if mc else "N/A"
    rows = [
        ("Exchange", info.get("exchange", "N/A")),
        ("Market Cap", mc_str),
        ("P/E Ratio (trailing)", info.get("trailingPE", "N/A")),
        ("EPS (trailing)", info.get("trailingEps", "N/A")),
        ("52-Week High", info.get("fiftyTwoWeekHigh", "N/A")),
        ("52-Week Low", info.get("fiftyTwoWeekLow", "N/A")),
        ("Dividend Yield", info.get("dividendYield", "N/A")),
        ("Beta", info.get("beta", "N/A")),
    ]
    story.append(_kv_table(rows))
    story.append(Spacer(1, 0.5 * cm))

    # ── SECTION 2: PRICE CHART ─────────────────────────────────────────────────
    story.append(Paragraph("2. Price Chart — 12 Months", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))
    story.extend(_embed_chart(chart_price_png, 16,
                               "12-month OHLCV candlestick with MA20/50/200, Bollinger Bands, and price targets"))

    # ── SECTION 3: TECHNICAL INDICATORS ───────────────────────────────────────
    story.append(Paragraph("3. Technical Analysis", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))
    story.append(Paragraph(f"<b>Overall Signal: {signal}</b>", st["body"]))
    for detail in technical_signal.get("details", []):
        story.append(Paragraph(f"• {detail}", st["bullet"]))
    story.append(Spacer(1, 0.3 * cm))
    story.extend(_embed_chart(chart_macd_png, 16, "MACD — Moving Average Convergence Divergence"))
    story.extend(_embed_chart(chart_rsi_png, 16, "RSI (14-period) — Relative Strength Index"))

    # ── SECTION 4: FUNDAMENTALS ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Fundamentals", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))

    def _fmt_pct(v):
        try:
            return f"{float(v)*100:.1f}%" if v else "N/A"
        except Exception:
            return "N/A"

    fund_rows = [
        ("Revenue (TTM)", f"${info.get('totalRevenue', 'N/A'):,}" if info.get("totalRevenue") else "N/A"),
        ("Gross Margin", _fmt_pct(info.get("grossMargins"))),
        ("Operating Margin", _fmt_pct(info.get("operatingMargins"))),
        ("Profit Margin", _fmt_pct(info.get("profitMargins"))),
        ("Return on Equity", _fmt_pct(info.get("returnOnEquity"))),
        ("Return on Assets", _fmt_pct(info.get("returnOnAssets"))),
        ("Debt/Equity", info.get("debtToEquity", "N/A")),
        ("Current Ratio", info.get("currentRatio", "N/A")),
        ("Quick Ratio", info.get("quickRatio", "N/A")),
        ("Free Cash Flow", f"${info.get('freeCashflow', 'N/A'):,}" if info.get("freeCashflow") else "N/A"),
    ]
    story.append(_kv_table(fund_rows))

    # ── SECTION 5: ANALYST CONSENSUS ──────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("5. Analyst Consensus", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))

    strong_buy = analyst_consensus.get("strong_buy", 0)
    buy = analyst_consensus.get("buy", 0)
    hold = analyst_consensus.get("hold", 0)
    sell = analyst_consensus.get("sell", 0)
    strong_sell = analyst_consensus.get("strong_sell", 0)

    analyst_rows = [
        ("Strong Buy", strong_buy),
        ("Buy", buy),
        ("Hold", hold),
        ("Sell", sell),
        ("Strong Sell", strong_sell),
        ("Mean Price Target", f"${price_targets.get('analyst_mean_target', 'N/A')}"),
        ("Low Target", f"${price_targets.get('analyst_low_target', 'N/A')}"),
        ("High Target", f"${price_targets.get('analyst_high_target', 'N/A')}"),
    ]
    story.append(_kv_table(analyst_rows))
    story.append(Spacer(1, 0.3 * cm))
    story.extend(_embed_chart(chart_analyst_png, 10, "Analyst ratings distribution"))

    # ── SECTION 6: AI ANALYSIS ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Investment Analysis & Recommendation", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))

    for line in ai_analysis_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.15 * cm))
        elif line.startswith("## ") or line.startswith("# "):
            story.append(Paragraph(line.lstrip("# "), st["h2"]))
        elif line.startswith("- ") or line.startswith("• "):
            story.append(Paragraph(line[2:], st["bullet"]))
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(f"<b>{line.strip('*')}</b>", st["body"]))
        else:
            # Convert inline **bold** markdown
            line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(line, st["body"]))

    # ── SECTION 7: NEWS DIGEST ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("7. News Digest (Last 12 Months)", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))
    story.extend(_embed_chart(chart_sentiment_png, 12, "Sentiment scores by source"))
    story.append(Spacer(1, 0.3 * cm))

    if news_scored:
        for article in news_scored[:12]:
            s_color = _sentiment_color(article.get("sentiment_label", "Neutral"))
            title = article.get("title", "")
            source = article.get("source", "")
            date = article.get("published_at", "")[:10]
            score = article.get("sentiment_score", 0)
            label = article.get("sentiment_label", "Neutral")
            story.append(Paragraph(
                f'<b>{title}</b>  <font color="grey" size=7.5>— {source} | {date} | '
                f'<font color="{s_color.hexval() if hasattr(s_color, "hexval") else "#424242"}">'
                f'{label} ({score:.2f})</font></font>',
                st["body"],
            ))
            desc = article.get("description", "")
            if desc:
                story.append(Paragraph(desc[:180], st["small"]))
            story.append(Spacer(1, 0.2 * cm))
    else:
        story.append(Paragraph("No news articles retrieved. Configure NEWS_API_KEY in .env.", st["body"]))

    # ── SECTION 8: SOCIAL SENTIMENT ───────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("8. Social Media Sentiment", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=6))

    story.append(Paragraph("<b>Reddit Posts</b>", st["h2"]))
    if reddit_posts:
        for post in reddit_posts[:8]:
            label = post.get("sentiment_label", "Neutral")
            score = post.get("sentiment_score", 0)
            story.append(Paragraph(
                f'• <b>{post.get("title", "")}</b>  '
                f'<font size=7.5 color="grey">r/{post.get("subreddit","")} | '
                f'{post.get("created_utc", "")} | ↑{post.get("score",0)} | '
                f'{label} ({score:.2f})</font>',
                st["bullet"],
            ))
    else:
        story.append(Paragraph("Reddit not configured. Add REDDIT_CLIENT_ID in .env.", st["body"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>StockTwits</b>", st["h2"]))
    if stocktwits_scored:
        for msg in stocktwits_scored[:8]:
            story.append(Paragraph(
                f'• {msg.get("body", "")}  '
                f'<font size=7.5 color="grey">@{msg.get("username","")} | '
                f'{msg.get("sentiment_label","Neutral")}</font>',
                st["bullet"],
            ))
    else:
        story.append(Paragraph("No StockTwits messages retrieved.", st["body"]))

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY, spaceAfter=4))
    story.append(Paragraph(
        "Disclaimer: This report is for informational purposes only and does not constitute "
        "financial advice. Past performance is not indicative of future results. "
        "Always conduct your own due diligence before making investment decisions.",
        st["footer"],
    ))

    doc.build(story)
    return buf.getvalue()
