"""
ai_summary.py
Uses OpenAI GPT-4o to generate a structured stock analysis summary.
Falls back to a rule-based summary if no API key is configured.
"""

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def _build_prompt(ticker: str, info: dict, price_targets: dict,
                  technical_signal: dict, sentiment_summary: dict,
                  analyst_consensus: dict, news_headlines: list[str]) -> str:
    """Assemble the GPT prompt from collected data."""
    name = info.get("longName", ticker)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market_cap = info.get("marketCap", "N/A")
    pe = info.get("trailingPE", "N/A")
    eps = info.get("trailingEps", "N/A")
    revenue = info.get("totalRevenue", "N/A")
    profit_margin = info.get("profitMargins", "N/A")
    debt_equity = info.get("debtToEquity", "N/A")
    description = info.get("longBusinessSummary", "")[:600]

    headlines_text = "\n".join(f"- {h}" for h in news_headlines[:10]) if news_headlines else "No headlines available."

    buy = analyst_consensus.get("buy", 0) + analyst_consensus.get("strong_buy", 0)
    hold = analyst_consensus.get("hold", 0)
    sell = analyst_consensus.get("sell", 0) + analyst_consensus.get("strong_sell", 0)

    prompt = f"""
You are a professional equity analyst. Based on the following data, provide a structured stock analysis report for {name} ({ticker}).

## Company Overview
- Sector: {sector} | Industry: {industry}
- Market Cap: {market_cap}
- Business: {description}

## Fundamentals
- P/E Ratio: {pe}
- EPS: {eps}
- Revenue: {revenue}
- Profit Margin: {profit_margin}
- Debt/Equity: {debt_equity}

## Technical Analysis
- Signal: {technical_signal.get('overall', 'N/A')}
- Details: {'; '.join(technical_signal.get('details', []))}

## Analyst Consensus
- Buy: {buy} | Hold: {hold} | Sell: {sell}
- Mean Price Target: {price_targets.get('analyst_mean_target', 'N/A')}
- Low Target: {price_targets.get('analyst_low_target', 'N/A')}
- High Target: {price_targets.get('analyst_high_target', 'N/A')}

## Sentiment (News + Social)
- Overall: {sentiment_summary.get('overall_label', 'N/A')} (score: {sentiment_summary.get('overall_score', 0):.2f})
- News sentiment avg: {sentiment_summary.get('news_avg', 0):.2f}
- Reddit sentiment avg: {sentiment_summary.get('reddit_avg', 0):.2f}

## Recent News Headlines (last 12 months)
{headlines_text}

---
Please provide:
1. **Executive Summary** (3-4 sentences covering the company's current position and outlook)
2. **Investment Thesis** (why this might be a good or bad investment)
3. **Key Risks** (3-5 bullet points)
4. **Overall Recommendation** (Strong Buy / Buy / Hold / Sell / Strong Sell) with a one-paragraph justification
5. **Price Targets**:
   - Suggested Buy Price: {price_targets.get('buy_price', 'N/A')}
   - Suggested Sell / Target Price: {price_targets.get('sell_price', 'N/A')}
   - Suggested Cut-Loss Price: {price_targets.get('cut_loss_price', 'N/A')}
   Confirm or adjust these based on your analysis and briefly explain your reasoning.

Keep the tone professional and data-driven. Limit total response to ~600 words.
"""
    return prompt.strip()


def generate_ai_analysis(
    ticker: str,
    info: dict,
    price_targets: dict,
    technical_signal: dict,
    sentiment_summary: dict,
    analyst_consensus: dict,
    news_headlines: list[str],
) -> str:
    """Call OpenAI GPT-4o and return the analysis text."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_key_here":
        return _fallback_summary(ticker, info, price_targets, technical_signal,
                                 sentiment_summary, analyst_consensus)

    try:
        import httpx
        from openai import OpenAI
        # Disable SSL verification for corporate proxy environments
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
        prompt = _build_prompt(ticker, info, price_targets, technical_signal,
                               sentiment_summary, analyst_consensus, news_headlines)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional equity research analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI analysis unavailable: {e}\n\n" + _fallback_summary(
            ticker, info, price_targets, technical_signal, sentiment_summary, analyst_consensus
        )


def _fallback_summary(ticker, info, price_targets, technical_signal,
                      sentiment_summary, analyst_consensus) -> str:
    """Rule-based summary when OpenAI is not available."""
    name = info.get("longName", ticker)
    sector = info.get("sector", "N/A")
    signal = technical_signal.get("overall", "Neutral")
    sentiment = sentiment_summary.get("overall_label", "Neutral")

    buy = analyst_consensus.get("buy", 0) + analyst_consensus.get("strong_buy", 0)
    hold = analyst_consensus.get("hold", 0)
    sell = analyst_consensus.get("sell", 0) + analyst_consensus.get("strong_sell", 0)
    total = buy + hold + sell or 1

    if buy / total > 0.5:
        consensus = "Buy"
    elif sell / total > 0.4:
        consensus = "Sell"
    else:
        consensus = "Hold"

    current = price_targets.get("current_price", "N/A")
    buy_p = price_targets.get("buy_price", "N/A")
    sell_p = price_targets.get("sell_price", "N/A")
    cut_p = price_targets.get("cut_loss_price", "N/A")

    return f"""## Executive Summary
{name} ({ticker}) operates in the {sector} sector. Based on available quantitative data, the stock shows a **{signal}** technical trend with **{sentiment}** sentiment across news and social media channels. Analyst consensus leans toward **{consensus}** ({buy} buy / {hold} hold / {sell} sell ratings).

## Investment Thesis
The stock's technical indicators suggest a {signal.lower()} momentum phase. Market sentiment is {sentiment.lower()}, which {"supports" if sentiment == "Positive" else "adds caution to"} a long position at current levels.

## Key Risks
- Market-wide volatility could affect short-term price action
- Macro headwinds (interest rates, inflation) may pressure valuations
- Company-specific risks detailed in the latest earnings call
- Liquidity risk if trading volumes decline

## Overall Recommendation: {consensus}
Based on the combination of technical signals ({signal}), analyst consensus ({consensus}), and market sentiment ({sentiment}), a **{consensus}** stance is appropriate at current levels.

## Price Targets
- **Current Price:** ${current}
- **Suggested Buy Price:** ${buy_p}
- **Suggested Sell / Target Price:** ${sell_p}
- **Suggested Cut-Loss Price:** ${cut_p}
"""
