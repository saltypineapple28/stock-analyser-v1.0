"""
technical.py
Computes technical indicators and generates buy/sell/cut-loss signals.
"""

import pandas as pd
import numpy as np


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame."""
    df = history.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Moving Averages ---
    df["MA5"]  = close.rolling(5).mean()
    df["MA14"] = close.rolling(14).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA30"] = close.rolling(30).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA200"] = close.rolling(200).mean()

    # --- Bollinger Bands (20-day, 2 std) ---
    df["BB_mid"] = df["MA20"]
    df["BB_std"] = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # --- ATR (Average True Range, 14-period) ---
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # --- Volume MA ---
    df["Vol_MA20"] = volume.rolling(20).mean()

    return df


def derive_price_targets(df: pd.DataFrame, analyst_targets: dict, info: dict) -> dict:
    """
    Derive buy / sell / cut-loss price recommendations.
    Uses analyst consensus targets and ATR-based risk management.
    """
    latest = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else None
    atr = df["ATR"].dropna().iloc[-1] if not df["ATR"].dropna().empty else None

    # Analyst price targets
    analyst_mean = None
    analyst_low = None
    analyst_high = None

    if isinstance(analyst_targets, dict):
        analyst_mean = analyst_targets.get("mean") or analyst_targets.get("targetMeanPrice")
        analyst_low = analyst_targets.get("low") or analyst_targets.get("targetLowPrice")
        analyst_high = analyst_targets.get("high") or analyst_targets.get("targetHighPrice")
    
    # Fallback to info dict
    if analyst_mean is None:
        analyst_mean = info.get("targetMeanPrice")
    if analyst_low is None:
        analyst_low = info.get("targetLowPrice")
    if analyst_high is None:
        analyst_high = info.get("targetHighPrice")

    # ── Buy Zone: MA-based support levels (user-selectable) ──────────────────
    def _ma_val(col):
        s = df[col].dropna()
        return round(float(s.iloc[-1]), 2) if not s.empty else None

    buy_zone_ma5  = _ma_val("MA5")
    buy_zone_ma14 = _ma_val("MA14")
    buy_zone_ma30 = _ma_val("MA30")
    buy_zone_ma60 = _ma_val("MA60")

    # Default buy_price = MA60 (medium-term support)
    buy_price = buy_zone_ma60 or buy_zone_ma30 or buy_zone_ma14 or buy_zone_ma5

    # Sell / target price: use analyst mean target, fallback to +20% of current
    if analyst_mean and latest:
        sell_price = round(float(analyst_mean), 2)
    elif latest:
        sell_price = round(latest * 1.20, 2)
    else:
        sell_price = None

    # Cut-loss price: current price minus 2x ATR (standard risk management)
    if latest and atr:
        cut_loss_price = round(latest - (2 * atr), 2)
    elif latest:
        cut_loss_price = round(latest * 0.92, 2)  # -8% fallback
    else:
        cut_loss_price = None

    return {
        "current_price": round(latest, 2) if latest else None,
        "buy_price": buy_price,
        "buy_zone_ma5":  buy_zone_ma5,
        "buy_zone_ma14": buy_zone_ma14,
        "buy_zone_ma30": buy_zone_ma30,
        "buy_zone_ma60": buy_zone_ma60,
        "sell_price": sell_price,
        "cut_loss_price": cut_loss_price,
        "analyst_mean_target": round(float(analyst_mean), 2) if analyst_mean else None,
        "analyst_low_target": round(float(analyst_low), 2) if analyst_low else None,
        "analyst_high_target": round(float(analyst_high), 2) if analyst_high else None,
        "atr": round(float(atr), 2) if atr else None,
    }


def get_technical_signal(df: pd.DataFrame) -> dict:
    """
    Generate a simple overall technical signal (Bullish / Bearish / Neutral)
    based on RSI, MACD, and MA crossover.
    """
    signals = []
    details = []

    try:
        rsi = df["RSI"].dropna().iloc[-1]
        if rsi < 30:
            signals.append(1)
            details.append(f"RSI {rsi:.1f} — Oversold (bullish)")
        elif rsi > 70:
            signals.append(-1)
            details.append(f"RSI {rsi:.1f} — Overbought (bearish)")
        else:
            signals.append(0)
            details.append(f"RSI {rsi:.1f} — Neutral")
    except Exception:
        pass

    try:
        macd = df["MACD"].dropna().iloc[-1]
        macd_sig = df["MACD_signal"].dropna().iloc[-1]
        if macd > macd_sig:
            signals.append(1)
            details.append("MACD above signal line — Bullish")
        else:
            signals.append(-1)
            details.append("MACD below signal line — Bearish")
    except Exception:
        pass

    try:
        close = df["Close"].dropna().iloc[-1]
        ma50 = df["MA50"].dropna().iloc[-1]
        ma200 = df["MA200"].dropna().iloc[-1]
        if close > ma50 > ma200:
            signals.append(1)
            details.append("Price > MA50 > MA200 — Bullish trend")
        elif close < ma50 < ma200:
            signals.append(-1)
            details.append("Price < MA50 < MA200 — Bearish trend")
        else:
            signals.append(0)
            details.append("Mixed MA alignment — Neutral trend")
    except Exception:
        pass

    score = sum(signals)
    if score >= 2:
        overall = "Bullish"
    elif score <= -2:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {"overall": overall, "score": score, "details": details}
