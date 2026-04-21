"""
data.py — yfinance data fetching and validation.
All external API calls live here. Nothing in this file touches Streamlit.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

RISK_FREE_RATE_FALLBACK = 4.3


# ============================================================
# VALIDATION HELPERS
# ============================================================

def is_etf(info: dict) -> bool:
    return info.get("quoteType", "").upper() == "ETF"


def info_is_valid(info: dict) -> bool:
    """
    Confirm yfinance returned real data for this ticker.

    Strategy: require at least 'longName' or 'shortName' as proof the symbol
    exists. We no longer require price fields here — those are fetched from
    history as a fallback because recent yfinance versions often omit
    currentPrice / regularMarketPrice from .info for valid tickers.
    """
    if not info:
        return False
    # Name fields — present for every real equity/ETF
    if info.get("longName") or info.get("shortName"):
        return True
    # quoteType — present for ETFs and some stocks
    if info.get("quoteType") and info["quoteType"] not in ("", "None"):
        return True
    # Price fields — accepted as fallback (older yfinance versions)
    if info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"):
        return True
    return False


def check_data_freshness(history: pd.DataFrame) -> bool:
    """Return True if the most recent price bar is within 4 calendar days."""
    if history is None or history.empty:
        return False
    try:
        last_date = history.index[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        return (datetime.today().date() - last_date).days <= 4
    except Exception:
        return False


# ============================================================
# SINGLE-TICKER FETCH
# ============================================================

def get_ticker_data(ticker: str) -> dict:
    """
    Fetch all data needed for one ticker.
    Returns {"valid": True, ...} on success or {"valid": False, "error": str} on failure.
    Never raises — all exceptions are caught and reported.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info_is_valid(info):
            return {
                "ticker": ticker,
                "valid": False,
                "error": "ticker not recognised — check the symbol and try again",
            }

        history = stock.history(period="1y")
        missing = []
        warnings_list = []
        ticker_is_etf = is_etf(info)

        if not check_data_freshness(history):
            warnings_list.append(f"{ticker}: price data may be stale (last update >4 days ago)")

        # Price — try info first, fall back to last close in history
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is None and not history.empty and "Close" in history.columns:
            closes = history["Close"].dropna()
            if len(closes) > 0:
                current_price = round(float(closes.iloc[-1]), 4)
        if current_price is None:
            missing.append("current price")

        # Sector / beta
        if ticker_is_etf:
            sector = "ETF (diversified)"
            beta = info.get("beta3Year") or info.get("beta") or 1.0
        else:
            sector = info.get("sector") or "Unknown"
            if sector == "Unknown":
                missing.append("sector")
            beta = info.get("beta")
            if beta is None:
                beta = 1.0
                missing.append("beta (defaulted to 1.0)")

        # History validation
        has_history = (
            history is not None
            and not history.empty
            and "Close" in history.columns
            and len(history) > 1
        )

        if has_history:
            close = history["Close"].dropna()
            if len(close) > 1:
                daily_returns = close.pct_change().dropna()
                volatility = round(float(daily_returns.std() * (252 ** 0.5) * 100), 2)
                max_drawdown = _calc_max_drawdown(close)
            else:
                volatility = None
                max_drawdown = None
                missing.append("volatility (insufficient price history)")
        else:
            history = pd.DataFrame()
            volatility = None
            max_drawdown = None
            missing.append("price history")

        return {
            "ticker": ticker,
            "current_price": current_price,
            "sector": sector,
            "beta": float(beta),
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "missing": missing,
            "warnings": warnings_list,
            "is_etf": ticker_is_etf,
            "history": history,
            "valid": True,
        }

    except Exception as e:
        return {"ticker": ticker, "valid": False, "error": str(e)}


# ============================================================
# BENCHMARK
# ============================================================

def get_spy_benchmark(existing_histories: dict = None) -> dict | None:
    """Return SPY volatility and 1-year return. Reuses history if already fetched."""
    try:
        if existing_histories and "SPY" in existing_histories:
            history = existing_histories["SPY"]
        else:
            history = yf.Ticker("SPY").history(period="1y")

        if history is None or history.empty or "Close" not in history.columns:
            return None

        close = history["Close"].dropna()
        if len(close) < 2:
            return None

        daily_returns = close.pct_change().dropna()
        spy_vol = round(float(daily_returns.std() * (252 ** 0.5) * 100), 2)
        spy_ret = round(float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100), 2)
        return {"volatility": spy_vol, "one_year_return": spy_ret, "beta": 1.0}
    except Exception:
        return None


# ============================================================
# RISK-FREE RATE
# ============================================================

def get_risk_free_rate() -> float:
    """Fetch live 13-week T-bill yield via ^IRX. Falls back to constant."""
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            rate = float(hist["Close"].iloc[-1])
            if 0 < rate < 20:
                return round(rate, 2)
        return RISK_FREE_RATE_FALLBACK
    except Exception:
        return RISK_FREE_RATE_FALLBACK


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _calc_max_drawdown(price_series: pd.Series) -> float | None:
    """Peak-to-trough % drop. Returns None on bad input."""
    try:
        if price_series is None or len(price_series) < 2:
            return None
        clean = price_series.dropna()
        if len(clean) < 2:
            return None
        running_peak = clean.cummax()
        drawdown = (clean - running_peak) / running_peak * 100
        result = drawdown.min()
        if pd.isna(result):
            return None
        return round(float(result), 2)
    except Exception:
        return None
