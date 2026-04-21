"""
analysis.py — portfolio math engine.
Pure calculations — no yfinance calls, no Streamlit, no AI.
All functions take DataFrames / dicts and return plain Python values.
"""

import pandas as pd
from data import _calc_max_drawdown


# ============================================================
# PORTFOLIO-LEVEL STATS
# ============================================================

def compute_portfolio_stats(
    df: pd.DataFrame,
    holdings: dict,
    ticker_histories: dict,
    weighted_volatility: float | None,
    risk_free_rate: float,
) -> tuple:
    """
    Compute portfolio 1-year return, max drawdown, and Sharpe ratio.
    Returns (port_return, port_drawdown, sharpe) — any can be None on failure.
    """
    try:
        close_frames = {}
        for ticker in df["Ticker"]:
            hist = ticker_histories.get(ticker)
            if hist is not None and not hist.empty and "Close" in hist.columns and len(hist) > 1:
                close_frames[ticker] = hist["Close"].dropna()

        if not close_frames:
            return None, None, None

        price_df = pd.DataFrame(close_frames).dropna()
        if len(price_df) < 2:
            return None, None, None

        total = df["Amount Invested"].sum()
        if total <= 0:
            return None, None, None

        portfolio_series = pd.Series(0.0, index=price_df.index)
        for ticker in close_frames:
            first_price = price_df[ticker].iloc[0]
            if first_price == 0:
                continue
            normalized = price_df[ticker] / first_price
            portfolio_series += normalized * holdings.get(ticker, 0)

        first_val = portfolio_series.iloc[0]
        last_val = portfolio_series.iloc[-1]
        if first_val == 0:
            return None, None, None

        port_return = round(float((last_val - first_val) / first_val * 100), 2)
        port_drawdown = _calc_max_drawdown(portfolio_series)

        sharpe = None
        if weighted_volatility and weighted_volatility > 0:
            sharpe = round((port_return - risk_free_rate) / weighted_volatility, 2)

        return port_return, port_drawdown, sharpe

    except Exception:
        return None, None, None


# ============================================================
# CORRELATION MATRIX
# ============================================================

def compute_correlation_matrix(ticker_histories: dict) -> tuple:
    """
    Build pairwise correlation matrix of daily returns.
    Returns (corr_matrix DataFrame, list of high-corr pair dicts).
    Returns (None, []) if fewer than 2 valid tickers or insufficient history.
    """
    try:
        returns_frames = {}
        for ticker, history in ticker_histories.items():
            if history is not None and not history.empty and "Close" in history.columns:
                close = history["Close"].dropna()
                if len(close) > 1:
                    returns_frames[ticker] = close.pct_change().dropna()

        if len(returns_frames) < 2:
            return None, []

        returns_df = pd.DataFrame(returns_frames).dropna()
        if len(returns_df) < 20:
            return None, []

        corr_matrix = returns_df.corr().round(2)
        high_corr_pairs = []
        tickers = corr_matrix.columns.tolist()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                val = float(corr_matrix.iloc[i, j])
                if val >= 0.8:
                    high_corr_pairs.append({
                        "pair": tickers[i] + " & " + tickers[j],
                        "correlation": val,
                    })
        high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        return corr_matrix, high_corr_pairs

    except Exception:
        return None, []


# ============================================================
# WEIGHTED METRICS
# ============================================================

def compute_weighted_beta(df: pd.DataFrame, total: float) -> float:
    df = df.copy()
    df["Weighted Beta"] = df["Beta"] * (df["Amount Invested"] / total)
    return round(float(df["Weighted Beta"].sum()), 3)


def compute_weighted_volatility(df: pd.DataFrame, total: float) -> float | None:
    valid = df.dropna(subset=["Volatility %"])
    if valid.empty:
        return None
    return round(float((valid["Volatility %"] * (valid["Amount Invested"] / total)).sum()), 2)


# ============================================================
# AI CONTEXT BUILDER
# ============================================================

def build_portfolio_context(
    df: pd.DataFrame,
    summary: dict,
    portfolio_return: float | None,
    portfolio_max_drawdown: float | None,
    sharpe_ratio: float | None,
    spy_data: dict | None,
    spy_sharpe: float | None,
    risk_free_rate: float,
    high_corr_pairs: list,
    corr_matrix,
) -> str:
    """
    Build a reusable plain-text block describing this portfolio.
    Used as context for all AI calls — initial analysis and follow-up Q&A.
    """
    total = summary["total"]
    weighted_beta = summary["weighted_beta"]
    weighted_volatility = summary["weighted_volatility"]
    sector_breakdown = summary["sector_breakdown"]

    corr_ctx = ""
    if high_corr_pairs:
        pairs_str = ", ".join(
            [p["pair"] + " (" + str(p["correlation"]) + ")" for p in high_corr_pairs]
        )
        corr_ctx = "High-correlation pairs (above 0.80): " + pairs_str + ". "
    elif corr_matrix is not None:
        corr_ctx = "No pairs above 0.80 correlation. "

    dd_ctx = ""
    if portfolio_max_drawdown is not None:
        dd_ctx += "Portfolio max drawdown: " + str(portfolio_max_drawdown) + "%. "
    valid_dd = df.dropna(subset=["Max Drawdown %"])
    if not valid_dd.empty:
        worst = valid_dd.loc[valid_dd["Max Drawdown %"].idxmin()]
        dd_ctx += "Worst holding: " + worst["Ticker"] + " at " + str(worst["Max Drawdown %"]) + "%. "

    sharpe_ctx = ""
    if sharpe_ratio is not None:
        sharpe_ctx = "Portfolio Sharpe: " + str(sharpe_ratio) + " (risk-free: " + str(risk_free_rate) + "%). "
    if spy_sharpe is not None:
        sharpe_ctx += "SPY Sharpe: " + str(spy_sharpe) + ". "

    bench_ctx = ""
    if spy_data:
        bench_ctx = (
            "SPY benchmark — volatility: " + str(spy_data["volatility"])
            + "%, 1-year return: " + str(spy_data["one_year_return"]) + "%. "
        )
    if portfolio_return is not None:
        bench_ctx += "Portfolio 1-year return: " + str(portfolio_return) + "%. "

    holdings_detail = df[
        ["Ticker", "Allocation %", "Beta", "Volatility %", "Max Drawdown %", "Type", "Sector"]
    ].to_dict(orient="records")

    return (
        "PORTFOLIO DATA:\n"
        "Total value: $" + "{:,.0f}".format(total) + ". "
        "Weighted beta: " + str(weighted_beta) + ". "
        "Weighted volatility: " + (str(weighted_volatility) + "%" if weighted_volatility else "N/A") + ". "
        + bench_ctx + corr_ctx + dd_ctx + sharpe_ctx
        + "Sector breakdown: " + str(sector_breakdown) + ". "
        + "Holdings: " + str(holdings_detail) + "."
    )
