import streamlit as st
import yfinance as yf
import pandas as pd
import anthropic
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Insight Tool", layout="centered")
st.title("Portfolio Insight Tool")
st.write("Enter your holdings below to get a full portfolio analysis.")

# --- Load API key from Streamlit secrets ---
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = None
    st.error("API key not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# --- Session state initialization ---
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = {}
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = ""
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "missing_data_notes" not in st.session_state:
    st.session_state.missing_data_notes = []

# --- Helper: detect ETF ---
def is_etf(info):
    quote_type = info.get("quoteType", "").upper()
    return quote_type == "ETF"

# --- Helper: fix dollar sign formatting from AI output ---
def fix_dollar_formatting(text):
    text = text.replace("`", "")
    return text

# --- Helper: check data freshness ---
def check_data_freshness(history):
    if history.empty:
        return False
    last_date = history.index[-1].date()
    today = datetime.today().date()
    days_since = (today - last_date).days
    return days_since <= 4

# --- Helper: calculate max drawdown for a single price series ---
# Takes a pandas Series of prices, returns the worst peak-to-trough drop as a percentage.
def calc_max_drawdown(price_series):
    if len(price_series) < 2:
        return None
    running_peak = price_series.cummax()
    drawdown = (price_series - running_peak) / running_peak * 100
    return round(drawdown.min(), 2)  # most negative value = worst drop


# --- Data fetch ---
def get_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")

        missing = []
        warnings = []
        ticker_is_etf = is_etf(info)

        if not check_data_freshness(history):
            warnings.append(ticker + ": price data may be stale — market could be closed or data delayed")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is None:
            missing.append("current price")

        if ticker_is_etf:
            sector = "ETF (diversified)"
            beta = info.get("beta3Year") or info.get("beta") or 1.0
        else:
            sector = info.get("sector", None)
            if sector is None:
                sector = "Unknown"
                missing.append("sector")
            beta = info.get("beta", None)
            if beta is None:
                beta = 1.0
                missing.append("beta (defaulted to 1.0)")

        if len(history) > 1:
            daily_returns = history["Close"].pct_change().dropna()
            volatility = round(daily_returns.std() * (252 ** 0.5) * 100, 2)
        else:
            volatility = None
            missing.append("volatility")

        # --- NEW: calculate max drawdown per ticker ---
        max_drawdown = calc_max_drawdown(history["Close"]) if len(history) > 1 else None

        return {
            "ticker": ticker,
            "current_price": current_price,
            "sector": sector,
            "beta": beta,
            "volatility": volatility,
            "max_drawdown": max_drawdown,   # ← NEW
            "missing": missing,
            "warnings": warnings,
            "is_etf": ticker_is_etf,
            "history": history,
            "valid": True
        }

    except Exception as e:
        return {"ticker": ticker, "valid": False, "error": str(e)}


# --- Fetch SPY benchmark data ---
def get_spy_benchmark(existing_histories=None):
    try:
        if existing_histories and "SPY" in existing_histories:
            history = existing_histories["SPY"]
        else:
            spy = yf.Ticker("SPY")
            history = spy.history(period="1y")

        if len(history) < 2:
            return None

        daily_returns = history["Close"].pct_change().dropna()
        spy_volatility = round(daily_returns.std() * (252 ** 0.5) * 100, 2)

        start_price = history["Close"].iloc[0]
        end_price = history["Close"].iloc[-1]
        spy_return = round((end_price - start_price) / start_price * 100, 2)

        return {
            "volatility": spy_volatility,
            "one_year_return": spy_return,
            "beta": 1.0,
            "close_series": history["Close"]
        }
    except Exception:
        return None


# --- Calculate portfolio 1-year return and portfolio-level max drawdown ---
# Builds a weighted daily portfolio value series, then computes both
# total return and the worst peak-to-trough drop across that combined series.
def get_portfolio_return(df, holdings, ticker_histories):
    try:
        close_frames = {}
        for ticker in df["Ticker"]:
            if ticker in ticker_histories and len(ticker_histories[ticker]) > 1:
                close_frames[ticker] = ticker_histories[ticker]["Close"]

        if not close_frames:
            return None, None

        price_df = pd.DataFrame(close_frames)
        price_df = price_df.dropna()

        if len(price_df) < 2:
            return None, None

        total = df["Amount Invested"].sum()

        portfolio_series = pd.Series(0.0, index=price_df.index)
        for ticker in close_frames:
            amount = holdings.get(ticker, 0)
            normalized = price_df[ticker] / price_df[ticker].iloc[0]
            portfolio_series += normalized * amount

        start_val = portfolio_series.iloc[0]
        end_val = portfolio_series.iloc[-1]
        portfolio_return = round((end_val - start_val) / start_val * 100, 2)

        # --- NEW: calculate portfolio-level max drawdown from the combined series ---
        portfolio_max_drawdown = calc_max_drawdown(portfolio_series)

        return portfolio_return, portfolio_max_drawdown

    except Exception:
        return None, None


# --- Calculate correlation matrix ---
def get_correlation_matrix(ticker_histories):
    try:
        returns_frames = {}
        for ticker, history in ticker_histories.items():
            if len(history) > 1:
                daily_returns = history["Close"].pct_change().dropna()
                returns_frames[ticker] = daily_returns

        if len(returns_frames) < 2:
            return None, []

        returns_df = pd.DataFrame(returns_frames)
        returns_df = returns_df.dropna()

        if len(returns_df) < 20:
            return None, []

        corr_matrix = returns_df.corr().round(2)

        high_corr_pairs = []
        tickers = corr_matrix.columns.tolist()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                val = corr_matrix.iloc[i, j]
                if val >= 0.8:
                    high_corr_pairs.append({
                        "pair": tickers[i] + " & " + tickers[j],
                        "correlation": val
                    })

        high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)

        return corr_matrix, high_corr_pairs

    except Exception:
        return None, []


# --- Portfolio analysis ---
def analyze_portfolio(holdings):
    rows = []
    missing_data_notes = []
    warnings = []
    skipped = []
    ticker_histories = {}

    progress = st.progress(0, text="Fetching data...")
    tickers = list(holdings.keys())

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text="Fetching data for " + ticker + "...")
        data = get_ticker_data(ticker)

        if not data["valid"]:
            skipped.append(ticker + " (" + data.get("error", "unknown error") + ")")
            continue

        if data["missing"]:
            if data["is_etf"]:
                missing_data_notes.append(ticker + " is an ETF — sector and some metrics are not applicable")
            else:
                missing_data_notes.append(ticker + " is missing: " + ", ".join(data["missing"]))

        if data["warnings"]:
            warnings.extend(data["warnings"])

        ticker_histories[ticker] = data["history"]

        rows.append({
            "Ticker": ticker,
            "Amount Invested": holdings[ticker],
            "Sector": data["sector"],
            "Beta": data["beta"],
            "Volatility %": data["volatility"],
            "Max Drawdown %": data["max_drawdown"],   # ← NEW
            "Type": "ETF" if data["is_etf"] else "Stock"
        })

    progress.empty()

    if not rows:
        st.error("No valid tickers found. Please check your inputs.")
        return

    df = pd.DataFrame(rows)
    total = df["Amount Invested"].sum()
    df["Allocation %"] = (df["Amount Invested"] / total * 100).round(2)
    df["Weighted Beta"] = df["Beta"] * (df["Amount Invested"] / total)
    weighted_beta = df["Weighted Beta"].sum().round(3)

    valid_vol = df.dropna(subset=["Volatility %"])
    if not valid_vol.empty:
        weighted_volatility = round(
            (valid_vol["Volatility %"] * (valid_vol["Amount Invested"] / total)).sum(), 2
        )
    else:
        weighted_volatility = None

    sector_breakdown = df.groupby("Sector")["Allocation %"].sum().round(2).to_dict()
    concentrated_sectors = {s: p for s, p in sector_breakdown.items() if p > 60 and s not in ["ETF (diversified)", "Unknown"]}

    spy_data = get_spy_benchmark(existing_histories=ticker_histories)

    # --- UPDATED: get_portfolio_return now returns both return and drawdown ---
    portfolio_return, portfolio_max_drawdown = get_portfolio_return(df, holdings, ticker_histories)

    corr_matrix, high_corr_pairs = get_correlation_matrix(ticker_histories)

    # Build correlation context for AI prompt
    corr_context = ""
    if high_corr_pairs:
        pair_strings = [p["pair"] + " (" + str(p["correlation"]) + ")" for p in high_corr_pairs]
        corr_context = "High correlation pairs (above 0.80): " + ", ".join(pair_strings) + ". "
    elif corr_matrix is not None:
        corr_context = "No holdings pairs have correlation above 0.80 — diversification across holdings is reasonable. "

    # --- NEW: build drawdown context for AI prompt ---
    drawdown_context = ""
    if portfolio_max_drawdown is not None:
        drawdown_context += "Portfolio max drawdown (worst peak-to-trough drop over past year): " + str(portfolio_max_drawdown) + "%. "
    valid_dd = df.dropna(subset=["Max Drawdown %"])
    if not valid_dd.empty:
        worst_holding = valid_dd.loc[valid_dd["Max Drawdown %"].idxmin()]
        drawdown_context += (
            "Worst individual holding drawdown: " + worst_holding["Ticker"] +
            " at " + str(worst_holding["Max Drawdown %"]) + "%. "
        )

    benchmark_context = ""
    if spy_data:
        benchmark_context = (
            "SPY Benchmark — Volatility: " + str(spy_data["volatility"]) + "%, "
            "1-Year Return: " + str(spy_data["one_year_return"]) + "%. "
        )
    if portfolio_return is not None:
        benchmark_context += "Portfolio 1-Year Return: " + str(portfolio_return) + "%. "

    prompt = (
        "You are a financial analyst explaining a retail investor portfolio. "
        "Be specific, direct, and avoid generic advice. Use plain text only — no markdown, no bold, no bullet points. "
        "Here is the portfolio data: "
        "Total Value: $" + "{:,.2f}".format(total) + ". "
        "Weighted Beta: " + str(weighted_beta) + ". "
        "Weighted Annualized Volatility: " + (str(weighted_volatility) + "%" if weighted_volatility else "unavailable") + ". "
        + benchmark_context
        + corr_context
        + drawdown_context +
        "Sector Breakdown: " + str(sector_breakdown) + ". "
        "Holdings: " + str(df[["Ticker", "Allocation %", "Beta", "Volatility %", "Max Drawdown %", "Type"]].to_dict(orient="records")) + ". "
        "Missing data notes: " + (", ".join(missing_data_notes) if missing_data_notes else "none") + ". "
        "Explain in 4-5 sentences: "
        "1. What this portfolio is concentrated in and what that means. "
        "2. What the beta, volatility, and max drawdown together tell us about the real downside risk, compared to SPY where available. "
        "3. How the risk is distributed across holdings — which positions are driving it, and whether high correlation creates hidden concentration risk. "
        "4. One specific thing this investor should be aware of or watch out for."
    )

    with st.spinner("Generating AI analysis..."):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_text = fix_dollar_formatting(message.content[0].text)
        except Exception as e:
            ai_text = "AI analysis failed: " + str(e)

    st.session_state.results_ready = True
    st.session_state.df = df
    st.session_state.summary = {
        "total": total,
        "weighted_beta": weighted_beta,
        "weighted_volatility": weighted_volatility,
        "sector_breakdown": sector_breakdown,
        "concentrated_sectors": concentrated_sectors,
        "skipped": skipped,
        "missing_data_notes": missing_data_notes,
        "warnings": warnings,
        "spy_data": spy_data,
        "portfolio_return": portfolio_return,
        "portfolio_max_drawdown": portfolio_max_drawdown,   # ← NEW
        "corr_matrix": corr_matrix,
        "high_corr_pairs": high_corr_pairs
    }
    st.session_state.ai_analysis = ai_text


# --- Input Section ---
input_method = st.radio("How would you like to enter your portfolio?", ["One by one", "Paste a list"])

holdings = {}

if input_method == "One by one":
    st.write("Add each ticker and the dollar amount you have invested.")
    num_holdings = st.number_input("How many holdings?", min_value=1, max_value=20, value=3, step=1)
    for i in range(int(num_holdings)):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", key="ticker_" + str(i), placeholder="e.g. AAPL").upper().strip()
        with col2:
            amount = st.number_input("Amount Invested ($)", min_value=0.0, key="amount_" + str(i), step=100.0)
        if ticker and amount > 0:
            holdings[ticker] = amount

else:
    st.write("Paste your portfolio below. One holding per line, formatted as: TICKER AMOUNT")
    st.code("AAPL 2000\nTSLA 1000\nSPY 3000")
    raw_input = st.text_area("Paste your portfolio here", height=150)
    if raw_input:
        for line in raw_input.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    ticker = parts[0].upper()
                    amount = float(parts[1])
                    if amount > 0:
                        holdings[ticker] = amount
                except ValueError:
                    st.warning("Skipped invalid line: " + line)

if st.button("Analyze Portfolio"):
    if not holdings:
        st.error("Please enter at least one valid holding.")
    else:
        analyze_portfolio(holdings)

# --- Display Results ---
if st.session_state.results_ready:
    df = st.session_state.df
    s = st.session_state.summary

    # --- Summary Metrics ---
    # NEW: now four columns — added Portfolio Max Drawdown
    st.subheader("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", "$" + "{:,.2f}".format(s["total"]))
    col2.metric("Weighted Beta", str(s["weighted_beta"]))
    col3.metric("Weighted Volatility", str(s["weighted_volatility"]) + "%" if s["weighted_volatility"] else "N/A")
    col4.metric(
        "Max Drawdown",
        str(s["portfolio_max_drawdown"]) + "%" if s["portfolio_max_drawdown"] is not None else "N/A"
    )

    # --- Benchmark Comparison ---
    spy = s.get("spy_data")
    port_return = s.get("portfolio_return")

    if spy:
        st.subheader("Benchmark Comparison (vs. SPY)")
        comparison_rows = [
            {"Metric": "Beta", "Your Portfolio": str(s["weighted_beta"]), "SPY (Benchmark)": "1.00"},
            {
                "Metric": "Annualized Volatility",
                "Your Portfolio": str(s["weighted_volatility"]) + "%" if s["weighted_volatility"] else "N/A",
                "SPY (Benchmark)": str(spy["volatility"]) + "%"
            },
            {
                "Metric": "1-Year Return",
                "Your Portfolio": (("+" if port_return >= 0 else "") + str(port_return) + "%") if port_return is not None else "N/A",
                "SPY (Benchmark)": ("+" if spy["one_year_return"] >= 0 else "") + str(spy["one_year_return"]) + "%"
            }
        ]
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df.set_index("Metric"), width='stretch')
        st.caption(
            "Beta measures sensitivity to market moves — SPY's beta is always 1.0 by definition. "
            "Volatility is annualized from 1 year of daily returns. "
            "1-Year Return is calculated from daily price history and may differ slightly from brokerage statements due to dividend handling."
        )

    # --- Correlation Matrix ---
    corr_matrix = s.get("corr_matrix")
    high_corr_pairs = s.get("high_corr_pairs", [])

    if corr_matrix is not None:
        st.subheader("Correlation Matrix")

        def color_correlation(val):
            if val == 1.0:
                return "background-color: #c0392b; color: white;"
            elif val >= 0.8:
                return "background-color: #e74c3c; color: white;"
            elif val >= 0.6:
                return "background-color: #e67e22; color: white;"
            elif val >= 0.4:
                return "background-color: #f39c12; color: black;"
            elif val >= 0.2:
                return "background-color: #d4efdf; color: black;"
            else:
                return "background-color: #27ae60; color: white;"

        styled_corr = corr_matrix.style.map(color_correlation).format("{:.2f}")
        st.dataframe(styled_corr, width='stretch')

        st.caption(
            "Correlation ranges from -1.0 to 1.0. "
            "1.0 means two holdings move in perfect lockstep. "
            "0.0 means no relationship. "
            "Negative values mean they tend to move in opposite directions. "
            "Pairs above 0.80 are highlighted in red — they may look diversified but behave as one position in a downturn."
        )

        if high_corr_pairs:
            for pair in high_corr_pairs:
                st.warning(
                    "High Correlation: " + pair["pair"] + " have a correlation of " +
                    str(pair["correlation"]) + " — these holdings tend to move together and may not provide the diversification they appear to."
                )

    # --- Portfolio Table ---
    # NEW: Max Drawdown % column added
    st.subheader("Portfolio Breakdown")
    st.dataframe(
        df[["Ticker", "Type", "Amount Invested", "Allocation %", "Sector", "Beta", "Volatility %", "Max Drawdown %"]].reset_index(drop=True),
        width='stretch'
    )

    # --- Sector Breakdown ---
    st.subheader("Sector Breakdown")
    sector_df = pd.DataFrame(list(s["sector_breakdown"].items()), columns=["Sector", "Allocation %"])
    st.dataframe(sector_df, width='stretch')

    if s["concentrated_sectors"]:
        for sector, pct in s["concentrated_sectors"].items():
            st.warning("Concentration Warning: " + str(pct) + "% of this portfolio is in " + sector)

    if s["warnings"]:
        for w in s["warnings"]:
            st.warning(w)

    if s["skipped"]:
        st.warning("Skipped tickers: " + ", ".join(s["skipped"]))

    if s["missing_data_notes"]:
        with st.expander("Data Notes"):
            for note in s["missing_data_notes"]:
                st.write("- " + note)

    st.subheader("AI Analysis")
    st.write(st.session_state.ai_analysis)
