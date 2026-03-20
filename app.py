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
    # Replace backtick-wrapped dollar amounts like `$1,770` with plain $1,770
    text = text.replace("`", "")
    return text

# --- Helper: check data freshness ---
def check_data_freshness(history):
    if history.empty:
        return False
    last_date = history.index[-1].date()
    today = datetime.today().date()
    days_since = (today - last_date).days
    # Allow up to 4 days (covers weekends + holidays)
    return days_since <= 4

# --- Data fetch ---
def get_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")

        missing = []
        warnings = []
        ticker_is_etf = is_etf(info)

        # Check data freshness
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

        return {
            "ticker": ticker,
            "current_price": current_price,
            "sector": sector,
            "beta": beta,
            "volatility": volatility,
            "missing": missing,
            "warnings": warnings,
            "is_etf": ticker_is_etf,
            "valid": True
        }

    except Exception as e:
        return {"ticker": ticker, "valid": False, "error": str(e)}


# --- Portfolio analysis ---
def analyze_portfolio(holdings):
    rows = []
    missing_data_notes = []
    warnings = []
    skipped = []

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

        rows.append({
            "Ticker": ticker,
            "Amount Invested": holdings[ticker],
            "Sector": data["sector"],
            "Beta": data["beta"],
            "Volatility %": data["volatility"],
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

    # Build AI prompt
    prompt = (
        "You are a financial analyst explaining a retail investor portfolio. "
        "Be specific, direct, and avoid generic advice. Use plain text only — no markdown, no bold, no bullet points. "
        "Here is the portfolio data: "
        "Total Value: $" + "{:,.2f}".format(total) + ". "
        "Weighted Beta: " + str(weighted_beta) + ". "
        "Weighted Annualized Volatility: " + (str(weighted_volatility) + "%" if weighted_volatility else "unavailable") + ". "
        "Sector Breakdown: " + str(sector_breakdown) + ". "
        "Holdings: " + str(df[["Ticker", "Allocation %", "Beta", "Volatility %", "Type"]].to_dict(orient="records")) + ". "
        "Missing data notes: " + (", ".join(missing_data_notes) if missing_data_notes else "none") + ". "
        "Explain in 4-5 sentences: "
        "1. What this portfolio is concentrated in and what that means. "
        "2. What the beta and volatility together tell us about the risk level. "
        "3. How the risk is distributed across holdings - which positions are driving it. "
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

    # Store in session state
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
        "warnings": warnings
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

# --- Display Results (persists via session state) ---
if st.session_state.results_ready:
    df = st.session_state.df
    s = st.session_state.summary

    st.subheader("Portfolio Breakdown")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", "$" + "{:,.2f}".format(s["total"]))
    col2.metric("Weighted Beta", str(s["weighted_beta"]))
    col3.metric("Weighted Volatility", str(s["weighted_volatility"]) + "%" if s["weighted_volatility"] else "N/A")

    st.dataframe(
        df[["Ticker", "Type", "Amount Invested", "Allocation %", "Sector", "Beta", "Volatility %"]].reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("Sector Breakdown")
    sector_df = pd.DataFrame(list(s["sector_breakdown"].items()), columns=["Sector", "Allocation %"])
    st.dataframe(sector_df, use_container_width=True)

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
