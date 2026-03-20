import streamlit as st
import yfinance as yf
import pandas as pd
import anthropic

st.set_page_config(page_title="Portfolio Insight Tool", layout="centered")
st.title("Portfolio Insight Tool")
st.write("Enter your holdings below to get a full portfolio analysis.")

# --- Input Method ---
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
    st.write("Example:")
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

# --- API Key ---
api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")

# --- Analyze Button ---
analyze = st.button("Analyze Portfolio")

# --- Data Functions ---
def get_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")

        missing = []

        current_price = info.get("currentPrice")
        if current_price is None:
            missing.append("current price")

        sector = info.get("sector", None)
        if sector is None:
            sector = "Unknown"
            missing.append("sector")

        beta = info.get("beta", None)
        if beta is None:
            beta = 1.0
            missing.append("beta")

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
            "valid": True
        }

    except Exception as e:
        return {"ticker": ticker, "valid": False, "error": str(e)}


def analyze_portfolio(holdings, api_key):
    rows = []
    missing_data_notes = []
    skipped = []

    progress = st.progress(0, text="Fetching data...")
    tickers = list(holdings.keys())

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text="Fetching data for " + ticker + "...")
        data = get_ticker_data(ticker)

        if not data["valid"]:
            skipped.append(ticker)
            continue

        if data["missing"]:
            missing_data_notes.append(ticker + " is missing: " + ", ".join(data["missing"]))

        rows.append({
            "Ticker": ticker,
            "Amount Invested": holdings[ticker],
            "Sector": data["sector"],
            "Beta": data["beta"],
            "Volatility %": data["volatility"]
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

    # --- Results: Table ---
    st.subheader("Portfolio Breakdown")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", "$" + "{:,.2f}".format(total))
    col2.metric("Weighted Beta", str(weighted_beta))
    col3.metric("Weighted Volatility", str(weighted_volatility) + "%" if weighted_volatility else "N/A")

    st.dataframe(
        df[["Ticker", "Amount Invested", "Allocation %", "Sector", "Beta", "Volatility %"]].reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("Sector Breakdown")
    sector_df = pd.DataFrame(list(sector_breakdown.items()), columns=["Sector", "Allocation %"])
    st.dataframe(sector_df, use_container_width=True)

    concentrated_sectors = {s: p for s, p in sector_breakdown.items() if p > 60}
    if concentrated_sectors:
        for sector, pct in concentrated_sectors.items():
            st.warning("Concentration Warning: " + str(pct) + "% of this portfolio is in " + sector)

    if skipped:
        st.warning("Skipped tickers (data unavailable): " + ", ".join(skipped))

    if missing_data_notes:
        with st.expander("Missing Data Notes"):
            for note in missing_data_notes:
                st.write("- " + note)

    # --- Results: AI Analysis ---
    st.subheader("AI Analysis")
    with st.spinner("Generating analysis..."):
        prompt = (
            "You are a financial analyst explaining a retail investor portfolio. "
            "Be specific, direct, and avoid generic advice. "
            "Here is the portfolio data: "
            "Total Value: $" + "{:,.2f}".format(total) + ". "
            "Weighted Beta: " + str(weighted_beta) + ". "
            "Weighted Annualized Volatility: " + (str(weighted_volatility) + "%" if weighted_volatility else "unavailable") + ". "
            "Sector Breakdown: " + str(sector_breakdown) + ". "
            "Holdings: " + str(df[["Ticker", "Allocation %", "Beta", "Volatility %"]].to_dict(orient="records")) + ". "
            "Missing data: " + (", ".join(missing_data_notes) if missing_data_notes else "none") + ". "
            "Explain in 4-5 sentences: "
            "1. What this portfolio is concentrated in and what that means. "
            "2. What the beta and volatility together tell us about the risk level. "
            "3. How the risk is distributed across holdings - which positions are driving it. "
            "4. One specific thing this investor should be aware of or watch out for."
        )

        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            st.write(message.content[0].text)
        except Exception as e:
            st.error("AI analysis failed: " + str(e))


# --- Run ---
if analyze:
    if not holdings:
        st.error("Please enter at least one valid holding.")
    elif not api_key:
        st.error("Please enter your Anthropic API key.")
    else:
        analyze_portfolio(holdings, api_key)
