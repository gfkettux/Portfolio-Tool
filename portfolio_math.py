import yfinance as yf
import pandas as pd
import anthropic

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
        print("Warning: Could not retrieve data for " + ticker + " - " + str(e))
        return {"ticker": ticker, "valid": False}


def analyze_portfolio(holdings):
    rows = []
    missing_data_notes = []
    skipped = []

    for ticker, amount in holdings.items():
        data = get_ticker_data(ticker)

        if not data["valid"]:
            skipped.append(ticker)
            continue

        if data["missing"]:
            missing_data_notes.append(ticker + " is missing: " + ", ".join(data["missing"]))

        rows.append({
            "Ticker": ticker,
            "Amount Invested": amount,
            "Sector": data["sector"],
            "Beta": data["beta"],
            "Volatility %": data["volatility"]
        })

    if not rows:
        print("No valid tickers found. Please check your inputs.")
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

    print("--- Portfolio Breakdown ---")
    print(df[["Ticker", "Amount Invested", "Allocation %", "Sector", "Beta", "Volatility %"]].to_string(index=False))
    print("Total Portfolio Value: $" + "{:,.2f}".format(total))
    print("Weighted Portfolio Beta: " + str(weighted_beta))
    if weighted_volatility is not None:
        print("Weighted Portfolio Volatility: " + str(weighted_volatility) + "%")
    print("Sector Breakdown: " + str(sector_breakdown))

    concentrated_sectors = {s: p for s, p in sector_breakdown.items() if p > 60}
    if concentrated_sectors:
        print("")
        for sector, pct in concentrated_sectors.items():
            print("Concentration Warning: " + str(pct) + "% of this portfolio is in " + sector)

    if skipped:
        print("Skipped tickers (data unavailable): " + ", ".join(skipped))

    if missing_data_notes:
        print("Missing data notes:")
        for note in missing_data_notes:
            print("  - " + note)

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

    client = anthropic.Anthropic(api_key="YOUR_API_KEY_HERE")
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    print("")
    print("--- AI Analysis ---")
    print(message.content[0].text)


holdings = {"AAPL": 2000, "TSLA": 1000, "SPY": 3000}
analyze_portfolio(holdings)
