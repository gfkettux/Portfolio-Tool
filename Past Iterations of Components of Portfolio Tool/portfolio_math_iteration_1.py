import yfinance as yf
import pandas as pd
import anthropic

def get_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice"),
        "sector": info.get("sector", "Unknown"),
        "beta": info.get("beta", 1.0)
    }

def analyze_portfolio(holdings):
    rows = []
    for ticker, amount in holdings.items():
        data = get_ticker_data(ticker)
        rows.append({
            "Ticker": ticker,
            "Amount Invested": amount,
            "Sector": data["sector"],
            "Beta": data["beta"]
        })

    df = pd.DataFrame(rows)
    total = df["Amount Invested"].sum()
    df["Allocation %"] = (df["Amount Invested"] / total * 100).round(2)
    df["Weighted Beta"] = df["Beta"] * (df["Amount Invested"] / total)
    weighted_beta = df["Weighted Beta"].sum().round(3)

    print("--- Portfolio Breakdown ---")
    print(df[["Ticker", "Amount Invested", "Allocation %", "Sector", "Beta"]].to_string(index=False))
    print("Total Portfolio Value: $" + "{:,.2f}".format(total))
    print("Weighted Portfolio Beta: " + str(weighted_beta))

    sector_breakdown = df.groupby("Sector")["Allocation %"].sum().to_dict()

    prompt = (
        "You are a financial analyst explaining a retail investor portfolio. "
        "Be specific, direct, and avoid generic advice. "
        "Here is the portfolio data: "
        "Total Value: $" + "{:,.2f}".format(total) + ". "
        "Weighted Beta: " + str(weighted_beta) + ". "
        "Sector Breakdown: " + str(sector_breakdown) + ". "
        "Holdings: " + str(df[["Ticker", "Allocation %", "Beta"]].to_dict(orient="records")) + ". "
        "Explain in 3-4 sentences: "
        "1. What this portfolio is concentrated in and what that means. "
        "2. What the beta tells us about the risk level. "
        "3. One specific thing this investor should be aware of."
    )

    client = anthropic.Anthropic(api_key="YOUR_API_KEY_HERE")
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    print("--- AI Analysis ---")
    print(message.content[0].text)

holdings = {"AAPL": 2000, "TSLA": 1000, "SPY": 3000}
analyze_portfolio(holdings)
