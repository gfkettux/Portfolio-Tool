import yfinance as yf

def get_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="1y")

    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice"),
        "sector": info.get("sector"),
        "beta": info.get("beta"),
        "history": history
    }

data = get_ticker_data("AAPL")
print("Ticker:", data["ticker"])
print("Price:", data["current_price"])
print("Sector:", data["sector"])
print("Beta:", data["beta"])
print("History rows:", len(data["history"]))
