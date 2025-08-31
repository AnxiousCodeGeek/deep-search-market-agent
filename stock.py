import os
import requests

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

companies = {
    "Nvidia": {"finnhub": "NVDA", "alpha": "NVDA"},
    "BYD": {"finnhub": "1211.HK", "alpha": "HKG:1211"},   # HK Exchange
    "Hyundai": {"finnhub": "005380.KQ", "alpha": "KRX:005380"}  # KOSDAQ
}

def fetch_finnhub(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def fetch_alpha(symbol):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

if __name__ == "__main__":
    for name, symbols in companies.items():
        print(f"\nFetching data for {name}")
        
        finnhub_data = fetch_finnhub(symbols["finnhub"])
        print("Finnhub:", finnhub_data)
        
        alpha_data = fetch_alpha(symbols["alpha"])
        print("Alpha Vantage:", alpha_data)


# import yfinance as yf

# def get_yfinance_data(symbol):
#     try:
#         ticker = yf.Ticker(symbol)
#         hist = ticker.history(period="5d")
        
#         if hist.empty:
#             return None
            
#         latest = hist.iloc[-1]
#         return {
#             "symbol": symbol,
#             "latest_date": latest.name.strftime("%Y-%m-%d"),
#             "open": float(latest["Open"]),
#             "high": float(latest["High"]),
#             "low": float(latest["Low"]),
#             "close": float(latest["Close"]),
#             "volume": int(latest["Volume"])
#         }
#     except Exception as e:
#         print(f"yFinance error for {symbol}: {e}")
#         return None

# # Test yfinance
# test_symbols = ["1211.HK", "005380.KS", "AAPL", "TSLA"]

# for symbol in test_symbols:
#     print(f"\n=== yFinance data for {symbol} ===")
#     data = get_yfinance_data(symbol)
#     if data:
#         print(data)
#     else:
#         print("No data from yFinance")