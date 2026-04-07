import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker: str, start: str = "2015-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker and save to CSV."""
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{ticker.lower()}.csv"
    df.to_csv(filepath)
    print(f"Saved {len(df)} rows to {filepath}")
    return df

def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load previously downloaded stock data from CSV."""
    filepath = f"data/{ticker.lower()}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No data found for {ticker}. Run download_stock_data() first.")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

if __name__ == "__main__":
    df = download_stock_data("SPY")
    print(df.head())
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
