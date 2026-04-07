import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer ML features from raw OHLCV data."""
    df = df.copy()

    # Returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Moving averages
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    # Price relative to moving averages
    df["ma_ratio_10"] = df["Close"] / df["ma_10"]
    df["ma_ratio_20"] = df["Close"] / df["ma_20"]
    df["ma_ratio_50"] = df["Close"] / df["ma_50"]

    # Volatility
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # Volume
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_10"] = df["Volume"].rolling(10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_10"]

    # RSI
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    # High/Low range
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Target: 1 if tomorrow close > today close, else 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaN values
    df = df.dropna()

    return df

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d",
    "ma_ratio_10", "ma_ratio_20", "ma_ratio_50",
    "volatility_10", "volatility_20",
    "volume_change", "volume_ratio",
    "rsi_14", "hl_range"
]

if __name__ == "__main__":
    from src.data_loader import load_stock_data
    df_raw = load_stock_data("SPY")
    df = build_features(df_raw)
    print(f"Features built: {len(df)} rows, {len(FEATURE_COLS)} features")
    print(df[FEATURE_COLS + ["target"]].tail())
    print(f"\nTarget balance: {df['target'].value_counts().to_dict()}")
