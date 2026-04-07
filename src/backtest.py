import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import load_stock_data
from src.features import build_features, FEATURE_COLS
from src.model import split_data, train_and_evaluate

def run_backtest(df_test, preds, proba, transaction_cost=0.001):
    df = df_test.copy()
    df["prediction"] = preds
    df["proba"] = proba

    # Market return each day
    df["market_return"] = df["Close"].pct_change()

    # Strategy: hold when prediction=1, cash when prediction=0
    # Shift by 1 so we act on NEXT day (no lookahead leakage)
    df["position"] = df["prediction"].shift(1)

    # Transaction cost whenever position changes
    df["position_change"] = df["position"].diff().abs().fillna(0)

    # Strategy return
    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - df["position_change"] * transaction_cost
    )

    # Cumulative returns
    df["cum_market"]   = (1 + df["market_return"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df

def compute_metrics(df):
    sr = df["strategy_return"].dropna()
    mr = df["market_return"].dropna()

    sharpe = (sr.mean() / sr.std()) * np.sqrt(252) if sr.std() > 0 else 0

    rolling_max = df["cum_strategy"].cummax()
    drawdown    = df["cum_strategy"] / rolling_max - 1
    max_dd      = drawdown.min()

    total_ret   = df["cum_strategy"].iloc[-1] - 1
    market_ret  = df["cum_market"].iloc[-1] - 1
    win_rate    = (sr > 0).sum() / len(sr)

    return {
        "Total Return (Strategy)": f"{total_ret:.2%}",
        "Total Return (Buy&Hold)": f"{market_ret:.2%}",
        "Sharpe Ratio":            f"{sharpe:.3f}",
        "Max Drawdown":            f"{max_dd:.2%}",
        "Win Rate":                f"{win_rate:.2%}",
        "Trading Days":            len(sr),
    }

def plot_results(df, save_path="data/backtest_results.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    axes[0].plot(df.index, df["cum_market"],   label="Buy & Hold", color="steelblue", linewidth=1.5)
    axes[0].plot(df.index, df["cum_strategy"], label="ML Strategy", color="darkorange", linewidth=1.5)
    axes[0].set_title("ML Strategy vs Buy & Hold — Cumulative Returns", fontsize=13)
    axes[0].set_ylabel("Portfolio Value (starting at 1.0)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Drawdown
    rolling_max = df["cum_strategy"].cummax()
    drawdown    = (df["cum_strategy"] / rolling_max - 1) * 100
    axes[1].fill_between(df.index, drawdown, 0, color="crimson", alpha=0.4)
    axes[1].set_title("Strategy Drawdown (%)", fontsize=13)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Chart saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    df_raw = load_stock_data("SPY")
    df     = build_features(df_raw)

    X_train, X_test, y_train, y_test, df_test = split_data(df, FEATURE_COLS)
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    best   = results[best_name]
    preds  = best["preds"]
    proba  = best["proba"]

    print(f"\nRunning backtest with: {best_name}")
    df_bt = run_backtest(df_test, preds, proba)

    metrics = compute_metrics(df_bt)
    print("\n=== Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_results(df_bt)
