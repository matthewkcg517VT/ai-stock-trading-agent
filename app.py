import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import load_stock_data, download_stock_data
from src.features import build_features, FEATURE_COLS
from src.model import split_data, train_and_evaluate
from src.backtest import run_backtest, compute_metrics

st.set_page_config(page_title="AI Stock Trading Agent", layout="wide", page_icon="📈")

st.title("📈 AI Stock Trading Agent")
st.markdown("ML-powered paper trading system with backtesting | Built with scikit-learn + Streamlit")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.selectbox("Stock Ticker", ["SPY", "AAPL", "MSFT", "NVDA", "QQQ"], index=0)
train_ratio = st.sidebar.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05)
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.05) / 100
model_choice = st.sidebar.selectbox("Model", ["LogisticRegression", "RandomForest", "GradientBoosting"])

run_btn = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Downloads historical OHLCV data")
st.sidebar.markdown("2. Engineers 12 technical features")
st.sidebar.markdown("3. Trains ML model on 80% of data")
st.sidebar.markdown("4. Backtests predictions on 20%")
st.sidebar.markdown("5. Computes Sharpe ratio & drawdown")

# --- Main ---
if run_btn:
    with st.spinner(f"Downloading {ticker} data..."):
        try:
            df_raw = load_stock_data(ticker)
        except FileNotFoundError:
            df_raw = download_stock_data(ticker)

    with st.spinner("Engineering features & training models..."):
        df = build_features(df_raw)
        X_train, X_test, y_train, y_test, df_test = split_data(df, FEATURE_COLS, train_ratio=train_ratio)
        results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    chosen = results[model_choice]
    preds = chosen["preds"]
    proba = chosen["proba"]

    with st.spinner("Running backtest..."):
        df_bt = run_backtest(df_test, preds, proba, transaction_cost=transaction_cost)
        metrics = compute_metrics(df_bt)

    # --- Metrics Row ---
    st.subheader("📊 Backtest Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy Return", metrics["Total Return (Strategy)"])
    c2.metric("Buy & Hold Return", metrics["Total Return (Buy&Hold)"])
    c3.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
    c4.metric("Max Drawdown", metrics["Max Drawdown"])
    c5.metric("Win Rate", metrics["Win Rate"])

    # --- Model Accuracy Row ---
    st.subheader("🤖 Model Comparison")
    acc_data = {name: f"{res['accuracy']:.2%}" for name, res in results.items()}
    acc_df = pd.DataFrame(acc_data.items(), columns=["Model", "Accuracy"])
    acc_df["Selected"] = acc_df["Model"].apply(lambda x: "✅" if x == model_choice else "")
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    # --- Equity Curve ---
    st.subheader("📈 Equity Curve")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(df_bt.index, df_bt["cum_market"], label="Buy & Hold", color="steelblue", linewidth=1.5)
    axes[0].plot(df_bt.index, df_bt["cum_strategy"], label=f"ML Strategy ({model_choice})", color="darkorange", linewidth=1.5)
    axes[0].set_title(f"{ticker} — ML Strategy vs Buy & Hold")
    axes[0].set_ylabel("Portfolio Value (starting 1.0)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rolling_max = df_bt["cum_strategy"].cummax()
    drawdown = (df_bt["cum_strategy"] / rolling_max - 1) * 100
    axes[1].fill_between(df_bt.index, drawdown, 0, color="crimson", alpha=0.4)
    axes[1].set_title("Strategy Drawdown (%)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Recent Predictions ---
    st.subheader("🔍 Recent Predictions")
    display_cols = ["Close", "prediction", "proba", "market_return", "strategy_return"]
    show = df_bt[display_cols].tail(20).copy()
    show["prediction"] = show["prediction"].map({1.0: "📈 Up", 0.0: "📉 Down", None: "—"})
    show["proba"] = show["proba"].apply(lambda x: f"{x:.1%}")
    show["market_return"] = show["market_return"].apply(lambda x: f"{x:.2%}")
    show["strategy_return"] = show["strategy_return"].apply(lambda x: f"{x:.2%}")
    st.dataframe(show, use_container_width=True)

else:
    st.info("👈 Configure settings in the sidebar and click **Run Backtest** to start.")
    st.image("data/backtest_results.png", caption="Last backtest results (SPY, Logistic Regression)")
