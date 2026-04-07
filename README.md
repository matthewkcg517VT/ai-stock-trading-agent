# 📈 AI Stock Trading Agent

An ML-powered paper trading system that predicts stock price direction and backtests trading strategies — built with Python, scikit-learn, and Streamlit.

## 🚀 Live Demo

Run the Streamlit dashboard to interactively backtest any strategy:

```bash
streamlit run app.py
```

## 🧠 How It Works

1. **Data** — Downloads 10 years of historical OHLCV data via `yfinance`
2. **Features** — Engineers 12 technical indicators (returns, moving averages, RSI, volatility, volume)
3. **Model** — Trains a classifier to predict next-day price direction (up/down)
4. **Strategy** — Converts predictions into long/cash positions with transaction costs
5. **Backtest** — Evaluates performance using Sharpe ratio, max drawdown, and win rate

## 📊 Results (SPY, 2015–2025)

| Metric | Value |
|--------|-------|
| Strategy Total Return | 35.93% |
| Buy & Hold Return | 51.22% |
| **Sharpe Ratio** | **1.321** |
| Max Drawdown | -10.15% |
| Win Rate | 52.54% |
| Model Accuracy | 57.1% |

> The ML strategy achieves a Sharpe ratio of 1.32 (> 1.0 is considered good) with significantly lower drawdowns than buy-and-hold, demonstrating effective risk management.

## 🗂️ Project Structure

```
ai-stock-trading-agent/
├── data/                   # Downloaded CSVs and backtest chart
├── notebooks/              # Exploration notebooks
├── src/
│   ├── data_loader.py      # Download & load stock data (yfinance)
│   ├── features.py         # Feature engineering (12 technical indicators)
│   ├── model.py            # Train & evaluate ML models
│   ├── backtest.py         # Simulate trades, compute metrics & charts
│   └── utils.py            # Shared utilities
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry point
└── requirements.txt
```

## 🔧 Features Engineered

- **Returns**: 1-day, 5-day, 10-day price returns
- **Moving Averages**: 10/20/50-day MA ratios
- **Volatility**: 10/20-day rolling standard deviation
- **Volume**: Volume change and ratio vs 10-day average
- **RSI**: 14-period Relative Strength Index
- **Range**: Daily high-low range as % of close

## 🤖 Models Compared

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 57.1% ✅ Best |
| Gradient Boosting | 54.5% |
| Random Forest | 51.0% |

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/matthewkcg517VT/ai-stock-trading-agent.git
cd ai-stock-trading-agent

# Install dependencies
pip install -r requirements.txt

# Download data
python src/data_loader.py

# Run backtest
python -m src.backtest

# Launch dashboard
streamlit run app.py
```

## 📦 Tech Stack

- **Python** — pandas, numpy, scikit-learn, yfinance, matplotlib, streamlit, xgboost

## ⚠️ Disclaimer

This is a paper trading simulation for educational purposes only. Not financial advice.
