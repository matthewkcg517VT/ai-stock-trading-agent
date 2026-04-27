import yfinance as yf, pandas as pd, numpy as np, requests
from datetime import datetime, timedelta
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

def get_vix_data(start='2015-01-01', end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    v = yf.download('^VIX', start=start, end=end, auto_adjust=True, progress=False)
    v.columns = v.columns.get_level_values(0)
    return v['Close'].rename('vix')

def add_vix_features(df, vix):
    df = df.copy()
    df['vix'] = vix.reindex(df.index).ffill()
    df['vix_change'] = df['vix'].pct_change()
    df['vix_ma10'] = df['vix'].rolling(10).mean()
    df['vix_ratio'] = df['vix'] / df['vix_ma10']
    df['high_fear'] = (df['vix'] > 25).astype(int)
    return df

def get_news_sentiment_today(api_key=None):
    if not VADER_AVAILABLE or not api_key:
        return 0.0
    try:
        yd = (datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')
        url = ('https://newsapi.org/v2/everything?q=stock+market+S%26P500'
               f'&from={yd}&language=en&sortBy=relevancy&pageSize=20&apiKey={api_key}')
        arts = requests.get(url, timeout=8).json().get('articles', [])
        a = SentimentIntensityAnalyzer()
        scores = [a.polarity_scores((x.get('title') or '')+' '+(x.get('description') or ''))['compound'] for x in arts]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_historical_sentiment(df_index, api_key=None):
    vix = get_vix_data(
        start=str(df_index[0].date()),
        end=str((df_index[-1]+timedelta(days=2)).date())
    ).reindex(df_index).ffill()
    sent = np.clip(-0.05*(vix-20), -1, 1).rename('sentiment')
    if api_key:
        sc = get_news_sentiment_today(api_key)
        if df_index[-1].date() >= (datetime.today()-timedelta(days=1)).date():
            sent.iloc[-1] = sc
    return sent

def get_todays_live_features(ticker='SPY', newsapi_key=None):
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today()-timedelta(days=150)).strftime('%Y-%m-%d')
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0)
    from src.features import build_features
    df = build_features(raw)
    vix = get_vix_data(start=start, end=end)
    df = add_vix_features(df, vix)
    sent = get_historical_sentiment(df.index, api_key=newsapi_key)
    df['sentiment'] = sent.reindex(df.index).ffill().fillna(0)
    return df

LIVE_FEATURE_COLS = [
    'return_1d','return_5d','return_10d',
    'ma_ratio_10','ma_ratio_20','ma_ratio_50',
    'volatility_10','volatility_20',
    'volume_change','volume_ratio',
    'rsi_14','hl_range',
    'vix','vix_change','vix_ratio','high_fear','sentiment',
]
