import numpy as np
import pandas as pd
import yfinance as yf


def fetch_etf_microstructure_features(start="2010-01-01", end="2026-03-31"):
    """獲取 ETF、基準指數與主要成分股數據，建構跨資產微結構特徵"""
    tickers = ["QQQ", "SPY", "AAPL"]
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    close = df["Close"].dropna()
    volume = df["Volume"]["QQQ"].dropna()
    returns = np.log(close / close.shift(1)).dropna()

    features = pd.DataFrame(index=returns.index)
    features["QQQ_Return"] = returns["QQQ"]

    # 特徵 A：結構相關性解耦 (Tracking Deviation)
    features["Bench_Spread"] = returns["QQQ"] - returns["SPY"]
    features["Comp_Spread"] = returns["QQQ"] - returns["AAPL"]

    # 特徵 B：市場微結構衝擊 (Volume Shock)
    vol_ma20 = volume.rolling(window=20).mean()
    features["Volume_Shock"] = (volume / vol_ma20).loc[features.index]

    # 特徵 C：波動率聚集群 (20日滾動標準差)
    features["Volatility"] = returns["QQQ"].rolling(window=20).std()

    features = features.dropna()

    train_mask = features.index <= "2018-12-31"
    val_mask = (features.index > "2018-12-31") & (features.index <= "2021-12-31")
    test_mask = features.index > "2021-12-31"

    return features.values, train_mask, val_mask, test_mask, features.index, features.columns
