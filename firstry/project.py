import yfinance as yf
import pandas as pd
import numpy as np

def fetch_etf_multivariate_data():
    """專題用：獲取 ETF、基準指數與主要成分股數據，建構多維特徵"""
    print("正在下載 ETF 與相關資產歷史資料...")
    # 選擇標的：QQQ (納斯達克ETF), SPY (大盤基準), AAPL (最大成分股)
    tickers = ["QQQ", "SPY", "AAPL"]
    df = yf.download(tickers, start="2010-01-01", auto_adjust=False)
    
    # 提取收盤價與成交量
    close = df["Close"].dropna()
    volume = df["Volume"]["QQQ"].dropna()
    
    # 1. 計算日報酬率
    returns = np.log(close / close.shift(1)).dropna()
    
    # 2. 特徵工程構造 (ETF 專屬微結構特徵)
    features = pd.DataFrame(index=returns.index)
    features["QQQ_Return"] = returns["QQQ"]
    
    # 特徵 A：結構相關性解耦 (Tracking Deviation)
    features["Bench_Spread"] = returns["QQQ"] - returns["SPY"]
    features["Comp_Spread"] = returns["QQQ"] - returns["AAPL"]
    
    # 特徵 B：市場微結構衝擊 (Volume Shock)
    # 使用 20 日移動平均量作為基準，計算當日爆量倍數
    vol_ma20 = volume.rolling(window=20).mean()
    features["Volume_Shock"] = (volume / vol_ma20).loc[features.index]
    
    # 特徵 C：波動率聚集群 (20日滾動標準差)
    features["Volatility"] = returns["QQQ"].rolling(window=20).std()
    
    # 移除 NaN
    features = features.dropna()
    
    # 3. 專題敏捷規劃的資料切分 (避免前視偏差)
    # 假設 2010-2018 訓練, 2019-2021 驗證/門檻, 2022-2026 測試(包含2022通膨跌與2024股災)
    train_mask = features.index <= "2018-12-31"
    val_mask = (features.index > "2018-12-31") & (features.index <= "2021-12-31")
    test_mask = features.index > "2021-12-31"
    
    print(f"特徵矩陣維度: {features.shape[1]} 維")
    print(f"資料筆數 - 訓練: {train_mask.sum()}, 驗證: {val_mask.sum()}, 測試: {test_mask.sum()}")
    
    return features.values, train_mask, val_mask, test_mask, features.index, features.columns

# 後續串接方式：
# feat_matrix, train_mask, val_mask, test_mask, dates, cols = fetch_etf_multivariate_data()
# model = TranAD(feats=5, d_model=64, nhead=8).to(DEVICE)  # 記得把 feats 改成 5
# anomalies, scores = train_and_eval_tranad(feat_matrix, train_mask, val_mask, test_mask)