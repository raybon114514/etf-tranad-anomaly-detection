import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf


class WindowDataset(Dataset):
    """滑動時間視窗 Dataset 封裝"""
    def __init__(self, data, window_size, stride=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.indices = np.arange(0, len(data) - window_size + 1, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        return self.data[start : start + self.window_size]


def fetch_multi_etf_data(
    tickers=None,
    start="2010-01-01",
    end="2026-03-31",
    train_end="2018-12-31",
    val_end="2021-12-31",
):
    """下載大盤與板塊 ETF 數據，並依時間切分訓練/驗證/測試集（防範前視偏差）"""
    if tickers is None:
        tickers = ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLI"]

    print(f"下載多檔 ETF 歷史數據: {tickers} ...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df["Close"]
    else:
        close_prices = df

    close_prices = close_prices[tickers].dropna()
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    close_prices = close_prices.loc[returns.index]

    train_mask = returns.index <= train_end
    val_mask = (returns.index > train_end) & (returns.index <= val_end)
    test_mask = returns.index > val_end

    return returns, close_prices, train_mask, val_mask, test_mask, tickers
