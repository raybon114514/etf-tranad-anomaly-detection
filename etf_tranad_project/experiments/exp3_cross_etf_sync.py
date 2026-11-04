import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import fetch_multi_etf_data
from src.models.tranad import TranAD
from src.utils.attribution import compute_scores_and_attribution

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_WINDOW = 30

ETF_9_LIST = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP"]


def main():
    returns, _, train_mask, val_mask, test_mask, tickers = fetch_multi_etf_data(tickers=ETF_9_LIST)
    features = returns.values
    feats_dim = features.shape[1]

    train_data = features[train_mask]
    mean, std = np.mean(train_data, axis=0), np.std(train_data, axis=0) + 1e-8
    features_norm = (features - mean) / std

    windows, target_dates = [], []
    for i in range(len(features_norm) - K_WINDOW + 1):
        windows.append(features_norm[i : i + K_WINDOW])
        target_dates.append(returns.index[i + K_WINDOW - 1])

    windows = np.array(windows, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)

    win_train = windows[train_mask.values[K_WINDOW - 1 :]]
    win_test = windows[test_mask.values[K_WINDOW - 1 :]]
    test_dates = target_dates[test_mask.values[K_WINDOW - 1 :]]

    train_loader = DataLoader(win_train, batch_size=64, shuffle=True)
    model = TranAD(feats=feats_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, 41):
        w_factor = 1.0 / epoch
        for w in train_loader:
            w = w.to(DEVICE)
            o1, o2 = model(w)
            focus = torch.abs(o1 - w)
            _, o2_hat = model(w, focus)

            loss1 = w_factor * nn.functional.mse_loss(o1, w) + (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)
            loss2 = w_factor * nn.functional.mse_loss(o2, w) - (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)

            optimizer.zero_grad()
            (loss1 + loss2).backward()
            optimizer.step()

    _, train_attr = compute_scores_and_attribution(model, win_train, DEVICE)
    per_etf_thresh = np.percentile(train_attr, 95, axis=0)

    test_scores, test_attr = compute_scores_and_attribution(model, win_test, DEVICE)
    sync_counts = np.sum(test_attr > per_etf_thresh, axis=1)

    sync_df = pd.DataFrame({
        "Date": test_dates,
        "Total_Anomaly_Score": test_scores,
        "Sync_ETF_Count": sync_counts,
        "Is_Cross_ETF_Event": sync_counts >= 6,
    })
    print("\n[跨 ETF 同步異常事件統計 (9 檔中 >= 6 檔同時超標)]")
    print(f"總計發生天數: {sync_df['Is_Cross_ETF_Event'].sum()} 天")
    print(sync_df[sync_df["Is_Cross_ETF_Event"]].tail(10))


if __name__ == "__main__":
    main()
