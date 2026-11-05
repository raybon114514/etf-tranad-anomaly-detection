import sys
import os
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import fetch_multi_etf_data
from src.models.tranad import TranAD
from src.utils.attribution import compute_scores_and_attribution, diagnose_recent_anomaly
from src.utils.visualization import plot_etf_anomaly_trend

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_WINDOW = 30


def main():
    returns, close_prices, train_mask, val_mask, test_mask, tickers = fetch_multi_etf_data()
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

    win_train = windows[np.asarray(train_mask)[K_WINDOW - 1 :]]
    win_test = windows[np.asarray(test_mask)[K_WINDOW - 1 :]]
    test_dates = target_dates[np.asarray(test_mask)[K_WINDOW - 1 :]]

    train_loader = DataLoader(win_train, batch_size=64, shuffle=True)
    model = TranAD(feats=feats_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n[啟動雙階段對抗訓練]")
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

    train_scores, _ = compute_scores_and_attribution(model, win_train, DEVICE)
    threshold = np.percentile(train_scores, 98)

    test_scores, test_attr = compute_scores_and_attribution(model, win_test, DEVICE)
    anomalies = test_scores > threshold

    print(f"-> 常態門檻值 (Train 98th): {threshold:.2f}")
    print(f"-> 測試集異常天數: {np.sum(anomalies)} / {len(test_scores)}")

    diagnose_recent_anomaly(test_dates, test_scores, test_attr, anomalies, tickers)
    plot_etf_anomaly_trend(test_dates, close_prices, test_scores, threshold, anomalies)

    results_df = pd.DataFrame({
        "Date": test_dates.strftime("%Y-%m-%d"),
        "AnomalyScore": test_scores,
        "IsAnomaly": anomalies,
    })
    results_path = "data/processed/tranad_anomaly_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"-> 測試集推論結果已儲存至: {results_path} (涵蓋至 {results_df['Date'].iloc[-1]})")


if __name__ == "__main__":
    main()
