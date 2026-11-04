import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yfinance as yf

from src.features.garch_volatility import GARCHModels
from src.models.tranad import TranAD
from src.utils.attribution import compute_scores_and_attribution

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_eval_tranad(features, train_mask, val_mask, test_mask, k_window=30):
    train_data = features[train_mask]
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8
    features_norm = (features - mean) / std

    windows, target_dates_idx = [], []
    for i in range(len(features_norm) - k_window + 1):
        windows.append(features_norm[i : i + k_window])
        target_dates_idx.append(i + k_window - 1)

    windows = np.array(windows, dtype=np.float32)
    target_dates_idx = np.array(target_dates_idx)

    win_train = windows[train_mask[target_dates_idx]]
    win_val = windows[val_mask[target_dates_idx]]
    win_test = windows[test_mask[target_dates_idx]]

    train_loader = DataLoader(win_train, batch_size=64, shuffle=True)
    model = TranAD(feats=features.shape[1], d_model=64, nhead=8).to(DEVICE)
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

    val_scores, _ = compute_scores_and_attribution(model, win_val, DEVICE)
    test_scores, _ = compute_scores_and_attribution(model, win_test, DEVICE)

    threshold = np.percentile(val_scores, 95)
    anomalies = test_scores > threshold
    return anomalies, test_scores


def main():
    print("正在下載 S&P 500 (^GSPC) 歷史資料驗證 GARCH vs GJR-GARCH 穩健性...")
    df = yf.download("^GSPC", start="1980-01-01", end="2026-04-01", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna().copy()

    train_mask = (df.index <= "2007-09-18").values
    val_mask = ((df.index > "2007-09-18") & (df.index <= "2016-12-19")).values
    test_mask = (df.index > "2016-12-19").values
    returns = df["return"].values

    garch_params = GARCHModels.fit_garch11(returns[train_mask])
    gjr_params = GARCHModels.fit_gjr_garch11(returns[train_mask])

    sigma_garch, z_garch = GARCHModels.extract_features(returns, "GARCH", garch_params)
    sigma_gjr, z_gjr = GARCHModels.extract_features(returns, "GJR", gjr_params)

    feat_garch = np.column_stack([returns, sigma_garch, z_garch])
    feat_gjr = np.column_stack([returns, sigma_gjr, z_gjr])

    anom_garch, _ = train_and_eval_tranad(feat_garch, train_mask, val_mask, test_mask)
    anom_gjr, _ = train_and_eval_tranad(feat_gjr, train_mask, val_mask, test_mask)

    intersection = np.sum(anom_garch & anom_gjr)
    union = np.sum(anom_garch | anom_gjr)
    jaccard = intersection / union if union > 0 else 1.0

    print(f"Jaccard 相似係數 (Jaccard Index): {jaccard:.4f}")


if __name__ == "__main__":
    main()
