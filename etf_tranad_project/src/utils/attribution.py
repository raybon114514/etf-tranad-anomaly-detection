import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_scores_and_attribution(model, win_array, device, batch_size=64):
    """計算視窗總異常分數 (Frobenius Norm MSE) 與觸發日各板塊誤差歸因 (XAI)"""
    model.eval()
    scores, attributions = [], []
    loader = DataLoader(win_array, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for w in loader:
            w = w.to(device)
            o1, _ = model(w)
            _, o2_hat = model(w, torch.abs(o1 - w))

            err1 = torch.sum((o1 - w) ** 2, dim=(1, 2))
            err2 = torch.sum((o2_hat - w) ** 2, dim=(1, 2))
            scores.extend((0.5 * err1 + 0.5 * err2).cpu().numpy())

            feat_err = (o2_hat[:, -1, :] - w[:, -1, :]) ** 2
            attributions.extend(feat_err.cpu().numpy())

    return np.array(scores), np.array(attributions)


def diagnose_recent_anomaly(test_dates, test_scores, test_attr, anomalies, tickers):
    """針對最近一次異常事件輸出跨板塊貢獻度與風險型態診斷"""
    if np.sum(anomalies) == 0:
        print("測試期間未偵測到超過門檻之異常日。")
        return

    anomaly_indices = np.where(anomalies)[0]
    recent_idx = anomaly_indices[-1]
    recent_date = test_dates[recent_idx].strftime("%Y-%m-%d")

    attr = test_attr[recent_idx]
    attr_pct = (attr / (attr.sum() + 1e-8)) * 100

    print("\n[近期異常事件歸因分析 (XAI)]")
    print(f"發生日期: {recent_date} | 總異常分數: {test_scores[recent_idx]:.2f}")
    print("板塊崩潰貢獻度:")
    for t, pct in zip(tickers, attr_pct):
        print(f" - {t}: {pct:>5.1f}%")

    max_ticker = tickers[np.argmax(attr_pct)]
    if max_ticker == "SPY":
        print("診斷結論: 屬於【宏觀系統性風險】，大盤 (SPY) 發生超預期脫鉤。")
    else:
        print(f"診斷結論: 屬於【板塊結構性破裂】，衝擊主要由 {max_ticker} 外溢，破壞了市場定價慣性。")
