import os
import sys

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
from src.utils.attribution import compute_scores_and_attribution
from src.utils.early_warning import EarlyWarningSystem
from src.utils.visualization import plot_early_warning_dashboard

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_WINDOW = 30


def main():
    print("=" * 65)
    print("啟動實驗 4：TranAD 跨 ETF 早期預警系統 (Early Warning System)")
    print("=" * 65)

    # 1. 載入動態最新資料
    returns, close_prices, train_mask, val_mask, test_mask, tickers = fetch_multi_etf_data()
    features = returns.values
    feats_dim = features.shape[1]

    # 2. 標準化 (僅使用訓練集統計量，杜絕前視偏差)
    train_data = features[train_mask]
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8
    features_norm = (features - mean) / std

    # 3. 視窗切分
    windows, target_dates = [], []
    for i in range(len(features_norm) - K_WINDOW + 1):
        windows.append(features_norm[i : i + K_WINDOW])
        target_dates.append(returns.index[i + K_WINDOW - 1])

    windows = np.array(windows, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)

    win_train = windows[np.asarray(train_mask)[K_WINDOW - 1 :]]
    win_test = windows[np.asarray(test_mask)[K_WINDOW - 1 :]]
    test_dates = target_dates[np.asarray(test_mask)[K_WINDOW - 1 :]]

    # 4. 模型訓練 (雙階段對抗訓練)
    train_loader = DataLoader(win_train, batch_size=64, shuffle=True)
    model = TranAD(feats=feats_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n[模型訓練中 - 40 Epochs 對抗學習]")
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

    # 5. 推論與建立早期預警基準
    print("\n[評估訓練集基準與建構預警門檻]")
    train_scores, _ = compute_scores_and_attribution(model, win_train, DEVICE)
    test_scores, test_attr = compute_scores_and_attribution(model, win_test, DEVICE)

    # 實例化早期預警系統 (90th 警戒門檻, 98th 臨界門檻)
    ews = EarlyWarningSystem(train_scores, watch_pct=90.0, critical_pct=98.0)
    print(f"-> 基準計算完成:")
    print(f"   🟢 安全常態區間: Score < {ews.t_watch:.2f}")
    print(f"   🟡 起頭警戒門檻: {ews.t_watch:.2f} <= Score < {ews.t_crit:.2f}")
    print(f"   🔴 臨界爆發門檻: Score >= {ews.t_crit:.2f}")

    # 6. 對盲測集進行分級預警
    df_signals = ews.evaluate_sequence(test_dates, test_scores, test_attr, tickers)

    # 7. 匯出預警成果 CSV
    csv_path = "data/processed/early_warning_signals.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_signals.to_csv(csv_path, index=False)
    print(f"\n-> 每日早期預警詳細紀錄已儲存至: {csv_path}")

    # 8. 產出視覺化儀表板
    fig_path = "results/figures/early_warning_dashboard.png"
    plot_early_warning_dashboard(df_signals, close_prices, ews.t_watch, ews.t_crit, save_path=fig_path)

    # 9. 印出最新狀態報告
    report_text = ews.generate_current_alert_report(df_signals)
    print("\n" + report_text)

    # 10. 展示「不對勁起頭」案例回測 (例如最近 10 個交易日的轉折推演)
    print("\n[近 10 個交易日之動態預警與起頭捕捉演變]")
    recent_tail = df_signals.tail(10)[["Date", "Score", "Velocity", "Level", "IsEarlyOnset", "TopContributor", "TopContribPct"]]
    print(recent_tail.to_string(index=False))

    # 統計盲測集表現
    level_counts = df_signals["Level"].value_counts().to_dict()
    print("\n[盲測集 (2022-2026) 預警分佈總結]")
    for lvl, cnt in level_counts.items():
        pct = (cnt / len(df_signals)) * 100
        print(f" - {lvl:<15}: {cnt:>4} 天 ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
