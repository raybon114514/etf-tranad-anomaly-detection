import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_etf_anomaly_trend(
    test_dates,
    close_prices,
    test_scores,
    threshold,
    anomalies,
    save_path="results/figures/etf_anomaly_trend.png",
):
    """繪製 ETF 走勢與 Transformer 重構誤差異常觸發對照圖"""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    if "QQQ" in close_prices.columns:
        plt.plot(test_dates, close_prices["QQQ"].loc[test_dates], label="QQQ (科技板塊)", color="navy", linewidth=1.5)
    if "SPY" in close_prices.columns:
        plt.plot(test_dates, close_prices["SPY"].loc[test_dates], label="SPY (大盤基準)", color="orange", linewidth=1.5)

    anomaly_dates = test_dates[anomalies]
    for d in anomaly_dates:
        plt.axvline(d, color="red", alpha=0.3, linestyle="--", linewidth=1)

    plt.title("ETF Prices and Detected Anomaly Events (2022-2026)", fontsize=14, fontweight="bold")
    plt.ylabel("Price (USD)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(test_dates, test_scores, label="Transformer Anomaly Score (MSE)", color="#1f77b4", alpha=0.8)
    plt.axhline(threshold, linestyle="-", color="crimson", linewidth=2, label=f"Threshold (Train 98th %ile: {threshold:.2f})")

    plt.scatter(
        test_dates[anomalies],
        test_scores[anomalies],
        color="red",
        s=40,
        zorder=5,
        label="Confirmed Anomaly",
    )
    plt.title("Reconstruction Error vs. Normalcy Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"視覺化圖表已儲存至: {save_path}")


def plot_early_warning_dashboard(
    df_signals,
    close_prices,
    t_watch,
    t_crit,
    save_path="results/figures/early_warning_dashboard.png",
):
    """繪製早期預警儀表板 (走勢分級標記、雙門檻、速度與加速度動量)"""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    dates = pd.to_datetime(df_signals["Date"])
    scores = df_signals["Score"].values
    velos = df_signals["Velocity"].values
    levels = df_signals["Level"].values

    is_watch = (levels == "ONSET_WATCH") | (levels == "ELEVATED_RISK")
    is_crit = levels == "CRITICAL"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # --- 1. ETF 走勢圖與分級預警底色區間 ---
    if "QQQ" in close_prices.columns:
        ax1.plot(dates, close_prices["QQQ"].loc[dates], label="QQQ (科技板塊)", color="#1e3a8a", linewidth=1.5)
    if "SPY" in close_prices.columns:
        ax1.plot(dates, close_prices["SPY"].loc[dates], label="SPY (大盤基準)", color="#ea580c", linewidth=1.5)

    # 標記起頭預警區間 (黃色) 與臨界爆發 (紅色)
    for d in dates[is_watch]:
        ax1.axvline(d, color="#eab308", alpha=0.25, linestyle="-", linewidth=1.5)
    for d in dates[is_crit]:
        ax1.axvline(d, color="#dc2626", alpha=0.25, linestyle="-", linewidth=1.5)

    ax1.set_title("ETF Prices with Multi-Tier Early Warning Highlights (2022-2026)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- 2. 雙重門檻與異常分數 ---
    ax2.plot(dates, scores, color="#2563eb", linewidth=1.3, label="TranAD Anomaly Score")
    ax2.axhline(t_watch, linestyle="--", color="#ca8a04", linewidth=1.8, label=f"Onset Watch Threshold ({t_watch:.2f})")
    ax2.axhline(t_crit, linestyle="-", color="#dc2626", linewidth=2.0, label=f"Critical Threshold ({t_crit:.2f})")

    # 突顯起頭提示點 (黃色散點)
    if np.sum(df_signals["IsEarlyOnset"]) > 0:
        early_dates = dates[df_signals["IsEarlyOnset"].values]
        early_scores = scores[df_signals["IsEarlyOnset"].values]
        ax2.scatter(early_dates, early_scores, color="#ca8a04", s=35, zorder=5, label="Early Onset Lead Signal (黃燈起頭)")

    ax2.set_title("Multi-Tier Thresholds: Catching Anomalies at Early Onset", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Anomaly Score (MSE)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # --- 3. 異常分數速度 (Velocity 一階差分動量) ---
    ax3.bar(dates, velos, color=np.where(velos >= 0, "#ef4444", "#10b981"), alpha=0.75, width=1.5, label="Score Velocity (ΔScore/day)")
    ax3.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax3.set_title("Score Velocity Momentum (Rate of Structural Breakdown)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Velocity")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"早期預警儀表板圖表已儲存至: {save_path}")
