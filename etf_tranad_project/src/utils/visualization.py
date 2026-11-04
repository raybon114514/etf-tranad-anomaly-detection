import matplotlib.pyplot as plt


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
    plt.savefig(save_path, dpi=300)
    print(f"視覺化圖表已儲存至: {save_path}")
