# 診斷、預警與視覺化工具 (`etf_tranad_project/src/utils/`)

本套件提供異常歸因分析、早期預警（EWS）訊號生成與實驗圖表繪製函式：

| 檔案 | 主要類別／函式 | 說明 |
| :--- | :--- | :--- |
| `__init__.py` | — | 套件初始化檔 |
| [`attribution.py`](attribution.py) | `compute_scores_and_attribution`、`diagnose_recent_anomaly` | 計算每日重建誤差異常分數與各維度（各 ETF / 特徵）歸因貢獻比例並輸出近期診斷 |
| [`early_warning.py`](early_warning.py) | `EarlyWarningSystem` | 整合異常速度（Velocity）、加速度（Acceleration）、非對稱波動率與共振度計算三層預警訊號 |
| [`visualization.py`](visualization.py) | `plot_etf_anomaly_trend`、`plot_early_warning_dashboard` | 繪製多 ETF 異常分數趨勢對照圖與 EWS 綜合預警儀表板 |
