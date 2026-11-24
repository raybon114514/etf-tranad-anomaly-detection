# 已處理實驗結果資料表 (`etf_tranad_project/data/processed/`)

本目錄存放由 `experiments/` 各實驗腳本輸出之中介與最終結構化 CSV 結果：

| 檔名 | 來源腳本 | 說明 |
| :--- | :--- | :--- |
| [`early_warning_signals.csv`](early_warning_signals.csv) | `experiments/exp4_early_warning_system.py` | 每日早期預警系統（EWS）綜合分數、速度／加速度動量與預警等級輸出表 |
| [`tranad_anomaly_results.csv`](tranad_anomaly_results.csv) | `experiments/exp1_multi_etf_baseline.py` | TranAD 模型於測試期間輸出之每日異常分數與異常判定結果表 |
| [`tranad_etf_scores.csv`](tranad_etf_scores.csv) | `experiments/exp1_multi_etf_baseline.py` | 核心 ETF 標的於異常區間之平均異常分數彙整表 |
