# 資料目錄 (`etf_tranad_project/data/`)

本目錄負責管理實驗過程中的市場歷史資料快取與模型輸出數據表，劃分為兩個子目錄：

| 子資料夾 | 說明 |
| :--- | :--- |
| [`raw/`](raw/README.md) | 預留存放透過 `yfinance` 下載之原始日線行情快取（`.csv` / `.parquet`），由 `.gitignore` 排除追蹤 |
| [`processed/`](processed/README.md) | 存放各實驗腳本產出之每日異常分數、ETF 平均分數與早期預警（EWS）訊號 CSV |
