# 核心原始碼模組 (`etf_tranad_project/src/`)

本目錄包含專題所有可重用 Python 模組，依職責劃分為資料載入、特徵工程、深度學習模型與分析工具：

## 模組架構

| 檔案／子套件 | 主要公開元件 | 說明 |
| :--- | :--- | :--- |
| `__init__.py` | — | `src` 套件初始化檔 |
| [`data_loader.py`](data_loader.py) | `WindowDataset`、`fetch_multi_etf_data` | 提供 PyTorch 滑動視窗資料集類別，並透過 `yfinance` 下載多檔 ETF 日線歷史資料 |
| [`features/`](features/README.md) | `GARCHModels`、`fetch_etf_microstructure_features` | GARCH(1,1) / GJR-GARCH(1,1) 波動率擬合與市場微結構特徵萃取 |
| [`models/`](models/README.md) | `PositionalEncoding`、`TranAD`、`DilatedCausalConv`、`VTT` | 多變量 Transformer 對抗重建架構與因果卷積混合架構 |
| [`utils/`](utils/README.md) | `EarlyWarningSystem`、`compute_scores_and_attribution`、`diagnose_recent_anomaly`、`plot_*` | 特徵歸因診斷、EWS 預警指標計算與視覺化繪圖函式 |
