# 波動率與微結構特徵工程 (`etf_tranad_project/src/features/`)

本套件負責建構金融時間序列的條件波動率特徵與跨標的市場微結構特徵：

| 檔案 | 主要函式／類別 | 說明 |
| :--- | :--- | :--- |
| `__init__.py` | — | 套件初始化檔 |
| [`garch_volatility.py`](garch_volatility.py) | `GARCHModels`（`fit_garch11`、`fit_gjr_garch11`、`extract_features`） | 實作 GARCH(1,1) 與含槓桿效果之 GJR-GARCH(1,1) 波動率估計，轉換生成模型輸入特徵 |
| [`microstructure.py`](microstructure.py) | `fetch_etf_microstructure_features` | 擷取 `QQQ`、`SPY` 與代表個股 `AAPL` 之日內高低價差、成交量衝擊與 20 日滾動波動率等微結構特徵 |
