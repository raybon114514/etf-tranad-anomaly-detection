# 實驗主腳本目錄 (`etf_tranad_project/experiments/`)

本目錄存放依研究路線分階段設計的四支可獨立執行實驗腳本。

> [!IMPORTANT]
> **執行位置與版本說明**
> - 請務必切換至 `etf_tranad_project/` 目錄後再執行（例如 `python experiments/exp4_early_warning_system.py`），因腳本內部使用 `data/processed/...` 與 `results/figures/...` 相對路徑讀寫檔案。
> - 目前這 4 支腳本在訓練迴圈上仍採用原版 `(loss1 + loss2).backward()` 合併反向傳播；已修正為獨立優化器 + `autograd.grad` 對抗訓練與第 `t` 天（`[:, -1, :]`）評分的版本請見 [`../../firstry/claude改進版.py`](../../firstry/claude改進版.py)。

## 實驗清單

| 腳本 | 實驗目的 | 主要輸出 |
| :--- | :--- | :--- |
| [`exp1_multi_etf_baseline.py`](exp1_multi_etf_baseline.py) | 載入 9 檔核心 ETF（SPY, QQQ, IWM, TLT, GLD, ^VIX, HYG, XLF, XLK），訓練基準 TranAD 並產生異常趨勢與歸因分析 | `data/processed/tranad_anomaly_results.csv`、`data/processed/tranad_etf_scores.csv` |
| [`exp2_garch_robustness.py`](exp2_garch_robustness.py) | 比較輸入標準 GARCH(1,1) 與非對稱 GJR-GARCH(1,1) 波動率特徵時，TranAD 模型之表現與穩健性 | 終端機實驗指標比較 |
| [`exp3_cross_etf_sync.py`](exp3_cross_etf_sync.py) | 結合 `QQQ` / `SPY` / `AAPL` 市場微結構特徵，檢驗跨標的風險同步與外溢診斷 | 微結構同步分析結果 |
| [`exp4_early_warning_system.py`](exp4_early_warning_system.py) | 整合 TranAD、GJR-GARCH 與 `EarlyWarningSystem`，評估異常速度、加速度與跨板塊共振之早期預警表現 | `data/processed/early_warning_signals.csv`、`results/figures/early_warning_dashboard.png` |
