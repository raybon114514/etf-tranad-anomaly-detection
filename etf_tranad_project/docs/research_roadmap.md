# 專題研究延伸方向與實驗藍圖 (Research Roadmap)

本文件整理自專題初期構想（`message.txt`），明確定義如何從基礎論文的「單點異常偵測」延伸出具備原創研究價值的「跨 ETF 異常事件與聯動慣性破缺分析」。

---

## 一、 核心研究定位

> **核心主軸**：以 `TranAD` 的多變量時間序列異常偵測為基礎，學習多檔 ETF 之間的歷史聯動慣性（Normal Pricing Dynamics），進一步研究**持續性異常（Persistent Anomaly）**、**跨 ETF 異常同步性（Cross-ETF Synchronization）**與**異常來源歸因（Sector Attribution）**。

---

## 二、 三階段實驗設計 (Three-Stage Experimental Plan)

### 第一階段：Baseline 跨 ETF 聯動慣性建模 (`exp1_multi_etf_baseline.py`)
* **輸入特徵**：大盤與板塊 ETF 之日對數報酬率（`SPY`, `QQQ`, `XLF`, `XLE`, `XLV`, `XLI`）。
* **核心任務**：
  1. 驗證 `TranAD` 能否在無前視偏差下捕捉常態定價慣性。
  2. 利用重構誤差分解（XAI Attribution）診斷異常日是由大盤（`SPY`）主導之宏觀風險，還是單一板塊外溢之結構性破裂。

### 第二階段：引入 GARCH 家族波動率與微結構特徵 (`exp2_garch_robustness.py`)
* **輸入特徵**：結合對數報酬率 $r_t$、條件波動率 $\sigma_t$ 與標準化殘差 $z_t$，以及價差解耦（`Bench_Spread`）與成交量衝擊（`Volume_Shock`）。
* **核心任務**：
  1. 比較對稱 `GARCH(1,1)` 與非對稱 `GJR-GARCH(1,1)` 對 `TranAD` 異常偵測結果的一致性（Jaccard Similarity）。
  2. 探討市場高波動期與 ETF 跨資產異常分數之動態關聯。

### 第三階段：從異常點（Point Anomaly）延伸至跨 ETF 異常事件（Anomaly Event） (`exp3_cross_etf_sync.py`)
* **多層次異常架構**：
  * **Level 1 (Point Anomaly)**：單日重構誤差超過常態 95% / 98% 門檻。
  * **Level 2 (Persistent Anomaly)**：異常是否連續數日發生，形成持續性事件區間（Sustained Episodes）。
  * **Level 3 (Cross-ETF Regime / Sync)**：9 檔 ETF 中有多少檔同時超過各自門檻？哪一檔 ETF 最早出現異常訊號？
