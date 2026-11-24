# 早期原型與獨立修正版腳本 (`firstry/`)

本目錄保留專題初期探索測試腳本、早期 CSV 輸出結果，以及已完成關鍵架構修正的獨立執行腳本。

## 檔案清單

| 檔案 | 類型 | 說明 |
| :--- | :--- | :--- |
| [`claude改進版.py`](claude改進版.py) | 改進版主程式 | **已修復版 TranAD + GJR-GARCH 腳本**：使用獨立優化器與 `torch.autograd.grad` 明確分離 Phase 1／Phase 2 對抗更新（對 `W` 取梯度後才清零或更新各參數，避免梯度互相抵消），並取視窗最後一步 `[:, -1, :]` 計算第 `t` 天異常分數，同時支援因果標準化視窗處理與即時圖表／診斷輸出 |
| [`etf_trend_anomaly_有缺陷.py`](etf_trend_anomaly_有缺陷.py) | 原版原型程式（有缺陷） | 早期未修正版本，存在 `(loss1 + loss2).backward()` 合併反向傳播造成對抗損失正負項抵消之設計問題 |
| [`etf_anomaly_trend_claude.png`](etf_anomaly_trend_claude.png) | 結果圖表 | `claude改進版.py` 產出之多 ETF 異常分數趨勢與重大市場事件對照圖 |
| [`project.py`](project.py) | 早期草稿 | 初期模型／資料流程測試草稿腳本 |
| [`vsd.py`](vsd.py) | 早期實驗腳本 | 初期較完整的實驗測試腳本 |
| [`t2.py`](t2.py) | 測試腳本 | 早期資料與特徵測試程式 |
| [`test.py`](test.py) | 測試腳本 | 早期環境／程式功能測試程式 |
| [`tranad_anomaly_results.csv`](tranad_anomaly_results.csv) | 早期結果 CSV | 早期模型輸出之每日異常分數與二元判定表（`Date,AnomalyScore,IsAnomaly`，自 `2022-01-03` 起） |
| [`tranad_etf_scores.csv`](tranad_etf_scores.csv) | 早期結果 CSV | 早期模型輸出之 9 檔 ETF 平均異常分數表（`ETF,Average_Anomaly_Score`） |
