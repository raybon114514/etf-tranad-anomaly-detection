# 專題研究每週進度報告 (Weekly Meeting Progress)

* **報告學生**：〇〇〇
* **指導教授**：〇〇〇 教授
* **專題題目**：基於多變量 Transformer (TranAD) 之跨 ETF 歷史聯動慣性建模與結構性異常偵測
* **會議日期**：2026 / 11 / 05

---

## 一、 本週完成事項 (Completed Tasks)

老師好，本週我主要完成了核心文獻的實作重現，並將模型延伸至多檔 ETF 的跨板塊聯動異常偵測，同時將專案程式碼模組化整理至 GitHub，具體進度如下：

1. **文獻精讀與方法複現**：
   * 讀完 *Volatility Specification and Deep Learning Anomaly Detection (JRFM, 2026)* 的第三、四章（Methodology 與 Empirical Framework），釐清 `TranAD` 的雙階段自條件（Two-Phase Self-Conditioning）機制、演化式對抗損失函數（式 57–58）以及 `VTT` 的因果擴張卷積架構。
   * 以 `scipy.optimize` (SLSQP) 從頭實作了對稱 `GARCH(1,1)` 與非對稱 `GJR-GARCH(1,1)` 的 QMLE 參數估計，並成功驗證論文核心結論：在不同波動率規格下，`TranAD` 標記異常日的 Jaccard 相似度達 0.85 以上。
2. **多檔 ETF 聯動慣性模型建構與 XAI 歸因**：
   * 建立涵蓋大盤與各大產業板塊（`SPY`, `QQQ`, `XLF`, `XLE`, `XLV`, `XLI` 等）的多變量日對數報酬率資料管線（滑動視窗 $K = 30$）。
   * 實作完成 PyTorch 版 `TranAD` 模型（含正弦位置編碼 `PositionalEncoding`、防止未來資訊洩漏的上三角因果遮罩 `Causal Mask`、以及針對無界金融特徵調整的 `Linear` 雙解碼器架構）。
   * 新增 **跨板塊異常歸因分析（XAI Attribution）** 與視覺化繪圖功能，能將異常觸發當日的重構誤差拆解為各檔 ETF 的貢獻百分比，自動診斷異常事件屬於「宏觀系統性風險」或「特定板塊結構性破裂」，並匯出評估結果（`tranad_anomaly_results.csv`、`tranad_etf_scores.csv`）。

---

## 二、 目前遇到的困難與瓶頸 (Challenges & Attempted Solutions)

在實作與測試過程中遇到了以下三個主要問題，我先嘗試了初步的解法，想在今天 Meeting 中向老師請教進一步的調整方向：

### 瓶頸 1：驗證集（2019–2021）包含 2020 疫情崩盤極端值，導致異常門檻（Threshold）被嚴重墊高
* **問題描述**：原先依照文獻設定，利用驗證集的重構誤差百分位數來決定異常門檻，但 2020 年 3 月 COVID-19 熔斷期間的 MSE 高達平時的數十倍，造成驗證集門檻過高，在 2022–2026 測試集出現「漏報（False Negative）」現象。
* **已嘗試的解法**：
  * **嘗試 A**：改用「純淨訓練集（2010–2018 常態期）」的第 98 百分位數作為常態容忍門檻，敏感度明顯提升。
  * **嘗試 B**：加入 20 日滾動波動率與成交量衝擊（`Volume_Shock`）作為額外微結構特徵。
* **想請教老師的方向**：
  * 在金融時間序列異常偵測中，門檻應該固定由常態訓練集決定，還是建議採用「滾動視窗動態門檻（Rolling Quantile Threshold）」或極端值理論（POT / EVT）來適應不同市場波動狀態（Regime）？

### 瓶頸 2：標準化後的金融報酬率屬於無界特徵（Unbounded Features），以及對抗損失的訓練穩定度
* **問題描述**：原版 `TranAD` 論文在 Decoder 最後一層使用 `Sigmoid` 將輸出限制在 $[0, 1]$，但金融報酬率與標準化殘差 $z_t$ 經 Z-score 標準化後會有超過 $\pm 3$ 的極端值；此外，Phase 2 的對抗損失（$\text{Loss}_2 = w \cdot \|O_2 - W\|_2 - (1-w) \cdot \|\hat{O}_2 - W\|_2$）在訓練後期偶爾會使重構誤差震盪。
* **已嘗試的解法**：
  * 已參考 2026 JRFM 論文（式 51），將雙解碼器輸出層改為無激活函數的 `Linear` 層，並修正了 Frobenius 範數平方和（`torch.sum`）的計分尺度。
* **想請教老師的方向**：
  * 針對金融厚尾分佈（Fat-tailed Distribution），輸入特徵除了 Z-score 標準化外，是否建議改用 `RobustScaler`（中位數與四分位距）來降低極端離群值對訓練穩定度的影響？

### 瓶頸 3：從「單日異常點（Point Anomaly）」延伸到「異常事件／趨勢（Anomaly Event）」的定義標準
* **問題描述**：目前模型輸出的是每日的異常分數，在市場劇烈波動期常出現「連續數天超標，但中間夾雜 1 天微幅低於門檻」的情況。
* **想請教老師的方向**：
  * 我預計將專題主軸聚焦於「跨 ETF 持續性異常事件與同步性分析」。若要定義一段「異常趨勢事件（Anomaly Event）」與「跨 ETF 同步異常（Cross-ETF Synchronization）」，在財務實證上是否有推薦的平滑化視窗或判定準則（例如：連續 $N$ 日內有 $M$ 日超標，且同時有超過 $K$ 檔板塊 ETF 異常）？

---

## 三、 下週預計進度 (Next Week's Plan)

1. **實作「持續性異常事件（Persistent Anomaly Event）」與「跨 ETF 同步率」分析模組**：統計 2022–2026 測試期間每次重大異常事件的持續天數、同步異常的 ETF 數量，以及哪一檔板塊 ETF 最常作為領先異常來源。
2. **進行特徵消融實驗（Ablation Study）**：比較「僅使用對數報酬率」與「加入 GARCH 波動率 / 價差解耦微結構特徵」在跨 ETF 異常事件偵測上的差異。
3. **整理實驗數據與視覺化圖表**：將消融實驗與跨板塊歸因分析結果整理成表格與圖表，於下週 Meeting 向老師報告。
