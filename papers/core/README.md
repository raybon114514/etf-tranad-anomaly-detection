# 核心研究論文 (`papers/core/`)

本目錄收錄直接支撐本專題模型架構（TranAD、VTT）與金融時間序列實驗設計之核心論文：

| 檔名 | 作者／發表來源 | 核心內容與專題關聯 |
| :--- | :--- | :--- |
| [`p1201-tuli.pdf`](p1201-tuli.pdf) | Tuli, Casale, Jennings (PVLDB 2022) | **TranAD 原論文**：提出自條件雙階段 Transformer 與對抗式重構訓練框架 |
| [`Volatility Specification and Deep Learning Anomaly Detection Robustness of Transformer Architectures to GARCH Model Choice (JRFM, 2026).pdf`](Volatility%20Specification%20and%20Deep%20Learning%20Anomaly%20Detection%20Robustness%20of%20Transformer%20Architectures%20to%20GARCH%20Model%20Choice%20%28JRFM,%202026%29.pdf) | JRFM (2026) | 檢驗 Transformer 異常偵測架構在不同 GARCH 波動率模型選擇下之穩健性（對應 [筆記 01](../../etf_tranad_project/docs/paper_notes/01_JRFM2026_Volatility_TranAD_Notes.md)） |
| [`Temporal Convolutional Networks for Anomaly Detection in Time Series (2019).pdf`](Temporal%20Convolutional%20Networks%20for%20Anomaly%20Detection%20in%20Time%20Series%20%282019%29.pdf) | He & Zhao (2019) | **TCN 時間序列異常偵測**：啟發本專案 `VTT` 中的膨脹因果卷積（Dilated Causal Conv）設計（對應 [筆記 02](../../etf_tranad_project/docs/paper_notes/02_TCN_TimeSeries_Anomaly_2019_Notes.md)） |
