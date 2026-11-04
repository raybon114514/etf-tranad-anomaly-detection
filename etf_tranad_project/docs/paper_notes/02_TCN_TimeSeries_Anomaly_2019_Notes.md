# 論文閱讀筆記 (2)：Temporal Convolutional Networks for Anomaly Detection in Time Series (2019)

* **論文標題**：*Temporal Convolutional Networks for Anomaly Detection in Time Series*
* **作者與年份**：Yangdong He, Jiabao Zhao (2019)
* **原始檔案位置**：[`Temporal Convolutional Networks for Anomaly Detection in Time Series (2019).pdf`](../../Temporal%20Convolutional%20Networks%20for%20Anomaly%20Detection%20in%20Time%20Series%20(2019).pdf)

---

## 1. 核心概念摘要
* **為什麼使用擴張因果卷積（Dilated Causal Convolution）？**
  1. **因果性（Causality）**：在時間點 $t$ 的輸出僅依賴於 $t$ 及過去的時間步 $t-1, t-2, \dots$，透過左側零填充（Left Zero-Padding: $(k-1) \times d$）杜絕未來資訊洩漏（Data Leakage）。
  2. **指數級擴展感受野（Exponentially Large Receptive Field）**：透過膨脹係數 $d = 1, 2, 4, \dots$，無需極深的網路層數即可捕捉長距離的時間序列依賴關係，且比傳統 RNN/LSTM 具備更高的平行運算效率與梯度穩定度。

---

## 2. 與本專題模型 (`VTT` / `TranAD`) 的關聯
* 在 2026 JRFM 論文與本專題的 `src/models/vtt.py` (`DilatedCausalConv`) 中，即採用了三組平行的因果擴張卷積分支（Kernel Size $k=(2,3,3)$，Dilation $d=(1,2,4)$，左側 Padding 分別為 $1, 4, 8$）作為多變量時間序列的局部多尺度特徵提取器（Local Multi-scale Feature Extractor），隨後再接上變數注意力（Variable Attention）與時間注意力（Temporal Attention）。
* **異常判定機制對照**：TCN 論文採用預測誤差向量搭配多元高斯分佈（Multivariate Gaussian Distribution）之馬氏距離（Mahalanobis Distance）作為異常分數；而在本專題之跨 ETF 架構中，我們比較了 Frobenius 重構誤差與各資產獨立誤差拆解的解釋性優劣。
