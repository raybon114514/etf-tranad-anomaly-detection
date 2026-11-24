# ETF 市場異常趨勢偵測模組化研究專案 (`etf_tranad_project/`)

本目錄為結構化設計的深度學習研究專案，將資料擷取、GARCH 波動率與市場微結構特徵、Transformer／TCN 模型、歸因與早期預警（EWS）工具模組化管理。

> [!IMPORTANT]
> **執行路徑與程式版本提示**
> 1. [`experiments/`](experiments/README.md) 內的腳本會透過相對路徑存取 `data/processed/` 與 `results/figures/`，請務必先切換至本目錄（`cd etf_tranad_project`）後再執行。
> 2. 目前 `exp1`–`exp4` 仍維持原版 `(loss1 + loss2).backward()` 合併梯度更新與全域視窗重建誤差；已修正為獨立優化器 + `autograd.grad` 對抗訓練及第 `t` 天評分的實作請參考上層 [`../firstry/claude改進版.py`](../firstry/claude改進版.py)。

---

## 📁 目錄架構

```text
etf_tranad_project/
├── .gitignore                        # 子專案忽略清單（快取、環境變數、raw CSV、checkpoints）
├── requirements.txt                  # Python 相依套件清單
├── README.md                         # 本專案說明文件
├── data/                             # 資料目錄（raw 原始快取、processed 已處理結果）
├── docs/                             # 專題進度、研究路線圖、投影片與論文筆記
├── experiments/                      # exp1 ~ exp4 實驗主腳本
├── results/                          # 模型權重預留目錄、圖表與公式圖片
└── src/                              # 核心模組原始碼（資料載入、特徵、模型、工具）
```

各子資料夾皆附有獨立 `README.md` 說明細節：
- [src/README.md](src/README.md)（含 [features/](src/features/README.md)、[models/](src/models/README.md)、[utils/](src/utils/README.md)）
- [experiments/README.md](experiments/README.md)
- [data/README.md](data/README.md)（含 [raw/](data/raw/README.md)、[processed/](data/processed/README.md)）
- [docs/README.md](docs/README.md)（含 [paper_notes/](docs/paper_notes/README.md)）
- [results/README.md](results/README.md)（含 [checkpoints/](results/checkpoints/README.md)、[figures/](results/figures/README.md)、[equations/](results/figures/equations/README.md)）

---

## 🔧 套件需求 (`requirements.txt`)

- `torch>=2.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.10.0`
- `scikit-learn>=1.2.0`
- `yfinance>=0.2.36`
- `matplotlib>=3.7.0`

---

## 🧪 實驗執行指令（於 `etf_tranad_project/` 下執行）

| 實驗腳本 | 研究主題 | 指令 |
| :--- | :--- | :--- |
| **`exp1_multi_etf_baseline.py`** | 9 檔核心 ETF（SPY, QQQ, IWM, TLT, GLD, ^VIX, HYG, XLF, XLK）基準異常趨勢偵測 | `python experiments/exp1_multi_etf_baseline.py` |
| **`exp2_garch_robustness.py`** | 比較 GARCH(1,1) 與 GJR-GARCH(1,1) 特徵對異常偵測之穩健性 | `python experiments/exp2_garch_robustness.py` |
| **`exp3_cross_etf_sync.py`** | QQQ / SPY / AAPL 微結構特徵與跨標的同步／外溢診斷 | `python experiments/exp3_cross_etf_sync.py` |
| **`exp4_early_warning_system.py`** | 結合速度、加速度、非對稱波動率與共振度之三層 EWS 預警系統 | `python experiments/exp4_early_warning_system.py` |

---

## 📚 對應參考文獻（位於上層 `../papers/core/`）

1. **TranAD**：[p1201-tuli.pdf](../papers/core/p1201-tuli.pdf) — Tuli, Casale, Jennings (PVLDB 2022), *TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data*
2. **JRFM 2026**：[Volatility Specification and Deep Learning Anomaly Detection (JRFM, 2026)](../papers/core/Volatility%20Specification%20and%20Deep%20Learning%20Anomaly%20Detection%20Robustness%20of%20Transformer%20Architectures%20to%20GARCH%20Model%20Choice%20%28JRFM,%202026%29.pdf)（筆記見 [`docs/paper_notes/01_JRFM2026_Volatility_TranAD_Notes.md`](docs/paper_notes/01_JRFM2026_Volatility_TranAD_Notes.md)）
3. **TCN Anomaly 2019**：[Temporal Convolutional Networks for Anomaly Detection in Time Series (2019)](../papers/core/Temporal%20Convolutional%20Networks%20for%20Anomaly%20Detection%20in%20Time%20Series%20%282019%29.pdf)（筆記見 [`docs/paper_notes/02_TCN_TimeSeries_Anomaly_2019_Notes.md`](docs/paper_notes/02_TCN_TimeSeries_Anomaly_2019_Notes.md)）
