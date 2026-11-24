# 多變量 ETF 市場異常趨勢偵測系統 (TranAD + GARCH / EWS)

本專案以 **TranAD (Deep Transformer Networks)** 與 **GARCH / GJR-GARCH 波動率模型** 為核心，建構跨美股大盤、類股、公債、黃金與恐慌指標等多維金融時間序列之**異常趨勢偵測（Anomaly Trend Detection）**與**早期預警系統（Early Warning System, EWS）**。

> [!NOTE]
> **分支與程式碼版本提醒**
> - 本倉庫已完成結構整理並同步 `main` 與 `feat/structured-repo` 分支。
> - 針對 TranAD 兩階段對抗訓練之**獨立優化器修正版**（使用 `torch.autograd.grad` 明確分離 Decoder1 / Decoder2 對抗目標，並以窗口最後一天 `[:, -1, :]` 作為第 `t` 天評分）目前位於 [`firstry/claude改進版.py`](firstry/claude改進版.py)；[`etf_tranad_project/experiments/`](etf_tranad_project/experiments/README.md) 中的 `exp1`–`exp4` 與 [`firstry/etf_trend_anomaly_有缺陷.py`](firstry/etf_trend_anomaly_有缺陷.py) 仍保留原始 `(loss1 + loss2).backward()` 合併回傳寫法以維持既有實驗紀錄，後續可視需求同步。

---

## 📁 資料夾結構導覽

| 目錄／檔案 | 說明 | 導覽連結 |
| :--- | :--- | :--- |
| **`etf_tranad_project/`** | 核心模組化研究專案（資料載入、GARCH／微結構特徵、TranAD／VTT 模型、`exp1`–`exp4`、圖表與文件） | [專案 README](etf_tranad_project/README.md) |
| **`firstry/`** | 早期原型腳本、探索測試檔，以及獨立修正版 `claude改進版.py` | [firstry README](firstry/README.md) |
| **`papers/`** | 依「核心必讀」與「經典基礎」分類的參考論文 PDF | [papers README](papers/README.md) |
| **`notes/`** | 研究筆記、EWS 動量推導、教授攻防 Q&A 與初期研究發想 `message.txt` | [notes README](notes/README.md) |
| **`報告進度/`** | 期中進度報告、改進版教授說明簡報（PPTX 與 HTML 預覽）、逐字講稿與圖表資源 | [報告進度 README](報告進度/README.md) |
| **`00_專題核心導航_MOC.md`** | Obsidian 全局導航地圖（彙整文獻、程式分支、筆記、簡報與待辦清單） | [MOC 導航](00_專題核心導航_MOC.md) |
| **`.gitignore`** | 排除 Python 快取（`__pycache__`、`*.pyc`）、虛擬環境、原始 CSV 與模型權重檔 | [.gitignore](.gitignore) |

---

## 🚀 快速開始

### 1. 安裝套件

```bash
pip install -r etf_tranad_project/requirements.txt
```

### 2. 執行模組化實驗（需於 `etf_tranad_project/` 目錄內執行）

`exp1`–`exp4` 會將結果寫入相對路徑 `data/processed/` 與 `results/figures/`，執行前請先切換目錄：

```bash
cd etf_tranad_project
python experiments/exp1_multi_etf_baseline.py
python experiments/exp2_garch_robustness.py
python experiments/exp3_cross_etf_sync.py
python experiments/exp4_early_warning_system.py
```

### 3. 執行獨立對抗修正版腳本

```bash
python firstry/claude改進版.py
```

---

## 📊 主要輸出圖表

- **EWS 早期預警儀表板**：[`etf_tranad_project/results/figures/early_warning_dashboard.png`](etf_tranad_project/results/figures/early_warning_dashboard.png)
- **多 ETF 異常分數趨勢圖**：[`etf_tranad_project/results/figures/etf_anomaly_trend.png`](etf_tranad_project/results/figures/etf_anomaly_trend.png)
- **改進版實驗趨勢圖**：[`firstry/etf_anomaly_trend_claude.png`](firstry/etf_anomaly_trend_claude.png)
