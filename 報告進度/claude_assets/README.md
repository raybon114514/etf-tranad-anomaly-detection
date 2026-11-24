# 教授說明簡報圖表與公式素材 (`報告進度/claude_assets/`)

本資料夾為 [`../claude改進版_教授說明簡報.pptx`](../claude改進版_教授說明簡報.pptx) 與 [`../claude改進版_簡報網頁預覽.html`](../claude改進版_簡報網頁預覽.html) 所引用的視覺素材庫，包含實驗對照圖表與數學公式圖片，請勿任意刪除以免網頁預覽破圖。

## 1. 實驗圖表與分析圖

| 檔名 | 說明 |
| :--- | :--- |
| [`orig_trend.png`](orig_trend.png) | 原版程式生成之異常趨勢圖副本（供網頁預覽直接載入對照） |
| [`claude_trend.png`](claude_trend.png) | `claude改進版.py` 生成之異常趨勢圖副本 |
| [`fig_timeline.png`](fig_timeline.png) | 樣本切分與重大市場事件時間軸示意圖 |
| [`fig_training_dynamics.png`](fig_training_dynamics.png) | 原版與改進版訓練過程 Loss / 對抗動態比較圖 |
| [`fig_sensitivity.png`](fig_sensitivity.png) | 模型關鍵參數／門檻敏感度分析圖 |
| [`fig_xai_cases.png`](fig_xai_cases.png) | 典型異常事件之特徵歸因（XAI）案例圖 |

## 2. 數學公式圖

| 檔名 | 說明 |
| :--- | :--- |
| [`eq_input.png`](eq_input.png) | 輸入多變量時間視窗與特徵向量定義公式 |
| [`eq_pe.png`](eq_pe.png) | 位置編碼（Positional Encoding）公式 |
| [`eq_attn_focus.png`](eq_attn_focus.png) | Self-Attention 與 Focus Score 計算公式 |
| [`eq_adv_losses.png`](eq_adv_losses.png) | Phase 1 / Phase 2 自條件對抗損失定義公式 |
| [`eq_cancel.png`](eq_cancel.png) | 原版合併反傳造成正負項對抗梯度抵消之說明公式 |
| [`eq_split.png`](eq_split.png) | 改進版分離式雙優化器更新規則公式 |
| [`eq_score_dayt.png`](eq_score_dayt.png) | 改進版第 `t` 天（最後時間步）單點異常評分公式 |
| [`eq_score_xai.png`](eq_score_xai.png) | 維度歸因貢獻與異常診斷評分公式 |
