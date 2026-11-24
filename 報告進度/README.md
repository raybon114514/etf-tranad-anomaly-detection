# 專題簡報與會議報告資源 (`報告進度/`)

本目錄存放專題期中報告簡報、原版 vs. 改進版教授說明簡報、HTML 網頁預覽、逐字講稿與自動生成腳本。

## 檔案清單

| 檔案／子資料夾 | 說明 |
| :--- | :--- |
| [`專題進度報告_ETF異常趨勢偵測.pptx`](專題進度報告_ETF異常趨勢偵測.pptx) | 期中專題進度簡報（由 `create_presentation.py` 生成；註：大綱文件規劃 12 頁，腳本實際建立 11 頁） |
| [`簡報網頁預覽_Slides.html`](簡報網頁預覽_Slides.html) | 期中專題進度簡報的網頁預覽檔 |
| [`create_presentation.py`](create_presentation.py) | 生成 `專題進度報告_ETF異常趨勢偵測.pptx` 的 Python（`python-pptx`）腳本 |
| [`簡報講稿與投影片大綱.md`](簡報講稿與投影片大綱.md) | 期中簡報 Slide 1–12 逐字講稿、時間分配與過場話術大綱 |
| [`claude改進版_教授說明簡報.pptx`](claude改進版_教授說明簡報.pptx) | 向指導教授報告「原版對抗梯度問題與改進版修正重點」之 13 頁簡報（由 `create_claude_presentation.py` 生成） |
| [`claude改進版_簡報網頁預覽.html`](claude改進版_簡報網頁預覽.html) | 教授說明簡報的 HTML 互動網頁預覽（使用 `claude_assets/` 圖檔） |
| [`create_claude_presentation.py`](create_claude_presentation.py) | 生成 `claude改進版_教授說明簡報.pptx` 之腳本（已更新原版對照圖預設候選路徑至 `../etf_tranad_project/results/figures/etf_anomaly_trend.png` 與 `claude_assets/orig_trend.png`） |
| [`equations/`](equations/README.md) | 期中簡報使用的 600dpi 高解析度數學公式 PNG 圖庫 |
| [`claude_assets/`](claude_assets/README.md) | 教授說明簡報與網頁預覽使用的圖表對照、時間軸與數學公式圖庫 |
