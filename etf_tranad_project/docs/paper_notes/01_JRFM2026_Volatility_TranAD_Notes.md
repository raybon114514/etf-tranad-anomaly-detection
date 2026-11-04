# 論文閱讀筆記 (1)：Volatility Specification and Deep Learning Anomaly Detection (JRFM, 2026)

* **論文標題**：*Volatility Specification and Deep Learning Anomaly Detection: Robustness of Transformer Architectures to GARCH Model Choice*
* **期刊與年份**：Journal of Risk and Financial Management (JRFM), 2026
* **原始檔案位置**：[`Volatility Specification and Deep Learning Anomaly Detection... (JRFM, 2026).pdf`](../../Volatility%20Specification%20and%20Deep%20Learning%20Anomaly%20Detection%20Robustness%20of%20Transformer%20Architectures%20to%20GARCH%20Model%20Choice%20(JRFM,%202026).pdf)

---

## 1. 研究問題與核心貢獻
* **核心問題**：在金融時間序列異常偵測中，將計量經濟學波動率模型（如 `GARCH`、`EGARCH`、`GJR-GARCH`、`APARCH`）萃取出的條件波動度 $\sigma_t$ 與標準化殘差 $z_t$ 餵入深度學習 Transformer 模型（`TranAD`、`VTT`）時，不同的 GARCH 模型選擇是否會顯著改變異常偵測的結果？
* **核心結論**：
  1. **高度穩健性（Robustness）**：`TranAD` 在不同 GARCH 家族規格下的異常日偵測高度一致（Jaccard Index > 0.85），顯示 Transformer 的自注意力與對抗機制能有效吸收波動率規格間的微小差異。
  2. **Point Anomaly vs. Regime Detection**：`TranAD` 擅長捕捉局部時間視窗（$K=30$）的點異常與短期波動衝擊；而 `VTT`（結合因果擴張卷積與變數/時間雙注意力，$N=100$）則傾向捕捉中長期的市場狀態轉換（Regime Shift）。

---

## 2. 方法論重點與公式實作對照 (對應 `src/models/tranad.py` & `src/features/garch_volatility.py`)

### (1) 防前視偏差之資料切分與標準化 (式 41–43)
* 僅使用訓練集計算均值 $\mu_{\text{train}}$ 與標準差 $\sigma_{\text{train}}$：
  $$\tilde{x}_t = \frac{x_t - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
* 滑動視窗長度 $K = 30$，每個視窗 $W_t = \{\tilde{x}_{t-K+1}, \dots, \tilde{x}_t\}$ 僅對齊最後一個時間點 $t$，確保推論時絕不使用未來資訊。

### (2) TranAD 雙階段自條件與演化對抗訓練 (式 44–59)
* **無界特徵修正（式 51）**：金融標準化特徵無上下界，因此 Decoder 輸出層必須移除原版 `TranAD` 的 `Sigmoid`，改用純 `Linear` 層輸出。
* **Phase 1**：輸入視窗 $W$ 與全零 Focus Score，獲得初步重構 $O_1, O_2$，計算殘差焦點分數 $\text{Focus} = |O_1 - W|$。
* **Phase 2**：將 $W$ 與 $\text{Focus}$ 拼接送入 Encoder 1，並在 Encoder 2 加上上三角因果遮罩（Causal Mask），輸出二階段重構 $\hat{O}_2$。
* **演化對抗損失（式 57–58）**：令演化權重 $w = \frac{1}{\text{epoch}}$：
  $$\mathcal{L}_1 = w \|O_1 - W\|_2^2 + (1 - w) \|\hat{O}_2 - W\|_2^2$$
  $$\mathcal{L}_2 = w \|O_2 - W\|_2^2 - (1 - w) \|\hat{O}_2 - W\|_2^2$$
* **異常分數計算（式 59）**：使用整個視窗的 Frobenius 範數平方和（在 PyTorch 中需使用 `torch.sum(..., dim=(1,2))` 而非 `mean`）：
  $$s_t = \frac{1}{2} \|O_1 - W\|_F^2 + \frac{1}{2} \|\hat{O}_2 - W\|_F^2$$

---

## 3. 對本專題的啟發與延伸
* 論文僅針對單一指數（S&P 500）之衍生特徵進行驗證；本專題將其擴展至**多檔跨板塊 ETF 的歷史聯動慣性（Cross-ETF Normal Pricing Dynamics）**，並利用最後一天的特徵重構誤差 $( \hat{O}_{2, K, :} - W_{K, :} )^2$ 進行**板塊異常歸因分析（XAI）**。
