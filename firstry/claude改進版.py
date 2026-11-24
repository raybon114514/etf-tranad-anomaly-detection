"""
claude改進版.py
基於 etf_trend_anomaly.py 的修正版本，對照 TranAD (Tuli et al., VLDB 2022)。

相對原版的修正：
  [修正1] 對抗訓練：L1 只更新 Encoder + Decoder1，L2 只更新 Decoder2
          （原版 loss1 + loss2 相加後 Ô2 項互相抵消，Phase 2 完全沒被訓練）
  [修正2] 異常分數只看視窗「最後一天」(時間點 t)，與論文一致
          （原版整個 30 天視窗加總，導致一次衝擊會連續 30 天被標為異常）
  [修正3] 改用 Adj Close（還原股利），避免除息日被誤判為下跌
  [修正4] 使用驗證集做 Early Stopping，並保存最佳權重
  [修正5] Focus Score 採論文的平方誤差 (O1 - W)^2
  [修正6] 歸因與分數口徑一致：同樣使用兩個解碼器、同樣只看最後一天
  [修正7] 門檻敏感度分析 (95/97/98/99 百分位)
  [修正8] 診斷規則加入「貢獻集中度」判斷系統性 vs 板塊性
"""
import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 0. 超參數
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_WINDOW = 30          # 滑動視窗天數 (論文預設 10，金融資料改 30)
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 8           # Early stopping：驗證損失連續幾個 epoch 沒進步就停
LR_MAIN = 1e-3         # Encoder + Decoder1 學習率
LR_DEC2 = 5e-4         # Decoder2 學習率 (略低，避免對抗項發散)
GRAD_CLIP = 1.0        # 梯度裁剪，穩定對抗訓練
THRESHOLD_PCT = 98     # 主門檻百分位

TICKERS = ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLI"]
TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"


# ==========================================
# 1. 資料獲取與特徵工程
# ==========================================
def fetch_multi_etf_data():
    print("下載大盤與 5 大板塊 ETF 數據...")
    df = yf.download(TICKERS, start="2010-01-01", auto_adjust=False, progress=False)

    # [修正3] 使用還原股利後的價格
    price_col = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
    prices = df[price_col][TICKERS].dropna()

    returns = np.log(prices / prices.shift(1)).dropna()
    prices = prices.loc[returns.index]

    train_mask = np.asarray(returns.index <= TRAIN_END)
    val_mask = np.asarray((returns.index > TRAIN_END) & (returns.index <= VAL_END))
    test_mask = np.asarray(returns.index > VAL_END)

    print(f"  價格欄位: {price_col} | 期間: {returns.index[0].date()} ~ {returns.index[-1].date()} "
          f"| 交易日數: {len(returns)}")
    return returns, prices, train_mask, val_mask, test_mask


# ==========================================
# 2. TranAD 架構
# ==========================================
class PositionalEncoding(nn.Module):
    """Vaswani et al. (2017) 正弦位置編碼，TranAD 沿用。"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TranAD(nn.Module):
    def __init__(self, feats, d_model=64, nhead=8):
        super().__init__()
        self.feats = feats
        self.proj1 = nn.Linear(feats * 2, d_model)   # [W ⊕ F] -> d_model
        self.proj2 = nn.Linear(feats, d_model)       # W -> d_model
        self.pe = PositionalEncoding(d_model)

        # 論文 Encoder：I1^2 = LN(I1^1 + FFN(I1^1)), I1^1 = LN(I1 + MHA(I1,I1,I1))
        self.enc1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True), num_layers=1)

        # 論文 Window Encoder：遮罩自注意力 + 交叉注意力
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))

        # 兩個獨立解碼器（Z-score 資料無界，故不加 Sigmoid）
        self.dec1 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feats))
        self.dec2 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feats))

    def forward(self, w, focus_score=None):
        B, K, _ = w.shape
        if focus_score is None:                       # Phase 1：F = 0
            focus_score = torch.zeros_like(w)

        # Encoder 分支（看得到 Focus Score）
        x1 = self.pe(self.proj1(torch.cat([w, focus_score], dim=-1)))
        i1 = self.enc1(x1)

        # Window Encoder 分支（因果遮罩：第 t 天只能看第 1~t 天）
        x2 = self.pe(self.proj2(w))
        mask = torch.triu(torch.full((K, K), float('-inf'), device=w.device), diagonal=1)
        a2, _ = self.self_attn2(x2, x2, x2, attn_mask=mask)
        h = self.ln1(x2 + a2)

        c2, _ = self.cross_attn(query=h, key=i1, value=i1)
        h = self.ln2(h + c2)                          # 論文 I2^3
        h = self.ln3(h + self.ffn(h))                 # 額外 FFN（標準 Transformer 做法）

        return self.dec1(h), self.dec2(h)


# ==========================================
# 3. 訓練工具
# ==========================================
def make_windows(features_norm, dates):
    """第 i 個視窗 = 第 i ~ i+K-1 天，代表日期 = 最後一天。"""
    n = len(features_norm) - K_WINDOW + 1
    windows = np.stack([features_norm[i:i + K_WINDOW] for i in range(n)]).astype(np.float32)
    target_dates = pd.DatetimeIndex(dates[K_WINDOW - 1:])
    return windows, target_dates


def two_phase_forward(model, w):
    """論文兩階段推論：Phase 1 得 O1, O2；Focus = (O1 - W)^2；Phase 2 得 Ô2。"""
    o1, o2 = model(w)
    focus = (o1 - w) ** 2                             # [修正5] 論文使用平方誤差
    _, o2_hat = model(w, focus)
    return o1, o2, o2_hat


def train_model(model, win_train, win_val):
    # [修正1] 參數分組：Decoder2 一組，其餘 (Encoder + Decoder1 + 投影層...) 一組
    dec2_params = list(model.dec2.parameters())
    dec2_ids = {id(p) for p in dec2_params}
    main_params = [p for p in model.parameters() if id(p) not in dec2_ids]

    opt_main = optim.Adam(main_params, lr=LR_MAIN)
    opt_dec2 = optim.Adam(dec2_params, lr=LR_DEC2)
    mse = nn.functional.mse_loss

    train_loader = DataLoader(win_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(win_val, batch_size=BATCH_SIZE, shuffle=False)

    best_val, best_state, bad_epochs = float("inf"), None, 0

    print("\n[啟動雙階段對抗訓練]")
    print(f"{'epoch':>5} | {'w':>6} | {'L1':>8} | {'L2':>8} | {'val_rec':>8}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        w_factor = 1.0 / epoch                        # 論文 ε^{-n}（官方程式碼用 1/n）
        sum_l1, sum_l2, nb = 0.0, 0.0, 0

        for w in train_loader:
            w = w.to(DEVICE)
            o1, o2, o2_hat = two_phase_forward(model, w)

            # 論文損失：
            #   L1 = w‖O1−W‖ + (1−w)‖Ô2−W‖   → Decoder1 要讓 Ô2 也重構得好（騙過 Decoder2）
            #   L2 = w‖O2−W‖ − (1−w)‖Ô2−W‖   → Decoder2 要放大 Ô2 的重構誤差
            loss1 = w_factor * mse(o1, w) + (1.0 - w_factor) * mse(o2_hat, w)
            loss2 = w_factor * mse(o2, w) - (1.0 - w_factor) * mse(o2_hat, w)

            # 各自對自己的參數求梯度（不再相加 → 對抗項不會抵消）
            g_main = torch.autograd.grad(loss1, main_params, retain_graph=True, allow_unused=True)
            g_dec2 = torch.autograd.grad(loss2, dec2_params, allow_unused=True)

            opt_main.zero_grad(set_to_none=True)
            opt_dec2.zero_grad(set_to_none=True)
            for p, g in zip(main_params, g_main):
                p.grad = g
            for p, g in zip(dec2_params, g_dec2):
                p.grad = g
            nn.utils.clip_grad_norm_(main_params, GRAD_CLIP)
            nn.utils.clip_grad_norm_(dec2_params, GRAD_CLIP)
            opt_main.step()
            opt_dec2.step()

            sum_l1 += loss1.item()
            sum_l2 += loss2.item()
            nb += 1

        # [修正4] 驗證：用「Phase 1 重構誤差」衡量模型對常態結構的學習程度
        # （不使用 Ô2，因為 Decoder2 的目標是放大誤差，不適合當收斂指標）
        model.eval()
        val_err, val_n = 0.0, 0
        with torch.no_grad():
            for w in val_loader:
                w = w.to(DEVICE)
                o1, _ = model(w)
                val_err += ((o1 - w) ** 2).mean(dim=(1, 2)).sum().item()
                val_n += len(w)
        val_rec = val_err / val_n

        print(f"{epoch:>5} | {w_factor:>6.3f} | {sum_l1 / nb:>8.4f} | {sum_l2 / nb:>8.4f} | {val_rec:>8.4f}")

        if val_rec < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_rec, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  -> Early stopping @ epoch {epoch}（最佳 val_rec = {best_val:.4f}）")
                break

    model.load_state_dict(best_state)
    return model


def compute_scores_and_attribution(model, win_array):
    """
    [修正2] 分數只取視窗最後一天 (時間點 t)：
        s_t = ½‖O1[t] − W[t]‖² + ½‖Ô2[t] − W[t]‖²
    [修正6] 歸因 = 同一公式的逐維度版本，因此 attributions.sum(axis=1) == scores
    """
    model.eval()
    scores, attributions = [], []
    loader = DataLoader(win_array, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for w in loader:
            w = w.to(DEVICE)
            o1, _, o2_hat = two_phase_forward(model, w)
            feat_err = 0.5 * (o1[:, -1, :] - w[:, -1, :]) ** 2 + 0.5 * (o2_hat[:, -1, :] - w[:, -1, :]) ** 2
            attributions.append(feat_err.cpu().numpy())
            scores.append(feat_err.sum(dim=1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(attributions)


def diagnose(attr_pct, tickers):
    """
    [修正8] 先看貢獻集中度，再看主導者：
      - 最大佔比 < 35%：多板塊同步失常 → 系統性風險
      - SPY 主導：大盤脫鉤
      - 其他：特定板塊結構性破裂
    """
    top = int(np.argmax(attr_pct))
    if attr_pct[top] < 35:
        return "【宏觀系統性風險】多個板塊同步偏離常態，無單一主導來源。"
    if tickers[top] == "SPY":
        return "【大盤脫鉤】大盤 (SPY) 與各板塊之間的定價關係發生超預期偏離。"
    return f"【板塊結構性破裂】衝擊主要來自 {tickers[top]}（佔 {attr_pct[top]:.1f}%），破壞了板塊間聯動慣性。"


# ==========================================
# 4. 主程式
# ==========================================
def main():
    returns, prices, train_mask, val_mask, test_mask = fetch_multi_etf_data()
    features = returns.values
    feats_dim = features.shape[1]

    # 標準化：只用訓練期統計量（杜絕前視偏差）
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    # 視窗化；視窗屬於哪個集合由其最後一天決定
    windows, target_dates = make_windows(features_norm, returns.index)
    m_train, m_val, m_test = (m[K_WINDOW - 1:] for m in (train_mask, val_mask, test_mask))
    win_train, win_val, win_test = windows[m_train], windows[m_val], windows[m_test]
    test_dates = target_dates[m_test]
    print(f"  視窗數 train/val/test = {len(win_train)}/{len(win_val)}/{len(win_test)}")

    model = TranAD(feats=feats_dim).to(DEVICE)
    model = train_model(model, win_train, win_val)

    # ---------- 門檻與偵測 ----------
    print("\n[計算門檻與預測]")
    train_scores, _ = compute_scores_and_attribution(model, win_train)
    test_scores, test_attr = compute_scores_and_attribution(model, win_test)

    # [修正7] 門檻敏感度分析
    print("門檻敏感度分析（訓練集分數百分位）:")
    for pct in (95, 97, 98, 99):
        th = np.percentile(train_scores, pct)
        n_anom = int((test_scores > th).sum())
        print(f"  P{pct}: 門檻 = {th:7.3f} | 測試集異常天數 = {n_anom:4d} / {len(test_scores)} "
              f"({n_anom / len(test_scores) * 100:5.1f}%)")

    threshold = np.percentile(train_scores, THRESHOLD_PCT)
    anomalies = test_scores > threshold
    print(f"\n-> 採用門檻 P{THRESHOLD_PCT}: {threshold:.3f}")
    print(f"-> 測試集異常天數: {anomalies.sum()} / {len(test_scores)}")

    # ---------- 歸因分析 ----------
    if anomalies.any():
        attr_pct_all = test_attr / (test_attr.sum(axis=1, keepdims=True) + 1e-8) * 100

        print("\n[測試期異常分數最高的 10 天]")
        header = "  日期        |  分數   | " + " | ".join(f"{t:>5}" for t in TICKERS)
        print(header)
        for idx in np.argsort(test_scores)[::-1][:10]:
            row = " | ".join(f"{p:5.1f}" for p in attr_pct_all[idx])
            print(f"  {test_dates[idx].date()} | {test_scores[idx]:7.2f} | {row}")

        print("\n[最近一次異常事件歸因分析 (XAI)]")
        recent_idx = np.where(anomalies)[0][-1]
        print(f"發生日期: {test_dates[recent_idx].date()} | 異常分數: {test_scores[recent_idx]:.2f}")
        print("板塊貢獻度:")
        for t, pct in zip(TICKERS, attr_pct_all[recent_idx]):
            print(f"  - {t}: {pct:>5.1f}%")
        print(f"[診斷結論]: {diagnose(attr_pct_all[recent_idx], TICKERS)}")

    # ---------- 視覺化 ----------
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 8))

    # 上圖：以測試期首日 = 100 的標準化價格，方便比較漲跌幅
    plt.subplot(2, 1, 1)
    p = prices.loc[test_dates]
    plt.plot(test_dates, p["QQQ"] / p["QQQ"].iloc[0] * 100, label="QQQ (科技板塊)", color="navy", linewidth=1.5)
    plt.plot(test_dates, p["SPY"] / p["SPY"].iloc[0] * 100, label="SPY (大盤基準)", color="orange", linewidth=1.5)
    for d in test_dates[anomalies]:
        plt.axvline(d, color="red", alpha=0.3, linestyle="--", linewidth=1)
    plt.title("ETF Prices (Rebased = 100) and Detected Anomaly Events", fontsize=14, fontweight="bold")
    plt.ylabel("Rebased Price")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    # 下圖：異常分數與門檻
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, test_scores, label="TranAD Anomaly Score (day t)", color="#1f77b4", alpha=0.8)
    plt.axhline(threshold, color="crimson", linewidth=2,
                label=f"Threshold (Train P{THRESHOLD_PCT}: {threshold:.2f})")
    plt.scatter(test_dates[anomalies], test_scores[anomalies], color="red", s=40, zorder=5,
                label="Detected Anomaly")
    plt.yscale("log")   # 分數為厚尾分布，對數座標較易閱讀
    plt.title("Anomaly Score vs. Normalcy Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score (log)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("etf_anomaly_trend_claude.png", dpi=300)
    print("\n圖表已儲存: etf_anomaly_trend_claude.png")
    plt.show()


if __name__ == "__main__":
    main()
