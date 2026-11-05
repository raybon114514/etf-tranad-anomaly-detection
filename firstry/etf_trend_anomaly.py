import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yfinance as yf
import matplotlib.pyplot as plt

# 確保可重現性
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_WINDOW = 30 # 滑動視窗天數

# ==========================================
# 1. 跨 ETF 資料獲取與特徵工程
# ==========================================
def fetch_multi_etf_data():
    print("下載大盤與 5 大板塊 ETF 數據...")
    tickers = ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLI"]
    df = yf.download(tickers, start="2010-01-01", auto_adjust=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df["Close"]
    else:
        close_prices = df

    # 取選定 ETF 欄位並計算對數報酬率
    close_prices = close_prices[tickers].dropna()
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    close_prices = close_prices.loc[returns.index]
    
    # 切分資料：2010-2018 訓練, 2019-2021 驗證, 2022-2026 測試
    train_mask = returns.index <= "2018-12-31"
    val_mask = (returns.index > "2018-12-31") & (returns.index <= "2021-12-31")
    test_mask = returns.index > "2021-12-31"
    
    return returns, close_prices, train_mask, val_mask, test_mask, tickers

# ==========================================
# 2. TranAD 架構與位置編碼
# ==========================================
class PositionalEncoding(nn.Module):
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
        self.proj1 = nn.Linear(feats * 2, d_model)
        self.proj2 = nn.Linear(feats, d_model)
        self.pe = PositionalEncoding(d_model)

        self.enc1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True), num_layers=1)
        
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))

        self.dec1 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feats))
        self.dec2 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feats))

    def forward(self, w, focus_score=None):
        B, K, _ = w.shape
        if focus_score is None: focus_score = torch.zeros_like(w)

        # Phase 1
        x1 = self.pe(self.proj1(torch.cat([w, focus_score], dim=-1)))
        i1 = self.enc1(x1)

        # Phase 2 (Causal Mask)
        x2 = self.pe(self.proj2(w))
        mask = torch.triu(torch.full((K, K), float('-inf'), device=w.device), diagonal=1)
        a2, _ = self.self_attn2(x2, x2, x2, attn_mask=mask)
        i2 = self.ln1(x2 + a2)

        c2, _ = self.cross_attn(query=i2, key=i1, value=i1)
        i2 = self.ln3(self.ln2(i2 + c2) + self.ffn(self.ln2(i2 + c2)))

        return self.dec1(i2), self.dec2(i2)

class WindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
    def __len__(self): return len(self.data) - self.window_size + 1
    def __getitem__(self, idx): return self.data[idx : idx + self.window_size]

# ==========================================
# 3. 專題核心：訓練、預測與歸因分析
# ==========================================
def main():
    returns, close_prices, train_mask, val_mask, test_mask, tickers = fetch_multi_etf_data()
    features = returns.values
    feats_dim = features.shape[1]

    # 標準化 (杜絕前視偏差)
    train_data = features[train_mask]
    mean, std = np.mean(train_data, axis=0), np.std(train_data, axis=0) + 1e-8
    features_norm = (features - mean) / std

    # 構造視窗對齊時間軸
    windows, target_dates = [], []
    for i in range(len(features_norm) - K_WINDOW + 1):
        windows.append(features_norm[i : i + K_WINDOW])
        target_dates.append(returns.index[i + K_WINDOW - 1])
    
    windows = np.array(windows, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)

    # 切分 Dataset (使用 np.asarray 確保相容 Series 與 ndarray)
    win_train = windows[np.asarray(train_mask)[K_WINDOW-1:]]
    win_val = windows[np.asarray(val_mask)[K_WINDOW-1:]]
    win_test = windows[np.asarray(test_mask)[K_WINDOW-1:]]
    test_dates = target_dates[np.asarray(test_mask)[K_WINDOW-1:]]

    train_loader = DataLoader(win_train, batch_size=64, shuffle=True)
    model = TranAD(feats=feats_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n[啟動雙階段對抗訓練]")
    model.train()
    for epoch in range(1, 41):
        w_factor = 1.0 / epoch
        for w in train_loader:
            w = w.to(DEVICE)
            o1, o2 = model(w)
            focus = torch.abs(o1 - w)
            _, o2_hat = model(w, focus)
            
            loss1 = w_factor * nn.functional.mse_loss(o1, w) + (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)
            # 對抗懲罰: 強制放大 o2_hat 與正常 o2 的差異
            loss2 = w_factor * nn.functional.mse_loss(o2, w) - (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)

            optimizer.zero_grad()
            (loss1 + loss2).backward()
            optimizer.step()

    # 計算分數與歸因 (XAI)
    def compute_scores_and_attribution(win_array):
        model.eval()
        scores, attributions = [], []
        loader = DataLoader(win_array, batch_size=64, shuffle=False)
        with torch.no_grad():
            for w in loader:
                w = w.to(DEVICE)
                o1, _ = model(w)
                _, o2_hat = model(w, torch.abs(o1 - w))
                
                # 計算總 MSE (單一純量分數)
                err1 = torch.sum((o1 - w)**2, dim=(1, 2))
                err2 = torch.sum((o2_hat - w)**2, dim=(1, 2))
                scores.extend((0.5 * err1 + 0.5 * err2).cpu().numpy())
                
                # 計算特徵歸因：擷取最後一天 (觸發日) 每個板塊的重構誤差
                feat_err = (o2_hat[:, -1, :] - w[:, -1, :])**2
                attributions.extend(feat_err.cpu().numpy())
                
        return np.array(scores), np.array(attributions)

    print("\n[計算門檻與預測]")
    # 【關鍵修正】：改用「純淨的訓練集 (2010-2018)」來定義市場常態的容忍邊界
    # 這樣可以避免門檻被 2020 疫情崩盤的極端值嚴重扭曲
    train_scores, _ = compute_scores_and_attribution(win_train)
    
    # 採用訓練集的 98 百分位作為嚴格門檻 (對應先前的修正邏輯)
    threshold = np.percentile(train_scores, 98) 

    # 將此常態門檻套用於測試集 (2022-2026) 進行盲測
    test_scores, test_attr = compute_scores_and_attribution(win_test)
    anomalies = test_scores > threshold

    print(f"-> 修正後常態門檻值: {threshold:.2f}")
    print(f"-> 測試集異常天數: {np.sum(anomalies)} / {len(test_scores)}")

    # 4. 專題亮點：異常歸因分析 (找戰犯)
    if np.sum(anomalies) > 0:
        print("\n[近期異常事件歸因分析 (XAI)]")
        # 抓出測試集中最近一次的異常日
        anomaly_indices = np.where(anomalies)[0]
        recent_idx = anomaly_indices[-1]
        recent_date = test_dates[recent_idx].strftime("%Y-%m-%d")
        
        # 算出該日各板塊誤差佔比
        attr = test_attr[recent_idx]
        attr_pct = (attr / (attr.sum() + 1e-8)) * 100
        
        print(f"發生日期: {recent_date} | 總異常分數: {test_scores[recent_idx]:.2f}")
        print("板塊崩潰貢獻度:")
        for t, pct in zip(tickers, attr_pct):
            print(f" - {t}: {pct:>5.1f}%")
            
        # 判斷異常型態
        max_ticker = tickers[np.argmax(attr_pct)]
        if max_ticker == "SPY":
            print(f"[診斷結論]: 屬於【宏觀系統性風險】，大盤 (SPY) 發生超預期脫鉤。")
        else:
            print(f"[診斷結論]: 屬於【板塊結構性破裂】，衝擊主要由 {max_ticker} 外溢，破壞了市場定價慣性。")

    # ============================================================
    # 5. 視覺化：ETF 走勢與異常觸發對照圖 (專題簡報必備)
    # ============================================================
    # 設定字型避免中文出現方塊亂碼
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 8))

    # --- 上半部：ETF 股價走勢與異常標記 ---
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, close_prices["QQQ"].loc[test_dates], label="QQQ (科技板塊)", color="navy", linewidth=1.5)
    plt.plot(test_dates, close_prices["SPY"].loc[test_dates], label="SPY (大盤基準)", color="orange", linewidth=1.5)
    
    # 在股價圖上畫出紅色垂直線，標示崩盤或異常發生的當下
    anomaly_dates = test_dates[anomalies]
    for d in anomaly_dates:
        plt.axvline(d, color="red", alpha=0.3, linestyle="--", linewidth=1)
        
    plt.title("ETF Prices and Detected Anomaly Events (2022-2026)", fontsize=14, fontweight="bold")
    plt.ylabel("Price (USD)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    # --- 下半部：Transformer 重構誤差與閾值 ---
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, test_scores, label="Transformer Anomaly Score (MSE)", color="#1f77b4", alpha=0.8)
    plt.axhline(threshold, linestyle="-", color="crimson", linewidth=2, label=f"Threshold (Train 98th %ile: {threshold:.2f})")
    
    # 標記高於門檻的紅點
    plt.scatter(
        test_dates[anomalies],
        test_scores[anomalies],
        color="red",
        s=40,
        zorder=5,
        label="Confirmed Anomaly"
    )
    plt.title("Reconstruction Error vs. Normalcy Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("etf_anomaly_trend.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
