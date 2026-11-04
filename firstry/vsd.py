import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

# 設定隨機種子以保證可重現性
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 資料獲取與前處理
# ==========================================
def fetch_and_preprocess_data():
    """取得 S&P 500 日線資料並計算報酬率與日內高低振幅特徵"""
    print("正在下載 S&P 500 歷史資料...")
    df = yf.download("^GSPC", start="1980-01-01", end="2026-04-01", auto_adjust=False)
    
    # 處理 yfinance 可能返回的 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    
    # 計算連續複合對數報酬率: r_t = ln(P_t / P_{t-1})
    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # 計算正規化日內高低價振幅 (論文式 75)
    df["h"] = (df["High"] - df["Close"]) / df["Close"]
    df["l"] = (df["Low"] - df["Close"]) / df["Close"]
    
    df = df.dropna().copy()
    
    # 根據論文切分時間點
    train_mask = df.index <= "2007-09-18"
    val_mask = (df.index > "2007-09-18") & (df.index <= "2016-12-19")
    test_mask = df.index > "2016-12-19"
    
    print(f"資料筆數 - 訓練集: {train_mask.sum()}, 驗證集: {val_mask.sum()}, 測試集: {test_mask.sum()}")
    return df, train_mask, val_mask, test_mask


# ==========================================
# 2. GARCH 與 GJR-GARCH QMLE 估計 (From Scratch)
# ==========================================
class GARCHModels:
    @staticmethod
    def fit_garch11(returns_train):
        """對稱 GARCH(1, 1) QMLE 估計"""
        r = returns_train * 100.0  # 縮放以穩定 SLSQP 數值條件
        mu = np.mean(r)
        e = r - mu
        T = len(r)
        var_sample = np.var(r)

        def nll(params):
            omega, alpha, beta = params
            sigma2 = np.zeros(T)
            sigma2[0] = var_sample
            for t in range(1, T):
                sigma2[t] = omega + alpha * (e[t - 1] ** 2) + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-6:
                    sigma2[t] = 1e-6
            return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (e ** 2) / sigma2)

        # 平穩性約束: 0.9999 - alpha - beta >= 0
        cons = ({'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2]})
        bounds = [(1e-6, None), (1e-6, 1.0), (1e-6, 1.0)]
        
        # 多起始點優化
        best_res, best_fun = None, np.inf
        for a_init, b_init in [(0.05, 0.90), (0.03, 0.94), (0.10, 0.85)]:
            w_init = var_sample * (1.0 - a_init - b_init)
            res = opt.minimize(nll, [w_init, a_init, b_init], method='SLSQP', bounds=bounds, constraints=cons)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res

        omega, alpha, beta = best_res.x
        omega /= (100.0 ** 2)  # 還原回原始報酬尺度
        return mu / 100.0, omega, alpha, beta

    @staticmethod
    def fit_gjr_garch11(returns_train):
        """非對稱 GJR-GARCH(1, 1) QMLE 估計"""
        r = returns_train * 100.0
        mu = np.mean(r)
        e = r - mu
        T = len(r)
        var_sample = np.var(r)

        def nll(params):
            omega, alpha, gamma, beta = params
            sigma2 = np.zeros(T)
            sigma2[0] = var_sample
            for t in range(1, T):
                I = 1.0 if e[t - 1] < 0 else 0.0
                sigma2[t] = omega + (alpha + gamma * I) * (e[t - 1] ** 2) + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-6:
                    sigma2[t] = 1e-6
            return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (e ** 2) / sigma2)

        # 平穩性約束: 0.9999 - alpha - beta - gamma/2 >= 0
        cons = ({'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[3] - p[2] / 2.0})
        bounds = [(1e-6, None), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
        
        best_res, best_fun = None, np.inf
        for a_init, g_init, b_init in [(0.04, 0.05, 0.88), (0.05, 0.08, 0.90), (0.06, 0.10, 0.85)]:
            w_init = var_sample * (1.0 - a_init - g_init / 2.0 - b_init)
            res = opt.minimize(nll, [w_init, a_init, g_init, b_init], method='SLSQP', bounds=bounds, constraints=cons)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res

        omega, alpha, gamma, beta = best_res.x
        omega /= (100.0 ** 2)
        return mu / 100.0, omega, alpha, gamma, beta

    @staticmethod
    def extract_features(returns_all, model_type, params):
        """固定訓練參數，向後遞迴外推產生全樣本的條件波動度 sigma 與標準化殘差 z"""
        T = len(returns_all)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns_all[:100])

        if model_type == "GARCH":
            mu, omega, alpha, beta = params
            e = returns_all - mu
            for t in range(1, T):
                sigma2[t] = omega + alpha * (e[t - 1] ** 2) + beta * sigma2[t - 1]
        elif model_type == "GJR":
            mu, omega, alpha, gamma, beta = params
            e = returns_all - mu
            for t in range(1, T):
                I = 1.0 if e[t - 1] < 0 else 0.0
                sigma2[t] = omega + (alpha + gamma * I) * (e[t - 1] ** 2) + beta * sigma2[t - 1]
        
        sigma = np.sqrt(np.maximum(sigma2, 1e-8))
        z = (returns_all - params[0]) / sigma
        return sigma, z

class PositionalEncoding(nn.Module):
    """為 Transformer 提供時間順序資訊的正弦位置編碼"""
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
# ==========================================
# 3. TranAD 模型架構 (針對無界特徵改為 Linear Decoder)
# ==========================================
class TranAD(nn.Module):
    def __init__(self, feats=3, d_model=64, nhead=8):
        super().__init__()
        self.feats = feats
        self.d_model = d_model
        
        # 1. 線性投影層：將特徵維度投影至 d_model (64)，確保可被 nhead (8) 整除
        self.proj1 = nn.Linear(feats * 2, d_model)
        self.proj2 = nn.Linear(feats, d_model)
        self.pe = PositionalEncoding(d_model)

        # 2. Encoder 1: 處理序列 C 拼接 Focus Score (論文式 44-45)
        enc1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.enc1 = nn.TransformerEncoder(enc1_layer, num_layers=1)

        # 3. Encoder 2: 處理視窗 W (因果遮罩自注意力式 48-49 與跨注意力式 50)
        self.self_attn2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 4. 雙解碼器: 依論文式 51 改為 Linear 輸出層
        self.dec1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feats)
        )
        self.dec2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feats)
        )

    def forward(self, w, focus_score=None):
        B, K, _ = w.shape
        if focus_score is None:
            focus_score = torch.zeros_like(w)

        # --- Encoder 1 ---
        x1 = self.pe(self.proj1(torch.cat([w, focus_score], dim=-1)))
        i1_2 = self.enc1(x1)  # [B, K, d_model]

        # --- Encoder 2 (帶因果遮罩防範前視偏差) ---
        x2 = self.pe(self.proj2(w))
        mask = torch.triu(torch.full((K, K), float('-inf'), device=w.device), diagonal=1)
        attn2_out, _ = self.self_attn2(x2, x2, x2, attn_mask=mask)
        i2_2 = self.ln1(x2 + attn2_out)

        # --- Cross-Attention (Q=i2_2, K=i1_2, V=i1_2) ---
        cross_out, _ = self.cross_attn(query=i2_2, key=i1_2, value=i1_2)
        i2_3 = self.ln2(i2_2 + cross_out)
        i2_3 = self.ln3(i2_3 + self.ffn(i2_3))

        # --- 雙解碼器輸出 ---
        o1 = self.dec1(i2_3)  # [B, K, feats]
        o2 = self.dec2(i2_3)  # [B, K, feats]
        return o1, o2

# ==========================================
# 4. VTT 架構 (因果擴張卷積 + 變數/時間雙注意力)
# ==========================================
class DilatedCausalConv(nn.Module):
    def __init__(self, in_channels=1, d_model=16):
        super().__init__()
        # 三組因果擴張卷積 branch (論文式 61: k=(2,3,3), d=(1,2,4))
        self.c1 = nn.Conv1d(in_channels, 1, kernel_size=2, dilation=1)
        self.c2 = nn.Conv1d(in_channels, 1, kernel_size=3, dilation=2)
        self.c3 = nn.Conv1d(in_channels, 1, kernel_size=3, dilation=4)
        self.proj = nn.Linear(3, d_model)

    def forward(self, x):
        # x: [B, N] -> [B, 1, N]
        x = x.unsqueeze(1)
        # 因果 padding: 左側補 (k-1)*d
        p1 = nn.functional.pad(x, (1, 0))
        p2 = nn.functional.pad(x, (4, 0))
        p3 = nn.functional.pad(x, (8, 0))
        h1 = self.c1(p1)
        h2 = self.c2(p2)
        h3 = self.c3(p3)
        h = torch.cat([h1, h2, h3], dim=1).permute(0, 2, 1)  # [B, N, 3]
        return self.proj(h)  # [B, N, d_model]


class VTT(nn.Module):
    def __init__(self, num_vars=5, seq_len=100, d_model=16, num_layers=2):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.conv_embeds = nn.ModuleList([DilatedCausalConv(1, d_model) for _ in range(num_vars)])
        
        # 變數注意力與時間注意力層
        self.var_attn = nn.MultiheadAttention(embed_dim=seq_len * d_model, num_heads=2, batch_first=True)
        self.time_attn = nn.MultiheadAttention(embed_dim=num_vars * d_model, num_heads=2, batch_first=True)
        
        self.ln1 = nn.LayerNorm(seq_len * d_model)
        self.ln2 = nn.LayerNorm(num_vars * d_model)
        self.decoder = nn.Linear(num_vars * d_model, num_vars)

    def forward(self, x):
        # x: [B, N, V]
        B, N, V = x.shape
        embeds = [self.conv_embeds[v](x[:, :, v]) for v in range(V)]
        E = torch.stack(embeds, dim=2)  # [B, N, V, d_model]

        # 1. 變數注意力: tokens 為變數 (V 個)
        X_var = E.permute(0, 2, 1, 3).reshape(B, V, N * self.d_model)
        var_out, _ = self.var_attn(X_var, X_var, X_var)
        X_var = self.ln1(X_var + var_out)

        # 2. 時間注意力: tokens 為時間步 (N 個)
        X_time = X_var.reshape(B, V, N, self.d_model).permute(0, 2, 1, 3).reshape(B, N, V * self.d_model)
        time_out, _ = self.time_attn(X_time, X_time, X_time)
        X_time = self.ln2(X_time + time_out)

        # 重建原始維度
        return self.decoder(X_time)  # [B, N, V]


# ==========================================
# 5. 滑動視窗 Dataset 封裝
# ==========================================
class WindowDataset(Dataset):
    def __init__(self, data, window_size, stride=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.indices = np.arange(0, len(data) - window_size + 1, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        return self.data[start : start + self.window_size]


# ==========================================
# 6. 主評估流程
# ==========================================
def train_and_eval_tranad(features, train_mask, val_mask, test_mask):
    """
    TranAD 訓練與評估 (精準重現版)
    Window = 30, Stride = 1, Features = 3 ([r, sigma, z])
    """
    # 1. 嚴格僅由訓練集提取均值與標準差進行標準化 (論文式 41)
    train_data = features[train_mask]
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8
    features_norm = (features - mean) / std

    # 2. 構建重疊視窗 (K = 30)
    K = 30
    T_total = len(features_norm)
    # 每個視窗 i 對應的時間點為 i + K - 1 (式 43)
    windows = []
    target_dates_idx = []
    for i in range(T_total - K + 1):
        windows.append(features_norm[i : i + K])
        target_dates_idx.append(i + K - 1)
    
    windows = np.array(windows, dtype=np.float32)
    target_dates_idx = np.array(target_dates_idx)

    # 依時間點劃分視窗所屬集合
    win_train = windows[train_mask[target_dates_idx]]
    win_val = windows[val_mask[target_dates_idx]]
    win_test = windows[test_mask[target_dates_idx]]

    # 3. 訓練 DataLoader (Batch Size = 64, 論文設定)
    train_loader = DataLoader(win_train, batch_size=64, shuffle=True, drop_last=False)
    
    model = TranAD(feats=3, d_model=64, nhead=8).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 訓練 40 Epochs
    model.train()
    for epoch in range(1, 41):
        w_factor = 1.0 / epoch
        for w in train_loader:
            w = w.to(DEVICE)
            
            # 【關鍵修正 1】：接住 Decoder 2 的 Phase 1 初始輸出 (o2)
            o1, o2 = model(w)
            
            # Phase 2: 以 Phase 1 重建殘差作為焦點分數 (式 54-55)
            focus = torch.abs(o1 - w)
            _, o2_hat = model(w, focus)
            
            # 演化式對抗損失 (式 57-58)
            loss1 = w_factor * nn.functional.mse_loss(o1, w) + (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)
            
            # 【關鍵修正 2】：第一項必須是 o2！對抗機制才能正常運作
            loss2 = w_factor * nn.functional.mse_loss(o2, w) - (1.0 - w_factor) * nn.functional.mse_loss(o2_hat, w)

            optimizer.zero_grad()
            (loss1 + loss2).backward()
            optimizer.step()
# 4. 推論計算異常分數 (論文式 59: 整個視窗之 Frobenius 範數均值)
    def compute_anomaly_scores(win_array):
        model.eval()
        scores = []
        loader = DataLoader(win_array, batch_size=64, shuffle=False)
        with torch.no_grad():
            for w in loader:
                w = w.to(DEVICE)
                o1, _ = model(w)
                focus = torch.abs(o1 - w)
                _, o2_hat = model(w, focus)
                
                # 【關鍵修正】：論文式 59 的 ||.||_2^2 為 Frobenius 範數平方，必須用 sum() 
                # 視窗內包含 30 天 * 3 個特徵 = 90 個元素，這會將數值尺度還原至論文水平
                err1 = torch.sum((o1 - w) ** 2, dim=(1, 2))
                err2 = torch.sum((o2_hat - w) ** 2, dim=(1, 2))
                score = 0.5 * err1 + 0.5 * err2
                scores.extend(score.cpu().numpy())
        return np.array(scores)

    val_scores = compute_anomaly_scores(win_val)
    test_scores = compute_anomaly_scores(win_test)

    # 5. 門檻設定在驗證集 (2007-2016) 的第 95 百分位數 (式 60)
    threshold = np.percentile(val_scores, 95)
    anomalies = test_scores > threshold

    print(f"  -> 驗證集 95th 門檻值: {threshold:.4f} (論文約 0.25~0.27)")
    print(f"  -> 測試集標記異常數: {np.sum(anomalies)} (論文約 87~91 天)")
    return anomalies, test_scores


def main():
    df, train_mask, val_mask, test_mask = fetch_and_preprocess_data()
    returns = df["return"].values
    
    # 1. 估計兩組波動度模型參數 (僅使用訓練集)
    print("\n--- 步驟 1: 擬合 GARCH 與 GJR-GARCH 參數 ---")
    garch_params = GARCHModels.fit_garch11(returns[train_mask])
    print(f"GARCH(1,1) 估計值: omega={garch_params[1]:.6f}, alpha={garch_params[2]:.4f}, beta={garch_params[3]:.4f}")
    
    gjr_params = GJR_GARCH_fit = GARCHModels.fit_gjr_garch11(returns[train_mask])
    print(f"GJR-GARCH(1,1) 估計值: omega={gjr_params[1]:.6f}, alpha={gjr_params[2]:.4f}, gamma={gjr_params[3]:.4f}, beta={gjr_params[4]:.4f}")

    # 2. 全樣本特徵萃取
    sigma_garch, z_garch = GARCHModels.extract_features(returns, "GARCH", garch_params)
    sigma_gjr, z_gjr = GARCHModels.extract_features(returns, "GJR", gjr_params)

    # 3. 測試 TranAD 跨模型一致性
    print("\n--- 步驟 2: 訓練 TranAD 並進行跨波動度模型比較 ---")
    feat_garch = np.column_stack([returns, sigma_garch, z_garch])
    feat_gjr = np.column_stack([returns, sigma_gjr, z_gjr])

    print("訓練 TranAD (輸入為對稱 GARCH)...")
    anom_garch, _ = train_and_eval_tranad(feat_garch, train_mask, val_mask, test_mask)
    
    print("訓練 TranAD (輸入為非對稱 GJR-GARCH)...")
    anom_gjr, _ = train_and_eval_tranad(feat_gjr, train_mask, val_mask, test_mask)

    # 4. 計算 Jaccard 相似度驗證論文核心結論
    intersection = np.sum(anom_garch & anom_gjr)
    union = np.sum(anom_garch | anom_gjr)
    jaccard = intersection / union if union > 0 else 1.0

    print("\n==========================================")
    print("實證結果驗證 (論文表 4 對應指標):")
    print(f"對稱 GARCH 標記異常日數: {np.sum(anom_garch)}")
    print(f"GJR-GARCH  標記異常日數: {np.sum(anom_gjr)}")
    print(f"兩者偵測重疊天數: {intersection}")
    print(f"Jaccard 相似係數 (Jaccard Index): {jaccard:.4f}")
    print("==========================================")
    if jaccard > 0.85:
        print(">> 成功重現論文核心結論：即便波動度計量規格不同，Transformer 仍展現高度共識 (Jaccard > 0.85)！")


if __name__ == "__main__":
    main()