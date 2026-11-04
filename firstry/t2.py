import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用裝置:", DEVICE)


# ============================================================
# 2. ETF 清單
# ============================================================

ETF_POOL = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "XLK",   # Technology
    "XLF",   # Financial
    "XLV",   # Healthcare
    "XLE",   # Energy
    "XLI",   # Industrial
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLB",   # Materials
]

ETF_LIST = random.sample(ETF_POOL, 9)

print("\n本次隨機選取的 ETF：")
print(ETF_LIST)


# ============================================================
# 3. 從 Yahoo Finance 下載資料 (改為 2y 確保樣本充足)
# ============================================================

print("\n正在下載 ETF 資料...")

data = yf.download(
    ETF_LIST,
    period="10y",  # 由 2mo 改為 2y 避免資料極度不足
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise RuntimeError("無法取得 Yahoo Finance 資料")

print("資料下載完成")


# ============================================================
# 4. 取得 Close Price
# ============================================================

close_data = data["Close"][ETF_LIST].copy()
close_data = close_data.dropna(how="all")
close_data = close_data.ffill()

print("\nClose Price：")
print(close_data.head())


# ============================================================
# 5. 計算每日報酬率
# ============================================================

returns = close_data.pct_change()
returns = returns.dropna()

print("\nDaily Return：")
print(returns.head())


# ============================================================
# 6. 先切分 Train / Test 再做 Standardization (避免 Data Leakage)
# ============================================================

split_idx = int(len(returns) * 0.7)

train_returns = returns.iloc[:split_idx]
test_returns = returns.iloc[split_idx:]

scaler = StandardScaler()
scaled_train = scaler.fit_transform(train_returns.values)
scaled_test = scaler.transform(test_returns.values)

scaled_train_df = pd.DataFrame(
    scaled_train,
    index=train_returns.index,
    columns=train_returns.columns
)

scaled_test_df = pd.DataFrame(
    scaled_test,
    index=test_returns.index,
    columns=test_returns.columns
)


# ============================================================
# 7. 建立時間序列 Sliding Window
# ============================================================

SEQ_LEN = 10


def create_sequences(data_df, seq_len):
    sequences = []
    dates = []
    values = data_df.values
    index = data_df.index

    for i in range(len(values) - seq_len + 1):
        window = values[i:i + seq_len]
        sequences.append(window)
        dates.append(index[i + seq_len - 1])  # 準確對齊到窗口最後一天

    return np.array(sequences), dates


X_train, train_dates = create_sequences(scaled_train_df, SEQ_LEN)
X_test, test_dates = create_sequences(scaled_test_df, SEQ_LEN)

print("\nSequence shape:")
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 8. PyTorch Dataset
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, X_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# ============================================================
# 9. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================================
# 10. TranAD (具備原版雙階段對抗機制)
# ============================================================

class TranAD(nn.Module):
    def __init__(
        self,
        num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.num_features = num_features

        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # 共享 Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder 1 (Phase 1: 粗重構)
        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=num_layers)
        self.output_layer1 = nn.Linear(d_model, num_features)

        # Decoder 2 (Phase 2: 對抗細化)
        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=num_layers)
        self.output_layer2 = nn.Linear(d_model, num_features)

    def forward(self, x):
        embedded = self.input_projection(x)
        embedded = self.positional_encoding(embedded)
        memory = self.encoder(embedded)

        # Phase 1
        decoded1 = self.decoder1(embedded, memory)
        out1 = self.output_layer1(decoded1)

        # Phase 2 (使用 out1 作為先驗輸入進行細化)
        out1_embedded = self.positional_encoding(self.input_projection(out1))
        decoded2 = self.decoder2(out1_embedded, memory)
        out2 = self.output_layer2(decoded2)

        return out1, out2


# ============================================================
# 11. 建立模型與雙優化器 (支援對抗訓練分步更新)
# ============================================================

NUM_FEATURES = len(ETF_LIST)

model = TranAD(
    num_features=NUM_FEATURES,
    d_model=64,
    nhead=4,
    num_layers=2
).to(DEVICE)

# 拆分 Phase 1 與 Phase 2 參數優化器以支援對抗博弈
optimizer1 = torch.optim.Adam(
    list(model.encoder.parameters()) + 
    list(model.decoder1.parameters()) + 
    list(model.output_layer1.parameters()),
    lr=0.001
)
optimizer2 = torch.optim.Adam(
    list(model.decoder2.parameters()) + 
    list(model.output_layer2.parameters()),
    lr=0.001
)

criterion = nn.MSELoss()


# ============================================================
# 12. Training (原版 TranAD 兩階段對抗訓練)
# ============================================================

EPOCHS = 50
print("\n開始訓練 TranAD (Adversarial Training)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, _ in train_loader:
        batch_x = batch_x.to(DEVICE)

        # ---------------- Phase 1: 訓練生成器 (粗重建) ----------------
        optimizer1.zero_grad()
        out1, out2 = model(batch_x)
        
        l1 = criterion(out1, batch_x)
        l2 = criterion(out2, batch_x)
        
        # 隨 epoch 增加，整體權重由 Phase 1 逐漸交接給 Phase 2
        loss1 = (1.0 / (epoch + 1)) * l1 + (1.0 - 1.0 / (epoch + 1)) * l2
        loss1.backward(retain_graph=True)
        optimizer1.step()

        # ---------------- Phase 2: 對抗細化 (判別放大) ----------------
        optimizer2.zero_grad()
        out1, out2 = model(batch_x)
        
        l2 = criterion(out2, batch_x)
        # 對抗懲罰項：拉開 out2 與 out1 的距離，迫使 Decoder 2 挖掘 Decoder 1 遺漏的特徵
        adv_penalty = criterion(out2, out1.detach())
        loss2 = (1.0 / (epoch + 1)) * l2 - (1.0 - 1.0 / (epoch + 1)) * 0.1 * adv_penalty

        loss2.backward()
        optimizer2.step()

        total_loss += loss2.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Adv Loss: {avg_loss:.6f}")


# ============================================================
# 13. 基於「訓練集」計算基準門檻 (杜絕 Data Leakage)
# ============================================================

model.eval()

with torch.no_grad():
    train_x = X_train_tensor.to(DEVICE)
    _, train_out2 = model(train_x)
    # 計算訓練集每個樣本窗口最後一天的重建誤差
    train_errors = (train_out2[:, -1, :] - train_x[:, -1, :]) ** 2
    train_scores = train_errors.mean(dim=-1).cpu().numpy()

# 以純淨訓練資料的 98% 邊界作為絕對門檻
threshold = np.percentile(train_scores, 98)
print(f"\n基準異常門檻 (依訓練集 98% 分位數): {threshold:.6f}")


# ============================================================
# 14. 測試集盲測與分數評定
# ============================================================

with torch.no_grad():
    test_x = X_test_tensor.to(DEVICE)
    _, test_out2 = model(test_x)
    test_errors = (test_out2[:, -1, :] - test_x[:, -1, :]) ** 2
    anomaly_scores = test_errors.mean(dim=-1).cpu().numpy()

# 測試集完全基於歷史訓練標準裁定
anomalies = anomaly_scores > threshold

print(f"測試集總樣本數：{len(anomaly_scores)}")
print(f"檢驗出的真實異常數量：{anomalies.sum()} (佔比: {anomalies.mean()*100:.2f}%)")


# ============================================================
# 15. 建立結果 DataFrame
# ============================================================

result = pd.DataFrame({
    "Date": test_dates,
    "AnomalyScore": anomaly_scores,
    "IsAnomaly": anomalies
})

print("\n偵測到的異常日期：")
print(result[result["IsAnomaly"] == True])


# ============================================================
# 16. 視覺化 Anomaly Score
# ============================================================

plt.figure(figsize=(14, 6))
plt.plot(result["Date"], result["AnomalyScore"], label="TranAD Test Anomaly Score", color="#1f77b4", alpha=0.8)
plt.axhline(threshold, linestyle="--", color="crimson", linewidth=1.5, label=f"Train Threshold (98th %ile: {threshold:.4f})")
plt.scatter(
    result.loc[result["IsAnomaly"], "Date"],
    result.loc[result["IsAnomaly"], "AnomalyScore"],
    color="red",
    s=40,
    zorder=5,
    label="Confirmed Anomaly"
)
plt.title("ETF Cross-Asset Anomaly Detection (TranAD Adversarial)")
plt.xlabel("Date")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()


# ============================================================
# 17. 個別 ETF 貢獻度分析
# ============================================================

# test_errors 的形狀是 (測試樣本數, ETF 數量)，即每個樣本窗口最後一天、
# 每檔 ETF 各自的重建誤差。要得到「每檔 ETF 在整個測試期間的平均異常程度」，
# 只需要對 dim=0（樣本維度）取平均，保留 dim=1（ETF/特徵維度）。
average_feature_error = test_errors.mean(dim=0).cpu().numpy()

feature_result = pd.DataFrame({
    "ETF": returns.columns,
    "Average_Anomaly_Score": average_feature_error
}).sort_values("Average_Anomaly_Score", ascending=False)

print("\n各 ETF 異常程度：")
print(feature_result)

plt.figure(figsize=(12, 6))
plt.bar(feature_result["ETF"], feature_result["Average_Anomaly_Score"])
plt.title("ETF Average Anomaly Contribution")
plt.xlabel("ETF")
plt.ylabel("Average Reconstruction Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 18. 儲存結果
# ============================================================

result.to_csv("tranad_anomaly_results.csv", index=False)
feature_result.to_csv("tranad_etf_scores.csv", index=False)

print("\n結果已儲存：tranad_anomaly_results.csv, tranad_etf_scores.csv")