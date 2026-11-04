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

# 隨機選擇 9 檔
ETF_LIST = random.sample(ETF_POOL, 9)

print("\n本次隨機選取的 ETF：")
print(ETF_LIST)


# ============================================================
# 3. 從 Yahoo Finance 下載最近兩個月資料
# ============================================================

print("\n正在下載 ETF 資料...")

data = yf.download(
    ETF_LIST,
    period="2mo",
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

# 移除完全沒有資料的日期
close_data = close_data.dropna(how="all")

# 缺失值向前填補
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
# 6. Standardization
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    returns.values
)

scaled_data = pd.DataFrame(
    scaled_data,
    index=returns.index,
    columns=returns.columns
)


# ============================================================
# 7. 建立時間序列 Sliding Window
# ============================================================

SEQ_LEN = 10


def create_sequences(data, seq_len):

    sequences = []

    values = data.values

    for i in range(len(values) - seq_len):

        window = values[i:i + seq_len]

        sequences.append(window)

    return np.array(sequences)


X = create_sequences(
    scaled_data,
    SEQ_LEN
)

print("\nSequence shape:")
print(X.shape)

# Shape:
# [樣本數, 時間長度, ETF數量]


# ============================================================
# 8. Train / Test Split
# ============================================================

train_size = int(len(X) * 0.7)

X_train = X[:train_size]
X_test = X[train_size:]

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 9. PyTorch Dataset
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    X_train_tensor,
    X_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# ============================================================
# 10. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            *
            (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# 11. Simplified TranAD
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

        # Feature embedding
        self.input_projection = nn.Linear(
            num_features,
            d_model
        )

        # Position encoding
        self.positional_encoding = PositionalEncoding(
            d_model
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Reconstruction layer
        self.output_layer = nn.Linear(
            d_model,
            num_features
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        embedded = self.input_projection(x)

        embedded = self.positional_encoding(
            embedded
        )

        memory = self.encoder(
            embedded
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        decoded = self.decoder(
            embedded,
            memory
        )

        output = self.output_layer(
            decoded
        )

        return output


# ============================================================
# 12. 建立模型
# ============================================================

NUM_FEATURES = len(ETF_LIST)

model = TranAD(
    num_features=NUM_FEATURES,
    d_model=64,
    nhead=4,
    num_layers=2
).to(DEVICE)

print("\nTranAD Model:")
print(model)


# ============================================================
# 13. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()


# ============================================================
# 14. Training
# ============================================================

EPOCHS = 50

print("\n開始訓練 TranAD...")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_x, _ in train_loader:

        batch_x = batch_x.to(DEVICE)

        optimizer.zero_grad()

        reconstruction = model(
            batch_x
        )

        loss = criterion(
            reconstruction,
            batch_x
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.6f}"
        )


# ============================================================
# 15. Anomaly Detection
# ============================================================

model.eval()

X_test_tensor = X_test_tensor.to(DEVICE)

with torch.no_grad():

    reconstructed = model(
        X_test_tensor
    )

# 每個時間點、每個 ETF 的 squared error
errors = (
    reconstructed - X_test_tensor
) ** 2

# 對 ETF 維度取平均
time_errors = errors.mean(
    dim=2
)

# 對 sequence window 取平均
anomaly_scores = time_errors.mean(
    dim=1
)

anomaly_scores = (
    anomaly_scores
    .cpu()
    .numpy()
)


# ============================================================
# 16. 設定異常門檻
# ============================================================

threshold = np.percentile(
    anomaly_scores,
    95
)

print("\nAnomaly Threshold:")
print(threshold)


anomalies = (
    anomaly_scores > threshold
)


print("\n異常數量：")
print(anomalies.sum())


# ============================================================
# 17. 找出對應日期
# ============================================================

test_start_index = (
    train_size + SEQ_LEN
)

test_dates = returns.index[
    test_start_index:
]


# 保證長度一致
min_len = min(
    len(test_dates),
    len(anomaly_scores)
)

test_dates = test_dates[:min_len]

anomaly_scores = anomaly_scores[:min_len]

anomalies = anomalies[:min_len]


# ============================================================
# 18. 建立結果 DataFrame
# ============================================================

result = pd.DataFrame({

    "Date": test_dates,

    "AnomalyScore":
        anomaly_scores,

    "IsAnomaly":
        anomalies

})

print("\n異常結果：")

print(
    result[
        result["IsAnomaly"] == True
    ]
)


# ============================================================
# 19. 找出最異常的日期
# ============================================================

top_anomalies = result.sort_values(
    "AnomalyScore",
    ascending=False
).head(10)

print("\nTop 10 Anomalies")
print(top_anomalies)


# ============================================================
# 20. 畫出 Anomaly Score
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    result["Date"],
    result["AnomalyScore"],
    label="TranAD Anomaly Score"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="95% Threshold"
)

plt.scatter(
    result.loc[
        result["IsAnomaly"],
        "Date"
    ],

    result.loc[
        result["IsAnomaly"],
        "AnomalyScore"
    ],

    label="Anomaly"
)

plt.title(
    "ETF TranAD Anomaly Detection"
)

plt.xlabel("Date")

plt.ylabel(
    "Anomaly Score"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 21. 個別 ETF 的異常貢獻分析
# ============================================================

# 使用測試資料重新計算
with torch.no_grad():

    feature_errors = (
        reconstructed - X_test_tensor
    ) ** 2

# 每個 ETF 在每個 window 的平均 error
feature_scores = feature_errors.mean(
    dim=1
)

feature_scores = (
    feature_scores
    .cpu()
    .numpy()
)

# 每個 ETF 的平均 anomaly contribution
average_feature_error = feature_scores.mean(
    axis=0
)

feature_result = pd.DataFrame({

    "ETF": returns.columns,

    "Average_Anomaly_Score":
        average_feature_error

})

feature_result = feature_result.sort_values(
    "Average_Anomaly_Score",
    ascending=False
)

print("\n各 ETF 異常程度：")

print(feature_result)


# ============================================================
# 22. 畫出 ETF 異常程度
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    feature_result["ETF"],
    feature_result["Average_Anomaly_Score"]
)

plt.title(
    "ETF Average Anomaly Contribution"
)

plt.xlabel("ETF")

plt.ylabel(
    "Average Reconstruction Error"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# ============================================================
# 23. 儲存結果
# ============================================================

result.to_csv(
    "tranad_anomaly_results.csv",
    index=False
)

feature_result.to_csv(
    "tranad_etf_scores.csv",
    index=False
)

print(
    "\n結果已儲存："
)

print(
    "tranad_anomaly_results.csv"
)

print(
    "tranad_etf_scores.csv"
)