import numpy as np
import torch
import torch.nn as nn


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


class TranAD(nn.Module):
    """多變量 Transformer (TranAD) 雙階段自條件對抗架構（Linear Decoder 版）"""
    def __init__(self, feats, d_model=64, nhead=8):
        super().__init__()
        self.feats = feats
        self.d_model = d_model
        self.proj1 = nn.Linear(feats * 2, d_model)
        self.proj2 = nn.Linear(feats, d_model)
        self.pe = PositionalEncoding(d_model)

        self.enc1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True),
            num_layers=1,
        )

        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.dec1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feats),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feats),
        )

    def forward(self, w, focus_score=None):
        B, K, _ = w.shape
        if focus_score is None:
            focus_score = torch.zeros_like(w)

        # Phase 1: Context Encoder with Focus Score
        x1 = self.pe(self.proj1(torch.cat([w, focus_score], dim=-1)))
        i1 = self.enc1(x1)

        # Phase 2: Window Encoder with Causal Mask
        x2 = self.pe(self.proj2(w))
        mask = torch.triu(torch.full((K, K), float('-inf'), device=w.device), diagonal=1)
        a2, _ = self.self_attn2(x2, x2, x2, attn_mask=mask)
        i2 = self.ln1(x2 + a2)

        c2, _ = self.cross_attn(query=i2, key=i1, value=i1)
        i2 = self.ln3(self.ln2(i2 + c2) + self.ffn(self.ln2(i2 + c2)))

        return self.dec1(i2), self.dec2(i2)
