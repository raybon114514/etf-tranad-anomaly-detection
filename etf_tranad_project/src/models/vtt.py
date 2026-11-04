import torch
import torch.nn as nn


class DilatedCausalConv(nn.Module):
    """三分支因果擴張卷積特徵提取器 (k=(2,3,3), d=(1,2,4))"""
    def __init__(self, in_channels=1, d_model=16):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels, 1, kernel_size=2, dilation=1)
        self.c2 = nn.Conv1d(in_channels, 1, kernel_size=3, dilation=2)
        self.c3 = nn.Conv1d(in_channels, 1, kernel_size=3, dilation=4)
        self.proj = nn.Linear(3, d_model)

    def forward(self, x):
        x = x.unsqueeze(1)
        p1 = nn.functional.pad(x, (1, 0))
        p2 = nn.functional.pad(x, (4, 0))
        p3 = nn.functional.pad(x, (8, 0))
        h1 = self.c1(p1)
        h2 = self.c2(p2)
        h3 = self.c3(p3)
        h = torch.cat([h1, h2, h3], dim=1).permute(0, 2, 1)
        return self.proj(h)


class VTT(nn.Module):
    """Variable-Temporal Transformer (因果擴張卷積 + 變數/時間雙注意力)"""
    def __init__(self, num_vars=5, seq_len=100, d_model=16, num_layers=2):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model

        self.conv_embeds = nn.ModuleList([DilatedCausalConv(1, d_model) for _ in range(num_vars)])
        self.var_attn = nn.MultiheadAttention(embed_dim=seq_len * d_model, num_heads=2, batch_first=True)
        self.time_attn = nn.MultiheadAttention(embed_dim=num_vars * d_model, num_heads=2, batch_first=True)

        self.ln1 = nn.LayerNorm(seq_len * d_model)
        self.ln2 = nn.LayerNorm(num_vars * d_model)
        self.decoder = nn.Linear(num_vars * d_model, num_vars)

    def forward(self, x):
        B, N, V = x.shape
        embeds = [self.conv_embeds[v](x[:, :, v]) for v in range(V)]
        E = torch.stack(embeds, dim=2)

        X_var = E.permute(0, 2, 1, 3).reshape(B, V, N * self.d_model)
        var_out, _ = self.var_attn(X_var, X_var, X_var)
        X_var = self.ln1(X_var + var_out)

        X_time = X_var.reshape(B, V, N, self.d_model).permute(0, 2, 1, 3).reshape(B, N, V * self.d_model)
        time_out, _ = self.time_attn(X_time, X_time, X_time)
        X_time = self.ln2(X_time + time_out)

        return self.decoder(X_time)
