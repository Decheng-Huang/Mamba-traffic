"""
Baseline模型定义：LSTM + Transformer
与V9共用同一套数据pipeline和输出头（MLP），公平对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== Positional Encoding（Transformer用） ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]


# ==================== LSTM Baseline ====================

class LSTMPredictor(nn.Module):
    """
    LSTM baseline，与V9结构对齐：
    - 输入: d0+d1 双特征
    - 输出: MLP(768→256→12) 同V9
    - 层数/维度: 与V9一致（3层，d_model=64）
    """
    def __init__(self, num_sensors, input_len=12, output_len=12,
                 d_model=64, n_layers=3, num_neighbors=0):
        super().__init__()
        self.output_len = output_len
        self.input_proj = nn.Linear(2 + num_neighbors, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(d_model)
        # 输出MLP（同V9）
        self.output_mlp = nn.Sequential(
            nn.Linear(input_len * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_len),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for m in [self.input_proj] + list(self.output_mlp):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, N, L, C)
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, L, C)
        x = self.input_proj(x)
        x, _ = self.lstm(x)             # (B*N, L, d_model)
        x = self.norm(x)
        x_flat = x.reshape(B * N, L * self.lstm.hidden_size)
        out = self.output_mlp(x_flat)
        return out.reshape(B, N, self.output_len)


# ==================== Transformer Baseline ====================

class TransformerPredictor(nn.Module):
    """
    Transformer baseline
    - 同V9: 3层TransformerEncoder, d_model=64, nhead=4
    - 输出MLP(768→256→12)
    """
    def __init__(self, num_sensors, input_len=12, output_len=12,
                 d_model=64, n_layers=3, nhead=4, num_neighbors=0,
                 dim_feedforward=256):
        super().__init__()
        self.output_len = output_len
        self.input_proj = nn.Linear(2 + num_neighbors, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len)

        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # 输出MLP（同V9）
        self.output_mlp = nn.Sequential(
            nn.Linear(input_len * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_len),
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.input_proj] + list(self.output_mlp):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, N, L, C)
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, L, C)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)          # (B*N, L, d_model)
        x = self.norm(x)
        x_flat = x.reshape(B * N, L * self.d_model)
        out = self.output_mlp(x_flat)
        return out.reshape(B, N, self.output_len)
