# -*- coding: utf-8 -*-
"""计算所有模型的标准指标：RMSE, MAE, MAPE, R² + 已有指标"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, gc, os, warnings, math
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

DATA_DIR = "数据集"
NUM_SENSORS = 50
NUM_NEIGHBORS = 5
IN_DIM = 1 + NUM_NEIGHBORS

# ===== 加载测试数据 =====
print('\n加载测试数据...')
test_50 = np.load(f"{DATA_DIR}/test_50.npy").astype(np.float32)
adj_50 = np.load(f"{DATA_DIR}/adj_50.npy").astype(np.float32)
T_te = test_50.shape[0] // NUM_SENSORS
test_data = test_50.reshape(NUM_SENSORS, T_te, 24)
test_x = test_data[..., :12]
test_y = test_data[..., 12:]

# 用训练集统计量标准化
train_50 = np.load(f"{DATA_DIR}/train_50.npy").astype(np.float32)
T_tr = train_50.shape[0] // NUM_SENSORS
train_data = train_50.reshape(NUM_SENSORS, T_tr, 24)
train_x = train_data[..., :12]
train_y = train_data[..., 12:]

# 取最后3000步的统计量（与Kaggle实验一致）
train_x_subset = train_x[:, -3000:]
train_y_subset = train_y[:, -3000:]
mean_y = train_y_subset.mean(axis=(1,2), keepdims=True)
std_y = train_y_subset.std(axis=(1,2), keepdims=True) + 1e-8
mean_x = train_x_subset.mean(axis=(1,2), keepdims=True)
std_x = train_x_subset.std(axis=(1,2), keepdims=True) + 1e-8

test_x_norm = (test_x - mean_x) / std_x
test_y_norm = (test_y - mean_y) / std_y

# 邻居索引
neighbor_indices = []
for i in range(NUM_SENSORS):
    nbrs = np.argsort(adj_50[i])[::-1]
    nbrs = nbrs[nbrs != i][:NUM_NEIGHBORS]
    if len(nbrs) < NUM_NEIGHBORS:
        nbrs = np.pad(nbrs, (0, NUM_NEIGHBORS-len(nbrs)), 'constant')
    neighbor_indices.append(nbrs)
neighbor_indices = np.array(neighbor_indices, dtype=np.int64)

def make_samples(xd, yd):
    S, T, _ = xd.shape
    Xl, Yl = [], []
    for t in range(T):
        xt = xd[:, t, :]
        xnb = xd[neighbor_indices, t, :]
        xm = np.stack([xt], axis=-1)
        xm = np.concatenate([xm, xnb.transpose(0,2,1)], axis=-1)
        Xl.append(xm); Yl.append(yd[:, t, :])
    return np.array(Xl), np.array(Yl)

# 只用前500个时间步（全量推理太慢）
N_EVAL = 500
BATCH_T = 32  # 每次推理32个时间步
X_te, Y_te = make_samples(test_x_norm[:, -N_EVAL:], test_y_norm[:, -N_EVAL:])
print(f'X_te: {X_te.shape}, Y_te: {Y_te.shape}')

# ===== 模型定义（与V10 Kaggle完全一致） =====
def selective_scan_fn(u, delta, A, B, C, D):
    Bs,L,D_in=u.shape; Ds=A.shape[1]
    delta=torch.clamp(delta,-4.0,4.0)
    deltaA=torch.exp(delta.unsqueeze(-1)*A)
    deltaB=delta.unsqueeze(-1)*B.unsqueeze(2)
    ue=u.unsqueeze(-1)
    x=torch.zeros(Bs,D_in,Ds,device=u.device,dtype=u.dtype)
    ys=torch.zeros(Bs,L,D_in,device=u.device,dtype=u.dtype)
    for i in range(L):
        x=deltaA[:,i]*x+deltaB[:,i]*ue[:,i]
        x=torch.clamp(x,-100.0,100.0)
        ys[:,i]=torch.sum(x*C[:,i].unsqueeze(1),dim=-1)
    return ys+D.unsqueeze(0).unsqueeze(0)*u

class MambaBlock(nn.Module):
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2):
        super().__init__()
        self.d_inner=int(expand*d_model); self.d_state=d_state
        self.in_proj=nn.Linear(d_model,self.d_inner*2,bias=False)
        self.conv1d=nn.Conv1d(self.d_inner,self.d_inner,d_conv,padding=d_conv-1,groups=self.d_inner,bias=True)
        self.x_proj=nn.Linear(self.d_inner,d_state*2+1,bias=False)
        self.dt_proj=nn.Linear(1,self.d_inner,bias=True)
        A_init=torch.linspace(0.1,1.0,d_state)
        self.A_log=nn.Parameter(A_init.log().repeat(self.d_inner,1))
        self.D=nn.Parameter(torch.ones(self.d_inner))
        self.out_proj=nn.Linear(self.d_inner,d_model,bias=False)
        self.dropout=nn.Dropout(0.1)
        for m in [self.in_proj,self.x_proj,self.out_proj,self.dt_proj]:
            if hasattr(m,'weight') and m.weight.dim()>=2: nn.init.xavier_uniform_(m.weight,gain=0.5)
            if isinstance(m,nn.Conv1d): nn.init.kaiming_normal_(m.weight)
            if hasattr(m,'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.uniform_(self.dt_proj.weight,-0.1,0.1); nn.init.zeros_(self.dt_proj.bias)
    @property
    def A(self): return torch.exp(torch.clamp(self.A_log,max=0.0))
    def forward(self,x):
        B,L,_=x.shape; xr=self.in_proj(x); xg,res=xr.split(self.d_inner,dim=-1)
        xc=xg.transpose(1,2); xc=self.conv1d(xc)[...,:L].transpose(1,2); xc=F.silu(xc)
        sp=self.x_proj(xc); dt,Bp,Cp=torch.split(sp,[1,self.d_state,self.d_state],dim=-1)
        delta=F.softplus(self.dt_proj(dt))
        y=selective_scan_fn(xc,delta,self.A,Bp,Cp,self.D)
        return self.dropout(self.out_proj(y*F.silu(res)))

class V8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.proj = nn.Linear(IN_DIM, 128)
        self.layers = nn.ModuleList([MambaBlock(128, d_state=16) for _ in range(2)])
        self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(12*128, 12)
        for m in [self.proj, self.output]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,2,1,3).reshape(B*N, L, C)
        x = self.proj(x)
        for l in self.layers: x = l(x) + x
        x = self.norm(x)
        return self.output(x.reshape(B*N, L*self.d_model)).reshape(B, N, 12)

class V9Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.proj = nn.Linear(IN_DIM, 128)
        self.layers = nn.ModuleList([MambaBlock(128, d_state=32) for _ in range(3)])
        self.norm = nn.LayerNorm(128)
        self.output = nn.Sequential(nn.Linear(12*128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        for m in self.output:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,2,1,3).reshape(B*N, L, C)
        x = self.proj(x)
        for l in self.layers: x = l(x) + x
        x = self.norm(x)
        return self.output(x.reshape(B*N, L*self.d_model)).reshape(B, N, 12)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.proj = nn.Linear(IN_DIM, 128)
        self.lstm = nn.LSTM(128, 128, 3, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(12*128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))
        for n, p in self.lstm.named_parameters():
            if 'weight' in n: nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in n: nn.init.zeros_(p)
        for m in [self.proj] + list(self.mlp):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,2,1,3).reshape(B*N, L, C)
        x = self.proj(x)
        x, _ = self.lstm(x)
        x = self.norm(x)
        return self.mlp(x.reshape(B*N, L*self.d_model)).reshape(B, N, 12)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.proj = nn.Linear(IN_DIM, 128)
        pe = torch.zeros(1, 100, 128)
        pos = torch.arange(0, 100).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000)/128))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        enc = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True, activation='relu')
        self.tf = nn.TransformerEncoder(enc, 3)
        self.norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(12*128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,2,1,3).reshape(B*N, L, C)
        x = self.proj(x) + self.pe[:, :L]
        x = self.tf(x)
        x = self.norm(x)
        return self.mlp(x.reshape(B*N, L*self.d_model)).reshape(B, N, 12)

# ===== 加载权重 & 评估 =====
MODELS = {
    'V8 (Mamba)': (V8Model, 'v8_v10.pt', '#1f77b4'),
    'V9 (Mamba++)': (V9Model, 'v9_v10.pt', '#d62728'),
    'LSTM': (LSTMModel, 'lstm_v10.pt', '#2ca02c'),
    'Transformer': (TransformerModel, 'transformer_v10.pt', '#ff7f0e'),
}

print(f'\n{"="*85}')
header = f'{"模型":<14}{"MSE":>8}{"RMSE":>8}{"MAE":>8}{"MAPE%":>8}{"R²":>8}{"覆盖%":>8}{"方差保留":>8}'
print(header)
print('-'*85)

def evaluate_batched(model, X_np, Y_np, batch_t=32):
    """分批推理（Mamba显存需求大，需要小batch）"""
    model.eval()
    T = X_np.shape[0]
    all_p, all_t = [], []
    for t0 in range(0, T, batch_t):
        t1 = min(t0 + batch_t, T)
        X_b = torch.FloatTensor(X_np[t0:t1]).to(DEVICE)
        Y_b = torch.FloatTensor(Y_np[t0:t1]).to(DEVICE)
        with torch.no_grad():
            p = model(X_b)
        all_p.append(p.cpu().numpy())
        all_t.append(Y_b.cpu().numpy())

    p = np.concatenate(all_p, axis=0)
    t = np.concatenate(all_t, axis=0)
    p_orig = p * std_y.reshape(1,50,1) + mean_y.reshape(1,50,1)
    t_orig = t * std_y.reshape(1,50,1) + mean_y.reshape(1,50,1)

    mse = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(np.mean((p_orig - t_orig) ** 2)))
    mae = float(np.mean(np.abs(p_orig - t_orig)))
    mask = t_orig > 1.0
    mape = float(np.mean(np.abs(p_orig[mask] - t_orig[mask]) / t_orig[mask]) * 100)
    ss_res = np.sum((p_orig - t_orig) ** 2)
    ss_tot = np.sum((t_orig - np.mean(t_orig)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    pr = (p_orig.min(), p_orig.max())
    tr = (t_orig.min(), t_orig.max())
    cov = (pr[1]-pr[0]) / max(tr[1]-tr[0], 1e-10) * 100
    sr = float(np.mean([p_orig[i].std() / max(t_orig[i].std(), 1e-10) for i in range(min(50, p_orig.shape[0]))]))
    return mse, rmse, mae, mape, r2, cov, sr

for name, (Cls, ckpt, _) in MODELS.items():
    gc.collect(); torch.cuda.empty_cache()
    try:
        model = Cls().to(DEVICE)
        sd = torch.load(f'results (3)/{ckpt}', map_location=DEVICE)
        model.load_state_dict(sd)

        # Mamba显存大，用小batch；LSTM/Transformer用大步长
        bt = 16 if 'Mamba' in name else 64
        mse, rmse, mae, mape, r2, cov, sr = evaluate_batched(model, X_te, Y_te, batch_t=bt)

        row = f'{name:<14}{mse:>8.4f}{rmse:>8.4f}{mae:>8.4f}{mape:>7.2f}%{r2:>8.4f}{cov:>7.1f}%{sr:>8.3f}'
        print(row)
    except Exception as e:
        print(f'{name:<14} 错误: {str(e)[:60]}')

print('='*85)
print(f'\n注：MAPE基于原始尺度(veh/5min)，过滤了t<1的样本避免除零')
print(f'R²基于原始尺度计算')
