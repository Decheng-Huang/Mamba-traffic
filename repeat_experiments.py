"""
重复实验脚本 — 每个模型跑3次取平均±标准差
验证结果稳定性，避免"恰好某次种子好"
用法: python repeat_experiments.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, gc, os, warnings, math, json

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {DEVICE}")

DATA_DIR = "数据集"
NUM_SENSORS = 50
NUM_NEIGHBORS = 5
IN_DIM = 1 + NUM_NEIGHBORS  # 6
INPUT_LEN = 12
OUTPUT_LEN = 12
N_SUBSAMPLE = 500
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 5      # 5个epoch足够看稳定性
N_REPEATS = 3

# ========== 数据加载 ==========
print("\n加载数据...")
train_50 = np.load(f"{DATA_DIR}/train_50.npy").astype(np.float32)
val_50 = np.load(f"{DATA_DIR}/val_50.npy").astype(np.float32)
test_50 = np.load(f"{DATA_DIR}/test_50.npy").astype(np.float32)
adj_50 = np.load(f"{DATA_DIR}/adj_50.npy").astype(np.float32)

T_tr = train_50.shape[0] // NUM_SENSORS
T_va = val_50.shape[0] // NUM_SENSORS
T_te = test_50.shape[0] // NUM_SENSORS

def reshape_data(arr, T):
    d = arr.reshape(NUM_SENSORS, T, 24)
    x = d[..., :12]; y = d[..., 12:]
    return x, y

train_x, train_y = reshape_data(train_50, T_tr)
val_x, val_y = reshape_data(val_50, T_va)
test_x, test_y = reshape_data(test_50, T_te)

# 取子集加速
train_x = train_x[:, -N_SUBSAMPLE:]
train_y = train_y[:, -N_SUBSAMPLE:]
print(f'子集: train_x {train_x.shape}')

# 标准化
mean_x = train_x.mean(axis=(1,2), keepdims=True)
std_x = train_x.std(axis=(1,2), keepdims=True) + 1e-8
mean_y = train_y.mean(axis=(1,2), keepdims=True)
std_y = train_y.std(axis=(1,2), keepdims=True) + 1e-8

train_x = (train_x - mean_x) / std_x
train_y = (train_y - mean_y) / std_y
test_x = (test_x - mean_x) / std_x
test_y = (test_y - mean_y) / std_y

# 邻居索引
neighbor_indices = []
for i in range(NUM_SENSORS):
    nbrs = np.argsort(adj_50[i])[::-1]
    nbrs = nbrs[nbrs != i][:NUM_NEIGHBORS]
    if len(nbrs) < NUM_NEIGHBORS:
        nbrs = np.pad(nbrs, (0, NUM_NEIGHBORS-len(nbrs)), 'constant')
    neighbor_indices.append(nbrs)
neighbor_indices = np.array(neighbor_indices)

def make_samples(x_data):
    """x_data: (S, T, 12) → (T, S, 12, 6)"""
    S, T, _ = x_data.shape
    X_list = []
    for t in range(T):
        x_t = x_data[:, t, :]
        x_nbr = x_data[neighbor_indices, t, :]
        x_multi = np.stack([x_t], axis=-1)
        x_multi = np.concatenate([x_multi, x_nbr.transpose(0,2,1)], axis=-1)
        X_list.append(x_multi)
    return np.array(X_list)

X_tr = make_samples(train_x)
Y_tr = train_y.transpose(1,0,2)  # (T, S, 12)
X_te = make_samples(test_x)
Y_te = test_y.transpose(1,0,2)

print(f"X_tr: {X_tr.shape}, Y_tr: {Y_tr.shape}")
del train_50, val_50, test_50, adj_50, train_x, train_y, test_x, test_y; gc.collect()

# ========== 模型定义（简化版，只保留必要的） ==========
def selective_scan_fn(u, delta, A, B, C, D):
    Bs, L, D_in = u.shape; D_state = A.shape[1]
    delta = torch.clamp(delta, -4.0, 4.0)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    x = torch.zeros(Bs, D_in, D_state, device=u.device, dtype=u.dtype)
    ys = torch.zeros(Bs, L, D_in, device=u.device, dtype=u.dtype)
    for i in range(L):
        x = deltaA[:,i] * x + deltaB[:,i] * u[:,i].unsqueeze(-1)
        ys[:,i] = torch.sum(x * C[:,i].unsqueeze(1), dim=-1)
    return ys + D.unsqueeze(0).unsqueeze(0) * u

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand*d_model); self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner*2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state*2+1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A_init = torch.linspace(0.1, 1.0, d_state)
        self.A_log = nn.Parameter(A_init.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)
        for m in [self.in_proj,self.x_proj,self.out_proj,self.dt_proj]:
            if hasattr(m,'weight') and m.weight.dim()>=2:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
            if isinstance(m,nn.Conv1d): nn.init.kaiming_normal_(m.weight)
            if hasattr(m,'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1); nn.init.zeros_(self.dt_proj.bias)
    @property
    def A(self): return torch.exp(torch.clamp(self.A_log, max=0.0))
    def forward(self, x):
        B,L,_=x.shape; xr=self.in_proj(x); xg,res=xr.split(self.d_inner,dim=-1)
        xc=xg.transpose(1,2); xc=self.conv1d(xc)[...,:L].transpose(1,2); xc=F.silu(xc)
        sp=self.x_proj(xc); dt,Bp,Cp=torch.split(sp,[1,self.d_state,self.d_state],dim=-1)
        delta=F.softplus(self.dt_proj(dt))
        y=selective_scan_fn(xc,delta,self.A,Bp,Cp,self.D)
        return self.dropout(self.out_proj(y*F.silu(res)))

class V8Model(nn.Module):
    def __init__(self):
        super().__init__(); self.d_model=128
        self.proj=nn.Linear(IN_DIM,128)
        self.layers=nn.ModuleList([MambaBlock(128,d_state=16) for _ in range(2)])
        self.norm=nn.LayerNorm(128); self.output=nn.Linear(12*128,12)
        for m in [self.proj,self.output]:
            nn.init.xavier_uniform_(m.weight,gain=0.5)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B,N,L,C=x.shape;x=x.permute(0,2,1,3).reshape(B*N,L,C)
        x=self.proj(x)
        for l in self.layers: x=l(x)+x
        x=self.norm(x); return self.output(x.reshape(B*N,L*self.d_model)).reshape(B,N,12)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__(); self.d_model=128
        self.proj=nn.Linear(IN_DIM,128)
        self.lstm=nn.LSTM(128,128,3,batch_first=True,dropout=0.1)
        self.norm=nn.LayerNorm(128)
        self.mlp=nn.Sequential(nn.Linear(12*128,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,12))
        for n,p in self.lstm.named_parameters():
            if 'weight' in n: nn.init.xavier_uniform_(p,gain=0.5)
            elif 'bias' in n: nn.init.zeros_(p)
        for m in [self.proj]+list(self.mlp):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight,gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B,N,L,C=x.shape;x=x.permute(0,2,1,3).reshape(B*N,L,C)
        x=self.proj(x);x,_=self.lstm(x);x=self.norm(x)
        return self.mlp(x.reshape(B*N,L*self.d_model)).reshape(B,N,12)

MODELS = {'V8': V8Model, 'LSTM': LSTMModel}

def train_model(model, X_np, Y_np, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    model.to(DEVICE)
    X = torch.FloatTensor(X_np).to(DEVICE)
    Y = torch.FloatTensor(Y_np).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    S = X.shape[0]; n_bs = (S+BATCH_SIZE-1)//BATCH_SIZE
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(S)
        for bi in range(n_bs):
            idx = perm[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
            opt.zero_grad()
            crit(model(X[idx]), Y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"    ep {ep+1}/{EPOCHS} done", flush=True)

@torch.no_grad()
def evaluate(model, X_np, Y_np):
    model.eval()
    X = torch.FloatTensor(X_np[:10]).to(DEVICE)
    Y = torch.FloatTensor(Y_np[:10]).to(DEVICE)
    pred = model(X)
    true = Y
    mse = nn.MSELoss()(pred, true).item()
    p = pred.cpu().numpy(); t = true.cpu().numpy()
    # 还原到原始尺度
    p_orig = p * std_y.reshape(1,50,1) + mean_y.reshape(1,50,1)
    t_orig = t * std_y.reshape(1,50,1) + mean_y.reshape(1,50,1)
    cov = (p_orig.max()-p_orig.min()) / max(t_orig.max()-t_orig.min(), 1e-10) * 100
    sr = [p_orig[i].std()/max(t_orig[i].std(),1e-10) for i in range(10)]
    return {'mse': mse, 'coverage': cov, 'std_ratio': np.mean(sr)}

# ========== 重复实验 ==========
print(f"\n重复实验 (每个模型 {N_REPEATS}次, 不同seed)")
print("="*50)

all_results = {}
for name, Cls in MODELS.items():
    print(f"\n--- {name} ---")
    seeds = [42, 123, 456]
    metrics = {'mse': [], 'coverage': [], 'std_ratio': [], 'params': 0}

    for i, seed in enumerate(seeds):
        gc.collect(); torch.cuda.empty_cache()
        model = Cls()
        metrics['params'] = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        train_model(model, X_tr, Y_tr, seed=seed)
        elapsed = time.time() - t0
        res = evaluate(model, X_te, Y_te)

        metrics['mse'].append(res['mse'])
        metrics['coverage'].append(res['coverage'])
        metrics['std_ratio'].append(res['std_ratio'])
        print(f"   run {i+1} seed={seed}: MSE={res['mse']:.4f}, 覆盖={res['coverage']:.1f}%, 耗时={elapsed:.0f}s")

    print(f"  → {name} 汇总 (mean±std):")
    print(f"     MSE:     {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    print(f"     覆盖:    {np.mean(metrics['coverage']):.1f}% ± {np.std(metrics['coverage']):.1f}")
    print(f"     方差保留: {np.mean(metrics['std_ratio']):.3f} ± {np.std(metrics['std_ratio']):.3f}")
    all_results[name] = metrics

print("\n" + "="*60)
print("重复实验汇总表 (mean ± std)")
print("="*60)
print(f"{'模型':<10} {'参数量':>8} {'MSE':>14} {'覆盖%':>14} {'方差保留':>12}")
print("-"*60)
for name, m in all_results.items():
    mu_mse = np.mean(m['mse']); sd_mse = np.std(m['mse'])
    mu_cov = np.mean(m['coverage']); sd_cov = np.std(m['coverage'])
    mu_sr = np.mean(m['std_ratio']); sd_sr = np.std(m['std_ratio'])
    print(f"{name:<10} {m['params']:>8,} {mu_mse:.4f}±{sd_mse:.4f}  {mu_cov:.1f}±{sd_cov:.1f}  {mu_sr:.3f}±{sd_sr:.3f}")
print("="*60)

# 保存结果
torch.save(all_results, 'repeat_results.pt')
print("\n结果已保存: repeat_results.pt")
