# ============================================================================
#  阶段1：官方 mamba-ssm vs 手写 Python 循环扫描 —— 对照实验
#
#  目的：回答"你的 Mamba 是不是真 Mamba、到底慢不慢"
#        用官方 CUDA 核重跑 V8/V9，与 kaggle_ffff12.py 的原实现对比速度与精度
#
#  运行：Kaggle（Linux + T4）新建 notebook
#       第1个 cell：  !pip install -q mamba-ssm causal-conv1d
#       第2个 cell：  把本文件内容整段粘进去
#
#  本地 Windows 跑不了：装 mamba-ssm 需要 nvcc + MSVC，本机都没有
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, gc, os, warnings, json

warnings.filterwarnings('ignore')
np.random.seed(42); torch.manual_seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}, 显存: {p.total_memory/1024**3:.1f}GB')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')

try:
    import mamba_ssm
    from mamba_ssm import Mamba
    print(f'mamba_ssm: {getattr(mamba_ssm, "__version__", "已加载")}')
except Exception as e:
    print(f'[失败] mamba_ssm 导入错误: {type(e).__name__}: {e}')
    print('  检查：pip install -q mamba-ssm causal-conv1d')
    raise

# ============================================================================
#  ⚠️ 替换时最关键的一点
#
#  官方 mamba_ssm.Mamba 内部 = RMSNorm → MambaBlock → 残差相加
#  你的 MambaBlock 只有 SSM 部分，所以原代码写 `x = l(x) + x` 手动加残差。
#
#  换成官方模块后必须写成 `x = l(x)`，否则残差加两次。
#  这是本文件与你原脚本唯一的实质差别。
#
#  另一处差异（初始化）：你的 A 初始化在 [0.1, 1.0]，
#  官方在 [1, d_state]。这是真实的架构差异，也是对比的意义之一。
# ============================================================================


# ========== 数据（与 kaggle_ffff12.py 完全一致） ==========
candidate_paths = [
    '/kaggle/input/datasets/huangdecheng/ffff12/',
    '/kaggle/input/datasets/huangdecheng/ffff12/ffff12/',
    '/kaggle/input/ffff12/',
]
DATA_DIR = None
for p in candidate_paths:
    if os.path.exists(p) and 'train_50.npy' in os.listdir(p):
        DATA_DIR = p; print(f'\n找到数据集: {p}'); break
if DATA_DIR is None:
    raise FileNotFoundError('FFFF12数据集未找到，请确认已添加为输入')

NUM_SENSORS = 50
NUM_NEIGHBORS = 5
IN_DIM = 1 + NUM_NEIGHBORS      # 6
BATCH_SIZE = 32
LR = 5e-4
D_MODEL = 128                   # ← 注意：你原代码是 128，不是 64

print('\n[1/4] 加载数据...')
t0 = time.time()


def load_split(fname):
    """
    读取 (24, 50*T) 布局的数据文件，返回 (50, T, 24)。

    实测确认：文件按 t*50+s 展平（时间优先），不是 s*T+t。
    判据是滑动窗口的相邻重叠必须为 0：
        x_t[1:12] == x_{t+1}[0:11]
    用错朝向不会报错，只会静默算出垃圾结果，所以这里强制自检。
    """
    a = np.load(os.path.join(DATA_DIR, fname)).astype(np.float32)
    assert a.ndim == 2 and a.shape[0] == 24, f'{fname} 形状异常: {a.shape}'
    T = a.shape[1] // NUM_SENSORS
    d = a.T.reshape(T, NUM_SENSORS, 24).transpose(1, 0, 2)   # → (50, T, 24)

    ov = np.abs(d[:, :-1, 1:12] - d[:, 1:, 0:11]).mean()
    if ov > 1e-3:
        raise ValueError(
            f'{fname} 布局自检失败：相邻窗口重叠误差={ov:.4f}（应为 0）。\n'
            f'  文件朝向可能又变了，先别继续跑，检查数据格式。')
    print(f'  {fname:14s} → {d.shape}  自检通过(重叠误差={ov:.2e})')
    return d


train_data = load_split('train_50.npy')
val_data   = load_split('val_50.npy')
test_data  = load_split('test_50.npy')
adj_mx     = np.load(os.path.join(DATA_DIR, 'adj_50.npy')).astype(np.float32)

T_tr, T_va, T_te = train_data.shape[1], val_data.shape[1], test_data.shape[1]

train_x, train_y = train_data[..., :12], train_data[..., 12:]
val_x,   val_y   = val_data[..., :12],   val_data[..., 12:]
test_x,  test_y  = test_data[..., :12],  test_data[..., 12:]

mean_x = train_x.mean(axis=(1, 2), keepdims=True)
std_x  = train_x.std(axis=(1, 2), keepdims=True) + 1e-8
mean_y = train_y.mean(axis=(1, 2), keepdims=True)
std_y  = train_y.std(axis=(1, 2), keepdims=True) + 1e-8

train_x = (train_x - mean_x) / std_x;  train_y = (train_y - mean_y) / std_y
val_x   = (val_x - mean_x) / std_x;    val_y   = (val_y - mean_y) / std_y
test_x  = (test_x - mean_x) / std_x;   test_y  = (test_y - mean_y) / std_y

neighbor_indices = []
for i in range(NUM_SENSORS):
    nbrs = np.argsort(adj_mx[i])[::-1]
    nbrs = nbrs[nbrs != i][:NUM_NEIGHBORS]
    if len(nbrs) < NUM_NEIGHBORS:
        nbrs = np.pad(nbrs, (0, NUM_NEIGHBORS - len(nbrs)), 'constant')
    neighbor_indices.append(nbrs)
neighbor_indices = np.array(neighbor_indices, dtype=np.int64)


def make_samples(x_data, y_data):
    S, T, _ = x_data.shape
    X_list, Y_list = [], []
    for t in range(T):
        x_t = x_data[:, t, :]
        y_t = y_data[:, t, :]
        x_nbr = x_data[neighbor_indices, t, :]          # (S, 5, 12)
        x_multi = np.stack([x_t], axis=-1)              # (S, 12, 1)
        x_multi = np.concatenate([x_multi, x_nbr.transpose(0, 2, 1)], axis=-1)
        X_list.append(x_multi); Y_list.append(y_t)
    return np.array(X_list), np.array(Y_list)


X_tr, Y_tr = make_samples(train_x, train_y)
X_va, Y_va = make_samples(val_x, val_y)
X_te, Y_te = make_samples(test_x, test_y)
X_tr_t = torch.FloatTensor(X_tr); Y_tr_t = torch.FloatTensor(Y_tr)
X_te_t = torch.FloatTensor(X_te); Y_te_t = torch.FloatTensor(Y_te)

# 阶段1 只需速度对照。速度是 per-step 属性，与样本总量无关，
# 所以用子集跑，把每 epoch 从 1140 步降到 ~157 步，省 7 倍时间。
N_SUB = int(os.environ.get('MAMBA_N_SUB', 5000))
if N_SUB and N_SUB < X_tr_t.shape[0]:
    X_tr_t, Y_tr_t = X_tr_t[:N_SUB], Y_tr_t[:N_SUB]
    print(f'  [子集模式] 训练样本 {N_SUB}，每 epoch ~{(N_SUB+BATCH_SIZE-1)//BATCH_SIZE} 步')
    print(f'  (要跑全量，设 MAMBA_N_SUB=0)')
del adj_mx; gc.collect()
print(f'X_tr: {X_tr_t.shape}, Y_tr: {Y_tr_t.shape}  ({time.time()-t0:.1f}s)')


# ========== 原实现：手写 Python 循环扫描 ==========
def selective_scan_fn(u, delta, A, B, C, D):
    """与 kaggle_ffff12.py 第149-159行逐字一致"""
    Bs, L, D_in = u.shape; D_state = A.shape[1]
    delta = torch.clamp(delta, -4.0, 4.0)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    x = torch.zeros(Bs, D_in, D_state, device=u.device, dtype=u.dtype)
    ys = torch.zeros(Bs, L, D_in, device=u.device, dtype=u.dtype)
    for i in range(L):                                  # ← 瓶颈在这
        x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
        ys[:, i] = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
    return ys + D.unsqueeze(0).unsqueeze(0) * u


class MambaBlockNaive(nn.Module):
    """你原来的手写 MambaBlock（逐字保留，用于对照）"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model); self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A_init = torch.linspace(0.1, 1.0, d_state)
        self.A_log = nn.Parameter(A_init.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.in_proj, self.x_proj, self.out_proj, self.dt_proj]:
            if hasattr(m, 'weight') and m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.dt_proj.bias)

    @property
    def A(self):
        return torch.exp(torch.clamp(self.A_log, max=0.0))

    def forward(self, x):
        B, L, _ = x.shape
        xr = self.in_proj(x); xg, res = xr.split(self.d_inner, dim=-1)
        xc = xg.transpose(1, 2); xc = self.conv1d(xc)[..., :L].transpose(1, 2)
        xc = F.silu(xc); sp = self.x_proj(xc)
        dt, Bp, Cp = torch.split(sp, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan_fn(xc, delta, self.A, Bp, Cp, self.D)
        return self.dropout(self.out_proj(y * F.silu(res)))


# ========== 新实现：官方 mamba-ssm ==========
class MambaBlockOfficial(nn.Module):
    """
    直接包一层官方 Mamba。
    官方内部已含 RMSNorm + MambaBlock + 残差，
    所以外层 forward 里写 `x = l(x)` 即可，与 V8/V9 原结构对应。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x)


# ========== 统一的 V8 / V9 骨架 ==========
class V8Model(nn.Module):
    """2层 / d_state=16 / 线性输出 —— 与 kaggle_ffff12.py 的 V8Model 结构一致"""
    def __init__(self, block_cls=MambaBlockNaive):
        super().__init__()
        self.d_model = D_MODEL
        self.proj = nn.Linear(IN_DIM, D_MODEL)
        self.layers = nn.ModuleList([block_cls(D_MODEL, d_state=16) for _ in range(2)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(12 * D_MODEL, 12)
        for m in [self.proj, self.output]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
        x = self.proj(x)
        for l in self.layers:
            x = l(x) + x                      # 手写块：需手动加残差
        x = self.norm(x)
        return self.output(x.reshape(B * N, L * self.d_model)).reshape(B, N, 12)


class V9Model(nn.Module):
    """3层 / d_state=32 / MLP输出 —— 与 kaggle_ffff12.py 的 V9Model 结构一致"""
    def __init__(self, block_cls=MambaBlockNaive):
        super().__init__()
        self.d_model = D_MODEL
        self.proj = nn.Linear(IN_DIM, D_MODEL)
        self.layers = nn.ModuleList([block_cls(D_MODEL, d_state=32) for _ in range(3)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Sequential(
            nn.Linear(12 * D_MODEL, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        for m in self.output:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
        x = self.proj(x)
        for l in self.layers:
            x = l(x) + x
        x = self.norm(x)
        return self.output(x.reshape(B * N, L * self.d_model)).reshape(B, N, 12)


class V8Official(V8Model):
    """官方版：残差已含在 Mamba 内部，不再手动 +x"""
    def __init__(self):
        super().__init__(block_cls=MambaBlockOfficial)

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
        x = self.proj(x)
        for l in self.layers:
            x = l(x)                          # ← 无手动残差
        x = self.norm(x)
        return self.output(x.reshape(B * N, L * self.d_model)).reshape(B, N, 12)


class V9Official(V9Model):
    def __init__(self):
        super().__init__(block_cls=MambaBlockOfficial)

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
        x = self.proj(x)
        for l in self.layers:
            x = l(x)
        x = self.norm(x)
        return self.output(x.reshape(B * N, L * self.d_model)).reshape(B, N, 12)


# ========== 训练 / 评估 ==========
def train_model(model, X, Y, epochs=30, tag=''):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    S = X.shape[0]
    n_batches = (S + BATCH_SIZE - 1) // BATCH_SIZE
    losses = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(S)
        ep_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(X[idx].to(DEVICE)), Y[idx].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / n_batches
        losses.append(avg)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  {tag} Ep {ep:3d}/{epochs}: loss={avg:.6f}  '
                  f'({(time.time()-t0)/(ep+1):.1f}s/ep)', flush=True)
    return losses, (time.time() - t0) / epochs


@torch.no_grad()
def evaluate(model, X, Y, bs=128):
    """
    分批评估。测试集有 10419×50 = 52 万条序列，
    一次送进去会爆显存（scan 的中间状态 520950×256×32 就有 17GB）。
    """
    model.eval()
    preds, trues = [], []
    for i in range(0, X.shape[0], bs):
        preds.append(model(X[i:i + bs].to(DEVICE)).cpu())
        trues.append(Y[i:i + bs])
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()

    mse = float(((pred - true) ** 2).mean())
    p = pred * std_y.reshape(1, NUM_SENSORS, 1) + mean_y.reshape(1, NUM_SENSORS, 1)
    t = true * std_y.reshape(1, NUM_SENSORS, 1) + mean_y.reshape(1, NUM_SENSORS, 1)
    mae = np.abs(p - t).mean()
    rmse = np.sqrt(((p - t) ** 2).mean())
    mape = np.mean(np.abs((p - t) / np.maximum(np.abs(t), 1e-3))) * 100
    return dict(mse=mse, mae=mae, rmse=rmse, mape=mape)


# ========== 跑对照 ==========
EPOCHS = int(os.environ.get("MAMBA_EPOCHS", 30))

print('=' * 74)
print(f'本次配置:  EPOCHS={EPOCHS}   训练样本={X_tr_t.shape[0]}   '
      f'每epoch步数={(X_tr_t.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE}')
print(f'预计时长:  V8手写~34min  V8官方~14min  V9手写~98min  V9官方~38min  合计~3h')
print('=' * 74)
if EPOCHS < 10:
    print('⚠️  当前 EPOCHS < 10，只适合测速度，精度结论不可用')
VARIANTS = [
    ('V8-手写',  V8Model),
    ('V8-官方',  V8Official),
    ('V9-手写',  V9Model),
    ('V9-官方',  V9Official),
]

results = {}
print('\n' + '=' * 74)
print('阶段1 对照实验：手写 Python 循环扫描  vs  官方 mamba-ssm CUDA 核')
print('=' * 74)

for name, ModelCls in VARIANTS:
    gc.collect(); torch.cuda.empty_cache()
    model = ModelCls()
    n_param = sum(p.numel() for p in model.parameters())
    print(f'\n--- {name}  ({n_param:,} 参数) ---', flush=True)
    try:
        losses, ep_t = train_model(model, X_tr_t, Y_tr_t, epochs=EPOCHS, tag=name)
        res = evaluate(model, X_te_t, Y_te_t)
        res['params'] = n_param; res['epoch_time'] = ep_t
        results[name] = res
        torch.save(model.state_dict(), f'{name.replace("-", "_")}.pt')
        # 每跑完一个就落盘，3 小时长跑中途断了不至于全丢
        with open('stage1_results.json', 'w', encoding='utf-8') as fj:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in results.items()}, fj, ensure_ascii=False, indent=2)
        print(f'[OK] {name}: MSE={res["mse"]:.4f} MAE={res["mae"]:.3f} '
              f'每epoch={ep_t:.1f}s', flush=True)
    except Exception as e:
        print(f'[失败] {name}: {type(e).__name__}: {e}', flush=True)
        import traceback; traceback.print_exc()

# ========== 汇总 ==========
print('\n' + '=' * 88)
print(f'{"模型":<10} {"参数量":>10} {"MSE":>9} {"MAE":>9} {"RMSE":>9} {"MAPE%":>8} {"s/epoch":>9} {"加速比":>8}')
print('-' * 88)
base = None
for name in ['V8-手写', 'V8-官方', 'V9-手写', 'V9-官方']:
    if name not in results: continue
    r = results[name]
    if name.endswith('手写'):
        base = r['epoch_time']
    spd = f'{base / r["epoch_time"]:.2f}x' if base else '-'
    print(f'{name:<10} {r["params"]:>10,} {r["mse"]:>9.4f} {r["mae"]:>9.3f} '
          f'{r["rmse"]:>9.3f} {r["mape"]:>7.1f}% {r["epoch_time"]:>8.1f}s {spd:>8}')
print('=' * 88)

print("""
怎么读这张表：
  · 加速比 > 1  → 官方核更快，你原实现确实慢在 Python 循环
  · 加速比 ≈ 1  → 瓶颈不在扫描，在别处（数据搬运/head）
  · MSE 差异     → 初始化不同（A 从 [0.1,1] vs [1,d_state]）带来的精度变化
  · 参数量应基本相同；官方略多是因为多了 RMSNorm 的参数

这两个数字写进论文都很值钱 —— 无论方向如何，都是"实测"而非"我以为"。
""")
