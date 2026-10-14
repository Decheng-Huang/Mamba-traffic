# -*- coding: utf-8 -*-
# ============================================================================
#  投稿版实验脚本 —— 效率 + 适用边界
#
#  论文主线：Mamba 做交通流预测，需不需要显式空间信息？
#     · 消融三档：真邻居 / 随机邻居 / 自邻居（后者信息量等于"无邻居"但参数量可比）
#     · 两个标准数据集：PeMS-BAY(325) + METR-LA(207)
#     · 标准指标：MAE / RMSE / MAPE 分 15 / 30 / 60 分钟三个 horizon
#     · 效率对照：手写 Python 循环扫描 vs 官方 mamba-ssm CUDA 核
#
#  在 Kaggle 上分两个 cell：
#    cell 1:  !pip install -q mamba-ssm causal-conv1d
#    cell 2:  把本文件整段粘进去
#
#  时间预算（关键，先看清再跑）
#  ---------------------------------------------------------------------------
#  手写 Python 循环扫描 在 325 节点 / 全量数据上 每 epoch 要 2 小时以上，不可行。
#  所以两个实验分开做，用的 batch 规模不同，这是刻意的设计，写论文时要说明：
#      · 效率实验  → 50 节点 + 子集（per-step 属性，与总量无关）  ⇒ 可跑手写版
#      · 精度/消融 → 全量节点 + 官方核                            ⇒ 手写版跑不动
#  ---------------------------------------------------------------------------
#
#  环境变量（都可以留空用默认）：
#     PAPER_MODE      matrix | quick | speed
#                       matrix = 跑完整消融矩阵（默认）
#                       quick  = 2 分钟冒烟测试，验证管线通了
#                       speed  = 只跑效率对照（50 节点 + 子集，手写 vs 官方）
#     PAPER_DATASETS  PeMS-BAY,METR-LA
#     PAPER_MODES     real,random,self
#     PAPER_EPOCHS    默认 30
#     PAPER_SEEDS     默认 1
# ============================================================================

import os, sys, gc, json, time, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认 GBK，打不出 ✓ ⚠️ 这类字符会直接抛 UnicodeEncodeError，
# 把真正的报错盖掉。改成 UTF-8（Kaggle 上本来就是 UTF-8，这行是空操作）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- 配置
MODE = os.environ.get('PAPER_MODE', 'matrix').strip()
EPOCHS = int(os.environ.get('PAPER_EPOCHS', 30))
SEEDS = [int(s) for s in os.environ.get('PAPER_SEEDS', '1').split(',') if s.strip()]
WANT_DS = [s.strip() for s in os.environ.get('PAPER_DATASETS', 'PeMS-BAY,METR-LA').split(',') if s.strip()]
WANT_MODES = [s.strip() for s in os.environ.get('PAPER_MODES', 'real,random,self').split(',') if s.strip()]
# 架构可选 V8(V9) / V9 / LSTM / TF。
# 论文的建议跑法：消融用 V8 跑满三种邻居设定；
# LSTM 和 TF 是神经网络基线，跑一次即可（PAPER_ARCH='V8,LSTM,TF'），不用跟着消融跑三遍。
WANT_ARCH = [s.strip() for s in os.environ.get('PAPER_ARCH', 'V8').split(',') if s.strip()]
WANT_BLOCK = [s.strip() for s in os.environ.get('PAPER_BLOCK', 'official').split(',') if s.strip()]

NUM_NEIGHBORS = 5
D_MODEL = 128
LR = 5e-4
MAX_SEQ_PER_BATCH = int(os.environ.get('PAPER_SEQ_PER_BATCH', 1600))  # 显存护栏
# 手写扫描会把 deltaA 物化成 (bs*N, L, d_inner, d_state)，官方核不物化，
# 所以官方核可以把每批序列数放大 4 倍而不爆显存。
SEQ_BUDGET = {'naive': MAX_SEQ_PER_BATCH, 'official': MAX_SEQ_PER_BATCH * 4}
HORIZONS = {15: 3, 30: 6, 60: 12}     # 分钟 -> 步长（数据是 5 分钟粒度）
MAPE_MIN = float(os.environ.get('PAPER_MAPE_MIN', 1.0))  # MAPE 只统计真值 > 此值的点
# 输入保护：标准化之后的 X 裁剪到 ±CLIP_NORM。0 = 关闭（默认）。
# 2026-09-23 加。背景：verify 协议原先三个划分都取【开头】2000 步，
# 于是 train = 1 月头一周、test = 6 月头一周，中间隔 5 个月。实测后果：
# 标准化后的输入出现 65σ 的极值（std_x 本身是健康的，min=1.011，
# 所以不是除以近零 std，是测试期原始值真的超出训练期范围），
# 三层网络把它放大成 ≤1e4 → 7.3e7 → 3.46e31，末尾 LayerNorm 一平方
# 就撞穿 float32 上限 3.4e38 → inf → nan。关掉官方 fast path 也照样炸，
# 所以不是核的问题，是输入的问题。
# 协议本身已修（train 改取末尾，见 load_dataset）。这里是第二道保险：
# 默认关闭，只有协议修完仍然出 nan 才打开。
# ★ 只裁输入 X，绝不裁目标 Y —— 裁 Y 等于改真值，指标就是假的。
CLIP_NORM = float(os.environ.get('PAPER_CLIP_NORM', 0))


def count_abs_gt(a, thr, chunk=8 << 20):
    """分块统计 |a| > thr 的元素个数。
    全量 train 有 3.6 万步 × 325 节点 × 12 通道 ≈ 1.4 亿个数，
    直接 np.abs(a) 会额外造一个 570 MB 的临时数组 —— 这台机器已经 OOM 过两次，
    不值得为一行诊断再冒一次险。"""
    f = np.asarray(a).reshape(-1)
    n = 0
    for i in range(0, f.size, chunk):
        n += int(np.count_nonzero(np.abs(f[i:i + chunk]) > thr))
    return n
PATIENCE = 8                          # 验证集早停耐心
# ★ 2026-09-26：允许用环境变量改落盘文件（与 gnn_baselines.py:145 同口径，
#   那边一直是可覆盖的，这边是硬编码，写重算结果时会覆写原始结果）。
RESULT_FILE = os.environ.get('PAPER_RESULT', 'paper_results.json')

# ★ 2026-09-26 新增：只重算指标、不训练。
#   背景：compute_metrics 的掩码原来判在**反标准化之后**（真值 0 经往返后不再
#   精确为 0），METR-LA 有 8.67% 的零值逃过掩码被当成有效点评分（任务书 §0.3）。
#   掩码已修；而训练/早停/检查点选择走的是 quick_eval，它的口径**本来就是对的**
#   → 结论：**不需要重新训练**，只要用已修好的 compute_metrics 拿现成的
#   `_best.pt` 重新推理一遍，指标就正确了。这个开关就是那条路。
RECOMPUTE = os.environ.get('PAPER_RECOMPUTE', '0').strip() == '1'

# verify 模式（真数据上验 permute 修复到底有没有用）用的规模。
# 325 节点 × 2000 时刻 ≈ 65 万条序列/epoch，T4 上约 190 秒/epoch。
VERIFY_STEPS = int(os.environ.get('PAPER_VERIFY_STEPS', 2000))
VERIFY_EPOCHS = int(os.environ.get('PAPER_VERIFY_EPOCHS', 20))
# 默认用官方核（快）。设成 naive 可作退路：本地没装 mamba_ssm，
# 或者 Kaggle 上编不出来时，仍能做完这个 A/B。
VERIFY_BLOCK = os.environ.get('PAPER_VERIFY_BLOCK', 'official').strip()

# 节点数不符时是报错还是只警告。默认报错：2026-09-23 有一次 Kaggle 数据集里
# 只有 50 节点的文件，脚本静默回退到 N=50 把整个 verify 跑完了（2.8 小时白烧），
# 而且是在那段"前 50 个传感器几乎不连通"的图上跑的，结论根本不能用。
# 确实想在小节点数上跑，就显式设 PAPER_STRICT_N=0。
STRICT_N = bool(int(os.environ.get('PAPER_STRICT_N', 1)))

# verify 模式跑哪几"臂"。2026-09-23 改：permute 修复已在 50 节点上被真数据
# 证过了（修好 MAE@60=2.720 vs 旧bug=4.401），再跑旧 bug 那臂纯属白费一半时间。
# 所以默认只跑"修好"。想重做那个 A/B 就设 PAPER_VERIFY_ARMS=fixed,oldbug。
VERIFY_ARMS = [({'fixed': True, 'oldbug': False}[a.strip()])
               for a in os.environ.get('PAPER_VERIFY_ARMS', 'fixed').split(',')
               if a.strip() in ('fixed', 'oldbug')]

# verify 跑哪些数据集。默认只跑 PeMS-BAY（Gate A 只需要它）。
VERIFY_DS = [s.strip() for s in
             os.environ.get('PAPER_VERIFY_DATASETS', 'PeMS-BAY').split(',')
             if s.strip()]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集登记表。pats 是文件名优先级：对齐好的在前，旧的兜底。
DATASETS = {
    'PeMS-BAY': dict(
        dirs=['数据集', '/kaggle/input/datasets/huangdecheng/ffff12',
              '/kaggle/input/ffff12'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=325, step_min=5),
    'PeMS-BAY-50': dict(
        dirs=['数据集', '/kaggle/input/datasets/huangdecheng/ffff12',
              '/kaggle/input/ffff12'],
        pats=['{s}_aligned50.npy', '{s}_50.npy'],
        N=50, step_min=5),
    'METR-LA': dict(
        dirs=['数据集/METR-LA', '/kaggle/input/metrla'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=207, step_min=5),
    # 2026-09-28 加：第二个数据集（METR-LA 因管线级不复现弃用后补位）。
    # 由 build_pemsd4.py 从 ASTGCN 的 pems04.npz 生成；切分 7:1:2 与 PeMS-BAY 统一
    # （注意 PeMS04 文献惯例是 6:2:2，这一条必须披露）。
    # ★ d0 的语义与 PeMS-BAY 不同：PeMS-BAY 的 d0 是速度(mph)，PeMS04 的 d0 是流量(veh/5min)。
    'PeMS04': dict(
        dirs=['数据集/PeMS04'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=307, step_min=5),
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 数据
def find_split_file(d, split, pats):
    """按优先级找数据文件"""
    for pat in pats:
        p = os.path.join(d, pat.format(s=split))
        if os.path.exists(p):
            return p
    return None


# parquet 的列顺序必须与 rebuild_data.py 一致，否则 x/y 通道会错位。
X_COLS = [f'x_t-{11 - i}_d0' for i in range(11)] + ['x_t+0_d0']
Y_COLS = [f'y_t+{i + 1}_d0' for i in range(12)]
ALL_COLS = X_COLS + Y_COLS

# ★ 2026-09-24 新增：时段通道（time-of-day）★
#   `_d1` 一直在 parquet 里（50 列里的那 24 个 _d1），但老代码只读 _d0，
#   一行没用上 —— IN_DIM 只有 1+k，**模型看不见"现在几点"**。
#
#   实测确认（本地 diag）：
#     · 取值范围 [0, 1)，288 个唯一值 = 一天的 288 个 5 分钟槽 → 就是 time-of-day
#     · **与节点无关**：同一时刻 325 个节点取值完全相同，步进恰为 1/288
#
#   为什么怀疑它是主因（A5 全量 V8 实测）：
#     · @15 落后文献 8.6%，@60 落后约 40% —— 差距随 horizon 急剧拉大
#     · 3.5 倍数据量只换来 1.4% 改善 → 缺的是**特征**不是样本
#   长程退化的机理：早上 6 点，模型看到过去一小时畅通，但它不知道
#   7 点早高峰要来。短期靠惯性够用，60 分钟后崩。
#   GWNet 那一类模型是显式喂时段协变量的，我们等于被绑住一只手。
#
#   PAPER_TOD=0 可关掉，用于 A/B（配置 ID 会带 |tod 区分，见主循环）。
TOD_COLS = [f'x_t-{11 - i}_d1' for i in range(11)] + ['x_t+0_d1']
TOD_DESC = '时段 = 一天内的位置，0.0=午夜 0.5=正午；288 个 5 分钟槽，与节点无关'
USE_TOD = bool(int(os.environ.get('PAPER_TOD', 1)))

# 强制所有划分都从 parquet 读（parquet 是 ground truth）。
# 做 A/B 时必开：开着时段通道的那一臂只能走 parquet，若另一臂因为
# pats 里 npy 在前而读了 npy，两臂就不是同一个数据源 —— 那就不是单变量对比了。
FORCE_PARQUET = bool(int(os.environ.get('PAPER_FORCE_PARQUET', 0)))


def load_from_parquet(path, N, tag):
    """直接从 parquet 读成 (N, T, 24)，等价于 rebuild_data.py 的产物。

    2026-09-23 加。起因：`数据集.zip` 里其实只有 3 个 parquet + adj_mx.npy，
    **没有** `*_aligned.npy`；而 load_dataset 只认 .npy，于是"数据传上去了"
    和"脚本找不到数据"同时成立。这几个 parquet 就是 ground truth，
    没有理由让用户为了跑实验再传 2.3 GB 的 npy。

    分批读（iter_batches），峰值内存只有一个 batch，不是整张表。
    """
    import pyarrow.parquet as pq          # 只在真要走这条路时才依赖 pyarrow
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    if n_total % N != 0:
        raise ValueError(f'{tag}: {n_total} 行不能被 N={N} 整除')
    T = n_total // N

    a = np.empty((n_total, 24), dtype=np.float32)
    # ★ 时段通道与节点无关（实测 325 个节点同值），所以每个时刻只留 node_id==0
    #   那一行 —— 存 (T, 12) 而不是 (N, T, 12)。全量 train 后者要 569 MB，
    #   前者只要 1.7 MB。
    tod = np.empty((T, 12), dtype=np.float32) if USE_TOD else None
    cols = ['node_id'] + ALL_COLS + (TOD_COLS if USE_TOD else [])
    off, seen, g1 = 0, False, 0
    for b in pf.iter_batches(batch_size=1 << 20, columns=cols):
        m = b.num_rows
        # 只抽查第一块：node_id 必须从 0 开始按 0..N-1 循环（= 时间优先布局）
        if not seen and m >= N:
            if not np.array_equal(np.asarray(b.column('node_id'))[:N], np.arange(N)):
                raise ValueError(
                    f'{tag}: node_id 开头不是 0..{N - 1}，parquet 不是时间优先布局，'
                    f'按 N={N} 解读会静默算出垃圾。')
            seen = True
        for k, c in enumerate(ALL_COLS):
            a[off:off + m, k] = np.asarray(b.column(c), dtype=np.float32)
        if USE_TOD:
            sel = np.nonzero(np.asarray(b.column('node_id')) == 0)[0]
            if len(sel):
                for j, c in enumerate(TOD_COLS):
                    tod[g1:g1 + len(sel), j] = np.asarray(
                        b.column(c), dtype=np.float32)[sel]
                g1 += len(sel)
        off += m
    if off != n_total:
        raise ValueError(f'{tag}: 只读到 {off} 行，metadata 说 {n_total}')
    if not seen:
        raise ValueError(f'{tag}: 没读到完整的一个时刻，N={N} 可能不对')
    if USE_TOD:
        # 自检：数量对不上、或取值跑出 [0,1)，说明读错列/读错行 —— 立刻报错，
        # 不要让它变成一个"看起来加了特征其实加的是垃圾"的实验。
        if g1 != T:
            raise ValueError(f'{tag}: 时段通道取到 {g1} 个时刻，应为 {T}')
        if not (float(tod.min()) >= 0.0 and float(tod.max()) < 1.0):
            raise ValueError(
                f'{tag}: 时段通道取值 {float(tod.min()):.4f}~{float(tod.max()):.4f} '
                f'越界，应落在 [0,1)。可能读错了列。')

    d = np.ascontiguousarray(a.reshape(T, N, 24).transpose(1, 0, 2))
    # 同一把尺子：与 load_layout 完全相同的重叠自检
    ov = float(np.abs(d[:, :-1, 1:12] - d[:, 1:, 0:11]).mean())
    if not (ov < 1e-3):
        raise ValueError(f'{tag}: parquet 布局自检失败（重叠误差={ov:.4f}，应为 0）')
    return d, ov, tod


class TodUnavailable(Exception):
    """要用时段通道，但这份数据里没有 _d1 列。

    和 NodeCountMismatch 同样的道理：故意不继承 ValueError，因为
    load_dataset 的候选循环里 `except ValueError: continue` 会把它吞掉，
    然后给出一个"所有目录都没通过布局自检"的错误结论 —— 那是误导。
    这类错误必须直接冒到用户面前。
    """


def load_layout(a, N, tag):
    """
    自适应两种布局，并强制自检。

    已知两种：
        (24, N*T)  时间优先，列号 = t*N + s      ← Kaggle 上 ffff12 那份
        (N*T, 24)  节点优先，行号 = s*T + t      ← rebuild_data.py 生成的 *_aligned.npy

    判据是滑动窗口的相邻重叠必须为 0：x_t[1:12] == x_{t+1}[0:11]
    用错朝向不会报错，只会静默算出垃圾，所以这里必须硬性拦下。
    """
    # ★ 2026-09-24：.npy 里只有 24 个 _d0 通道，没有 _d1 时段列。
    #   开着时段通道却走 npy 路径 = "以为加了特征、其实没加" 的假实验 ——
    #   跑完还会得出"时段特征没用"的错误结论。必须报错，不许静默降级。
    if USE_TOD:
        raise TodUnavailable(
            f'{tag}: 已开启时段通道（PAPER_TOD=1，默认），而 .npy 里没有 _d1 列。\n'
            f'    时段通道只能从 parquet 读。请改用 parquet 数据，'
            f'或设 PAPER_TOD=0 关掉时段通道。\n'
            f'    （这条 if 是特意加的：走 npy 却以为加了时段特征，'
            f'跑完会得出"时段没用"的错误结论。）')
    if a.shape[0] == 24:
        T = a.shape[1] // N
        d = a.T.reshape(T, N, 24).transpose(1, 0, 2).copy()
    elif a.shape[1] == 24:
        T = a.shape[0] // N
        d = a.reshape(N, T, 24).copy()
    else:
        raise ValueError(f'{tag}: shape {a.shape} 既不是 24 行也不是 24 列')
    ov = float(np.abs(d[:, :-1, 1:12] - d[:, 1:, 0:11]).mean())
    if not (ov < 1e-3):
        raise ValueError(
            f'{tag}: 布局自检失败（重叠误差={ov:.4f}，应为 0）。\n'
            f'  文件形状 {a.shape}，按 N={N} 解读。\n'
            f'  可能原因：(1) N 猜错了；(2) 这个 npy 的样本顺序被打乱过（旧版 train_50.npy 就有这毛病）。\n'
            f'  对策：跑 rebuild_data.py 从 parquet 重建 *_aligned.npy。')
    # 第三个返回值是时段通道。npy 路径没有 → None（上面已拦下 USE_TOD 的情况）
    return d, ov, None


class NodeCountMismatch(Exception):
    """声明的节点数在数据里读不通，且 STRICT_N 要求报错。

    故意不继承 ValueError：load_dataset 内部用 `except ValueError: continue`
    来跳过读不通的候选 N，继承的话会被吞掉。
    """


def _has_any_train(d, cfg):
    """这个目录里有没有任何可用的训练数据（npy 或 parquet）。"""
    return bool(find_split_file(d, 'train', cfg['pats'])
                or os.path.exists(os.path.join(d, 'train.parquet')))


def _discover_kaggle_dirs(cfg):
    """在 /kaggle/input 下自己找可能的数据集目录。

    2026-09-23 加：Kaggle 上的目录名由上传者自己定，写死路径很容易对不上，
    而"找不到数据"的代价是白跑一整个会话。这里按文件名认人，优先级：
        train_aligned.npy  >  train.parquet  >  只有 train_50.npy
    parquet 排在 _50.npy 前面，因为它就是 ground truth（_50.npy 有旧版被打乱的，
    节点数也只有 50）。
    """
    root = '/kaggle/input'
    if not os.path.isdir(root):
        return []
    aligned, parq, fallback = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if not x.startswith('.')]
        if not _has_any_train(dirpath, cfg):
            continue
        if 'train_aligned.npy' in filenames:
            aligned.append(dirpath)
        elif 'train.parquet' in filenames:
            parq.append(dirpath)
        else:
            fallback.append(dirpath)
    return aligned + parq + fallback


def load_dataset(name, quick=False, subset=None):
    """返回 (train, val, test, adj, N)。每份形状 (N, T, 24)。"""
    cfg = DATASETS[name]

    # 候选目录 = 登记表里存在的 + /kaggle/input 下自动扫到的（去重保序）
    cands = [c for c in cfg['dirs']
             if os.path.exists(c) and _has_any_train(c, cfg)]
    for extra in _discover_kaggle_dirs(cfg):
        if extra not in cands:
            cands.append(extra)
    if not cands:
        raise FileNotFoundError(
            f'{name}: 找不到数据。找过 {cfg["dirs"]}，文件名优先级 {cfg["pats"]}，'
            f'/kaggle/input 下也没扫到。')

    # 两轮：先用【声明的 N】试遍所有候选目录（这样"另一个目录里才有全量节点"
    # 的情况也能救回来）；全试完都不行，再依次试别的 N 并大声警告。
    cand_N = [cfg['N']] + [n for n in (50, 207, 325) if n != cfg['N']]
    out = None
    for k, N in enumerate(cand_N):
        for d in cands:
            try:
                splits, ovs, used, tods = [], [], [], []
                for s in ['train', 'val', 'test']:
                    fp = os.path.join(d, f'{s}.parquet')
                    # ★ 2026-09-24：开着时段通道时优先用 parquet。
                    #   _d1 时段列只存在于 parquet，npy 里没有。pats 的优先级
                    #   原本是 npy 在前，于是"明明有一份能读出时段的 parquet，
                    #   却被 npy 抢先、然后 load_layout 报错" —— 没必要。
                    #   若 parquet 不存在，仍走 npy，由 load_layout 抛
                    #   TodUnavailable 大声拦住（不再静默降级成"没加特征"）。
                    if (USE_TOD or FORCE_PARQUET) and os.path.exists(fp):
                        f = None
                        if USE_TOD:
                            # 别让它退化成下面那条"节点数不符"的报错 —— 那句话看不出
                            # 根因（真正的根因是：时段列只有 325 节点那份 parquet 有，
                            # 而 quick/speed 模式声明的 N=50 配不出时段通道）。
                            import pyarrow.parquet as _pq
                            _nr = _pq.ParquetFile(fp).metadata.num_rows
                            if _nr % N:
                                raise TodUnavailable(
                                    f'{name}: PAPER_TOD=1 需要 parquet 里的 _d1 时段列，'
                                    f'但 {os.path.basename(fp)} 是 {_nr} 行，'
                                    f'不能被声明的 N={N} 整除（本数据的 parquet 只有 325 节点）。\n'
                                    f'    这个节点数的实验想跑：设 PAPER_TOD=0 关掉时段通道。\n'
                                    f'    想加时段特征：用 N=325 的数据集（PeMS-BAY / METR-LA）。')
                    else:
                        f = find_split_file(d, s, cfg['pats'])
                    if f is None:
                        # 没有 npy 就退回 parquet（ground truth）。
                        # 读不通就抛错，交给下面的 except 换下一个候选目录。
                        if not os.path.exists(fp):
                            raise FileNotFoundError(fp)
                        dd, ov, td = load_from_parquet(fp, N, f'{s}.parquet')
                        f = fp
                    else:
                        a = np.load(f).astype(np.float32)
                        # USE_TOD 时这里抛 TodUnavailable —— 不在 except 的
                        # 捕获列表里，会直接冒到用户面前（要的就是这个）。
                        dd, ov, td = load_layout(a, N, os.path.basename(f))
                    splits.append(dd)
                    ovs.append(ov)
                    tods.append(td)
                    used.append(os.path.basename(f))
            except (ValueError, OSError, FileNotFoundError):
                continue          # 这个目录按这个 N 读不通，换下一个
            log(f'  {name}: 目录 {d}')
            log(f'    文件 {used[0]}   N={N}  ' + '  '.join(
                f'{s}={t.shape[1]:,}(自检{ov:.1e})' for s, t, ov in
                zip(['train', 'val', 'test'], splits, ovs)))
            if k > 0:
                # ★ 2026-09-23 加：曾经因为 Kaggle 数据集里只有 50 节点文件，
                #   脚本静默回退到 N=50 跑完了整个 verify（2.8 小时白烧）。
                #   节点数不符是配置错误，默认直接报错，别让它悄悄降级。
                #   注意这句 raise 在 try 之外，否则会被上面的 except 吞掉。
                msg = (f'{name}: 声明的是 N={cfg["N"]}，实际只有 N={N} 的文件读得通 '
                       f'（用的是 {used[0]}）。\n'
                       f'    这不是警告是错误：节点数不对，空间消融和"全量节点"的结论都不成立。\n'
                       f'    确认要按 N={N} 跑就设 PAPER_STRICT_N=0。')
                if STRICT_N:
                    raise NodeCountMismatch(msg)
                log(f'    ⚠️  {msg}')
            # 时段通道要么三个划分都有，要么都没有（混着来是数据配置出了问题）
            if any(t is not None for t in tods) and any(t is None for t in tods):
                raise TodUnavailable(
                    f'{name}: train/val/test 里只有一部分能取到时段通道 '
                    f'({[None if t is None else t.shape for t in tods]})。\n'
                    f'    多半是有的划分是 parquet、有的划分是 npy —— 这种混合配置不可信，'
                    f'先统一数据来源。')
            out = (splits, N, tods if any(t is not None for t in tods) else None)
            break
        if out is not None:
            break
    if out is None:
        raise ValueError(
            f'{name}: 候选目录 {cands} × 候选 N={cand_N} 都没通过布局自检。\n'
            f'    先跑 rebuild_data.py 修数据；若数据在 Kaggle 上，'
            f'确认 train_aligned.npy 的确传上去了。')

    splits, N, tods = out
    # 邻接矩阵
    adj = None
    for cand in [os.path.join(d, 'adj_mx.npy'),
                 os.path.join(d, 'sensor_graph', 'adj_mx.npy'),
                 os.path.join(d, 'adj_50.npy')]:
        if os.path.exists(cand):
            adj = np.load(cand).astype(np.float64)
            break
    if adj is None:
        raise FileNotFoundError(f'{name}: 找不到 adj_mx.npy')
    if adj.shape[0] != N:
        # adj 是全量节点、数据是前 N 个，取子阵
        adj = adj[:N, :N]
    np.fill_diagonal(adj, 0)

    if quick:
        splits = [s[:, :800] for s in splits]
        if tods is not None:
            tods = [t[:800] for t in tods]
    elif subset:
        # ★ 2026-09-23 改：train 取【末尾】subset 步，val/test 取【开头】。
        #   原来三个划分都取开头，于是 train = 1 月头一周、val = 5 月头一周、
        #   test = 6 月头一周，train 和测试之间空了 5 个月。后果实测：
        #   标准化后的输入里出现 65σ 的值（std_x 本身健康，min=1.011，
        #   所以不是除以近零 std），三层网络把它滚成 ≤1e4 → 7.3e7 → 3.46e31，
        #   末尾 LayerNorm 一平方 → 1.2e63 > float32 上限 3.4e38 → inf → nan。
        #   关掉官方 fast path 照样炸 —— 不是核的锅，是输入分布自己跑飞了。
        #   改成 train 取末尾（≈4 月底–5 月 7 日）后，train 末尾与 val 开头首尾相接，
        #   train→test 的间隔从 5 个月缩到 3 周，漂移回到与正式协议同一量级。
        #   注意：这只是让 verify 更接近正式协议，不是调参凑指标；
        #   正式 matrix 模式根本不走这里（没有 subset）。
        splits = [s[:, -subset:] if i == 0 else s[:, :subset]
                  for i, s in enumerate(splits)]
        # 时段通道必须跟着同一把切刀走，否则特征和标签错位 —— 那是比不加特征
        # 更坏的情况：模型会学到"这个时段配的是那段路况"的假对应。
        if tods is not None:
            tods = [t[-subset:] if i == 0 else t[:subset]
                    for i, t in enumerate(tods)]
    return splits[0], splits[1], splits[2], adj, N, tods


# ---------------------------------------------------------------- 邻居
def build_neighbors(adj, mode, k=NUM_NEIGHBORS, seed=0):
    """
    四种空间信息设定，是论文的核心消融。

      real   : 邻接矩阵里权重最高的 k 个 —— 真空间信息
      random : 随机 k 个别的传感器   —— 有"别的传感器"但语义是错的
      self   : k 个位置全填自己       —— 信息量 = 无邻居，但输入维度不变，参数量可比
      none   : 完全不给邻居，IN_DIM=1
    """
    N = adj.shape[0]
    rng = np.random.RandomState(seed)
    if mode == 'none':
        return None, 0
    idx = np.zeros((N, k), dtype=np.int64)
    short = 0
    for i in range(N):
        if mode == 'real':
            # 只取真邻居（权重严格大于 0）。
            # 注意：不能像原代码那样 np.pad(..., 'constant') 补索引 0 ——
            # 那等于把 0 号传感器的数据当成邻居喂进去，是主动注入错误信息。
            # 本数据集的邻接矩阵很稀疏（PeMS-BAY 稀疏度 97.8%），
            # 24% 的节点真邻居不足 k 个，所以这条分支是有实际影响的。
            order = np.argsort(adj[i])[::-1]
            order = order[(order != i) & (adj[i][order] > 0)][:k]
            if len(order) < k:
                short += 1
                # 缺的位置补「自己」：中性，既不注入假信息也不改变输入维度
                order = np.concatenate([order, np.full(k - len(order), i)])
        elif mode == 'random':
            pool = np.array([j for j in range(N) if j != i])
            order = rng.permutation(pool)[:k]
        elif mode == 'self':
            order = np.full(k, i, dtype=np.int64)
        else:
            raise ValueError(f'未知邻居模式: {mode}')
        idx[i] = order
    if mode == 'real':
        deg = (adj > 0).sum(axis=1)
        # ★ 2026-09-24：这里原先只印节点级比例，与论文/Gate A 的口径不是一回事。
        #   任务书写「真邻居 88% 而非 21%」，那 88%/21.2% 是**通道级**：
        #   平均每节点 k 个输入通道里有几个是真邻居 = mean(min(deg,k))/k。
        #   （回算 50 节点：通道级 21.20%，与任务书的 21.2% 精确吻合；
        #     节点级则是 0%，对应任务书「100% 节点不足 k 个」。）
        #   两个数一起印，免得读 log 的人拿 75.7% 去对论文的 88% 以为出错。
        chan_ratio = float(np.minimum(deg, k).mean()) / k
        log(f'    真邻居比例（通道级，= Gate A 判据/论文口径）: {100*chan_ratio:.2f}% '
            f'（阈值 85%，平均每节点 {np.minimum(deg, k).mean():.2f}/{k} 个通道是真邻居）')
        log(f'    真邻居不足 {k} 个的节点: {short}/{N} '
            f'({100*short/N:.1f}%，节点级 —— 与上面那个数是不同口径) '
            f'，这些位置已用「自己」补齐（未注入假信息）')
        log(f'    邻接矩阵: 稀疏度 {100*(adj==0).mean():.1f}%  '
            f'邻居数 最小={deg.min()} 中位={int(np.median(deg))} 最大={deg.max()}')
        if int((deg == 0).sum()):
            log(f'    ⚠️  {int((deg == 0).sum())} 个节点的真邻居数为 0 —— '
                f'它们 k 个通道全填自己，在 real 设定下与 self 设定完全等价（论文需披露）')
        if short / N > 0.5:
            log(f'    ⚠️  超过半数节点的真邻居不足 {k} 个 —— 空间消融在这种图上没有意义，'
                f'换全量节点再跑')
    return idx, k


def make_samples(x_data, y_data, nbr_idx, tod=None):
    """(N, T, 12) -> (T, N, 12, 1+k[+1]) 与 (T, N, 12)。预分配，避免 list 翻倍内存

    tod：None 或 (T, 12) 的时段通道。与节点无关，所以直接广播到 N 个节点。
         tod[t] = [x_t-11_d1 ... x_t+0_d1]，与 x_data[:, t, :] 是同一个样本，
         下标必须严格同源 —— 错位比不加特征更坏。
    """
    N, T, L = x_data.shape
    k = 0 if nbr_idx is None else nbr_idx.shape[1]
    C = 1 + k + (1 if tod is not None else 0)
    X = np.empty((T, N, L, C), dtype=np.float32)
    for t in range(T):
        X[t, :, :, 0] = x_data[:, t, :]
        if k:
            X[t, :, :, 1:1 + k] = x_data[nbr_idx, t, :].transpose(0, 2, 1)
    if tod is not None:
        if tod.shape[0] != T:
            raise ValueError(
                f'make_samples: 时段通道有 {tod.shape[0]} 个样本，速度有 {T} 个 —— '
                f'切分/裁剪时没同步，特征与标签会错位。')
        if tod.shape[1] != L:
            raise ValueError(f'make_samples: 时段通道宽 {tod.shape[1]}，应为 {L}')
        X[:, :, :, -1] = tod[:, None, :]          # (T,1,12) 广播到 (T,N,12)
    Y = y_data.transpose(1, 0, 2).copy()          # (T, N, 12)
    return X, Y


# ---------------------------------------------------------------- 模型
def selective_scan_fn(u, delta, A, B, C, D):
    """手写 Python 循环扫描 —— 与 kaggle_ffff12.py 逐字一致，保留用于效率对照"""
    Bs, L, D_in = u.shape
    D_state = A.shape[1]
    delta = torch.clamp(delta, -4.0, 4.0)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    x = torch.zeros(Bs, D_in, D_state, device=u.device, dtype=u.dtype)
    ys = torch.zeros(Bs, L, D_in, device=u.device, dtype=u.dtype)
    for i in range(L):
        x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
        ys[:, i] = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
    return ys + D.unsqueeze(0).unsqueeze(0) * u


class RMSNorm(nn.Module):
    """和 mamba_ssm 内部的 RMSNorm 同款，用来把手写块的结构和官方块对齐"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 先转 float32 再算，避免输入本身已在 float32 边界时平方溢出
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class MambaBlockNaive(nn.Module):
    """
    手写 Python 循环扫描。

    ★ 2026-09-23 修的一个真 bug ★
      原写法没有块内归一化，也没有块内残差，而官方 mamba_ssm.Mamba 的
      forward 是「先 RMSNorm → 混合器 → 加残差」。后果有两个：
        (1) 数值爆炸。实测（全量 325 节点、真实数据、初始权重、未训练）：
            各层 out_proj 输出 absmax = 1.3e4 → 2.3e12 → 3.4e38，
            第三层正好撞上 float32 上限 3.4e38 → nan。
            V8 只有 2 层（1e8×1e8=1e16）所以侥幸没炸，V9 三层必炸。
            这就是项目里「V9 的 MSE=nan」的根因，与 permute bug 无关。
        (2) 对比不公平。官方块带 pre-norm+残差、手写块不带，
            两者根本不是同一个网络，「手写 vs 官方」的效率数字不可比。
      补上 RMSNorm 和块内残差后：三层输出稳定在 8.8e2 → 2.6e3 → 3.0e3，
      不再逐层放大；且与官方块结构等价，效率对比才成立。
      注意：调用方要相应用 `x = l(x)`，不能再补 `+ x`。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        # ★ 2026-09-23 对齐 dt_rank ★
        #   官方 mamba_ssm 的 dt_rank = ceil(d_model/16)，d_model=128 时是 8；
        #   原来这里写死成 1（dt 退化成标量），等于每次只算一个步长再广播给
        #   d_inner 个通道 —— dt 是选择性机制的核心，压成标量是实打实的容量损失，
        #   既让"手写 vs 官方"的效率对比不对等，也可能压低精度。
        #   旧写法：x_proj → Linear(d_inner, d_state*2+1)，dt_proj → Linear(1, d_inner)
        self.dt_rank = int(math.ceil(d_model / 16))
        self.norm = RMSNorm(d_model)          # ← 与官方块对齐的 pre-norm
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2,
                                bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.linspace(0.1, 1.0, d_state)
        self.A_log = nn.Parameter(A_init.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for m in [self.in_proj, self.x_proj, self.out_proj, self.dt_proj]:
            if m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.dt_proj.bias)

    @property
    def A(self):
        return torch.exp(torch.clamp(self.A_log, max=0.0))

    def forward(self, x):
        residual = x                       # ← 与官方块一致：块内残差
        x = self.norm(x)                   # ← 与官方块一致：pre-norm
        B, L, _ = x.shape
        xg, res = self.in_proj(x).split(self.d_inner, dim=-1)
        xc = self.conv1d(xg.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        dt, Bp, Cp = torch.split(self.x_proj(xc),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan_fn(xc, delta, self.A, Bp, Cp, self.D)
        # 官方 MambaBlock 内部 dropout 默认 0，所以这里也不加，保持结构等价；
        # 残差在这里加（官方块也是块内加），调用方不要再补 + x
        return self.out_proj(y * F.silu(res)) + residual


HAS_OFFICIAL = False
MambaBlockOfficial = None          # 没装 mamba_ssm 时保持 None，下面 Predictor 靠它判断
try:
    from mamba_ssm import Mamba

    class MambaBlockOfficial(nn.Module):
        """
        官方 mamba_ssm 的 Mamba **混合器** + 与手写块逐项对应的 pre-norm / 残差。

        ★ 2026-09-26 修的一个真 bug ★
          原写法是 `return self.mamba(x)`，注释声称「Mamba 内部已含 RMSNorm +
          MambaBlock + 残差」。**这条注释是错的**：mamba_ssm 的 `Mamba`
          （`modules/mamba_simple.py`）只含混合器，pre-norm 与残差在
          `modules/block.py` 的 `Block` 里 —— `Block` 包住 `Mamba` 才有。
          实测（4090 / mamba_ssm 2.2.2）逐键对比 state_dict：
            MambaBlockOfficial 每块 116,480 参数，**没有任何 norm 相关的键**；
            MambaBlockNaive    每块 116,608 参数，多一个 `norm.weight (128,)`。
          即 official 支路当时跑的是**没有归一化、也没有残差的网络** —— 正是
          2026-09-23 在手写支路修掉的那次数值爆炸（1.3e4 → 2.3e12 → 3.4e38）
          的成因。后果不只是数值：`论文_实验设置_EN.md` §4.3 声称两条支路
          「compute the same function; they differ only in how the scan is
          executed」，而当时它们差着归一化和残差 —— 是两个不同的网络，
          「official vs naive」的差里捆着 kernel 与架构两样东西。
          补上 norm + 残差后两边**参数量与结构逐项相同**（各 116,608），
          差的只剩扫描实现，§4.3 那句话才成立。
          ★ 影响面：主力模型 V8 的 official 行也是走这条路，全部需要重跑。

        ★ 两边共用本文件的 `RMSNorm`（不换成 mamba_ssm 自己那份融合核版本）：
          目的是让「唯一变量是扫描实现」这件事严格成立。
        """

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.norm = RMSNorm(d_model)           # ← 与手写块同一个实现
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

        def forward(self, x):
            residual = x                           # ← 与手写块一致：块内残差
            return self.mamba(self.norm(x)) + residual

    HAS_OFFICIAL = True
except Exception as _e:
    log(f'[提示] 官方 mamba_ssm 不可用（{type(_e).__name__}），只能跑手写块。')
    log('       需要官方核请先跑：!pip install -q mamba-ssm causal-conv1d')


class Predictor(nn.Module):
    """
    四个模型共用同一条数据通路和同一个输出头，保证对比公平：

      V8        2 层 / d_state=16 / 线性输出        —— 本文的 Mamba 基线
      V9        3 层 / d_state=32 / MLP 输出        —— 加深加宽的 Mamba
      LSTM      3 层 LSTM(128) + 同款 MLP 输出      —— 循环网络基线
      TF        3 层 TransformerEncoder + 同款 MLP  —— 注意力基线

    ★ 这里修掉了原脚本的一个真 bug ★
      原写法： x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
      permute 之后再 reshape，PyTorch 只按当前逻辑顺序重新线性化，
      不会"按维度重排后再折叠"，结果把 (批次, 传感器, 时刻) 三者搅在一起——
      每一行喂给模型的根本不是某个传感器的 12 步时间序列。
      正确写法就是直接 reshape，不要 permute。
      这个写法在本项目 12 个文件里都有，凡是带 .contiguous() 的更隐蔽，
      因为 contiguous 只是让内存连续，并不会把维度转回来，结果一样是错的。
    """
    def __init__(self, in_dim, arch='V8', block_cls=None, d_state=None,
                 fix_permute=True):
        super().__init__()
        self.arch = arch
        self.fix_permute = fix_permute      # False 只用于 verify 模式的对照
        self.d_model = D_MODEL
        self.official = False

        if arch in ('V8', 'V9'):
            if block_cls is None:
                block_cls = MambaBlockOfficial if HAS_OFFICIAL else MambaBlockNaive
            self.official = (block_cls is MambaBlockOfficial)
            self.n_layers = 2 if arch == 'V8' else 3
            if d_state is None:
                d_state = 16 if arch == 'V8' else 32
            self.proj = nn.Linear(in_dim, D_MODEL)
            self.layers = nn.ModuleList([block_cls(D_MODEL, d_state=d_state)
                                         for _ in range(self.n_layers)])
        elif arch == 'LSTM':
            self.proj = nn.Linear(in_dim, D_MODEL)
            self.lstm = nn.LSTM(D_MODEL, D_MODEL, 3, batch_first=True,
                                dropout=0.1)
        elif arch == 'TF':
            self.proj = nn.Linear(in_dim, D_MODEL)
            pe = torch.zeros(1, 64, D_MODEL)
            pos = torch.arange(0, 64).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, D_MODEL, 2).float()
                            * (-math.log(10000.0) / D_MODEL))
            pe[0, :, 0::2] = torch.sin(pos * div)
            pe[0, :, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe)
            layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, 0.1,
                                               batch_first=True, activation='relu')
            self.tf = nn.TransformerEncoder(layer, 3)
        else:
            raise ValueError(f'未知架构 {arch}，可选 V8 / V9 / LSTM / TF')

        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(12 * D_MODEL, 12) if arch == 'V8' else \
            nn.Sequential(nn.Linear(12 * D_MODEL, 256), nn.ReLU(),
                          nn.Dropout(0.1), nn.Linear(256, 12))

        for m in [self.proj] + (list(self.output) if isinstance(self.output, nn.Sequential)
                                else [self.output]):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, L, C = x.shape
        if self.fix_permute:
            x = x.reshape(B * N, L, C)             # ← 修好的写法
        else:
            x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)   # 旧 bug，仅作 A/B 对照
        if self.arch in ('V8', 'V9'):
            x = self.proj(x)
            for l in self.layers:
                x = l(x)      # 两个块内部都自带 pre-norm 和残差，这里不再补 +x
        elif self.arch == 'LSTM':
            x, _ = self.lstm(self.proj(x))
        else:                                       # TF
            x = self.tf(self.proj(x) + self.pe[:, :L])
        x = self.norm(x)
        return self.output(x.reshape(B * N, L * self.d_model)).reshape(B, N, 12)


# ---------------------------------------------------------------- 指标
def compute_metrics(pred, true, std_y, mean_y, N):
    """
    先反标准化回真实速度，再分 horizon 算 MAE / RMSE / MAPE。

    ★ 2026-09-23 改为**掩零口径**（等于 DCRNN 官方的 null_val=0）★
      起因：把 HI 基线和文献比，METR-LA 差 91%，掩掉真值里的零之后降到 24%。
      查证后确认 DCRNN 官方的 masked_mae / masked_rmse / masked_mape 都用
      null_val=0，即**真值为 0 的点不计入指标**。本仓库原来的实现是全点计入，
      和文献不同口径，直接放同一张表就是不可比的。

      影响面：PeMS-BAY 真值非零占 100.0%，这一改几乎不动；
      METR-LA 真值非零占 87.9%（12.1% 是缺失被填的 0），改完差异很大。

      未掩零的值一并保留为 *_all，方便审稿人核对，但论文表里用掩零值。
    """
    sd = std_y.reshape(1, N, 1)
    mn = mean_y.reshape(1, N, 1)
    p = pred * sd + mn
    t = true * sd + mn
    # ★★ 2026-09-26 修：掩码必须判在**标准化空间**，与 quick_eval / MaskedMSELoss 一致 ★★
    #   原实现是 `m_null = tt > 0`（tt = 反标准化后的真值）。真值 0 经
    #   「标准化 → 反标准化」往返后不再精确等于 0：实测 METR-LA 的 2,060,359 个
    #   零值里有 **178,626 个（8.67%）** 变成 +3.5e-15~+7.1e-15 而**逃过掩码**，
    #   被当成有效点评 —— 那些点真值是 0、预测约等于节点均值，单点误差约 56 mph。
    #   后果：持久性 @60 被抬到 5.9624（正确值 5.3634，+11.2%），且**随浮点舍入
    #   浮动、不可复现** —— 任务书 §0.3(a) 里那个"复现不出来的 3.989/4.684/5.826"
    #   就是同一缺陷口径下逃逸集略有不同的产物，从来不是良定义的量。
    #   PeMS-BAY 只有 264 个点逃逸（0.0006%）→ 3.0487 vs 3.0483，可忽略，主线不受影响。
    #   修法与 quick_eval:873 / MaskedMSELoss:895 逐字相同：在标准化空间里跟
    #   原值 0 的位置 (-mean_y/std_y) 比 —— 那里 0 是精确表示的，不可能逃逸。
    null_norm = ((-mean_y) / std_y).reshape(1, N)
    res = {}
    for hmin, idx in HORIZONS.items():
        pp, tt = p[:, :, idx - 1], t[:, :, idx - 1]
        d = pp - tt
        key = str(hmin)
        m_all = np.ones_like(tt, dtype=bool)
        m_null = true[:, :, idx - 1] > null_norm   # ← 标准化空间，精确
        if not m_null.any():
            m_null = m_all
        res[f'mae_{key}'] = float(np.abs(d[m_null]).mean())
        res[f'rmse_{key}'] = float(np.sqrt((d[m_null] ** 2).mean()))
        # MAPE 另有近零真值问题：真值 0.1 时单点比值可达数百，能把整体从
        # 9% 抬到 400%（实测 399.1% 就是这个原因，不是代码 bug）。所以
        # 在掩零基础上再要求真值 > MAPE_MIN。
        mm = m_null & (tt > MAPE_MIN)
        res[f'mape_{key}'] = float(np.abs(d[mm] / tt[mm]).mean() * 100) \
            if mm.any() else float('nan')
        res[f'mape_raw_{key}'] = float(np.abs(d / np.maximum(np.abs(tt), 1e-3)).mean() * 100)
        res[f'mape_cov_{key}'] = float(mm.mean())
        # 未掩零的对照值
        res[f'mae_all_{key}'] = float(np.abs(d[m_all]).mean())
        res[f'rmse_all_{key}'] = float(np.sqrt((d[m_all] ** 2).mean()))
        res[f'null_frac_{key}'] = float(1.0 - m_null.mean())
    res['n_eval'] = int(pred.shape[0])
    return res


# ---------------------------------------------------------------- 基线
# ★★ 2026-09-27：HA 口径修正（这是一个**实现错误**，不是口径选择）★★
#
#   旧实现是 `np.zeros_like(Yte)` —— 在标准化空间里恒等于 0，也就是
#   **逐传感器常数**（训练集逐节点均值）。它**不是** DCRNN 论文表里的 HA。
#
#   DCRNN 官方 `scripts/eval_baseline_methods.py` 定义的是
#   **逐传感器 × 逐「一周中的同一相位」**：
#
#       period = 7 * 24 * 12                      # 2016 个 5 分钟槽 = 一周
#       for i in range(n_train, min(n_sample, n_train + period)):
#           inds = [j for j in range(i % period, n_train, period)]
#           historical = df.iloc[inds, :]
#           y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
#
#   三条要点，缺一条就复现不出已发表值：
#     1. 相位周期是 **2016（一周）**，不是 288（一天）、更不是常数。
#     2. 求均值时 **剔除 0**（`historical != null_val`，null_val=0）—— 缺测不参与。
#     3. 所以结果对 12 个时域**恒定** —— 这正是 DCRNN Table 1 里 HA 行
#        在 15/30/60 三档印着同一个 4.16 的原因。
#
#   实测（`ha_official.py`，四个独立锚点）：
#     METR-LA   4.1467 / 4.1466 / 4.1467   已发表 4.16（旧口径 7.506）
#     PeMS-BAY  2.8835 / 2.8813 / 2.8787   已发表 2.88（旧口径 5.434）
#     还原出的序列长度 34,272 / 52,116 与标准数据集逐位相同。
#
#   病因精确到**剖面粒度**（逐节点常数 vs 逐节点逐周相位），**不是**零值处理：
#   把官方口径的剔零去掉，数只动 0.01；把剖面退化成常数，就回到 7.5。
HA_PERIOD = 7 * 24 * 12     # 2016 = 一周的 5 分钟槽数，官方 period
BL_VERSION = 2              # 1 = 旧 HA（实现错误）；2 = 官方逐周相位。
                            # 用于让 results 文件里的旧基线缓存失效 —— 没有它，
                            # 续跑逻辑会认成「已有结果」，报出来的仍是 7.506。


def raw_speed_series(d):
    """把 (N, T, 24) 的滑窗还原成 (N, T+23) 的**原始**速度序列（不填缺测）。

    ★ 这是**步长 1** 的滑窗，不是不重叠切片：窗口 t 覆盖原始时刻 [t, t+23]，
      x[n,t,j] = raw[t+j]，y[n,t,h] = raw[t+12+h]，全序列长度 T+23。

      （第一版按 [x0 | y.reshape(N,-1)] 拼，长度变成 12+12T —— 时间轴被膨胀
       12 倍，`i % 2016` 于是等价于 14 小时周期而不是一周，逐周剖面因此失效。
       自检条件就是长度：METR-LA 应为 34,272、PeMS-BAY 应为 52,116。）
    """
    N, T, _ = d.shape
    x, y = d[:, :, :12], d[:, :, 12:]
    s = np.empty((N, T + 23), dtype=np.float32)
    s[:, :T] = x[:, :, 0]                 # raw[0 .. T-1]
    s[:, T:T + 12] = y[:, T - 12:, 0]     # raw[T .. T+11]
    s[:, T + 12:] = y[:, T - 1, 1:]       # raw[T+12 .. T+22]
    return s


def ha_profile(hist, pool_end):
    """官方 HA：逐节点 × 逐相位（period=HA_PERIOD）的均值，**剔除 0**。

    hist 必须是**原始**速度序列（未标准化、未填缺测）—— 填过缺测就无从剔零，
    标准化过就丢掉了速度量纲。
    """
    N = hist.shape[0]
    prof = np.full((N, HA_PERIOD), np.nan, dtype=np.float64)
    for p in range(HA_PERIOD):
        idx = np.arange(p, pool_end, HA_PERIOD)
        if idx.size == 0:
            continue
        seg = hist[:, idx].astype(np.float64)
        m = seg > 0
        n = m.sum(axis=1)
        tot = np.where(m, seg, 0.0).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            prof[:, p] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    # 兜底：某个 (节点, 相位) 在整个历史池里一个有效值都没有 → 退到该节点的
    # 全池均值；该节点整条都缺测 → 退到全数据均值。不兜底的话 NaN 会静默
    # 污染 MAE（NaN 参与均值 → 结果整体变 NaN，看起来像「跑挂了」）。
    bad = np.isnan(prof)
    if bad.any():
        pool = hist[:, :pool_end].astype(np.float64)
        pm = pool > 0
        pn = np.maximum(pm.sum(axis=1), 1)
        node_mu = np.where(pm, pool, 0.0).sum(axis=1) / pn
        gmu = float(node_mu[pm.any(axis=1)].mean()) if pm.any() else 0.0
        fb = np.where(pm.any(axis=1), node_mu, gmu)[:, None]
        prof = np.where(bad, fb, prof)
        log(f'  [HA] {int(bad.sum())} 个 (节点,相位) 无有效历史，已用节点均值兜底')
    return prof


def run_baselines(Xte, Yte, std_y, mean_y, N, hist_raw):
    """
    两条「不用学」的基线，必须和模型放在同一张表里。

      持久性(Persistence)：把最后观测到的速度直接当成未来 12 步的预测。
          交通速度自相关极高，这条基线在 15/30 分钟上非常强，
          论文里省略它会被审稿人直接质疑。

      历史均值(HA)：**逐传感器 × 逐「一周中的同一相位」**，取历史池均值、
          剔除缺测。定义见本节顶部的长注释。

    hist_raw 是必需参数（形状 (N, L) 的**原始**速度序列，train+val 拼接）：
    HA 的定义里就含「历史」，不把它传进来在物理上就只能是错的 —— 旧实现
    正是因为拿不到历史才退化成「逐传感器常数」。**不要给它默认值**，
    那样就又能静默地算错一次。
    """
    out = {}
    # Xte 是 (T, N, L, C)：取「最后一个观测时刻」且「特征 0（流量/速度）」那个标量
    last = Xte[:, :, 11, 0]                       # (T, N)
    persist = np.repeat(last[:, :, None], 12, axis=2)   # (T, N, 12)

    out['基线-持久性'] = compute_metrics(persist, Yte, std_y, mean_y, N)

    # ---- 历史均值：官方逐周相位口径 ----
    # 历史池 = train+val 拼出来的原始序列（长度 hl，末尾与 test 重叠 23 步）。
    # 官方取 n_train = round(0.8 * 全序列长)，这里照抄；hl 已含 23 步重叠，
    # 所以全序列长 = hl + Tte。
    Tte = Yte.shape[0]
    hl = hist_raw.shape[1]
    pool_end = min(hl, int(round((hl + Tte) * 0.8)))
    prof = ha_profile(hist_raw, pool_end)

    # test 局部窗 t 的第 h 步目标，落在原始时刻 (hl-23) + t + 12 + h。
    # hl-23 = Ttr+Tva = 第一个 test 窗口的全局窗口号。
    tidx = np.arange(Tte)
    mn = mean_y.reshape(N, 1)
    sd = std_y.reshape(N, 1)
    ha_pred = np.empty((Tte, N, 12), dtype=np.float32)
    for h in range(12):
        abs_idx = (hl - 23) + tidx + 12 + h
        raw = prof[:, abs_idx % HA_PERIOD]        # (N, Tte) 真实速度
        ha_pred[:, :, h] = ((raw - mn) / sd).T    # → 标准化空间，与 Yte 同空间

    out['基线-历史均值'] = compute_metrics(ha_pred, Yte, std_y, mean_y, N)
    for v in out.values():
        v.update(params=0, best_epoch=0, val_mae=0.0,
                 epoch_time=0.0, epochs_run=0, bl_version=BL_VERSION)
    return out


@torch.no_grad()
def infer(model, X, bs):
    model.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def quick_eval(model, X, Y, std_y, mean_y, N, bs):
    """只取前 1024 个样本 —— 早停/选最优用，不用于最终报告。
    口径与 compute_metrics 一致：真值为 0（缺失）的点不计。"""
    pred = infer(model, X[:1024], bs)
    true = Y[:1024]
    if not np.isfinite(pred).all():
        return float('inf'), pred
    d = (pred - true) * std_y.reshape(1, N, 1)     # 均值在差分里抵消
    m = true > ((-mean_y) / std_y).reshape(1, N, 1)
    if not m.any():
        m = np.ones_like(true, dtype=bool)
    return float(np.abs(d[m]).mean()), pred


class MaskedMSELoss(nn.Module):
    """DCRNN 口径的掩码 MSE：真值为 0（缺失）的点不计入损失。

    训练和评价必须同一个口径。只在评价时掩零、训练时仍去拟合那些 0，
    模型会把容量浪费在学"缺失"上，评出来的数也就没法和文献比。
    """

    def __init__(self, null_norm):
        super().__init__()
        # register_buffer 只收 torch.Tensor，numpy 数组会直接抛 TypeError。
        # ★ 必须 reshape 成 (1,N,1) 再注册：调用方给的 (-mean_y)/std_y 本身是 (N,1,1)，
        #    直接广播会与 (B,N,12) 在 dim 0 撞车（报 "tensor a (4) vs tensor b (325)"）。
        v = np.asarray(null_norm, dtype=np.float32).reshape(-1)
        self.register_buffer('null_norm', torch.as_tensor(v).reshape(1, -1, 1))

    def forward(self, pred, target):
        m = (target > self.null_norm).float()
        n = m.sum()
        if n < 1:
            return F.mse_loss(pred, target)
        return ((pred - target) ** 2 * m).sum() / n


# ---------------------------------------------------------------- nan 尸检
def nan_autopsy(model, X, tag, bs):
    """前向出非有限值时，定位到具体批次和具体模块，返回可写日志的若干行。

    2026-09-23 加。起因：一次 1 小时的 verify 跑完只留下一句
    "[警告] 测试集预测含 nan —— 该配置结果作废"，没有任何线索，
    只能再花一轮手搓诊断格去查（还因为 OOM / 重启清目录白跑了两轮）。
    让失败自己带上证据，比事后追查便宜得多。

    三件事一次拿到：
      1. 哪些批次出问题（分批扫，用与训练一致的 bs，避免一次性铺开爆显存）
      2. 逐模块的 absmax —— 第一个爆掉的模块名就是定性依据
      3. 同输入关掉官方 fast path 再跑一次 —— 区分「核的病」和「模型的病」
    """
    lines = []
    fails = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        try:
            with torch.no_grad():
                y = model(xb)
            if not bool(torch.isfinite(y).all()):
                fails.append(i)
            del y
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lines.append(f'    [{tag}] 第 {i} 批显存不足，跳过')
        finally:
            del xb
        if len(fails) >= 3:
            break
    if not fails:
        lines.append(f'    [{tag}] 逐批扫过没找到非有限输出 —— '
                     f'可能是批次划分方式不同才触发，不一定是数据问题')
        return lines

    lines.append(f'    [{tag}] 出问题的批次起点: {fails}')
    i0 = fails[0]
    xb = torch.from_numpy(X[i0:i0 + bs]).to(DEVICE)
    bad = []

    def mk(name):
        def f(mod, inp, outp):
            t = outp[0] if isinstance(outp, tuple) else outp
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                fin = bool(torch.isfinite(t).all())
                amax = float(t.abs().max()) if fin else float('inf')
                if (not fin) or amax > 1e4:
                    bad.append((name, type(mod).__name__, fin, amax, tuple(t.shape)))
        return f

    hs = [m.register_forward_hook(mk(n)) for n, m in model.named_modules() if n]
    with torch.no_grad():
        y = model(xb)
    for h in hs:
        h.remove()
    ok = bool(torch.isfinite(y).all())
    lines.append(f'    第 {i0} 批输出有限={ok}  '
                 f'absmax={float(y.abs().max()):.4g}')
    for name, cls, fin, amax, shp in bad[:20]:
        lines.append(f'      {name:<44} {cls:<18} finite={fin} '
                     f'absmax={amax:.4g} {shp}')
    del y
    torch.cuda.empty_cache()

    # 决定性对照：同一批输入、同一份权重，只切换官方快路核
    n_off = 0
    for mod in model.modules():
        if hasattr(mod, 'use_fast_path'):
            mod.use_fast_path = False
            n_off += 1
    if n_off:
        with torch.no_grad():
            y2 = model(xb)
        ok2 = bool(torch.isfinite(y2).all())
        a2 = float(y2.abs().max()) if ok2 else float('inf')
        lines.append(f'    关掉 fast path（{n_off} 个模块）后 输出有限={ok2} '
                     f'absmax={a2:.4g}')
        lines.append('      → 关掉后变有限 = 官方 CUDA 核的数值问题，模型本身没问题；'
                     '仍非有限 = 模型结构的放大问题')
        del y2
        torch.cuda.empty_cache()
    del xb
    torch.cuda.empty_cache()
    return lines


# ---------------------------------------------------------------- 训练
def train_one(cfg, data, seed, epochs, bs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val, test, std_y, mean_y, N = data
    Xtr, Ytr = train
    Xva, Yva = val
    Xte, Yte = test

    # in_dim 显式传进来（main 里按实际造出来的通道数算），不再由 1+k 反推 ——
    # 反推的写法在加了时段通道之后会静默对不上，然后报一个难懂的 shape 错误。
    model = Predictor(cfg.get('in_dim', 1 + cfg['k']), arch=cfg['arch'],
                      block_cls=cfg['block_cls'],
                      fix_permute=cfg.get('fix_permute', True)).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # 掩码 MSE：真值 0 = 缺失，不计入。null 在标准化空间的位置是 -mean_y/std_y。
    # ★ 必须 .to(DEVICE)：crit 不是 model 的子模块，model.to(cuda) 带不上它，
    #    buffer 留在 CPU 上会撞 "found at least two devices, cuda:0 and cpu"。
    crit = MaskedMSELoss((-mean_y) / std_y).to(DEVICE)

    # ★★ 2026-09-26 新增：PAPER_RECOMPUTE=1 时**跳过训练**，只载入已有检查点重算指标 ★★
    #   为什么这样是安全的：早停与选点走 quick_eval（:873），它的掩码口径本来就是
    #   在标准化空间判的（**本来就对**），所以修 compute_metrics 不改变任何一次
    #   权重更新 —— 已有 `_best.pt` 的权重可以原样复用，不必重训。
    #   为什么单独一个分支而不是在循环里加 if：这样重算路径**碰不到**任何训练语句，
    #   也就不可能覆写 `_last.pt`/`_best.pt`。重算不该产生副作用。
    #   失败一律**大声报错 + 返回 None**，不写入结果文件 —— 缺键是可查的，
    #   悄悄写一个貌似成功的数字才是这个项目反复踩的坑。
    if RECOMPUTE:
        ck = f'{cfg["id"]}_best.pt'
        if not os.path.exists(ck):
            log(f'  [重算] 缺 {ck} —— 跳过（没有检查点就无从重算）')
            return None
        try:
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
        except Exception as e:
            log(f'  [重算] {ck} 与配置 {cfg["id"]} 不符 —— 跳过：'
                f'{type(e).__name__}: {e}')
            return None
        log(f'  [重算] 载入 {ck}，只推理、不训练、不覆写任何 .pt')
        pred = infer(model, Xte, bs)
        if not np.isfinite(pred).all():
            log(f'  [重算] {cfg["id"]} 预测含 nan/inf —— 作废，不写入结果')
            return None
        m = compute_metrics(pred, Yte, std_y, mean_y, N)
        m.update(params=n_param, best_epoch=-1, val_mae=float('nan'),
                 epoch_time=0.0, epochs_run=0, n_train=0, recomputed=True)
        log(f'  [重算] {cfg["id"]}  MAE@15={m["mae_15"]:.4f}  '
            f'MAE@30={m["mae_30"]:.4f}  MAE@60={m["mae_60"]:.4f}')
        return m

    S = Xtr.shape[0]
    n_b = (S + bs - 1) // bs
    best_val, best_state, best_ep, bad = float('inf'), None, -1, 0
    log(f'  参数 {n_param:,}   训练样本 {S:,}   每 epoch {n_b:,} 步   bs={bs}')
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(S)
        tot = 0.0
        for bi in range(n_b):
            idx = perm[bi * bs:(bi + 1) * bs]
            opt.zero_grad()
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                log(f'  [中止] ep{ep} 第 {bi} 步 loss 非有限 —— '
                    f'权重已存 {cfg["id"]}_last.pt')
                torch.save(model.state_dict(), f'{cfg["id"]}_last.pt')
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        tr_loss = tot / n_b

        v_mae, _ = quick_eval(model, Xva, Yva, std_y, mean_y, N, bs)
        flag = ''
        if v_mae < best_val:
            best_val, best_ep, bad = v_mae, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            flag = '  ★'
        else:
            bad += 1
        log(f'  ep {ep:3d}/{epochs}  train_loss={tr_loss:.6f}  '
            f'val_MAE={v_mae:.4f}  ({time.time()-t0:.0f}s){flag}')

        if bad >= PATIENCE:
            log(f'  早停：验证集 {PATIENCE} 轮没改善，最优在 ep{best_ep}')
            break

    # 两个权重都存。2026-09-23 教训：一次 nan 作废之后只剩 _best，
    # 而 nan 恰恰是 _best 在测试集上出的 —— 想做「好轮次 vs 坏轮次」的对比时，
    # 坏轮次的权重已经没了。重启 Kaggle 会话又把 _best 一起清掉，彻底查不了。
    torch.save(model.state_dict(), f'{cfg["id"]}_last.pt')
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f'{cfg["id"]}_best.pt')

    pred = infer(model, Xte, bs)
    if not np.isfinite(pred).all():
        log(f'  [警告] 测试集预测含 nan —— 该配置结果作废')
        log('  --- 自动尸检：定位到批次与模块 ---')
        try:
            for ln in nan_autopsy(model, Xte, 'test', bs):
                log(ln)
        except Exception as e:
            log(f'    尸检本身失败（{type(e).__name__}: {e}），权重已存 '
                f'{cfg["id"]}_last.pt 可事后查')
        return None
    m = compute_metrics(pred, Yte, std_y, mean_y, N)
    m.update(params=n_param, best_epoch=best_ep, val_mae=best_val,
             epoch_time=(time.time() - t0) / max(1, ep + 1), epochs_run=ep + 1,
             n_train=S)
    return m


# ---------------------------------------------------------------- 主流程
# ★ 本文件没有 __main__ 守卫：从第 741 行往下全是顶层代码，import 一下就会真跑整个矩阵。
#   2026-09-23 踩过一次 —— 用 python -c "import paper_experiments" 做单测，
#   结果在 4050 上真起了一轮实验并覆写了 paper_results.json。
#   所以这里立一道闸：被 import 且没人显式设 PAPER_MODE 时直接报错。
#   baseline_only.py / diag_*.py 都设了 PAPER_MODE=checkdata，不受影响。
if __name__ != '__main__' and 'PAPER_MODE' not in os.environ:
    raise RuntimeError(
        'paper_experiments 是从 import 进来的，且环境里没有 PAPER_MODE。\n'
        '  本模块一旦开始执行就会跑完整个实验矩阵（可能十几个 GPU 小时）。\n'
        '  只想复用函数：先设 PAPER_MODE=checkdata 再 import。\n'
        '  真要跑实验：python paper_experiments.py 直接执行。')

log('=' * 78)
log(f'投稿版实验   模式={MODE}   设备={DEVICE}')
if DEVICE == 'cuda':
    p = torch.cuda.get_device_properties(0)
    log(f'  {torch.cuda.get_device_name(0)}  {p.total_memory/1024**3:.1f}GB   '
        f'torch {torch.__version__}')
log(f'  官方 mamba_ssm: {"可用" if HAS_OFFICIAL else "不可用（只跑手写块）"}')
# 特征配置打在头上：A/B 是靠两份日志比出来的，配置必须自己写进日志里，
# 不能靠"我记得当时开没开"。
log(f'  时段通道(PAPER_TOD)={"开" if USE_TOD else "关"}'
    f'  IN_DIM={1 + NUM_NEIGHBORS + (1 if USE_TOD else 0)}（real/self/random 设定下）'
    f'  强制读parquet(PAPER_FORCE_PARQUET)={"开" if FORCE_PARQUET else "关"}')
if MODE == 'quick':
    log('  ⚠️  quick 模式：训练集和测试集都被截断到前 800 步，')
    log('      用途只是验证管线跑得通。这一模式的任何数字都不得进论文。')
if MODE == 'verify':
    log(f'  verify 模式：PeMS-BAY 全量 325 节点，但各划分只取前 {VERIFY_STEPS} 个时刻、')
    log(f'      跑 {VERIFY_EPOCHS} 轮 —— 用途是 A/B 验证 permute 修复，')
    log('      目的是「学不学得起来」这个是非题，不是精度。数字不得进论文。')
log('=' * 78)

results = {}
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, encoding='utf-8') as f:
        results = json.load(f)
    log(f'续跑：已有 {len(results)} 条结果')


def save_results():
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# 组装待跑清单
if MODE == 'checkdata':
    # 只验证数据读不读得通，不训练。Kaggle 上先跑这个：
    # 几秒钟就能知道目录找没找到、节点数对不对，不用赌一小时。
    log('数据自检（不训练）')
    for ds in WANT_DS:
        try:
            tr, va, te, adj, N, tds = load_dataset(ds)
            deg = (adj > 0).sum(axis=1)
            log(f'    ✓ 形状 train{tr.shape} val{va.shape} test{te.shape}')
            if tds is None:
                log(f'      时段通道: 无（PAPER_TOD=0，或数据里没有 _d1 列）')
            else:
                log(f'      时段通道: {[t.shape for t in tds]}  '
                    f'取值范围 {float(tds[0].min()):.4f}~{float(tds[0].max()):.4f}')
            log(f'      邻接非零 {int((adj > 0).sum()):,}  '
                f'邻居数 中位{int(np.median(deg))}/最大{int(deg.max())}  '
                f'孤立节点 {int((deg == 0).sum())}')
        except Exception as e:
            log(f'    ✗ {ds}: {type(e).__name__}: {e}')
    # ★ 这里【不能】用 sys.exit(0)。在 Jupyter/Kaggle 的 cell 里它抛 SystemExit，
    #   被 IPython 吞掉后 cell 直接结束，看起来就像"什么都没输出"——
    #   2026-09-23 用户就是这么被误导的。改成 jobs=[] 让主循环空转，
    #   后面的写文件与汇总再用 MODE 判断跳过，输出一定能出来。
    jobs = []
elif MODE == 'quick':
    # 四种架构 + 两种邻居设定都过一遍，确保每条代码路径都跑得到
    jobs = [(ds, m, arch, 'naive', 1, dict(quick=True, epochs=2))
            for ds in ['PeMS-BAY-50'] for m in ['real', 'self']
            for arch in ['V8', 'V9', 'LSTM', 'TF']]
elif MODE == 'speed':
    # 效率对照必须在完全相同的批大小下比，否则加速比没有意义，
    # 所以这里统一用 naive 的预算（手写版跑得动的最紧那档）。
    jobs = [(ds, 'real', arch, blk, 1,
             dict(quick=False, epochs=EPOCHS, subset=5000,
                  seq_budget=MAX_SEQ_PER_BATCH))
            for ds in ['PeMS-BAY-50'] for arch in ['V8', 'V9']
            for blk in ['naive', 'official']]
elif MODE == 'verify':
    # 便宜的真数据 A/B：同一份数据、同一个种子，只差 permute 修没修。
    # 目的不是出精度数字，是回答「修完之后模型到底学不学得起来」——
    # 这个前提不成立的话，后面几百个 GPU 小时全是白烧。
    # 所以只用全量节点里的一小段时刻、跑 VERIFY_EPOCHS 轮。
    # ★ 2026-09-23：suffix 里必须带上 subset 步数和轮数。
    #   原来只有 '|修好'，于是把 PAPER_VERIFY_STEPS 从 2000 改成 10419 之后
    #   配置 ID 没变，续跑逻辑认成「已有结果」直接跳过 —— 白跑一次，
    #   而且报出来的是旧协议 2000 步的数字，看起来像新跑的结果。
    #   基线那边有 n_eval 防呆，模型这边当时没有，现在把协议编进 ID。
    jobs = [(ds, 'real', arch, VERIFY_BLOCK, 1,
             dict(quick=False, epochs=VERIFY_EPOCHS, subset=VERIFY_STEPS,
                  fix_permute=fx,
                  # ★ 2026-09-24：前面加 '|缺测修'。
                  #   改了标准化（缺测 0 不再当真实车速）之后，旧 ID 下的结果作废，
                  #   但 ID 不变的话续跑逻辑会认成「已有结果」直接跳过 ——
                  #   报出来的还是旧口径的数字，看起来却像新跑的。
                  suffix='|缺测修' + ('|修好' if fx else '|旧bug')
                         + f'|n{VERIFY_STEPS}ep{VERIFY_EPOCHS}'))
            for ds in VERIFY_DS for arch in ['V9']
            for fx in VERIFY_ARMS]
else:
    jobs = [(ds, m, arch, blk, sd, dict(quick=False, epochs=EPOCHS))
            for ds in WANT_DS for m in WANT_MODES
            for arch in WANT_ARCH for blk in WANT_BLOCK for sd in SEEDS]

log(f'\n共 {len(jobs)} 个配置')

for ds_name, mode, arch, blk, seed, opt_ in jobs:
    # ★ 2026-09-24：时段通道要写进 ID。
    #   否则 ...|s1 和开了时段通道的 ...|s1 撞成同一个 ID，续跑时会认成
    #   「已有结果」直接跳过 —— 报出来的还是旧数字，看起来却像新跑的。
    # PAPER_SUFFIX：给整批加一个标记。做 A/B 时用得上 ——
    # 加了时段通道那次（|tod）和没加那次必须各跑一遍，且必须是**同一份
    # 代码**跑出来的两个数，不能拿旧版本留下的结果当对照：
    # 旧结果的代码和数据管道可能已经变了，那就不是单变量对比了。
    cid = (f'{ds_name}|{mode}|{arch}|{blk}|s{seed}'
           + ('|tod' if USE_TOD else '')
           + os.environ.get('PAPER_SUFFIX', '')
           + opt_.get('suffix', ''))
    if cid in results:
        log(f'\n[跳过] {cid}（已有结果）')
        continue
    log(f'\n{"=" * 78}\n[{len(results)+1}/{len(jobs)}] {cid}\n{"=" * 78}')
    try:
        # --- 数据（每个配置重新载，避免多个配置交叉占内存）
        train3, val3, test3, adj, N, tods = load_dataset(
            ds_name, quick=opt_.get('quick', False), subset=opt_.get('subset'))
        # 反标准化参数
        # ★ 2026-09-24 改：缺测（原始值 0）不再当成真实车速 ★
        #   评价指标和训练损失（MaskedMSELoss）都已按 DCRNN 口径 null_val=0 掩零，
        #   唯独**输入这一侧**没有 —— 两半口径不一致，就是这个 bug。
        #   实测后果：node 117 平时 66.7 mph、std 仅 1.07，一个缺测点标准化后是
        #   -55 ~ -66σ（train/test 均报 55.67σ，正出自该节点）。三层网络把它滚到
        #   1e31，LayerNorm 求方差时平方溢出 → nan，ep5 第 226 步中止。
        #   （2026-09-23 那次「train 改取末尾」并没修这个，只是换了窗口，
        #     把 66σ 碰巧降到 55.67σ；所以关掉官方 fast path 照样炸 —— 不是核的病。）
        #   现在：逐节点统计量只按有效值（>0）算，输入里的缺测用该节点均值填，
        #   填完标准化恰好是 0（「无信息」），而不是「该路段瞬间瘫痪」。
        tx = train3[..., :12]
        ty = train3[..., 12:]

        def _stat(a):
            """逐节点 mean/std，缺测（0）不计入。整节点全缺测时退化为 mean=0/std=1。"""
            m = a > 0
            ok = m.any(axis=(1, 2), keepdims=True)
            n = np.maximum(m.sum(axis=(1, 2), keepdims=True), 1)
            mu = (a * m).sum(axis=(1, 2), keepdims=True) / n
            var = (((a - mu) ** 2) * m).sum(axis=(1, 2), keepdims=True) / n
            mu = np.where(ok, mu, 0.0)
            sd = np.where(ok, np.sqrt(var), 1.0) + 1e-8
            return mu, sd

        def _fill(a, mu):
            """缺测填成该节点均值（用的是 train 的均值，val/test 同样填 train 的）。"""
            return np.where(a > 0, a, mu)

        mean_x, std_x = _stat(tx)
        mean_y, std_y = _stat(ty)
        # ★ 只填 x，**不填 y**：y 里的 0 是掩零口径的锚点。
        #   指标判缺失用的是 `true > (-mean_y/std_y)`；若把 y 的缺测填成节点均值，
        #   归一化后恰好是 0，而 0 大于那个负数阈值 → 会被判成**有效点**，
        #   等于把缺失当真值评，比不填更糟。
        tr_x = (_fill(tx, mean_x) - mean_x) / std_x
        va_x = (_fill(val3[..., :12], mean_x) - mean_x) / std_x
        te_x = (_fill(test3[..., :12], mean_x) - mean_x) / std_x
        tr_y = (ty - mean_y) / std_y
        va_y = (val3[..., 12:] - mean_y) / std_y
        te_y = (test3[..., 12:] - mean_y) / std_y
        # 分布漂移仪表：不管裁不裁，都先把各划分的越界比例和 absmax 打出来。
        # 这一次 nan 就是靠"train 极差 4.8 而测试值到 65.8"这条线索定位的，
        # 以后每个配置都留一行，免得下次再靠猜。
        for _tag, _a in (('train', tr_x), ('val', va_x), ('test', te_x)):
            _mx = max(abs(float(_a.min())), abs(float(_a.max())))
            _n10 = count_abs_gt(_a, 10.0)
            log(f'  [分布] {_tag}: absmax={_mx:.4g}  |值|>10 的有 {_n10:,}/{_a.size:,} '
                f'({100.0 * _n10 / _a.size:.4f}%)')
        if CLIP_NORM > 0:
            for _tag, _a in (('train', tr_x), ('val', va_x), ('test', te_x)):
                _n = count_abs_gt(_a, CLIP_NORM)
                if _n:
                    log(f'  [输入保护] {_tag}: 裁剪 {_n:,} 个值到 ±{CLIP_NORM}')
            tr_x = np.clip(tr_x, -CLIP_NORM, CLIP_NORM)
            va_x = np.clip(va_x, -CLIP_NORM, CLIP_NORM)
            te_x = np.clip(te_x, -CLIP_NORM, CLIP_NORM)
        tr_n = (tr_x, tr_y)
        va_n = (va_x, va_y)
        te_n = (te_x, te_y)
        # ★ 2026-09-27：HA 官方口径要的是**原始**历史序列（未标准化、未填缺测）——
        #   官方是在原始 DataFrame 上按 period=2016 取相位均值并剔除 0 的。
        #   这里在 train3/val3 被 del 之前拼出来：**只有这一处**还拿得到历史，
        #   所以必须在这里算，不能挪到 run_baselines 里去（那边只有 test）。
        #   两段末尾重叠 23 步，衔接处只取一次。
        hist_raw = np.concatenate(
            [raw_speed_series(train3), raw_speed_series(val3)[:, 23:]], axis=1)
        del train3, val3, test3, tx, ty
        gc.collect()

        nbr_idx, k = build_neighbors(adj, mode, seed=seed)
        del adj

        # 时段通道：USE_TOD 开着却拿不到，说明数据的 _d1 列没读到 ——
        # 这时静默按"没加特征"跑完，会得出"时段特征没用"的错误结论。
        if USE_TOD and tods is None:
            raise TodUnavailable(
                f'{cid}: PAPER_TOD=1（默认开），但 load_dataset 没给出时段通道。')
        in_dim = 1 + k + (1 if tods is not None else 0)
        if tods is not None:
            log(f'  时段通道: train{tods[0].shape} val{tods[1].shape} '
                f'test{tods[2].shape}  范围 '
                f'{float(tods[0].min()):.4f}~{float(tods[0].max()):.4f}（{TOD_DESC}）')
        log(f'  邻接设定={mode}  邻居数={k}  IN_DIM={in_dim}')
        te_pair = make_samples(*te_n, nbr_idx, None if tods is None else tods[2])

        # 基线每个数据集只算一次（与邻居设定无关）。
        # ★ 必须校验规模：verify/quick 模式只取很少的时刻，若把那种小规模基线
        #   缓存下来，全量跑时会当成「已有结果」直接跳过，表格里就是错的基线。
        bkey = f'{ds_name}|基线-持久性'
        n_te = te_pair[0].shape[0]
        cached = results.get(bkey)
        if cached is not None and cached.get('n_eval') != n_te:
            log(f'  [注意] 已缓存的基线是按 {cached.get("n_eval")} 条测试样本算的，'
                f'当前是 {n_te} 条 —— 规模不符，重算')
            cached = None
        # ★ 2026-09-27：HA 口径修正（逐传感器常数 → 官方逐周相位）之后，
        #   旧结果文件里的 `基线-历史均值` 必须作废。只比 n_eval 是挡不住的 ——
        #   n_eval 一样，于是续跑逻辑认成「已有结果」，报出来的仍是 7.506。
        #   bl_version 由此而来：老行没有这个键，一律重算。
        if cached is not None and cached.get('bl_version') != BL_VERSION:
            log(f'  [注意] 已缓存的基线是 bl_version='
                f'{cached.get("bl_version")}（旧 HA 口径），当前 {BL_VERSION} —— 重算')
            cached = None
        if cached is None:
            bl = run_baselines(te_pair[0], te_pair[1], std_y, mean_y, N, hist_raw)
            for nm, v in bl.items():
                results[f'{ds_name}|{nm}'] = v
            save_results()
            log('  ' + '  '.join(
                f'{n}: MAE@15={v["mae_15"]:.3f} @60={v["mae_60"]:.3f}'
                for n, v in bl.items()))
        del hist_raw

        data = (make_samples(*tr_n, nbr_idx, None if tods is None else tods[0]),
                make_samples(*va_n, nbr_idx, None if tods is None else tods[1]),
                te_pair, std_y, mean_y, N)
        del tr_n, va_n, te_n, tods
        gc.collect()

        # --- 显存护栏：控制「每批序列数」≈ seq_budget，而不是控制样本数
        #     手写扫描会把 deltaA 展开成 (bs*N, L, d_inner, d_state)，
        #     N=325 时用 1600 就已经 4GB 起；官方核不展开，可以放大 4 倍。
        if blk == 'official' and not HAS_OFFICIAL:
            log('  [跳过] 官方核不可用（先 !pip install mamba-ssm causal-conv1d）')
            continue
        budget = opt_.get('seq_budget') or SEQ_BUDGET[blk]
        bs = 8 if opt_.get('quick') else max(1, budget // N)
        block_cls = MambaBlockOfficial if blk == 'official' else MambaBlockNaive
        log(f'  批大小 bs={bs}（每批 {bs*N:,} 条序列，预算 {budget:,}）')

        mcfg = dict(id=cid.replace('|', '_').replace('-', ''), k=k,
                    in_dim=in_dim,
                    arch=arch, block_cls=block_cls,
                    fix_permute=opt_.get('fix_permute', True))
        m = train_one(mcfg, data, seed, opt_['epochs'], bs)
        del data
        gc.collect()
        torch.cuda.empty_cache()
        if m is None:
            continue
        m['cfg'] = dict(dataset=ds_name, neighbors=mode, arch=arch,
                        block=blk, seed=seed, N=N, bs=bs)
        results[cid] = m
        save_results()
        log(f'  ✓ MAE@15={m["mae_15"]:.3f} @30={m["mae_30"]:.3f} @60={m["mae_60"]:.3f}  '
            f'RMSE@60={m["rmse_60"]:.3f}  MAPE@60={m["mape_60"]:.2f}% '
            f'(未过滤 {m["mape_raw_60"]:.1f}%)')
    except Exception as e:
        log(f'  [失败] {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------------- 汇总
if MODE == 'checkdata':
    # 自检已经打印完了，汇总对它是噪音；更不能写文件。
    log('')
    log('（checkdata 模式：只做数据自检，不训练、不写文件、不做汇总）')
elif not results:
    log('')
    log('没有结果。')
    # 这里也【不用】sys.exit —— notebook 里 SystemExit 会把 cell 输出吞掉。
else:
    log('\n' + '=' * 100)
    log('汇总（MAPE 为过滤近零真值后的值，括号内是未过滤值）')
    log('=' * 100)
    hdr = (f'{"配置":<34} {"参数":>9} '
           f'{"MAE@15":>7} {"MAE@30":>7} {"MAE@60":>7} '
           f'{"RMSE@60":>8} {"MAPE@60":>14} {"vs持久性@60":>11} {"s/ep":>7}')
    log(hdr)
    log('-' * len(hdr))
    for cid, m in sorted(results.items()):
        ds = m.get('cfg', {}).get('dataset')
        b = results.get(f'{ds}|基线-持久性') if ds else None
        # 相对持久性基线的 MAE@60 降幅，正数 = 比"什么都不做"好
        imp = f'{(b["mae_60"] - m["mae_60"]) / b["mae_60"] * 100:>10.1f}%' \
            if (b and m['params']) else f'{"—":>11}'
        log(f'{cid:<34} {m["params"]:>9,} '
            f'{m["mae_15"]:>7.3f} {m["mae_30"]:>7.3f} {m["mae_60"]:>7.3f} '
            f'{m["rmse_60"]:>8.3f} '
            f'{m["mape_60"]:>7.2f}%({m["mape_raw_60"]:>5.1f}%) '
            f'{imp} {m["epoch_time"]:>7.1f}')
    log('=' * 100)
    log(f'结果已写入 {RESULT_FILE}；权重存为 <配置ID>_best.pt')
    log("""
    论文里怎么用这张表：

      · 消融主线：同一数据集、同一架构下比较 real / random / self 三行的 MAE@60。
          若 random ≈ real  → Mamba 并没有真正利用空间结构，显式邻居特征对它增益有限
          若 real << random → 空间信息确实被用上了
        两种结果都能写，都是实测结论；后者更常规，前者更有意思。

      · self 那一行是关键对照：它和 real 参数量完全一致、输入维度完全一致，
        唯一的区别是那 5 路"邻居"换成自己 —— 所以 real 相对 self 的提升
        可以干净地归因给空间信息，不含参数量混淆。

      · MAPE 一定要报过滤后的值，并在文中写明过滤阈值和覆盖率
        （mape_cov 字段），否则那 0.006% 的零值点会把 MAPE 抬到 400%。

      · 效率结论来自 speed 模式或 probe_throughput.py，规模与消融实验不同，
        写的时候要说明"速度是 per-step 属性"，不能把两处的数字混在一张表里比。
        ⚠️ 2026-09-23：手写块的 dt_rank 已从写死的 1 对齐到官方的 ⌈d_model/16⌉=8，
        且补了块内 pre-norm 与残差 —— 两个块现在才结构等价。
        **此前所有加速比（2.58×/2.71×/2.74×/2.60×/2.79×/2.63×）都要作废重测。**
    """)

    # ---------------------------------------------------------------- verify 判定
    if MODE == 'verify':
        log('\n' + '=' * 100)
        log('verify 判定：permute 修复到底有没有让模型学起来')
        log('=' * 100)
        base = results.get('PeMS-BAY|基线-持久性')
        # ★ 2026-09-24 改 endswith → in：
        #   2026-09-23 往 suffix 后面加了 '|n{STEPS}ep{EPOCHS}' 之后，ID 结尾不再
        #   是 '|修好'，这几处 endswith 就再也匹配不到东西 —— 于是**两条臂都跑成功
        #   也会打印「没有可判定的结果」**，看起来像炸了。上次那句报错很可能就是这个。
        rows = [(c, m) for c, m in sorted(results.items())
                if '|修好' in c or '|旧bug' in c]

        if not rows:
            # 2026-09-23 改：原来这里写"两条对照都没跑完"，但最常见的情形其实是
            # 「跑完了、但前向出 nan 被 train_one 作废」，配置压根没进 results。
            # 把话说准，否则读日志的人会往错的方向查。
            log('  没有可判定的结果。两种可能，看上面：')
            log('    (a) 出现过 [警告] 测试集预测含 nan —— 那是前向数值溢出，'
                '要查模型不是查协议；')
            log('    (b) 没跑到最后就被中断了。')
        else:
            log(f'  {"对照":<10} {"MAE@15":>8} {"MAE@30":>8} {"MAE@60":>8} '
                f'{"最优轮次":>9} {"总轮数":>7} {"vs持久性@60":>12}')
            log('  ' + '-' * 74)
            for c, m in rows:
                tag = '修好' if '|修好' in c else '旧bug'
                imp = (f'{(base["mae_60"] - m["mae_60"]) / base["mae_60"] * 100:>11.1f}%'
                       if base else f'{"—":>12}')
                log(f'  {tag:<10} {m["mae_15"]:>8.3f} {m["mae_30"]:>8.3f} '
                    f'{m["mae_60"]:>8.3f} {m["best_epoch"]:>9d} '
                    f'{m["epochs_run"]:>7d} {imp}')
            if base:
                log(f'  {"持久性":<10} {base["mae_15"]:>8.3f} {base["mae_30"]:>8.3f} '
                    f'{base["mae_60"]:>8.3f}')

            good = next((m for c, m in rows if '|修好' in c), None)
            bad = next((m for c, m in rows if '|旧bug' in c), None)
            log('')
            if bad:
                log(f'  旧bug：最优轮次 ep{bad["best_epoch"]} / 共 {bad["epochs_run"]} 轮，'
                    f'val_MAE 最好 {bad["val_mae"]:.3f}。')
            if good:
                log(f'  修好：最优轮次 ep{good["best_epoch"]} / 共 {good["epochs_run"]} 轮。')
            log('')
            # ★ 规模门槛：小规模下模型本来就学不动，这时候打勾是假阳性。
            #   （本脚本第一次冒烟就是这么被骗的：40 个样本 2 轮，也打了 ✓）
            n_tr = max((m.get('n_train') or 0) for _, m in rows)
            enough = n_tr >= 2000 and max(m['epochs_run'] for _, m in rows) >= 10
            if not enough:
                log(f'  ⚠️  本次规模太小（训练样本 {n_tr:,}、最多 '
                    f'{max(m["epochs_run"] for _, m in rows)} 轮），模型本来就没法学。')
                log('     这一跑只能说明「管线通了、nan 没了」，不能判定 permute 修复的效果。')
                log(f'     要出判定，至少 PAPER_VERIFY_STEPS≥2000、'
                    f'PAPER_VERIFY_EPOCHS≥10。')
            elif good and base:
                if good['mae_60'] < base['mae_60']:
                    log('  ✓ 修好那一行 MAE@60 低于持久性基线 —— 模型确实学起来了。')
                    log('     结论：permute 修复有效，全量实验值得按之前的估算排。')
                else:
                    log('  ✗ 修好那一行仍打不过持久性基线 —— 问题不止 permute 一个。')
                    log('     结论：先别烧 GPU。把这条结果和 val_mae 随轮次的变化贴回来重新定位。')
            log("""
      怎么读：

        · 旧bug 那一行的 MAE@15/@30/@60 三档都明显差于持久性基线
            → 复现了「模型只会输出均值、没学到输入到输出的映射」。这就是 bug 的指纹。
          （别拿「最优轮次」当判据：2026-09-23 实测旧bug 的最优轮次落在 ep16，
            val_MAE 也从 5.48 缓降到 5.22 —— 它在慢慢拟合边缘均值，只是学不会
            输入。所以唯一靠得住的是和持久性基线的对比，不是轮次。）

        · 修好那一行的 MAE@60 明显低于旧bug，且低于持久性
            → 修复有效。注意两条用的是同一份数据、同一个种子，唯一的差别
              就是 forward 里那一行 reshape —— 所以提升可以干净地归因给修复本身。

        · 两条差不多，或者修好的也学不起来
            → permute 不是唯一的问题。别继续往下跑，先回来查。
    """)
