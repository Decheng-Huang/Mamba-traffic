# -*- coding: utf-8 -*-
"""§3.5 合成图依赖门 —— 在花任何钱之前，先证明"这个任务真的需要图"。

为什么需要它：
  MambaST 的 ON/OFF 两臂跑出什么数都可以解释。如果没有一个**已知需要图**的任务，
  那么"两臂没差别"既可能是模型的问题，也可能是**任务本身不需要图** ——
  两种解释在真实数据上无法区分，而这个消融的全部结论就压在那一处差值上。
  所以先造一个图依赖是**构造出来**的任务：在这个任务上 ON 必须明显赢 OFF。
  赢了 → 后面的真实数据结论才有解释力；没赢 → 先修生成器，别去租卡。

任务构造：
  N=32 的**环形图**（i 与 i±1 相连）加 4 条长程边，边权随机且**逐节点不同**。
  这点是刻意的：两臂共用一个 Linear，OFF 臂对全部节点学**同一条**时序规则；
  只有当"每个节点该信哪些邻居"因节点而异时，共享的逐节点规则才不可能拟合，
  ON 臂的优势才会明确。边权若全一样，OFF 臂能靠一条规则蒙对，门就失去鉴别力。

    z_t = 0.05·z_{t-1} + 0.95·(W z_{t-12}) + 0.05·ε     W = 行归一化邻接
    ★ 耦合滞后 **12 步 = L** 是 2026-09-26 实测定的，不是随手写的：滞后 1 步的旧版
      经实测没有鉴别力（ON/OFF = 0.951/0.943/0.986）。机制与实测数据见 gen_series。
  W 的每一行不同 → 邻居怎么加权逐节点不同。仿射到 [1,100] 并截断，保证严格为正
  （compute_metrics 与 MaskedMSELoss 的掩零口径都以"真值 > 0"为准）。

判据（先写死再看数；**全部集中在 `verdict()` 这一个纯函数里**）：
  通过：mae_on < 0.6 · mae_off
  且两臂都要好于持久性（否则是训练没跑起来，不是图的问题）
  且三时域符号一致
  且任务形状断言成立 —— oracle 的 own+agg MAE 随 h 单调增（**判在 oracle 上，见下**）
  若 OFF 已经追平 ON（比值 > 0.9）→ **报错说生成器无效**，而不是报"图没用"。

★ 判据经过一次修订（2026-09-26，经作者裁定），修订记录**必须留在案**：
  · 原第 4 条判在「训练出来的 ON 臂」上、且要求**严格**单调（mae_15<mae_30<mae_60）。
  · 实测它是**功率不足**：任务真实趋势只有 **+0.09%/档**（oracle 2.287→2.289→2.291），
    而单 seed 的 ON 臂档间跳动 **±0.9%** —— 要求一次训练去分辨比自己噪声小十倍的
    趋势，**与机制真假无关，本来就会随机挂**。
  · 改为：任务形状判在 oracle（确定性、无训练噪声）上；模型侧只查"有没有明显反向"，
    容差 MONO_TOL。若哪天要重新收紧，必须**先**给出比上面这份更大的功效论证。

⚠️ **本门的适用范围以它的构造为限**：它证明的是「图通路能学出逐节点的空间混合」，
   即**空间读出能力**；**不**覆盖长时域的时空外推（各时域都已在噪声地板上，
   看不到随 h 的难度增长）。论文引用时必须带上这句限定。

用法：
    PAPER_GNN_DEVICE=cpu python gate_gnn_graph.py
"""
import os

os.environ.setdefault('PAPER_GNN_DEVICE', 'cpu')

# ---- 规模（可用环境变量缩小做冒烟；正式跑用默认值）
N = int(os.environ.get('GATE_N', 32))
# ★ 2026-09-26：800 → 2000。ON 臂必须**自己学出**图算子（它拿到的是对隐状态做
#   2 跳混合的 GWNGraphConv，不是精确的 Wz），801 个训练样本对 384k 参数远远不够 ——
#   实测 ON 只吃到了线性 oracle 差距的约 29%（oracle 预测 ON/OFF 0.849，真跑 0.951）。
#   两臂同样多数据，所以这不是偏袒 ON，而是让 ON 的结构优势有机会表现出来。
T_TR = int(os.environ.get('GATE_TTRAIN', 2000))
T_VA = int(os.environ.get('GATE_TVAL', 250))
T_TE = int(os.environ.get('GATE_TTEST', 350))
EPOCHS = int(os.environ.get('GATE_EPOCHS', 50))
BS = int(os.environ.get('GATE_BS', 16))
PATIENCE = int(os.environ.get('GATE_PATIENCE', 12))
SEED = 42
L = 12

# ★ 生成器参数（2026-09-26 重做，理由与实测数据见 gen_series 的 docstring）
#   K_LAG 必须 == L：这样 h=1..12 每个时域所需的 (W z) 都落在输入窗口内。
K_LAG = 12
A_SELF = float(os.environ.get('GATE_A_SELF', 0.05))    # 自项系数
C_NB = float(os.environ.get('GATE_C_NB', 0.95))        # 邻居耦合系数
#   (0.05, 0.95) 是实测里「既稳定、oracle 余量又较大」的一档（o/nb≈1.89）。
#   相邻的 (0.05,0.99)/(0.1,0.95)/(0.2,0.9) 全部发散 —— 别随手往上调。
# ★ 2026-09-26 新增（默认 0 = 不改现有行为）：滞后 **1** 步的**额外**邻居耦合。
#   为什么想加它：K_LAG=12 让 h=1..12 的驱动格全在窗口内 → ON 臂被压到噪声地板
#   （实测 2.3535 对解析值 2.356），MAE 于是**不随步长增长**，门的单调性检查必挂。
#   而滞后 1 步的项对 h≥2 落在窗口**外**，逼两臂都得往前滚 → 噪声逐步累积 → 才有
#   误差随 h 增长。目标：让「在窗口内的鉴别力」与「随 h 增长的误差」同时存在。
#   ⚠️ 但它同时会把空间平滑度推高（(Wz)_i 变得可由 z_i 推出）→ 鉴别力可能崩。
#      所以**必须**先过 check_generator() 的 o/nb，再谈它有没有用。
D_NB1 = float(os.environ.get('GATE_D_NB1', 0.0))

# ★ 这两个必须在 import gnn_baselines **之前**设好：它是按 env 初始化的模块，
#   而 make_optimizer 用的是**模块级** EPOCHS（不是 train_one 收到的那个 epochs 参数）
#   —— 只传参数不设 env，余弦退火会按 100 轮铺开而我们只跑 50 轮，等于中途掐断。
#   两臂受同样影响，所以不偏袒谁，但没有理由留着一个错的调度。
os.environ.setdefault('PAPER_GNN_EPOCHS', str(EPOCHS))
os.environ.setdefault('PAPER_GNN_PATIENCE', str(PATIENCE))

import numpy as np
import torch

import gnn_baselines as G
assert G.EPOCHS == EPOCHS and G.PATIENCE == PATIENCE, \
    (G.EPOCHS, EPOCHS, G.PATIENCE, PATIENCE)

PASS_RATIO = 0.60      # mae_on 必须低于 0.6×mae_off
DEAD_RATIO = 0.90      # mae_off/mae_on 高于这个值 → 生成器没有鉴别力，报错
# ★ 2026-09-26 新增：模型侧单调性的容差。依据是**实测**的档间噪声 ±0.9%，
#   取 ≈2×。见 verdict() 里 (b) 段的说明 —— 严格单调那条检查是功率不足的。
MONO_TOL = float(os.environ.get('GATE_MONO_TOL', 0.02))


def verdict(off, on, pers, om):
    """门的全部判据，做成**纯函数**。

    ★ 为什么抽出来：判据本身被修订过（2026-09-26，经作者裁定）。抽成纯函数后可以拿
      **已记录的** `gate_out3.log` 数字直接复核修订后的判据，不必再烧一次训练 —— 而
      "改判据"正是最容易变成事后挑口径的动作，必须能被独立复核。

    off/on: {15: MAE, 30: MAE, 60: MAE}（测试集）
    pers:   同键的持久性 MAE
    om:     `_oracle_mae()` 的返回；**键是时域下标 3/6/12**，不是 15/30/60
    返回失败原因列表；空列表 = 门通过。
    """
    bad = []
    r60 = on[60] / off[60]
    if r60 > DEAD_RATIO:
        bad.append(f'★ 生成器没有鉴别力：mae_on/mae_off = {r60:.3f} > {DEAD_RATIO}。'
                   f'这说明这个合成任务**不需要图**，先修生成器再谈后面的结论')
    elif r60 >= PASS_RATIO:
        bad.append(f'ON 的领先不够：mae_on/mae_off = {r60:.3f}，要求 < {PASS_RATIO}')
    for key, m in (('ON', on), ('OFF', off)):
        if m[60] >= pers[60]:
            bad.append(f'{key} 臂 MAE@60={m[60]:.4f} 没赢过持久性 {pers[60]:.4f} '
                       f'—— 是训练没跑起来，不是图的问题')
    sgn = [np.sign(off[h] - on[h]) for h in G.HORIZONS]
    if len(set(sgn)) != 1:
        bad.append(f'★ ON/OFF 的符号在三个时域上不一致（{sgn}）—— '
                   f'这是整个 run 的红旗，不是"部分成立"的结果')

    # ---- (a) 任务形状断言：挂在 **oracle** 上，不挂在训练出来的臂上 ----
    #   理由：这条要问的是"任务本身有没有随步长变难"，而它用训练噪声去测是测不准的。
    #   oracle 的 own+agg 是确定性的（固定 seed、无训练），是问这个问题的正确工具。
    #   它抓的是"长时域反而更容易"这种病（如耦合滞后与预测时域共振）。
    mn = [om[G.HORIZONS[h]][1] for h in G.HORIZONS]
    if not (mn[0] < mn[1] < mn[2]):
        bad.append(f'★ 任务没有随步长变难的形状：oracle 的 own+agg MAE = '
                   f'{" / ".join(f"{v:.3f}" for v in mn)}（@15/@30/@60）不单调增 '
                   f'—— 这是**生成器**的问题，不是训练的问题')

    # ---- (b) 模型侧只查"有没有明显反向"，容差 MONO_TOL ----
    #   ★ 2026-09-26 修订（原作者裁定）。原版要求 `mae_15 < mae_30 < mae_60` 严格成立，
    #   实测证明**功率不足**：任务真实趋势只有 **+0.09%/档**（oracle 2.287→2.289→2.291），
    #   而单 seed 训练出来的 ON 臂档间跳动 **±0.9%** —— 噪声是趋势的十倍。
    #   要求一次训练去分辨比自己噪声小十倍的趋势，**与机制真假无关，本来就会随机挂**。
    for a, b in ((15, 30), (30, 60)):
        if on[b] < on[a] * (1.0 - MONO_TOL):
            bad.append(f'ON 臂 MAE @{b} 比 @{a} 低了 {(1 - on[b] / on[a]) * 100:.1f}% '
                       f'（超过容差 {MONO_TOL:.0%}）—— 长时域反而明显更准，'
                       f'训练或任务有问题')
    return bad


def build_graph(seed=0):
    """环形图 + 4 条长程边，边权随机（逐节点不同 —— 见文件头）。"""
    rng = np.random.RandomState(seed)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in ((i - 1) % N, (i + 1) % N):
            A[i, j] = rng.uniform(0.5, 1.5)
    for _ in range(4):                       # 长程边
        i, j = rng.randint(0, N, 2)
        if i != j:
            A[i, j] = A[j, i] = rng.uniform(0.5, 1.5)
    A = np.maximum(A, A.T)                   # 对称化（无向图）
    np.fill_diagonal(A, 0.0)
    row = A.sum(axis=1, keepdims=True)
    assert (row > 0).all(), '有孤立节点，环形图不该出现'
    return A, A / row                        # (邻接, 行归一化转移)


def gen_series(W, T, seed):
    """z_t = A_SELF·z_{t-1} + C_NB·(W z_{t-K_LAG}) + D_NB1·(W z_{t-1}) + 0.05 ε，再仿射到 [1,100]。

    ★★ 2026-09-26 重做：耦合从「滞后 1 步」改成「滞后 K_LAG = L = 12 步」★★
    旧版（`0.40 z_{t-1} + 0.55 W z_{t-1}`）经实测**没有鉴别力**，门判 ON/OFF = 0.951/0.943/0.986
    （要求 < 0.6）。根因不是"图没用"，是**生成器的结构**：

      预测第 h 步需要 (W z) 在 t+L-1+h-1 处。滞后 1 步时，h=1 要的那一格恰好在窗口
      末尾（t+L-1）✓，而 h≥2 要的格子**落在窗口之外**（超到 t+L 及以后）→ 两臂都只能
      "往前滚"动力学，而每步注入的噪声经 W 传播放大，把差距抹平。
      滞后改成 K_LAG = L = 12 之后，h=1..12 所需的 (W z) 全部落在 t..t+L-1 内 ——
      图臂能直接读到，逐节点臂结构上读不到。

    线性 oracle 实测（`gate_gen_screen.py` 的口径，判据 o/nb > 1.667 = ON/OFF < 0.6）：
      旧版滞后 1 步              o/nb = 1.179 / 1.045 / 1.048   ✗
      滞后 K=12, a=0.0, c=0.99   o/nb = 2.073 / 2.096 / 2.117   ✓（余量最大，但离发散最近）
      滞后 K=12, a=0.05,c=0.95   o/nb = 1.917 / 1.933 / 1.890   ✓ ← 本函数采用
      滞后 K=12, a=0.1, c=0.9    o/nb = 1.816 / 1.850 / 1.754   ✓

    ⚠️ **稳定性边界极紧**：K=12 时 (a,c) = (0.05,0.99)/(0.1,0.95)/(0.2,0.9) 全部**发散**
    （sd 到 1e26~1e147），而且**短序列看不出来** —— 发散慢到 800 步的窗口里 sd 还是正常的，
    再经「除以样本自身 std」的仿射一归一，屏幕上看起来完全合理。所以本文件在生成之后
    **必须**跑 `check_generator()`：它拿 T=20000 判稳定性，再跑一遍 oracle 判鉴别力。
    这两条不过就直接报错，不允许往下训练。
    """
    z = _sim(W, T + 2 * L + K_LAG, seed)
    s = 20.0 + 5.0 * z / max(1e-9, z.std())
    return np.clip(s, 1.0, 100.0).astype(np.float32)


def _sim(W, T_all, seed):
    """递推本体，返回**仿射前**的 z，形状 (N, T_all - K_LAG)（已去掉热启动段）。

    ★ 抽成一个函数是刻意的：`gen_series` 与 `check_generator` 原来各写了一份逐字相同的
      递推。**加一个耦合项时两份必然漂移**，而"稳定性自检跑的是另一份递推"恰好是这个
      项目反复踩的坑（自检说稳定、训练用的却是另一个过程）。两处只准调这一个。
    ★ `D_NB1=0` 时 `0.0 * (W @ z)` 恒为 0，`v + 0.0 == v` 在 IEEE 下逐位成立 →
      默认参数下本函数与重构前**逐位相同**（已用 sd=0.09456 与 o/nb 三个值验证）。
    """
    rng = np.random.RandomState(seed)
    z = np.zeros((N, T_all), dtype=np.float64)
    for t in range(K_LAG, T_all):
        z[:, t] = (A_SELF * z[:, t - 1] + C_NB * (W @ z[:, t - K_LAG])
                   + D_NB1 * (W @ z[:, t - 1])
                   + 0.05 * rng.randn(N))
    return z[:, K_LAG:]


def _oracle_mae(seed_tr=1, seed_te=3, T_tr=820, T_te=374):
    """线性 oracle：只用自己 12 步 vs 自己 12 步 + (Wz) 12 步，逐时域取 MAE 之比。

    `own` 是 OFF 臂的**下界**（线性最优），`nb` 是 ON 臂的**上界**（它被直接喂了精确的
    Wz，而真 ON 臂只有一个要自己学出来的 GWNGraphConv）。所以这个比值**高估**真门能拿到的
    差距 —— 它是必要条件，不是充分条件。留着当"生成器有没有鉴别力"的快速判据。
    """
    hs = sorted(G.HORIZONS.values())
    _, W = build_graph(seed=0)
    tr, te = gen_series(W, T_tr, seed_tr), gen_series(W, T_te, seed_te)

    def mk(q):
        T = q.shape[1] - (L - 1) - max(hs)
        return (np.stack([q[:, t:t + L].T for t in range(T)]).transpose(0, 2, 1),
                np.stack([(W @ q)[:, t:t + L].T for t in range(T)]).transpose(0, 2, 1))

    o_tr, g_tr = mk(tr)
    o_te, g_te = mk(te)
    out = {}
    for h in hs:
        ytr = tr[:, L - 1 + h:][:, :o_tr.shape[0]].T.reshape(-1)
        yte = te[:, L - 1 + h:][:, :o_te.shape[0]].T.reshape(-1)
        mo = _ridge_mae(o_tr.reshape(-1, L), ytr, o_te.reshape(-1, L), yte)
        mn = _ridge_mae(np.concatenate([o_tr, g_tr], 2).reshape(-1, 2 * L), ytr,
                        np.concatenate([o_te, g_te], 2).reshape(-1, 2 * L), yte)
        out[h] = (mo, mn)
    return out


def _oracle_ratio(**kw):
    """`_oracle_mae` 的比值视图：{h: own/nb}。判据 > 1.667（见 check_generator）。

    ★ 2026-09-26 拆出 `_oracle_mae`：门挂掉的第二条检查是「ON 臂 MAE 不随步长单调」，
      而 ON 臂的好坏正对应 oracle 的 `mn`（own+agg）。只看比值 `mo/mn` 会把
      「两边一起变大」这种情形完全藏住 —— 必须能单独看到 mn 随 h 的形状。
    """
    return {h: mo / mn for h, (mo, mn) in _oracle_mae(**kw).items()}


def _ridge_mae(X_tr, y_tr, X_te, y_te, lam=1e-6):
    def fit(X, y):
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ y)
    w = fit(X_tr, y_tr)
    return float(np.abs(np.concatenate(
        [X_te, np.ones((X_te.shape[0], 1))], axis=1) @ w - y_te).mean())


def check_generator():
    """生成器自检：先证稳定，再证有鉴别力。**不过就报错，不训练。**

    为什么必须分开查：稳定性失败会**伪装成**"门通过/不通过"—— 发散序列经样本 std
    一归一，屏幕上一切正常。2026-09-26 就是先选了一个 a=0.2,c=0.9,K=12 的发散配置，
    oracle 打出 6.42/11.56/3.92 的"漂亮"余量，全部是假的。
    """
    _, W = build_graph(seed=0)
    # 仿射前的原始 sd 才是稳定性的判据：仿射后必然被归一化，看不出来
    # ★ 走 _sim，与 gen_series **同一个递推** —— 自检和训练必须是同一个过程
    raw = _sim(W, 20000 + 2 * L + K_LAG, 1)
    sd_raw = float(raw.std())
    print(f'  [生成器自检] 稳定性：T=20000 原始 sd={sd_raw:.5f} '
          f'（须有限且 < 100）')
    bad = []
    if not (np.isfinite(raw).all() and sd_raw < 100):
        bad.append(f'发散：T=20000 的 sd={sd_raw:.4g}，说明 (a,c)=({A_SELF},{C_NB}) '
                   f'在 K={K_LAG} 下不稳定。短序列看不出来，必须用 20000 步判')
    r = _oracle_ratio()
    print('  [生成器自检] 鉴别力 o/nb = ' + '  '.join(
        f'@{h}={v:.3f}' for h, v in r.items()) + '   （须 > 1.667）')
    if min(r.values()) <= 1.667:
        bad.append(f'生成器没有鉴别力：最差时域 o/nb={min(r.values()):.3f} ≤ 1.667 '
                   f'（= 门的 ON/OFF 0.6）。**改生成器，不要改判据**')
    if bad:
        print('\n✗ 生成器自检失败 —— 不训练：')
        for b in bad:
            print('   ✗', b)
        raise SystemExit(2)
    print('  [生成器自检] 通过：既稳定，又有鉴别力。\n')


def to_d24(s):
    """(N, T+L-1) 的序列 -> (N, T, 24)：前 12 列历史、后 12 列未来，窗口**连续**。

    连续是硬要求：prepare_split 里的 assert_window_overlap 会核对
    X[t,1:12] == X[t+1,0:11]，窗口不连续会直接抛。
    """
    T = s.shape[1] - (2 * L - 1)
    d24 = np.empty((N, T, 2 * L), dtype=np.float32)
    for t in range(T):
        d24[:, t, :L] = s[:, t:t + L]
        d24[:, t, L:] = s[:, t + L:t + 2 * L]
    return d24


def persistence_mae(d24):
    """持久性：把最后一个观测值原样重复 12 步。与指标同口径（未掩零点）。"""
    last = d24[:, :, L - 1][:, :, None]           # (N, T, 1)
    err = np.abs(last - d24[:, :, L:])            # (N, T, 12)
    return {h: float(err[:, :, idx - 1].mean()) for h, idx in G.HORIZONS.items()}


def main():
    A, W = build_graph(seed=0)
    deg = (A > 0).sum(axis=1)
    print(f'图：N={N}  边={int((A > 0).sum())}  度 中位{int(np.median(deg))}'
          f'/最大{int(deg.max())}  孤立{int((deg == 0).sum())}')

    # ★ 训练之前先自证：这个生成器既稳定、又有鉴别力。不过就 SystemExit，不烧 GPU。
    check_generator()
    s_tr = gen_series(W, T_TR + 2 * L, seed=1)
    s_va = gen_series(W, T_VA + 2 * L, seed=2)
    s_te = gen_series(W, T_TE + 2 * L, seed=3)
    d24s = [to_d24(x) for x in (s_tr, s_va, s_te)]
    print(f'序列：train{d24s[0].shape} val{d24s[1].shape} test{d24s[2].shape}  '
          f'真值范围 {min(float(d.min()) for d in d24s):.2f}~'
          f'{max(float(d.max()) for d in d24s):.2f}（必须 > 0）')

    tx = d24s[0][:, :, :L]
    n = tx.shape[1]
    mu = tx.mean(axis=(1, 2), keepdims=True).astype(np.float32)
    sd = (tx.std(axis=(1, 2), keepdims=True) + 1e-8).astype(np.float32)
    # ★ y 侧刻意用 0/1：这样指标直接落在**原始单位**上，与持久性可比。
    z = np.zeros((N, 1, 1), dtype=np.float32)
    o = np.ones((N, 1, 1), dtype=np.float32)

    splits = []
    for d24, tag in zip(d24s, ('gate-train', 'gate-val', 'gate-test')):
        splits.append(G.prepare_split(d24.copy(), mu, sd, z, o, tag, in_dim=1))
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = splits
    data = ((Xtr, Ytr), (Xva, Yva), (Xte, Yte), o.reshape(-1), z.reshape(-1), N)

    pers = persistence_mae(d24s[2])
    print('持久性 MAE：' + '  '.join(f'@{h}={pers[h]:.4f}' for h in G.HORIZONS))

    res = {}
    for arm in ('MAMBAST', 'MAMBASTG'):
        print(f'\n---- {arm} ----')
        torch.manual_seed(SEED)
        model = G.build_model(arm, A, N, in_dim=1)
        m = G.train_one(arm, model, data, SEED, EPOCHS, BS, f'gate_{arm}')
        if m is None:
            print(f'  {arm} 返回 None（发散或 nan）—— 门失败')
            raise SystemExit(1)
        res[arm] = m
        print(f'  {arm}: ' + '  '.join(
            f'MAE@{h}={m[f"mae_{h}"]:.4f}' for h in G.HORIZONS) +
            f'   best_ep={m["best_epoch"]}  epochs_run={m["epochs_run"]}  '
            f'{m["epoch_time"]:.1f}s/轮')

    print('\n' + '=' * 70)
    for h in G.HORIZONS:
        on, off, pe = (res['MAMBASTG'][f'mae_{h}'], res['MAMBAST'][f'mae_{h}'],
                       pers[h])
        print(f'  @{h:>2}min   持久性 {pe:7.4f}   OFF {off:7.4f}   ON {on:7.4f}'
              f'   ON/OFF={on / off:.3f}   ON/持久={on / pe:.3f}')
    print('=' * 70)

    om = _oracle_mae()
    mn = [om[G.HORIZONS[h]][1] for h in G.HORIZONS]
    per = 100 * (mn[2] / mn[0] - 1) / 2
    print(f'  任务随 h 的难度趋势（oracle 的 own+agg，确定性、无训练噪声）：'
          + ' → '.join(f'{v:.3f}' for v in mn) + f'   每档 {per:+.2f}%')
    print(f'  ⚠️ 这个趋势的**绝对量级**就是你看到的样子 —— 它确实单调增，但小到'
          f'±0.9% 的单 seed 噪声之下，\n'
          f'     所以模型侧的单调性只能按容差判（MONO_TOL={MONO_TOL:.0%}）。')

    r60 = res['MAMBASTG']['mae_60'] / res['MAMBAST']['mae_60']
    bad = verdict({h: res['MAMBAST'][f'mae_{h}'] for h in G.HORIZONS},
                  {h: res['MAMBASTG'][f'mae_{h}'] for h in G.HORIZONS},
                  pers, om)

    if bad:
        print('\n门失败：')
        for b in bad:
            print('  ✗', b)
        raise SystemExit(1)
    print(f'\n合成图依赖门通过：mae_on/mae_off = {r60:.3f} < {PASS_RATIO}，'
          f'且两臂都赢过持久性、三时域符号一致。')
    print(f'→ 这个任务确实需要图机制，MambaST 的 ON/OFF 消融有解释力。')
    print(f'⚠️ **本门的适用范围以它的构造为限**：它证明的是「图通路能学出逐节点的空间混合」，'
          f'即**空间读出能力**；\n'
          f'   它**不**覆盖长时域的时空外推能力（本任务各时域都已在噪声地板上，'
          f'看不到随 h 的难度增长）。\n'
          f'   论文里引用本门时必须带上这句限定，不能拿它当作 @60 主张的依据。')


if __name__ == '__main__':
    main()
