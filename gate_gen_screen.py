# -*- coding: utf-8 -*-
"""gate_gen_screen.py —— 用**线性 oracle** 筛查合成图生成器的鉴别力（秒级，纯 CPU）。

为什么需要它（2026-09-26）：
  `gate_gnn_graph.py` 的判据是 `mae_on/mae_off < 0.6`。实测两臂各跑 17~20 轮后得到
  `0.951/0.943/0.986` —— 门不通过。但门不通过只说明"这个合成任务不需要图"，
  它**不说明**"图没用"，也不告诉我**生成器哪里不对**。
  直接去改生成器再上 GPU 重跑，是拿 10 分钟一格的试错去猜一个可以算出来的量。

  这里做的事：把 ON/OFF 两臂各自**能达到的最好水平**用一个线性 oracle 估出来。
    · `own`  = 只用节点自己的 12 步历史，逐时域最小二乘（各节点共享一套权重，
               —— 这正是 OFF 臂的结构：一个共享的逐节点时序模型）
    · `nb`   = 自己历史 + `(Wz)` 聚合的 12 步（= ON 臂能算出来、OFF 臂算不出来的那部分）
  两者都是**线性最优**，所以 `own` 是 OFF 臂的**下界**（真模型只会更差或持平）。

  ★ 判据方向（2026-09-26 更正，第一版写反了）：
    `o/nb = own/nb = 无图/有图 ≈ OFF/ON`，而门的判据是 `ON/OFF < 0.6`，
    所以这里要的是 **`o/nb > 1/0.6 = 1.667`**，不是"< 0.6"。
    写成 "< 0.6" 会把「完全没有鉴别力（o/nb≈1.0）」误读成接近通过。

  这正是 gate 判据 `mae_on/mae_off < 0.6` 在线性层面的对应物，且**不用训练**。
  ⚠️ 它给的是 ON 的**上界**（线性 oracle 被直接喂了精确的 Wz，而真 ON 臂只有一个
     需要自己学的 GWNGraphConv）→ **o/nb 通过是必要条件，不是充分条件**。
  ⚠️⚠️ 必须配稳定性检查：发散序列经"除以样本自身 std"的仿射一归一，屏幕上完全正常。
     2026-09-26 就因此选出过一个发散的 a=0.2,c=0.9,K=12 配置，oracle 打出
     6.42/11.56/3.92 的假余量。本文件**不做**稳定性检查，别用它下结论 ——
     判稳定性用 `gate_gnn_graph.py:check_generator()`（T=20000）。

  同时打印 `corr(z_i,t , (Wz)_i,t)` —— 空间平滑度。它接近 1 就意味着
  「邻居均值 ≈ 自己」，OFF 臂不需要邻居也能猜中，这是生成器失效的**机制**。

用法：
    python gate_gen_screen.py             # 扫全部候选生成器
"""
import sys

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

import numpy as np

N = 32
L = 12
T_TR = 800
T_TE = 350
SEED = 42


def build_graph(seed=0):
    """与 gate_gnn_graph.py 逐行相同（环形 + 4 条长程边，边权逐节点不同）。"""
    rng = np.random.RandomState(seed)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in ((i - 1) % N, (i + 1) % N):
            A[i, j] = rng.uniform(0.5, 1.5)
    for _ in range(4):
        i, j = rng.randint(0, N, 2)
        if i != j:
            A[i, j] = A[j, i] = rng.uniform(0.5, 1.5)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    row = A.sum(axis=1, keepdims=True)
    assert (row > 0).all()
    return A, A / row


def gen(W, T, seed, a, b, s, rho=0.0, r=0.0):
    """候选生成器：z_t = a·z_{t-1} + b·(W z_{t-1}) + r·η_t + s·ε_t
                              η_t = rho·η_{t-1} + ε_t   （逐节点，空间上互不相关）

    r/rho 这一项是**候选的修正**：给每个节点一个自己独有的、空间粗糙的持续分量，
    把空间平滑度压下去，让 (Wz)_i 不再能由 z_i 推出来。
    """
    rng = np.random.RandomState(seed)
    z = np.zeros((N, T + 1))
    eta = np.zeros(N)
    for t in range(1, T + 1):
        eps = rng.randn(N)
        eta = rho * eta + eps
        z[:, t] = a * z[:, t - 1] + b * (W @ z[:, t - 1]) + r * eta + s * eps
    return z[:, 1:]


def windows(s, L, hs):
    """把 (N,T) 序列切成「12 步历史 → 第 h 步真值」的样本。

    返回 own (M, L) 与 agg (M, L)，agg = 把 own 沿节点轴过一遍 W。
    """
    T = s.shape[1] - (L - 1) - max(hs)
    M = T
    own = np.empty((M, L, N), dtype=np.float64)
    for i in range(M):
        own[i] = s[:, i:i + L].T                       # (N, L) 节点在前
    return own


def _with_bias(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def ridge_fit(X, y, lam=1e-6):
    X1 = _with_bias(X)
    G = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(G, X1.T @ y)


def ridge_mae(X_tr, y_tr, X_te, y_te):
    """拟合 + 预测 + 平均绝对误差，一步到位。

    ★ 截距必须在**训练和测试两边都补上**：第一版只在 fit 里补，
      预测时拿 12 列的 X 去乘 13 维权重，直接 matmul 形状错。
      （这个坑不隐蔽，但它提醒了：oracle 的数字必须自己先对得上形状。）
    """
    w = ridge_fit(X_tr, y_tr)
    return float(np.abs(_with_bias(X_te) @ w - y_te).mean())


def screen(name, a, b, s, rho=0.0, r=0.0, hs=(3, 6, 12)):
    A, W = build_graph(seed=0)
    s_tr = gen(W, T_TR + 2 * L, 1, a, b, s, rho, r)
    s_te = gen(W, T_TE + 2 * L, 3, a, b, s, rho, r)

    # 空间平滑度：corr(z_i, (Wz)_i)
    Wz = W @ s_tr
    cc = float(np.corrcoef(s_tr.ravel(), Wz.ravel())[0, 1])

    # 样本：own = 自己的 12 步；agg = (Wz) 的 12 步（= ON 能算、OFF 算不出的信息）
    def make(seq):
        T = seq.shape[1] - (L - 1) - max(hs)
        # ★ 轴序必须是 (T, N, L)：np.stack 出来的是 (T, L, N)，要 transpose。
        #   写成 (T,L,N) 时 `reshape(-1, L)` 不会报错 —— 它会沿错轴分组，
        #   算出一组形状完全合法、含义完全错误的 oracle 数。这个坑只能靠
        #   「把轴序写明在注释里 + m_p 那行拿 (T,N) 去减做交叉检查」抓住。
        own = np.stack([seq[:, t:t + L].T for t in range(T)]).transpose(0, 2, 1)
        agg = np.stack([(W @ seq)[:, t:t + L].T
                        for t in range(T)]).transpose(0, 2, 1)
        assert own.shape == (T, N, L), own.shape
        return own, agg
    o_tr, g_tr = make(s_tr)
    o_te, g_te = make(s_te)

    out = {}
    for h in hs:
        y_tr = s_tr[:, L - 1 + h:][:, :o_tr.shape[0]].T
        y_te = s_te[:, L - 1 + h:][:, :o_te.shape[0]].T
        Xo_tr = o_tr.reshape(-1, L); yo = y_tr.reshape(-1)
        Xn_tr = np.concatenate([o_tr, g_tr], axis=2).reshape(-1, 2 * L)
        Xo_te = o_te.reshape(-1, L); yo_te = y_te.reshape(-1)
        Xn_te = np.concatenate([o_te, g_te], axis=2).reshape(-1, 2 * L)
        m_o = ridge_mae(Xo_tr, yo, Xo_te, yo_te)
        m_n = ridge_mae(Xn_tr, yo, Xn_te, yo_te)
        # 持久性：把最后一个观测原样重复
        m_p = float(np.abs(o_te[:, :, L - 1].reshape(-1) - yo_te).mean())
        out[h] = (m_p, m_o, m_n)
    return cc, out


VARIANTS = [
    ('基线（现门在用）',        dict(a=0.40, b=0.55, s=0.05)),
    ('削弱自项 a',              dict(a=0.10, b=0.85, s=0.05)),
    ('a=0 纯邻居驱动',          dict(a=0.00, b=0.90, s=0.05)),
    ('a=0 + 逐节点粗糙分量',    dict(a=0.00, b=0.60, s=0.05, rho=0.9, r=0.6)),
    ('a=0.1 + 逐节点粗糙分量',  dict(a=0.10, b=0.60, s=0.05, rho=0.9, r=0.6)),
    ('a=0 + 强粗糙分量',        dict(a=0.00, b=0.40, s=0.05, rho=0.95, r=1.2)),
]


def main():
    print(__doc__.split('用法')[0].strip().splitlines()[0])
    print(f'N={N}  L={L}  train={T_TR}  test={T_TE}\n')
    print('判据：o/nb = own/nb ≈ OFF/ON 需要 > 1.667（= 门的 ON/OFF < 0.6）。')
    print('      own 是 OFF 臂的**下界**（线性最优）；nb 是 ON 臂的**上界**'
          '（被直接喂了精确 Wz）—— 通过是必要条件，不是充分条件。')
    print('      ⚠️ 本文件不查稳定性；发散序列在这里看起来完全正常，'
          '判稳定性请用 gate_gnn_graph.py:check_generator()。\n')
    hdr = (f'{"生成器":<24}{"corr":>7} '
           + ' '.join(f'{"@" + str(h):>19}' for h in (3, 6, 12)))
    print(hdr)
    print(f'{"":<24}{"":>7} ' + ' '.join(
        f'{"持久":>6}{"own":>6}{"o/nb":>7}' for _ in (3, 6, 12)))
    print('-' * len(hdr))
    for name, kw in VARIANTS:
        cc, res = screen(name, **kw)
        row = f'{name:<24}{cc:>7.3f} '
        for h in (3, 6, 12):
            p, o, n = res[h]
            row += f'{p:>6.2f}{o:>6.2f}{o / n:>7.3f} '
        print(row)
    print(f'\n（@3/@6/@12 对应门里的 15/30/60 分钟：门的 L=12 步 × 5 分钟 = 60 分钟）')


if __name__ == '__main__':
    main()
