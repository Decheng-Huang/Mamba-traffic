# -*- coding: utf-8 -*-
"""
把 ASTGCN 的 pems04.npz 转成本仓库 parquet 口径（第二个数据集，2026-09-28 加）。

为什么需要它
----
论文原计划用 PeMS-BAY + METR-LA 两个数据集，但 METR-LA 在 2026-09-27 被判定
「管线级不复现」（HA 复现是准的 4.146 vs 已发表 4.16，但我们所有训练模型 ≈2×
已发表值）而弃用 → 结果只剩一个数据集。任务书 §0「≥2 数据集硬指标目前不满足」
是投稿前第一条未解项。**这里补的就是它。**

每一条约定都是先从现有产物量出来的，不是推断的
----
| 约定            | 值                                          | 出处 |
|-----------------|---------------------------------------------|------|
| 列名 schema     | x_t-11_d0, x_t-11_d1, …, y_t+12_d0, y_t+12_d1 | 读 `数据集/train.parquet` 实测（50 列） |
| `_d1` = 时段    | (h*60+min)/1440                             | 逐位复现 PeMS-BAY，最大绝对差 **0.000e+00** |
| 行序            | 时间优先：行号 = t0_idx*N + node            | `rebuild_data.py` 断言 1 |
| 窗口            | t0 ∈ [11, T_raw-13]，共 T_raw-23 个         | 52116-23 = 52093 对上了 parquet 实测 |
| 切分            | 7:1:2，train=floor(.7W) val=floor(.1W) test=余 | 36465:5209:10419 实测 = 0.7000:0.1000:0.2000 |

邻接口径（**与 PeMS-BAY 不同，这是必须披露的一条**）
----
PeMS-BAY / METR-LA 用的是 DCRNN 高斯核加权图（含自环、对角=1、不归一化），
2026-09-28 已验证 `数据集/adj_mx.npy` 与官方 `adj_mx_bay.pkl` **逐位相同（差 0）**。

而 **PeMS04 的社区标准图是稀疏二值连通图**，两处独立确认：
  · ASTGCN `lib/utils.py:get_adjacency_matrix` —— 只 `A[i,j]=1`，**cost 列整个被忽略**；
  · BasicTS `scripts/data_preparation/PEMS04/generate_adj_mx.py` —— 对称化、并**明确不加自环**
    （脚本自己打印 "kindly note that there is no self loop"）。
没有更稠密的「标准」版本，所以照用，**不做任何偏离**：

  ★ 自环问题（2026-09-28 查清，一处差点写错的声明）：我原以为必须补自环（A + I）才能与
    PeMS-BAY 的算子语义对齐，因为 DCRNN 官方 `adj_mx_bay.pkl` 的对角是 1。**实测这是多余的**：
    三个消费者全部无条件 `np.fill_diagonal(adj, 0)`（`paper_experiments.py:468`、
    `gnn_baselines.py:644`、`rebuild_data.py:92`）→ 连 PeMS-BAY 的自环也进不了模型。
    所以**两个数据集的有效图都不含自环，语义本来就一致**，论文里不需要声明任何偏离。
    BasicTS 把"缺自环"标为已知瑕疵，但在这个仓库里它不影响任何东西。

  ★ 图规模差异要写进论文（这是先验预期，不是事后解释）。
    **必须用"有效图"= 剥掉对角之后的图**（见上条，两个数据集都要剥）：
      PeMS-BAY  无向边 1184 / 平均度 7.29 / 密度 0.0224 / 零度节点 12
      PeMS04    无向边  340 / 平均度 2.21 / 密度 0.0072 / 零度节点  0
      ★ 2026-10-10 更正（**不改上面那两行**）：PeMS-BAY 的「无向边 1184」是把有向图
        当无向图取 `B.sum() // 2` 的产物，对一个有向图**没有定义**；真值是 **2369 条弧**
        （其中 1789 条只有一个方向、2079 对互异节点相连）。留着不改，是因为本节是
        「**跑之前写死的先验预期**」，涂改它就等于涂改预注册记录。平均度 7.29 = 弧数/
        节点数，这一列**未变**，所以 3.3× 与 H1–H5 的判据一个字都没动。
        详情见 `先验预期_PeMS04.md` §3 的两条订正注记。
    ⇒ **有效平均度差 3.3×。机制若为真，PeMS04 上的效应应当更小但不为零。此预期在跑之前写死。**
    ★ 另有一条同向的削弱因素（也在跑之前写死）：`build_neighbors:529-532` 在真邻居不足
      k=5 时用「自己」补齐，于是两臂的处理强度差得很远 ——
      `real` 臂里真邻居通道占比 PeMS-BAY 88.25% vs PeMS04 **43.97%**（95.4% 的
      PeMS04 节点真邻居不足 5 个，中位度只有 2）。处理强度比 0.498。
      两个因素相乘 ≈ 0.30 × 0.50 ≈ 0.15。详见 `先验预期_PeMS04.md`。

用法
----
    python build_pemsd4.py
输出到 `数据集/PeMS04/{train,val,test}.parquet` 与 `数据集/PeMS04/adj_mx.npy`
"""
import csv
import os
import numpy as np
import pandas as pd

SRC_NPZ = os.path.join('_pemsd4', 'pems04.npz')
SRC_DIST = os.path.join('_pemsd4', 'distance.csv')
OUT_DIR = os.path.join('数据集', 'PeMS04')

N = 307
STEP_MIN = 5
# PeMS04 原始序列起于 2018-01-01 00:00，16992 个 5 分钟 = 恰好 59 天。
# 不含夏令时切换（2018 年美国 DST 起于 3 月 11 日），所以墙钟 tod 无跳变。
START = pd.Timestamp('2018-01-01 00:00:00')

TRAIN_FRAC, VAL_FRAC = 0.7, 0.1
IN_STEPS = OUT_STEPS = 12


def tod(ts):
    """时段特征：与 PeMS-BAY 逐位相同的定义 —— (h*60+min)/1440，落在 [0,1)。"""
    return (ts.hour * 60 + ts.minute) / 1440.0


def build_adjacency():
    """PeMS04 标准二值连通图（对称化、**无自环**）。依据见文件头注释。

    ★ 2026-09-28 更正：本函数原先把 `A + I` 当作「唯一偏离」写进来，是为了对齐
      `build_gwn_supports` 的 A_f = D⁻¹A 语义。**这个偏离是无效的，已删除** —— 三条
      消费路径全部无条件剥掉对角线：
        · `paper_experiments.py:468`  `np.fill_diagonal(adj, 0)`
        · `gnn_baselines.py:644`      `np.fill_diagonal(adj, 0)`
        · `rebuild_data.py:92`        `np.fill_diagonal(adj, 0)`（断言 3 用）
      所以 PeMS-BAY 的 adj_mx.npy（与 DCRNN 官方 pkl 逐位相同、**对角为 1**）在进模型前
      自环也被剥掉了 → 两个数据集进模型的有效图**都不含自环**，算子语义本来就一致。
      于是既不需要补自环、也不该在论文里声明一个不存在的偏离。照 BasicTS 的原样（无自环）。
    """
    with open(SRC_DIST, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        assert header[:2] == ['from', 'to'], f'距离文件表头不对: {header}'
        edges = [(int(a), int(b)) for a, b, _ in r]
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in edges:
        assert 0 <= i < N and 0 <= j < N, f'节点索引越界: {i},{j}'
        A[i, j] = 1.0
        A[j, i] = 1.0                      # 与 ASTGCN/BasicTS 一致：对称化
    n_undirected = int((A != 0).sum() // 2)
    assert not (np.diag(A) != 0).any(), '对称化后对角应为 0（本文件无自环对）'
    deg = (A != 0).sum(1)
    print(f'  邻接 {A.shape} | 无向边 {n_undirected} | 密度 {(A != 0).mean():.4f} '
          f'| 度 {deg.mean():.2f} (min={deg.min()} max={deg.max()}) | 孤立节点 {(deg == 0).sum()}')
    assert (deg == 0).sum() == 0, 'PeMS04 出现了孤立节点，与已知不符'
    return A


def main():
    z = np.load(SRC_NPZ, allow_pickle=True)
    assert 'data' in z.files, f'npz 里没有 data，实际键: {z.files}'
    data = z['data']
    assert data.shape == (16992, N, 3), f'data 形状意外: {data.shape}'
    flow = np.asarray(data[:, :, 0], dtype=np.float64)      # 通道 0 = 流量
    T_raw = flow.shape[0]
    print(f'  原始序列 通道0(流量): {flow.shape} | 非零率 {(flow != 0).mean():.4f} '
          f'| min={flow.min():.1f} max={flow.max():.1f}')

    # ★ 生成器自带的信息量自检：通道 0 若是流量，应与通道 2（速度）负相关。
    #   若相关为正/近零，说明通道顺序不是 (流量,占有率,速度)，整份数据作废。
    occ, speed = data[:, :, 1], data[:, :, 2]
    r_fs = float(np.corrcoef(flow.ravel(), speed.ravel())[0, 1])
    r_fo = float(np.corrcoef(flow.ravel(), occ.ravel())[0, 1])
    print(f'  通道体检: corr(通道0, 速度)={r_fs:+.3f} (流量-速度应为负) '
          f'| corr(通道0, 占有率)={r_fo:+.3f}')
    assert r_fs < 0, '流量与速度不呈负相关 → 通道顺序可能不是 (流量,占有率,速度)，停手'

    ts_axis = pd.date_range(START, periods=T_raw, freq=f'{STEP_MIN}min')

    # 窗口：t0 需要 12 个历史 + 12 个未来 → t0 ∈ [11, T_raw-13]
    # t0 从索引 11 起（第 12 个时刻）—— PeMS-BAY 实测 train 起点 00:55 = 索引 11，据此定案
    t0_ids = np.arange(IN_STEPS - 1, T_raw - OUT_STEPS)
    W = len(t0_ids)
    print(f'  时间轴 {ts_axis[0]} → {ts_axis[-1]} | 有效窗口 {W} (期望 {T_raw - 23})')
    assert W == T_raw - 23

    tr = int(W * TRAIN_FRAC)
    va = int(W * VAL_FRAC)
    te = W - tr - va
    print(f'  切分 7:1:2 → {tr}:{va}:{te}  ({tr / W:.4f}:{va / W:.4f}:{te / W:.4f})')

    # 列名与现有 parquet 逐字一致。★ 第 12 个（当前）时刻是 `x_t+0`（正号），不是 `x_t-0`
    # —— 以 parquet 实际列名为准。注意 `preprocess_local.py:37` 写的是 `x_t-0`，
    # 与真实 parquet 不符（那个脚本对现有数据是失效的，未被使用）。
    past = [f'x_t-{k}' for k in range(IN_STEPS - 1, 0, -1)] + ['x_t+0']
    assert len(past) == IN_STEPS, past
    cols = ['node_id', 't0_timestamp']
    cols += [f'{nm}_d{j}' for nm in past for j in range(2)]
    cols += [f'y_t+{i + 1}_d{j}' for i in range(OUT_STEPS) for j in range(2)]
    assert len(cols) == 50, len(cols)

    splits = {'train': (0, tr), 'val': (tr, tr + va), 'test': (tr + va, W)}
    for name, (lo, hi) in splits.items():
        sub = t0_ids[lo:hi]
        m = len(sub)
        n_feat = 2 * (IN_STEPS + OUT_STEPS)
        # blk[i, s, :] = 第 i 个窗口、节点 s 的 48 个特征（= parquet 一行）
        blk = np.empty((m, N, n_feat), dtype=np.float32)
        for i, t0 in enumerate(sub):
            h = flow[t0 - IN_STEPS + 1:t0 + 1, :].T          # (N, 12) 历史流量
            f = flow[t0 + 1:t0 + OUT_STEPS + 1, :].T         # (N, 12) 未来流量
            ht = tod(ts_axis[t0 - IN_STEPS + 1:t0 + 1])      # (12,)
            ft = tod(ts_axis[t0 + 1:t0 + OUT_STEPS + 1])
            blk[i, :, 0:2 * IN_STEPS:2] = h
            blk[i, :, 1:2 * IN_STEPS:2] = ht                 # 广播到 N 个节点
            blk[i, :, 2 * IN_STEPS::2] = f
            blk[i, :, 2 * IN_STEPS + 1::2] = ft
        # 行序：时间优先（行号 = i*N + node），与 rebuild_data.py 断言 1 一致。
        # ★ 不要用 np.empty 再逐列填 —— 2026-09-28 踩过：node_id 那一列忘了写，
        #   而 np.empty 拿到的是零页 → node_id 全 0，且 rebuild_data 的 N 是从这个
        #   坏列自己算出来的（得 1），于是断言 1 拿 np.arange(1)=[0] 去比 [0]，空转通过。
        #   这里直接按行序 tile 出来，并自带一条自检，不依赖下游。
        df = pd.DataFrame(blk.reshape(m * N, n_feat), columns=cols[2:])
        node_id = np.tile(np.arange(N, dtype=np.int64), m)
        df.insert(0, 'node_id', node_id)
        df.insert(1, 't0_timestamp', np.repeat(ts_axis[sub], N).astype(str))
        assert df.columns.tolist() == cols, f'列序不对: {df.columns.tolist()[:4]}…'
        assert np.array_equal(df['node_id'].to_numpy(),
                              np.tile(np.arange(N), m).astype(np.int64))
        assert (df['node_id'].to_numpy()[:N] == np.arange(N)).all()
        pq_path = os.path.join(OUT_DIR, f'{name}.parquet')
        df.to_parquet(pq_path, index=False)
        d1 = df[[c for c in cols if c.endswith('_d1')]].to_numpy()
        print(f'  [{name:5s}] {len(df):,} 行 × {len(df.columns)} 列 → '
              f'{os.path.getsize(pq_path) / 1e6:.1f} MB | '
              f'_d1 ∈ [{d1.min():.4f}, {d1.max():.4f})')
        assert 0.0 <= d1.min() and d1.max() < 1.0, '_d1 越界，应落在 [0,1)'
        del blk, df, node_id

    A = build_adjacency()
    np.save(os.path.join(OUT_DIR, 'adj_mx.npy'), A)
    print(f'  → {os.path.join(OUT_DIR, "adj_mx.npy")}')
    print('\n完成。下一步：python rebuild_data.py PeMS04  （它自带三条对齐断言）')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=' * 72)
    print('生成 PeMS04（ASTGCN 口径 → 本仓库口径）')
    print('=' * 72)
    main()
