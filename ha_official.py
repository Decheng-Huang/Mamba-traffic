# -*- coding: utf-8 -*-
"""ha_official.py —— 按 DCRNN 官方定义重算 HA 基线，独立复现已发表值。

为什么单独写一个脚本：`paper_experiments.py:run_baselines` 里的 HA 是
**逐传感器常数**（`np.zeros_like(Yte)` 在标准化空间 = 训练集逐节点均值），
而 DCRNN 官方（`scripts/eval_baseline_methods.py`）的 HA 是
**逐传感器 × 逐「一周中的同一相位」**：

    period = 7 * 24 * 12                      # 2016 个 5 分钟槽 = 一周
    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()

三条要点：
  1. 相位周期是 2016（一周），**不是** 288（一天），也不是逐传感器常数。
  2. 求均值时 **`historical != null_val`（剔除 0）** —— 这是最关键的一行。
  3. 所以它对 12 小时时域是**恒定**的 —— 这解释了 DCRNN Table 1 里
     HA 行为什么在 15/30/60 三档印着同一个 4.16。

本脚本的目的**不是**产出论文数字，而是**验证**：把官方定义跑在**我们自己的
数据**上，能不能复现已发表的 4.16（METR-LA）/ 2.88（PeMS-BAY）。
如果复现得上 → 我们的 HA 是实现错误，不是口径差异，修就完了。
如果复现不上 → 说明数据本身和作者用的不是同一份，那要先查数据。

用法：  python ha_official.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

os.environ.setdefault('PAPER_GNN_SOURCE', 'parquet')
os.environ.setdefault('PAPER_GNN_TOD', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                  # noqa: E402
import gnn_baselines as gb                          # noqa: E402

PERIOD = 7 * 24 * 12                                # 2016，官方的 period

# 已发表值，一手来源：
#   METR-LA  4.16 —— DCRNN Table 1, p.5 (arXiv:1707.01926v3)
#   PeMS-BAY 2.88 —— 同上
PUBLISHED = {'METR-LA': 4.16, 'PeMS-BAY': 2.88}


def series_from_windows(d):
    """把 (N, T, 24) 的滑窗还原成 (N, L) 的原始序列。

    ★ 关键：这是**步长 1** 的滑窗，不是不重叠的切片。
      窗口 t 覆盖原始时刻 [t, t+23]：输入 = raw[t..t+11]，目标 = raw[t+12..t+23]。
      所以 x[n,t,j] = raw[t+j]，y[n,t,h] = raw[t+12+h]，全序列长度 L = T + 23。

      （第一版写成 [x0 | y.reshape(N,-1)]，长度变成 12+12T —— **把时间轴膨胀了
       12 倍**，i%2016 于是等价于 14 小时周期而不是一周，逐周剖面因此失效。
       自检条件就是长度：METR-LA 应为 34,272、PeMS-BAY 应为 52,116。）
    """
    N, T, _ = d.shape
    x, y = d[:, :, :12], d[:, :, 12:]
    s = np.empty((N, T + 23), dtype=np.float32)
    s[:, :T] = x[:, :, 0]                 # raw[0..T-1]
    s[:, T:T + 12] = y[:, T - 12:, 0]     # raw[T..T+11]
    s[:, T + 12:] = y[:, T - 1, 1:]       # raw[T+12..T+22]
    # 自检：同一个原始时刻在两条路径下必须一致 —— raw[t] 既能从 x[t,0] 取，
    # 也能从 y[t-12,0] 取（t>=12）。
    seg = float(np.abs(x[:, 12:, 0] - y[:, :-12, 0]).max())
    return s, seg


def ha_profile(s, pool_end):
    """官方 HA：逐节点 × 逐相位（period=2016）的均值，**剔除 0**。"""
    N = s.shape[0]
    prof = np.full((N, PERIOD), np.nan, dtype=np.float64)
    for p in range(PERIOD):
        idx = np.arange(p, pool_end, PERIOD)
        if idx.size == 0:
            continue
        seg = s[:, idx].astype(np.float64)
        m = seg > 0
        n = m.sum(axis=1)
        tot = np.where(m, seg, 0.0).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            prof[:, p] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    return prof


def masked_mae(pred, true):
    """官方 lib/metrics.py 口径：只对 label != 0 的位点算 |误差| 的均值。

    `mask /= np.mean(mask)` 让分母变成有效点个数 —— 等价于只对有效点求均值。
    """
    m = true != 0
    if m.sum() == 0:
        return float('nan')
    return float(np.abs(pred[m] - true[m]).mean())


for name in ['METR-LA', 'PeMS-BAY']:
    print('=' * 78)
    print(f'=== {name} ===')
    train, val, test, adj, N, tods = gb.load_dataset(name)
    Ttr, Tva, Tte = train.shape[1], val.shape[1], test.shape[1]
    s_tr, c1 = series_from_windows(train)
    s_va, c2 = series_from_windows(val)
    s_te, c3 = series_from_windows(test)
    print(f'  N={N}  窗口数 train/val/test = {Ttr:,}/{Tva:,}/{Tte:,}')
    print(f'  [自检] 滑窗重叠差  train={c1:.3e}  val={c2:.3e}  test={c3:.3e}'
          f'   （应 ~0）')

    # 拼成完整序列：三段重叠 23 步，衔接处只取一次
    full = np.concatenate([s_tr, s_va[:, 23:], s_te[:, 23:]], axis=1)
    off_te = Ttr + Tva                    # test 第一个窗口的全局窗口号
    EXPECT_L = {'METR-LA': 34272, 'PeMS-BAY': 52116}
    ok = '✓' if full.shape[1] == EXPECT_L[name] else '✗★'
    print(f'  完整序列 L={full.shape[1]:,}（标准长度 {EXPECT_L[name]:,}）{ok}'
          f'   test 起点窗口号={off_te:,}')

    # 官方口径：n_train = 80%（把 val 折进历史池）。原文 n_train = round(L*0.8)。
    pool_end = min(off_te + 24, int(round(full.shape[1] * 0.8)))
    print(f'  历史池 = train+val = 前 {pool_end:,} 步'
          f'（{pool_end / full.shape[1]:.1%}；官方 n_train=round(L*0.8)='
          f'{int(round(full.shape[1] * 0.8)):,}）')

    prof = ha_profile(full, pool_end)
    nan_nodes = int(np.isnan(prof).all(axis=1).sum())
    print(f'  相位剖面 {prof.shape}  全 NaN 的节点数={nan_nodes}')

    # 目标下标：test 局部窗 t 的全局窗号 = off_te + t，目标 h 落在 raw[窗号+12+h]
    tidx = np.arange(Tte)
    for h_label, h in (('@15', 0), ('@30', 6), ('@60', 11)):
        abs_idx = off_te + tidx + 12 + h
        true = full[:, abs_idx]                     # (N, Tte)
        pred = prof[:, abs_idx % PERIOD]            # 逐周相位剖面
        mae = masked_mae(pred, true)
        nz = 100.0 * float((true == 0).mean())
        print(f'  HA {h_label}:  MAE={mae:.4f}   零值占比={nz:.2f}%')

    # 逐传感器常数（我们生产里的口径），只用来对照「差多少」
    m = full[:, :pool_end] > 0
    mu = (np.where(m, full[:, :pool_end], 0.0).sum(axis=1)
          / np.maximum(m.sum(axis=1), 1))
    for h_label, h in (('@15', 0), ('@30', 6), ('@60', 11)):
        abs_idx = off_te + tidx + 12 + h
        true = full[:, abs_idx]
        pred = np.repeat(mu[:, None], Tte, axis=1)
        print(f'  [对照] 逐传感器常数 {h_label}:  MAE={masked_mae(pred, true):.4f}')

    pub = PUBLISHED[name]
    print(f'  ★ 已发表 HA（DCRNN Table 1, p.5）= {pub}')
    print()

    del train, val, test, s_tr, s_va, s_te, full, prof
