# -*- coding: utf-8 -*-
"""
从 parquet 重建「时刻对齐」的数据文件。支持 PeMS-BAY(325) 与 METR-LA(207)。

用法
----
    python rebuild_data.py                # 两个数据集都跑
    python rebuild_data.py PeMS-BAY       # 只跑 PeMS-BAY
    python rebuild_data.py METR-LA        # 只跑 METR-LA

问题（为什么需要重建）
----
PeMS-BAY 现成的 train_50.npy 布局是 node-major（行号 = s*T + t，脚本的 reshape 假设没错），
但【每个节点内部的样本顺序被打乱过，且各节点打乱方式不同】。证据：
    块均值(b=0..11) = 67.629, 59.175, 59.179 ... 与逐节点 parquet 均值小数点后三位吻合
    但 相邻传感器在同一行号 t 下的读数相关系数 = -0.001（对齐时应有 +0.5）
kaggle_ffff12.py:121 用 x_data[neighbor_indices, t, :] 取邻居，前提是
「所有节点的第 t 行是同一时刻」。该前提不成立 → 5 个邻居实为 5 个随机时刻，是噪声。

parquet 的布局
----
时间优先：行号 = t*N + s。即 [时刻0的N个节点][时刻1的N个节点]...
所以 reshape(T, N, 24) 再 transpose(1,0,2) 就是对齐的 (N, T, 24)。

输出（写入各数据集目录）
----
    train_aligned.npy   (N*T, 24)   全量节点，drop-in 替代现成的 train_50.npy（列含义不变）
    val_aligned.npy
    test_aligned.npy
若该数据集节点数 > 50，另存一份前 50 节点的 *_aligned50.npy 便于快速实验。
"""
import os
import gc
import sys
import numpy as np
import pyarrow.parquet as pq

X_COLS = [f'x_t-{11 - i}_d0' for i in range(11)] + ['x_t+0_d0']
Y_COLS = [f'y_t+{i + 1}_d0' for i in range(12)]
ALL_COLS = X_COLS + Y_COLS

DATASETS = {
    'PeMS-BAY': '数据集',
    'METR-LA': os.path.join('数据集', 'METR-LA'),
    # 2026-09-28 加：第二个数据集（METR-LA 因管线级不复现被弃用后补位）。
    # 数据由 build_pemsd4.py 从 ASTGCN 的 pems04.npz 生成，切分统一 7:1:2。
    'PeMS04': os.path.join('数据集', 'PeMS04'),
}


def rebuild(name, d):
    print(f'\n{"=" * 72}\n{name}   ({d})\n{"=" * 72}')
    counts = {}
    for split in ['train', 'val', 'test']:
        t = pq.read_table(os.path.join(d, f'{split}.parquet'),
                          columns=['node_id'] + ALL_COLS)
        nid = np.asarray(t.column('node_id'))
        n_total = len(nid)
        # 从 train 推断节点数，之后各 split 复用同一 N
        if split == 'train':
            N = int(nid[:2000].max()) + 1 if len(np.unique(nid[:2000])) > 1 else 1
            # 更稳的做法：node_id 从 0 开始连续，N = 唯一值个数
            N = int(np.unique(nid).max()) + 1
            counts['N'] = N
        N = counts['N']
        T = n_total // N
        assert n_total % N == 0, f'{split}: {n_total} 不能被 {N} 整除'

        # 断言 1：node_id 必须是 0..N-1 循环（= 时间优先布局）
        for probe in {0, 1, T // 2, T - 1}:
            seg = nid[probe * N:(probe + 1) * N]
            if not np.array_equal(seg, np.arange(N)):
                raise AssertionError(
                    f'{split}: node_id 在第 {probe} 个时刻不是 0..{N - 1} 顺序')

        a = np.empty((n_total, 24), dtype=np.float32)
        for k, c in enumerate(ALL_COLS):
            a[:, k] = np.asarray(t.column(c), dtype=np.float32)
        del t, nid
        gc.collect()

        a = a.reshape(T, N, 24).transpose(1, 0, 2).copy()   # (N, T, 24)

        # 断言 2：同节点相邻样本必须是相邻时刻
        # ★ 2026-09-28：探测上限原来硬编 2000，T<2000 的划分会广播失败（PeMS04 的 val T=1696）。
        #   改成 min(2000, T)：对 T≥2000（PeMS-BAY/METR-LA 全部划分）切片逐位不变，是恒等变换。
        t_probe = min(2000, T)
        err = float(np.abs(a[:, 1:t_probe, 0:23] - a[:, 0:t_probe - 1, 1:24]).mean())

        # 断言 3：相邻传感器同时刻读数应强相关
        adj = np.load(os.path.join(d, 'sensor_graph', 'adj_mx.npy')) \
            if os.path.exists(os.path.join(d, 'sensor_graph', 'adj_mx.npy')) \
            else np.load(os.path.join(d, 'adj_mx.npy'))
        adj = np.asarray(adj, dtype=np.float64)
        np.fill_diagonal(adj, 0)
        rs = []
        for i in range(min(N, 50)):
            j = int(np.argmax(adj[i]))
            x, y = a[i, :, 0], a[j, :, 0]
            if x.std() > 1e-6 and y.std() > 1e-6:
                rs.append(np.corrcoef(x, y)[0, 1])
        r = float(np.mean(rs))

        flag1 = '✓' if err < 1e-4 else '✗'
        flag3 = '✓' if r > 0.2 else '✗'
        print(f'  [{split:5s}] N={N} T={T:,}  连续性={err:.6f}{flag1}  '
              f'相邻相关={r:+.3f}{flag3}')

        out = os.path.join(d, f'{split}_aligned.npy')
        np.save(out, a.reshape(N * T, 24))
        extra = ''
        if N > 50:
            out50 = os.path.join(d, f'{split}_aligned50.npy')
            np.save(out50, a[:50].reshape(50 * T, 24))
            extra = f'   (+ {os.path.basename(out50)} {os.path.getsize(out50) / 1e6:.0f}MB)'
        print(f'          → {os.path.basename(out)} '
              f'{os.path.getsize(out) / 1e6:.0f} MB{extra}')
        del a
        gc.collect()


if __name__ == '__main__':
    targets = sys.argv[1:] or list(DATASETS)
    print('=' * 72)
    print('从 parquet 重建对齐数据：' + ', '.join(targets))
    print('=' * 72)
    for nm in targets:
        if nm not in DATASETS:
            print(f'未知数据集 {nm}，可选：{list(DATASETS)}')
            continue
        rebuild(nm, DATASETS[nm])
    print('\n完成。')
