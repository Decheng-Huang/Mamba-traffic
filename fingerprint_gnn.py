# -*- coding: utf-8 -*-
"""gnn_baselines.py 的结构指纹 —— 任何改动之后跑一遍，与文件末尾记录的基准比对。

为什么要这个脚本：本仓库最贵的错误不是崩溃，而是"不报错、只让数整体偏一点"
（把输入通道从 2 砍成 1、把 A 转置、把 (B,L,N) 和 (B,N,L) 弄混……）。
这些改动**都能跑通**，只有数值指纹能当场抓住。

★ 为什么必须显式固定随机数流 ★
  规划里那份指纹脚本把输入张量建在任何 torch.manual_seed **之前**（在循环外），
  于是每次运行的输入都不同：报告值 DCRNN -54.777428 与复现值 -54.791961 对不上，
  而参数量却对得上。**不可复现的指纹等于没有指纹。**
  这里把张量创建放在 manual_seed 之后，同一版本任何时候跑都得到同一组数。

★ 关于"改前基线" ★
  这份脚本是在 MambaST 三臂已经加进去之后才写的，**改前的副本没有留存**，
  所以"没动到 DCRNN/STGCN/GWNET"这句只能靠参数量来背书：
  298,049 / 38,460 / 365,052 这三个数在改动之前由两次独立测量得到（--selftest 与
  规划 Agent 的探索脚本），改动之后逐位不变。参数量是架构的指纹 —— 结构一动它必变。
  从今往后，本文件末尾那组值是**唯一**基准。

用法：
    PAPER_GNN_DEVICE=cpu python fingerprint_gnn.py
"""
import os
os.environ.setdefault('PAPER_GNN_DEVICE', 'cpu')

import numpy as np
import torch
import gnn_baselines as G

N, B = 20, 2

# 基准：name -> (参数量, 输出 sum, 输出 absmax)。
# 前三个是**原始模型**，它们的参数量 298,049 / 38,460 / 365,052 在 MambaST 加入
# 之前就由两次独立测量得到，加完之后逐位不变 —— 这就是"没有动到老模型"的证据。
# 后三个是本次新增的对照臂：MAMBAST 与 MAMBASTG 参数量**必须相等**（恒等对照的
# 全部意义所在），MAMBASTN 少一个图模块（131,840 个参数）。
BASELINE = {
    'DCRNN':    (298049, 11.485078, 0.024511),
    'STGCN':    (38460, 76.923233, 0.970567),
    'GWNET':    (365052, 2.705269, 0.211358),
    'MAMBAST':  (384140, -56.776554, 1.797619),
    'MAMBASTG': (384140, -132.740662, 1.593520),
    'MAMBASTN': (252300, -4.886843, 2.158014),
}


def main():
    rng = np.random.RandomState(0)
    A = rng.rand(N, N)
    A = (A + A.T) / 2.0
    A[A < 0.5] = 0.0
    np.fill_diagonal(A, 0.0)
    print(f'设备={G.DEVICE}  PAPER_GNN_TOD={int(G.GNN_TOD)}  '
          f'PAPER_GNN_MAMBA_BLOCK={G.MAMBA_BLOCK}')
    rows = []
    for name in ['DCRNN', 'STGCN', 'GWNET'] + list(G.MAMBAST_ARMS):
        # ★ 先 manual_seed 再造输入：顺序反了就不是指纹了（见文件头）
        torch.manual_seed(0)
        x1 = torch.randn(B, 12, N, 1)
        x2 = torch.randn(B, 12, N, 2)
        m = G.build_model(name, A, N).eval()
        with torch.no_grad():
            y = m(x2 if G.OFFICIAL_IN_DIM[name] > 1 else x1)
        n_par = m.num_params
        rows.append((name, n_par, float(y.sum()), float(y.abs().max())))
        print(f'{name:<9} params={n_par:>10,}  sum={float(y.sum()):>18.6f}  '
              f'absmax={float(y.abs().max()):.6f}')
    return rows


if __name__ == '__main__':
    rows = main()
    bad = []
    for nm, p, s, a in rows:
        ref = BASELINE[nm]
        # 参数量必须**完全**相等（架构指纹）；sum/absmax 是浮点聚合，给一点余量。
        if p != ref[0]:
            bad.append(f'{nm} 参数量 {p:,} != 基准 {ref[0]:,} —— 结构被改动了')
        elif abs(s - ref[1]) > 1e-4 or abs(a - ref[2]) > 1e-4:
            bad.append(f'{nm} 数值漂移: sum {s:.6f}/{ref[1]:.6f}  '
                       f'absmax {a:.6f}/{ref[2]:.6f}')
    print('\n---- 基准（2026-09-25，MambaST 三臂加入之后）----')
    for nm in BASELINE:
        p, s, a = BASELINE[nm]
        print(f'    {nm:<9} {p:>10,}  {s:>18.6f}  {a:.6f}')
    if bad:
        print('\n指纹不匹配：')
        for b in bad:
            print('  ✗', b)
        raise SystemExit(1)
    print('\n指纹与基准一致 ✓')
