# -*- coding: utf-8 -*-
# ============================================================================
#  零成本基线 —— 回答任务书 Q1
#
#  为什么单独写一个脚本：
#    任务书 Q1 的问题是「持久性基线到底是多少」。论文核心主张是
#    "V8 用 31% 参数达到可比精度"，但四个模型 MSE 挤在 11.50~11.83 的
#    3% 区间里 —— 如果持久性也在 11.5 附近，"精度可比"的真实含义就变成
#    "大家都不会"，核心主张当场崩塌。
#    这个问题不需要 GPU、不需要训练，几分钟就能算完。而它决定论文
#    走路线 A（参数效率）还是路线 B（适用边界实证），所以必须在烧
#    GPU 之前先拿到。
#
#  设计原则：
#    1. 复用 paper_experiments.load_dataset，不另写一套数据读取
#       —— 两套实现迟早会漂移，那时候基线就和模型不在同一份数据上了。
#    2. 全部在**原始速度空间**算指标，与 compute_metrics 的口径一致
#       （同一组 HORIZONS、同一个 MAPE_MIN）。
#    3. 不构造 make_samples 的 X 张量：325 节点全量下那个张量约 1GB，
#       而基线只需要窗口里的最后一步。直接从 (N,T,24) 取，省内存。
#
#  用法：
#      python baseline_only.py                       # 两个数据集
#      PAPER_DATASETS=PeMS-BAY python baseline_only.py
# ============================================================================

import os
import sys
import json

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 只借数据和常量，不触发它的主流程 —— 所以这里不能用 import 的副作用。
# paper_experiments.py 顶层会执行主流程，因此改用 exec 取定义部分太脏；
# 直接 import 但设一个空模式让它不跑任何 job。
os.environ.setdefault('PAPER_MODE', 'checkdata')
import paper_experiments as P          # noqa: E402

HORIZONS = P.HORIZONS                  # {15:3, 30:6, 60:12}
MAPE_MIN = P.MAPE_MIN
OUT = 'baseline_results.json'


def log(*a):
    print(*a, flush=True)


def metrics(pred, true):
    """pred/true: (T, N, 12) 原始速度空间。口径与 compute_metrics 一致。"""
    res = {}
    for hmin, idx in HORIZONS.items():
        pp, tt = pred[:, :, idx - 1], true[:, :, idx - 1]
        d = pp - tt
        m = tt > MAPE_MIN
        k = str(hmin)
        res[f'mae_{k}'] = float(np.abs(d).mean())
        res[f'rmse_{k}'] = float(np.sqrt((d ** 2).mean()))
        res[f'mape_{k}'] = float(np.abs(d[m] / tt[m]).mean() * 100) if m.any() else float('nan')
        res[f'mape_raw_{k}'] = float(np.abs(d / np.maximum(np.abs(tt), 1e-3)).mean() * 100)
    res['n_eval'] = int(pred.shape[0])
    return res


def series_of(w):
    """(N,T,24) 滑窗 -> 连续序列 (N, T+23)。步长按 load_layout 的自检是 1。"""
    return np.concatenate([w[:, 0, :], w[:, 1:, 23]], axis=1)


def seasonal(S, Yte, lag):
    """季节性朴素：用 lag 步之前的同时刻值当预测。

    返回 (pred, mask)。lag 越大，前面能算的样本越少 ——
    这些样本只用在「四条基线在同一子集上互比」那张表里。
    """
    T = Yte.shape[0]
    pred = np.full_like(Yte, np.nan)
    ok = np.zeros(T, dtype=bool)
    for t in range(T):
        src = t + 12 + np.arange(12) - lag       # 目标绝对步减去 lag
        if src[0] < 0:
            continue
        pred[t] = S[:, src]                      # (N,12)：S[:, 索引数组] 已经就是 (N,12)
        ok[t] = True
    return pred, ok


def main():
    todo = [s.strip() for s in
            os.environ.get('PAPER_DATASETS', 'PeMS-BAY,METR-LA').split(',') if s.strip()]
    allout = {}

    log('=' * 100)
    log('零成本基线（原始速度空间，指标口径同 compute_metrics）')
    log('=' * 100)

    for ds in todo:
        log(f'\n{"-" * 100}\n{ds}\n{"-" * 100}')
        try:
            train3, val3, test3, adj, N = P.load_dataset(ds)
        except Exception as e:
            log(f'  [失败] {type(e).__name__}: {e}')
            continue

        # 目标序列：test3[:, t, 12:24] 是样本 t 的 12 步真值
        Yte = test3[:, :, 12:].transpose(1, 0, 2).copy()      # (T, N, 12)

        # 训练集每个传感器的均值，作为「历史均值」基线
        ha = train3[:, :, 12:].mean(axis=(1, 2))              # (N,)
        ha_pred = np.repeat(ha[None, :, None], Yte.shape[0], axis=0)
        ha_pred = np.repeat(ha_pred, 12, axis=2)

        # 窗口内可得的两条
        last = test3[:, :, 11]                                # 最后观测 (N,T)
        persist = np.repeat(last.T[:, :, None], 12, axis=2)   # (T,N,12)
        ma12 = test3[:, :, :12].mean(axis=2).T[:, :, None]
        ma12 = np.repeat(ma12, 12, axis=2)

        cands = {'持久性(last)': persist, '历史均值(train mean)': ha_pred,
                 '近12步均值(MA-12)': ma12}

        # 季节性朴素需要窗口外的历史，从头算会吃掉前面一段样本，
        # 所以它们单独成一组，并且和上面三条在**同一子集**上再比一次。
        S = series_of(test3)
        # 自检：重建的连续序列必须能还原出原滑窗，否则季节基线算的是另一条
        # 序列，数字再好看也是假的。用几个分散的位置验，别只看 t=0。
        for t in (0, 1, 777, test3.shape[1] // 2, test3.shape[1] - 1):
            assert np.array_equal(S[:, t:t + 24], test3[:, t, :]), \
                f'series_of 重建失败于 t={t}'
        log(f'  [自检] 连续序列重建一致：{S.shape}（逐个窗口比对通过）')
        for lag, nm in ((288, '日周期(lag=288)'), (2016, '周周期(lag=2016)')):
            if S.shape[1] <= lag:
                log(f'  [跳过] {nm}：测试集序列只有 {S.shape[1]} 步，不够 {lag}')
                continue
            pr, ok = seasonal(S, Yte, lag)
            cands[nm] = (pr, ok)

        log(f'  测试集 {Yte.shape[0]:,} 个时刻 × {N} 节点 = '
            f'{Yte.shape[0] * N * 12:,} 个预测点\n')

        log(f'  {"基线":<22} {"MAE@15":>8} {"MAE@30":>8} {"MAE@60":>8} '
            f'{"RMSE@60":>9} {"MAPE@60":>9} {"样本":>9}')
        log('  ' + '-' * 80)
        full = {}
        for nm, v in cands.items():
            if isinstance(v, tuple):
                continue
            m = metrics(v, Yte)
            full[nm] = m
            log(f'  {nm:<22} {m["mae_15"]:>8.3f} {m["mae_30"]:>8.3f} {m["mae_60"]:>8.3f} '
                f'{m["rmse_60"]:>9.3f} {m["mape_60"]:>8.2f}% {m["n_eval"]:>9,}')

        # 季节性基线的公共子集：所有 lag 都能算的那些样本
        season = {k: v for k, v in cands.items() if isinstance(v, tuple)}
        if season:
            common = np.ones(Yte.shape[0], dtype=bool)
            for _, (_, ok) in season.items():
                common &= ok
            log(f'\n  含季节性基线的同子集对比（共同可算样本 {int(common.sum()):,} / '
                f'{Yte.shape[0]:,}，即 {common.mean() * 100:.1f}%）：')
            log(f'  {"基线":<22} {"MAE@15":>8} {"MAE@30":>8} {"MAE@60":>8} '
                f'{"RMSE@60":>9} {"MAPE@60":>9}')
            log('  ' + '-' * 80)
            sub = {}
            sub['持久性(last)'] = metrics(persist[common], Yte[common])
            sub['历史均值(train mean)'] = metrics(ha_pred[common], Yte[common])
            sub['近12步均值(MA-12)'] = metrics(ma12[common], Yte[common])
            for nm, (pr, _) in season.items():
                sub[nm] = metrics(pr[common], Yte[common])
            for nm, m in sub.items():
                log(f'  {nm:<22} {m["mae_15"]:>8.3f} {m["mae_30"]:>8.3f} {m["mae_60"]:>8.3f} '
                    f'{m["rmse_60"]:>9.3f} {m["mape_60"]:>8.2f}%')

        allout[ds] = dict(N=N, n_test=int(Yte.shape[0]), full=full, subset=sub if season else {})

        # 自检：历史均值预测恒定值，它的 MAE 应该落在目标序列自身标准差的
        # 同一量级（对高斯恰好是 0.8×std）。差一个数量级说明取数串了。
        sd = float(Yte.std())
        ha_mae = metrics(ha_pred, Yte)['mae_60']
        log(f'\n  [自检] 目标序列 std={sd:.3f}，历史均值 MAE@60={ha_mae:.3f}，'
            f'比值 {ha_mae / sd:.2f}（期望 ~0.8，差一个数量级就是取数串了）')
        del train3, val3, test3, Yte

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(allout, f, ensure_ascii=False, indent=2)
    log(f'\n已写入 {OUT}')

    log("""
怎么读：
  · 拿这张表去和模型的 MAE@15/30/60 比。任务书 Gate A 第 2 条要求
    「模型三档都不劣于持久性」。这里就能看出那条线卡在哪。
  · 周周期那条通常很强（交通流有稳定的周内节律）。如果模型连它都打不过，
    说明模型没学到任何输入→输出的映射，只有均值水平 —— 那就是路线 B 的信号。
  · 注意「同子集」那张表才是四条基线互比的公平表；上面那张里季节性基线的
    样本数和别人不同，不能横向比。
""")


if __name__ == '__main__':
    main()
