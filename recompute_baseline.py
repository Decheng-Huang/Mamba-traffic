# -*- coding: utf-8 -*-
"""recompute_baseline.py —— 用**当前生产代码**重算持久性基线，把它的定义钉死。

为什么必须重算（任务书 §0.3(a)）：
  仓库里 METR-LA 的持久性有**三个互相冲突的数字**，论文表里只能有一个：

    · 调查记录的表里       3.989 / 4.684 / 5.826   ← 在任何现存产物里都搜不到出处
    · 本地 paper_results   4.015 / 5.109 / 6.835   ← 09-23 代码状态；且同一文件里
                                                      PeMS-BAY 那格的 n_eval=300
                                                      （quick 子集），已被污染
    · baseline_only.py     4.056 / 5.377 / 7.422   ← 另一套口径（填 x/全点）

  靠读文件已经解决不了 —— 必须用**当前代码**实跑一次。

生产定义（逐行核自 paper_experiments.py:1225-1253 与 :781-825）：
  1. mean_x/std_x = 逐节点统计 train 的 x 区（12 列），**只用 >0 的有效值**
  2. te_x = (_fill(test_x, mean_x) - mean_x) / std_x    缺测填 train 节点均值
  3. te_y = (test_y - mean_y) / std_y                   ★ y 不填（0 是掩零锚点）
  4. 预测 = Xte[:, :, 11, 0]（标准化空间里的「最后一个观测」），广播 12 步
  5. compute_metrics 用 std_y/mean_y 反标准化，并**掩零**（tt > 0 = DCRNN null_val=0）

★ 第 4/5 步有一处可疑，本脚本顺手量出来：
    预测值活在 **x 空间**（用 mean_x/std_x 标准化），反标准化却用 **y 空间**的
    std_y/mean_y。只有 std_x == std_y 时它才等于「最后观测速度」。持久性是全篇
    论文的主标尺（"+17.8% vs 持久性"），所以这个量必须打印出来。

★ 保真性：指标与基线走的是 **真正的 P.run_baselines / P.compute_metrics**，
  只把 _stat/_fill 抄一份（它们是函数内的闭包，导不出来）。抄写是否等价的
  判据在 `verify_fill_fidelity`：把生产口径的数字与一条独立直算路径对账。

用法：
    python recompute_baseline.py                # 两个数据集
    python recompute_baseline.py METR-LA        # 只算一个
"""
import os
import sys

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

os.environ.setdefault('PAPER_MODE', 'checkdata')   # 防止 import 触发主流程
os.environ.setdefault('PAPER_TOD', '0')            # 基线不需要时段通道
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                  # noqa: E402
import paper_experiments as P                       # noqa: E402


# ---------------------------------------------------------------- 从生产代码抄来
# 出处 paper_experiments.py:1227-1241（函数内闭包，无法 import）。**未改一字。**
def node_stat(a):
    """逐节点 mean/std，缺测（0）不计入。整节点全缺测时退化为 mean=0/std=1。"""
    m = a > 0
    ok = m.any(axis=(1, 2), keepdims=True)
    n = np.maximum(m.sum(axis=(1, 2), keepdims=True), 1)
    mu = (a * m).sum(axis=(1, 2), keepdims=True) / n
    var = (((a - mu) ** 2) * m).sum(axis=(1, 2), keepdims=True) / n
    mu = np.where(ok, mu, 0.0)
    sd = np.where(ok, np.sqrt(var), 1.0) + 1e-8
    return mu, sd


def node_fill(a, mu):
    """缺测填成该节点均值（用的是 train 的均值，val/test 同样填 train 的）。"""
    return np.where(a > 0, a, mu)


def hr(t=''):
    print('\n' + '=' * 78)
    if t:
        print(t)
        print('=' * 78)


def run(ds_name):
    hr(f'{ds_name}')
    out = P.load_dataset(ds_name)
    train3, val3, test3, adj, N = out[:5]
    print(f'  N={N}  train={train3.shape}  val={val3.shape}  test={test3.shape}')

    tx, ty = train3[..., :12], train3[..., 12:]
    mean_x, std_x = node_stat(tx)
    mean_y, std_y = node_stat(ty)

    # ★ 可疑处一：x 空间 vs y 空间的统计量差多少
    d_mu = float(np.abs(mean_x - mean_y).max())
    d_sd = float(np.abs(std_x - std_y).max())
    r_sd = float((std_x / std_y).max())
    print(f'  [x/y 空间统计量] max|mean_x-mean_y|={d_mu:.6g}  '
          f'max|std_x-std_y|={d_sd:.6g}  max(std_x/std_y)={r_sd:.6f}')
    print(f'                   std_x 中位={np.median(std_x):.4f}  '
          f'std_y 中位={np.median(std_y):.4f}  '
          f'（若两者差异大，预测值反标准化会缩放它）')

    te_x = (node_fill(test3[..., :12], mean_x) - mean_x) / std_x
    te_y = (test3[..., 12:] - mean_y) / std_y

    # nbr_idx=None -> C=1。基线只看通道 0，所以不需要邻居，省 13 倍内存。
    Xte, Yte = P.make_samples(te_x, te_y, None, None)
    print(f'  Xte={Xte.shape}  Yte={Yte.shape}')
    # ★ 2026-09-27：run_baselines 现在多一个必需参数 hist_raw（HA 官方口径要
    #   原始历史序列）。本脚本只读 ['基线-持久性']，但签名变了就得跟着传。
    hist_raw = np.concatenate(
        [P.raw_speed_series(train3), P.raw_speed_series(val3)[:, 23:]], axis=1)
    del train3, val3, tx, ty
    import gc
    gc.collect()

    m = P.run_baselines(Xte, Yte, std_y, mean_y, N, hist_raw)['基线-持久性']

    hr(f'{ds_name} —— 持久性（生产口径：缺测填节点均值 + 掩零）')
    print(f'  MAE@15={m["mae_15"]:.4f}  MAE@30={m["mae_30"]:.4f}  '
          f'MAE@60={m["mae_60"]:.4f}')
    print(f'  RMSE@15={m["rmse_15"]:.4f}  RMSE@30={m["rmse_30"]:.4f}  '
          f'RMSE@60={m["rmse_60"]:.4f}')
    print(f'  MAPE@60={m["mape_60"]:.4f}%   掩零比例 '
          f'@15={m["null_frac_15"] * 100:.4f}% @60={m["null_frac_60"] * 100:.4f}%')
    print(f'  （未掩零对照）mae_all_15={m["mae_all_15"]:.4f} '
          f'mae_all_30={m["mae_all_30"]:.4f} mae_all_60={m["mae_all_60"]:.4f}')
    print(f'  n_eval={m["n_eval"]}')

    # ---------------- 2×2 口径表：四种可辩护定义一次摊开 ----------------
    # 预测值有两个来源：fill（生产：缺测填 train 节点均值）与 raw（缺测留 0）。
    # 指标有两个口径：掩零（生产：tt > 0，= DCRNN null_val=0）与全点。
    # 生产的组合是 ④。这张表就是任务书 §0.3(a) 要的那张。
    last_z = Xte[:, :, 11, 0]                        # (T,N) 标准化空间（已填）
    last_fill = last_z * std_y.reshape(1, N) + mean_y.reshape(1, N)
    last_raw = test3[:, :, 11].T.astype(np.float64)  # (T,N) 未填，缺测就是 0
    Yr = test3[:, :, 12:].transpose(1, 0, 2).astype(np.float64)   # (T,N,12) 原始
    n_last_missing = int((last_raw <= 0).sum())
    print(f'\n  [缺测] 最后一帧 <=0 的占 {100.0 * n_last_missing / last_raw.size:.4f}%'
          f'（{n_last_missing:,}/{last_raw.size:,}）—— fill 与 raw 的差别全在这里')

    print(f'\n  {"时域":<6} {"①raw/全点":>11} {"②fill/全点":>12} '
          f'{"③raw/掩零":>11} {"④fill/掩零":>12}')
    print('  ' + '-' * 60)
    for hmin, idx in P.HORIZONS.items():
        yy = Yr[:, :, idx - 1]
        mk = yy > 0
        r = [float(np.abs(last_raw - yy).mean()),
             float(np.abs(last_fill - yy).mean()),
             float(np.abs(last_raw - yy)[mk].mean()),
             float(np.abs(last_fill - yy)[mk].mean())]
        print(f'  MAE@{hmin:<2} {r[0]:>11.4f} {r[1]:>12.4f} '
              f'{r[2]:>11.4f} {r[3]:>12.4f}')

    # 保真性自检：两条独立路径必须走到同一个 ④@60。
    #   A = 本脚本直算（test3 原始张量 → 掩零 → 取均值）
    #   B = P.run_baselines（make_samples + compute_metrics 的真实生产函数）
    # 不等说明两条路径读的不是同一份张量，那么上面的表也不能信。
    d60 = np.abs(last_fill - Yr[:, :, 11])
    m60 = Yr[:, :, 11] > 0
    a60, b60 = float(d60[m60].mean()), float(m['mae_60'])
    flag = 'OK  ' if abs(a60 - b60) < 1e-3 else '不同!!'
    print(f'\n  [自检] {flag} 直算 ④@60 = {a60:.4f}   '
          f'生产 run_baselines = {b60:.4f}   差 {a60 - b60:+.2e}')
    if flag != 'OK  ':
        print('         >1e-3 的差 = 两条路径读的不是同一份张量，上表不可信。')
    del test3, Yr, last_raw, last_fill, Xte, Yte
    gc.collect()
    return m


if __name__ == '__main__':
    todo = sys.argv[1:] or ['METR-LA', 'PeMS-BAY']
    for nm in todo:
        try:
            run(nm)
        except Exception as e:
            print(f'\n[{nm}] 失败 {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
