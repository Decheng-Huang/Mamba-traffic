# -*- coding: utf-8 -*-
"""时段通道 A/B 判决 —— 本地跑，不花一分钱租卡。

用法：
    1. 从盒子上把 paper_results.json 拷回本地
    2. python ab_verdict.py

为什么要单独一个脚本：**判决标准必须在看到数字之前就定死**。
如果等 A/B 两个数出来再议"差多少才算有用"，那就是事后挑选口径，
正是"不能靠着错误硬编"要挡的事。所以判据写在这里，先定后用。

判据（2026-09-24 定，见文件末尾 RULES）：
    Δ@60 ≥ +5%  →  采纳时段通道（B 为论文主模型，A 降级为消融行）
    Δ@60 ≤  0   →  不采纳（论文报一个诚实的阴性结果）
    中间地带    →  **不能只凭一个种子下结论**，要么补第二个种子，要么如实写"未见显著差异"
"""
import json
import os
import sys

RESULT_FILE = 'paper_results.json'
DS = 'PeMS-BAY'
BASE = f'{DS}|real|V8|official|s1'
ARM_A = BASE + '|notod'      # PAPER_TOD=0 PAPER_SUFFIX=notod
ARM_B = BASE + '|tod'        # PAPER_TOD=1（不设 PAPER_SUFFIX）

HZ = [15, 30, 60]

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def load():
    if not os.path.exists(RESULT_FILE):
        sys.exit(f'找不到 {RESULT_FILE} —— 先从盒子上下载回来。')
    with open(RESULT_FILE, encoding='utf-8') as f:
        return json.load(f)


def row(res, key):
    m = res.get(key)
    if m is None:
        return None
    return dict(mae={h: m[f'mae_{h}'] for h in HZ},
                rmse60=m['rmse_60'], mape60=m['mape_60'],
                params=m.get('params'), best_epoch=m.get('best_epoch'),
                epochs_run=m.get('epochs_run'), n_train=m.get('n_train'),
                epoch_time=m.get('epoch_time'), cfg=m.get('cfg'))


def main():
    res = load()
    a, b = row(res, ARM_A), row(res, ARM_B)

    print('=' * 78)
    print('时段通道 A/B 判决')
    print('=' * 78)
    for k in (ARM_A, ARM_B):
        print(('  ✓ ' if k in res else '  ✗ 缺 ') + k)
    if a is None or b is None:
        print('\n两臂还没跑齐，先等等。')
        return

    # ---- 单变量自检：除了 tod 那一维，其他必须完全一致
    print('\n--- 单变量自检（不通过的话下面的比较无意义）---')
    ok = True
    for f, why in [('n_train', '训练样本数'), ('epochs_run', '实际跑了几轮')]:
        va, vb = a.get(f), b.get(f)
        same = va == vb
        ok &= same
        print(f'  {"✓" if same else "✗"} {why}: A={va}  B={vb}')
    pa, pb = a.get('params'), b.get('params')
    dp = (pb - pa) if (pa and pb) else None
    # in_dim 6→7，第 0 层的 in_proj 多一列 → 参数差应恰好等于 d_inner
    print(f'  {"✓" if dp == 128 else "✗"} 参数量: A={pa:,}  B={pb:,}  '
          f'差={dp}（应恰为 d_inner=128）')
    ok &= (dp == 128)
    for tag in ('best_epoch', 'epoch_time'):
        print(f'    {tag}: A={a.get(tag)}  B={b.get(tag)}  '
              f'（不必相同，训练随机性）')
    if not ok:
        print('\n  ⚠️ 单变量自检没过 —— 这两臂不是"只差一个时段通道"，'
              '不要拿去当消融证据。')

    # ---- 主表
    print('\n--- 指标 ---')
    print(f'{"":>10} {"@15":>8} {"@30":>8} {"@60":>8} {"RMSE@60":>9} {"MAPE@60":>9}')
    for tag, r in (('A 无时段', a), ('B 有时段', b)):
        print(f'{tag:>10} {r["mae"][15]:>8.3f} {r["mae"][30]:>8.3f} '
              f'{r["mae"][60]:>8.3f} {r["rmse60"]:>9.3f} {r["mape60"]:>8.2f}%')

    d = {h: (a['mae'][h] - b['mae'][h]) / a['mae'][h] * 100 for h in HZ}
    print(f'{"B 相对 A":>10} ' + ' '.join(f'{d[h]:>+7.2f}%' for h in HZ) +
          '   （正 = 时段通道让误差变小）')

    # ---- 参照物
    pers = res.get(f'{DS}|基线-持久性')
    if pers:
        print('\n--- 参照（同一口径）---')
        print(f'{"持久性":>10} {pers["mae_15"]:>8.3f} {pers["mae_30"]:>8.3f} '
              f'{pers["mae_60"]:>8.3f}')
        for tag, r in (('A 无时段', a), ('B 有时段', b)):
            g = [(pers[f'mae_{h}'] - r['mae'][h]) / pers[f'mae_{h}'] * 100 for h in HZ]
            print(f'{tag+" vs":>10} ' + ' '.join(f'{x:>+7.2f}%' for x in g) +
                  '   （正 = 赢持久性）')

    # ---- 判决
    print('\n' + '=' * 78)
    print('判决（判据先定后用，不事后改口径）')
    print('=' * 78)
    if d[60] >= 5.0 and d[15] >= 0.0:
        v = (f'采纳时段通道：@60 改善 {d[60]:.2f}%（≥5%），@15 未退化（{d[15]:+.2f}%）。\n'
             f'  → 论文主模型用 B（含时段通道），A 作为"去掉时段特征"的消融行。')
    elif d[60] <= 0.0:
        v = (f'不采纳：@60 变化 {d[60]:+.2f}%（≤0）。\n'
             f'  → 论文主模型用 A（不含时段特征），并如实报告：'
             f'加入交通领域的标准时段协变量在本设定下未见收益。')
    else:
        v = (f'**中间地带**：@60 改善 {d[60]:+.2f}%，落在 0~5% 之间；'
             f'@15 {d[15]:+.2f}%。\n'
             f'  → 只有 1 个种子，无法把这点差异和种子噪声分开。两条路：\n'
             f'     (a) 补种子 2（两臂各多跑一次，约 ¥5.6）后再判；\n'
             f'     (b) 不补，论文用更简单的 A，并写"未观察到显著差异"。\n'
             f'  → **不允许**把 {d[60]:+.2f}% 直接说成"时段特征有效"。')
    print('  ' + v.replace('\n', '\n  '))
    print('=' * 78)


if __name__ == '__main__':
    main()
