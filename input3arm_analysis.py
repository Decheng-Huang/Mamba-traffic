# -*- coding: utf-8 -*-
"""三臂「输入侧」消融配对分析：real / random / self（PeMS-BAY）。

用法（把两个 JSON 放在同目录）：
    python input3arm_analysis.py [input3arm.json] [normfix.json]

背景 / 纪律
-----------
* 本脚本**直读结果 JSON 的原始字段**，不做任何手工誊录。
  2026-09-28 教训：用 3 位小数的日志誊录值算 t 会算错（t≈51 vs 真值 30.0）。
  ⇒ 一切数字只能取自结果文件原始字段。
* 三臂必须**逐条核对** `epochs_run` / `params` / `bs` 才能比 ——
  `cid` 带 `e{EPOCHS}`，轮数不同是不同 key，能跑通且看不出来。
* ⚠️ 本表是**输入侧**消融（喂进那 5 个邻居通道的是什么）。
  **不得与 `MAMBASTG`/`MAMBAST` 的配对消融（算子侧，有没有图算子）混在一张表里比。**
"""
import json
import statistics as st
import sys

IN3 = sys.argv[1] if len(sys.argv) > 1 else 'paper_results_input3arm_0928.json'
NRM = sys.argv[2] if len(sys.argv) > 2 else 'paper_results_normfix_0928_0414.json'
H = (15, 30, 60)
T_CRIT = 4.303          # df=2, 双尾 p<0.05


def grab(path, ds='PeMS-BAY'):
    """按 (neighbors, seed) 归集模型行；两条基线单独取。"""
    d = json.load(open(path, encoding='utf-8'))
    out = {}
    for k, v in d.items():
        if not k.startswith(ds + '|'):
            continue
        cfg = v.get('cfg')
        if not cfg:                      # 基线行没有 cfg
            if '持久性' in k:
                out['持久性'] = v
            elif '历史均值' in k:
                out['历史均值'] = v
            continue
        out.setdefault(cfg['neighbors'], {})[cfg['seed']] = v
    return out


A = grab(IN3)
B = grab(NRM)
# real 只在 NRM 里，random/self 只在 IN3 里；合并以兼容两文件都有的情况
arms = {a: {**B.get(a, {}), **A.get(a, {})} for a in ('real', 'random', 'self')}
base = {**B, **A}

print('== 可比性检查（必须逐条相同才能比）==')
for a, seeds in arms.items():
    for s in sorted(seeds):
        v = seeds[s]
        print(f'  {a:<7} s{s}  epochs_run={v["epochs_run"]:<3} '
              f'params={v["params"]}  bs={v["cfg"]["bs"]}')

print('\n== 每臂 mean ± sd（n=3 seed）==')
print(f'{"arm":<9}' + ''.join(f'{f"@{h}":>22}' for h in H))
for a in ('random', 'self', 'real'):
    row = f'{a:<9}'
    for h in H:
        v = [arms[a][s][f'mae_{h}'] for s in sorted(arms[a])]
        row += f'{st.mean(v):>13.6f} ±{st.stdev(v):.6f}'
    print(row)
for b in ('持久性', '历史均值'):
    print(f'{b:<8}' + ''.join(f'{base[b][f"mae_{h}"]:>22.6f}' for h in H))

print(f'\n== 逐 seed 配对差 %（|t|>{T_CRIT} 才 p<0.05）==')
for x, y in (('self', 'real'), ('random', 'real'), ('random', 'self')):
    for h in H:
        d = [(arms[y][s][f'mae_{h}'] - arms[x][s][f'mae_{h}']) / arms[x][s][f'mae_{h}'] * 100
             for s in sorted(arms[x])]
        m, sd = st.mean(d), st.stdev(d)
        t = m / (sd / 3 ** 0.5)
        print(f'  ({y}-{x})/x @{h:<2}: ' + ' '.join(f'{q:+7.4f}' for q in d)
              + f'   mean {m:+7.4f}% ±{sd:.4f}  t={t:+6.2f}'
              + ('  ★显著' if abs(t) > T_CRIT else '  (不显著)'))

print('\n== val_mae（模型选点用的那个 —— 与 test 排名不一致时必须一起报）==')
for a in ('real', 'random', 'self'):
    v = [arms[a][s]['val_mae'] for s in sorted(arms[a])]
    print(f'  {a:<7} {st.mean(v):.6f} ±{st.stdev(v):.6f}   '
          + ' '.join(f'{q:.6f}' for q in v))
