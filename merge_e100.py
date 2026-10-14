#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把盒子上下载的 `res_*.json` 里**天花板 100** 的那批键，合并成
   `box_results/paper_results_e100.json`（make_tables.py 的输入之一）。

为什么要有这一步
----------------
`cid` 不含轮数（paper_experiments.py:1399-1402），轮数靠 `PAPER_SUFFIX` 带进去。
所以 e30 与 e100 是**两个不同的键**、混在同一个 `res_*.json` 里：
    res_V8_real_s1.json   { PeMS-BAY|real|V8|official|s1|tod            ← e30
                            PeMS-BAY|real|V8|official|s1|tod|e100 }     ← e100
make_tables.py 只认 `|e100`，所以先把它们挑出来单独放一个文件 —— 这样
「表里写 e100、数却是 e30」这个错误在**结构上**就不可能再发生。

用法
----
    python merge_e100.py                    # 扫 box_results/
    BOX_DIR=/tmp/sand python merge_e100.py  # 沙盒（测试用）

★ 只搬运，不加工：每个字段都按**原样**复制，不做任何重命名或计算。
   出现同名键而值不同时**直接报错**，不"以后者为准"—— 那正是静默覆盖的入口。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BOX = os.environ.get('BOX_DIR') or os.path.join(ROOT, 'box_results')
OUT_NAME = 'paper_results_e100.json'
SUFFIX = '|e100'


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    if not os.path.isdir(BOX):
        raise SystemExit('★ 目录不存在：%s' % BOX)

    # 只看**盒子上下载的原始 per-job 文件**。不扫聚合文件（它们只是本脚本的产物，
    # 扫进来会自我循环），也不扫 gnn_results（那是另一条线，轮数在键里自带 e100）。
    srcs = sorted(n for n in os.listdir(BOX)
                  if n.startswith('res_') and n.endswith('.json'))
    if not srcs:
        raise SystemExit('★ %s 下没有 res_*.json —— 先把盒子上的 res_json.tgz '
                         '解到这里（解之前先核 sha256）。' % BOX)

    merged, origin, took = {}, {}, []
    for n in srcs:
        try:
            d = load(os.path.join(BOX, n))
        except Exception as e:
            raise SystemExit('★ 读不了 %s：%s: %s' % (n, type(e).__name__, e))
        got = [k for k in d if k.endswith(SUFFIX)]
        for k in got:
            if k in merged and merged[k] != d[k]:
                # ★ 同名不同值：绝不静默取一个。把差异点打印到字段级。
                a, b = merged[k], d[k]
                diff = sorted({f for f in set(a) | set(b) if a.get(f) != b.get(f)})
                detail = '\n'.join('      %-12s %s=%r   %s=%r'
                                   % (f, origin[k], a.get(f), n, b.get(f)) for f in diff)
                raise SystemExit('★ 键 %s 在两个文件里值不同 —— 停下。\n'
                                 '   来源 A: %s\n   来源 B: %s\n   差异字段:\n%s\n'
                                 '  （同一个 cid 两次跑出不同结果：先查是不是并发写坏 / '
                                 'different EPOCHS）' % (k, origin[k], n, detail))
            merged[k] = d[k]
            origin[k] = n
        took.append((n, len(d), len(got)))

    print('扫描 %s' % BOX)
    print('  %-28s %6s %8s' % ('文件', '总键', '|e100 键'))
    for n, tot, got in took:
        print('  %-28s %6d %8d%s' % (n, tot, got, '' if got else '   ← 没有 |e100 键'))
    print('  %-28s %6s %8d' % ('合计（去重后）', '', len(merged)))

    if not merged:
        raise SystemExit('\n★ 一个 %s 键都没有 —— 这批还没跑，或者下错文件了。' % SUFFIX)

    # 独立锚点：这些键都必须真的是本批的（不许混进 e30 键）
    bad = [k for k in merged if not k.endswith(SUFFIX)]
    assert not bad, '筛出逻辑坏了，混进非 e100 键：%s' % bad
    cid_parts = {len(k.split('|')) for k in merged}
    print('\n按 cid 段数分组：%s' % {p: sum(1 for k in merged if len(k.split('|')) == p)
                                    for p in sorted(cid_parts)})
    for k in sorted(merged):
        v = merged[k]
        print('  %-46s ep_run=%-4s best=%-4s params=%-8s mae15=%s'
              % (k, v.get('epochs_run', '-'), v.get('best_epoch', '-'),
                 v.get('params', '-'),
                 ('%.6f' % v['mae_15']) if isinstance(v.get('mae_15'), float) else '-'))

    fp = os.path.join(BOX, OUT_NAME)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write('\n')

    # ★ 回读对质：写出去再读回来必须逐键相等（不是"看起来对"）
    back = load(fp)
    assert back == merged, '回读与内存不一致：写坏了'
    print('\n落盘 %s   %d 键   %d 字节' % (fp, len(back), os.path.getsize(fp)))
    print('回读逐键相等 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
