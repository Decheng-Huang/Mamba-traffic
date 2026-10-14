# -*- coding: utf-8 -*-
"""
最终判定：读 t0_timestamp（ISO 字符串），直接看
  1. 单个节点内部，时间戳是不是连续递增、等间隔
  2. train / val / test 在时间上是不是前后相接（而非交错）
"""
import pyarrow.parquet as pq
import numpy as np

P = '数据集'
OUT = []


def p(s=''):
    print(s)
    OUT.append(s)


info = {}
for name in ['train', 'val', 'test']:
    t = pq.read_table(f'{P}/{name}.parquet', columns=['node_id', 't0_timestamp'])
    nid = np.asarray(t.column('node_id'))
    ts = np.asarray(t.column('t0_timestamp')).astype('datetime64[s]')
    info[name] = (nid, ts)

# ---------- 1. 单节点内部 ----------
p('=' * 78)
p('1. 节点 0 内部：时间戳是否连续递增、等间隔？')
p('=' * 78)
for name in ['train', 'val', 'test']:
    nid, ts = info[name]
    ts0 = ts[nid == 0]
    d = np.diff(ts0).astype('int64')
    p(f'\n  [{name}] 节点0 样本数 = {len(ts0):,}')
    p(f'    前 6 个:  {[str(x) for x in ts0[:6]]}')
    p(f'    后 3 个:  {[str(x) for x in ts0[-3:]]}')
    p(f'    间隔唯一值: {np.unique(d)[:8]}  （秒）')
    p(f'    严格递增? {bool((d > 0).all())}   全部等间隔? {bool((d == d[0]).all())}')
    if not (d == d[0]).all():
        bad = np.where(d != d[0])[0]
        p(f'    ★ 有 {len(bad)} 处间隔异常，前 5 处位置: {bad[:5]}  间隔: {d[bad[:5]]}')

# ---------- 2. 跨文件时间范围 ----------
p()
p('=' * 78)
p('2. 三个文件的时间范围（节点 0）')
p('=' * 78)
rng = {}
for name in ['train', 'val', 'test']:
    nid, ts = info[name]
    ts0 = ts[nid == 0]
    rng[name] = (ts0.min(), ts0.max(), len(ts0))
    p(f'  {name:6s}  {ts0.min()}  →  {ts0.max()}   ({len(ts0):,} 个样本)')

tr_min, tr_max, _ = rng['train']
va_min, va_max, _ = rng['val']
te_min, te_max, _ = rng['test']
p()
p(f'    train.max < val.min ?   {tr_max} < {va_min}   →  {bool(tr_max < va_min)}')
p(f'    val.max   < test.min?   {va_max} < {te_min}   →  {bool(va_max < te_min)}')
p()
if tr_max < va_min and va_max < te_min:
    p('  ✓✓ 三集合在时间上严格前后相接，无重叠 → 无信息泄漏（标准按时间划分）')
else:
    p('  ✗✗ 时间范围有重叠 → 信息泄漏，指标偏乐观')

# ---------- 3. 全部 325 个节点 ----------
p()
p('=' * 78)
p('3. 全部 325 个节点逐一检查（不是只看节点0）')
p('=' * 78)
leak = 0
gaps = []
cnts = {'train': [], 'val': [], 'test': []}
for n in range(325):
    r = {}
    for name in ['train', 'val', 'test']:
        nid, ts = info[name]
        m = nid == n
        r[name] = (ts[m].min(), ts[m].max(), int(m.sum()))
        cnts[name].append(int(m.sum()))
    if not (r['train'][1] < r['val'][0] and r['val'][1] < r['test'][0]):
        leak += 1
    gaps.append(int((r['val'][0] - r['train'][1]).astype('int64')))

p(f'  时间范围有重叠的节点数 = {leak} / 325')
gapu, gapc = np.unique(gaps, return_counts=True)
p(f'  train末→val首 的间隔: 唯一值={gapu[:6]} 秒，出现次数={gapc[:6]}')
p(f'    （间隔 = 300 秒说明 val 紧接 train 的下一个时间步，正常）')

# ---------- 4. 每节点样本数 ----------
p()
p('=' * 78)
p('4. 每节点样本数（确认 70/10/20）')
p('=' * 78)
for name in ['train', 'val', 'test']:
    c = np.array(cnts[name])
    p(f'  {name:6s} 节点数=325  每节点样本数 min={c.min():,} max={c.max():,} '
      f'{"（全部相同）" if c.min() == c.max() else "★ 不一致"}')
tot = sum(sum(v) for v in cnts.values())
for name in ['train', 'val', 'test']:
    p(f'  {name:6s} 占比 = {100 * sum(cnts[name]) / tot:.2f}%')

# ★ newline='\n' 不能省：本文件在《证据清单》里，`.gitattributes` 又是 `* -text`
#   （字节一致，不做任何换行转换）。文本模式默认会把 '\n' 翻成平台换行，
#   于是在 Windows 上跑一次就把 50 行全变成 CRLF、sha256 随之变 —— 而清单记的是
#   这个哈希。结果就是「清单只在写它的那台机器上验得过」。2026-10-10 实测过一次。
open('split_report.txt', 'w', encoding='utf-8', newline='\n').write('\n'.join(OUT))
p()
p('（已写入 split_report.txt）')
