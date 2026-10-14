# -*- coding: utf-8 -*-
"""
下载 METR-LA（207 传感器，洛杉矶，2012-03-01 ~ 2012-06-30）。

源：Hugging Face 镜像 hf-mirror.com（huggingface.co 在国内不通，这是清华镜像）
    数据集 abhishek8aiml/METR-LA，MIT 许可
    文件结构与本地 PeMS-BAY 完全一致：train/val/test.parquet + sensor_graph/adj_mx.npy
    出处：由 DCRNN 官方仓库 (github.com/liyaguang/DCRNN) 派生
"""
import os
import sys
import urllib.request

BASE = 'https://hf-mirror.com/datasets/abhishek8aiml/METR-LA/resolve/main'
DST = os.path.join('数据集', 'METR-LA')
FILES = [
    'train.parquet',
    'val.parquet',
    'test.parquet',
    'sensor_graph/adj_mx.npy',
    'sensor_graph/sensor_locations.csv',
    'sensor_graph/distances.csv',
    'sensor_graph/adj_mx_mapping.json',
]


def fetch(rel):
    url = f'{BASE}/{rel}'
    out = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'  跳过（已存在） {rel}  {os.path.getsize(out) / 1e6:.2f} MB')
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get('Content-Length', 0))
            tmp = out + '.part'
            got = 0
            with open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        sys.stdout.write(f'\r    {rel}  {pct:5.1f}%  '
                                         f'{got / 1e6:7.2f}/{total / 1e6:.2f} MB')
                        sys.stdout.flush()
            os.replace(tmp, out)
        print(f'\r  ✓ {rel:42s} {os.path.getsize(out) / 1e6:8.2f} MB')
        return True
    except Exception as e:
        print(f'\r  ✗ {rel:42s} 失败: {e}')
        return False


print('=' * 72)
print('下载 METR-LA  (hf-mirror.com)')
print('=' * 72)
ok = True
for f in FILES:
    ok &= fetch(f)

print()
print('=' * 72)
if ok:
    print(f'完成 → {DST}')
    for root, _, files in os.walk(DST):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f'  {os.path.relpath(p, DST):42s} {os.path.getsize(p) / 1e6:8.2f} MB')
else:
    print('有文件失败，重跑本脚本会自动续传（已完成的会跳过）')
print('=' * 72)
