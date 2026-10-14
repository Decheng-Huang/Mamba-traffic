# -*- coding: utf-8 -*-
"""check_box.py —— 只读盘点盒子（AutoDL）上已经跑出了什么。

设计约束：
  · **只读**：不写、不删、不改任何文件，只 print。盘点上不该有副作用。
  · **自定位**：盒子上的项目路径我没法从本地确认，所以自己从 /root /home
    /workspace /mnt 往下找带 paper_experiments.py 或 gnn_baselines.py 的目录。
  · **有界**：walk 限深 5、限时 150 s，跳过 conda/site-packages 等大目录。
  · **可粘贴**：全部 stdlib，无第三方依赖。

用法（在盒子上任意目录）：
    python3 check_box.py
"""
import glob
import json
import os
import subprocess
import sys
import time

# 中文 Windows 的 stdout 是 GBK，遇到 GBK 里没有的符号会直接 UnicodeEncodeError
# 崩掉整个盘点 —— 本地实测崩过一次（是那个"打勾"符号）。盒子是 UTF-8 没事，
# 但本地也要能跑（本地也能拿它核对）。
# ★ 只改 errors，**不改 encoding**：硬改成 utf-8 会让中文在 GBK 控制台变乱码。
#   保留原编码 + errors='replace' = 中文正常、个别符号退化成 '?'、永不崩。
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

# 本地（写这份脚本时）的哈希，用来判断盒子上那份代码是不是旧的。
# 改本地脚本后要同步更新这里，否则会误报"盒子代码是新的"。
LOCAL_MD5 = {
    'gnn_baselines.py':    'd08dfa578cdb7045f25de87adb10558e',
    'paper_experiments.py': 'b96fb74282a18465bf0ae44fd823c2e8',
}
# 要核对的 GNN 格：数据集、模型、轮数。
# ★ 必须**精确到轮数**：只核 `ds|mdl|` 前缀会把"e100 那格压根没跑"判成 OK
#   —— 2026-09-26 第一次跑就是这样：脚本对 METR-LA|DCRNN 打了 OK，
#   而那格其实死在 ep97/100、没有 .pt、没有 json 记录。前缀匹配不算核对。
GNN_EXPECT = [('PeMS-BAY', 'GWNET'), ('PeMS-BAY', 'DCRNN'),
              ('METR-LA', 'GWNET'), ('METR-LA', 'DCRNN')]
GNN_EPOCHS = [100]
NUMKEYS = ('mae_15', 'mae_30', 'mae_60', 'rmse_60', 'mape_60',
           'params', 'best_epoch', 'epochs_run', 'epoch_time', 'epochs')


def flat(m, prefix='', depth=0):
    """把嵌套 dict 展平成 (路径, 标量) 列表。

    为什么需要：三份 json 的形状并不一样 —— gnn_results.json 是
    'cid -> 指标 dict'，而 baseline_results.json 是 '数据集 -> 模型 -> 指标'。
    只按第一种写，第二种会被打印成"(无已知指标键)"，等于什么都没看到。
    """
    out = []
    for k, v in m.items():
        kk = f'{prefix}{k}'
        if isinstance(v, dict) and depth < 2:
            out += flat(v, kk + '.', depth + 1)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out.append((kk, v))
    return out


def sh(cmd, t=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=t)
        out = (r.stdout or '').strip()
        if not out and r.stderr:
            out = '(stderr) ' + r.stderr.strip()[:300]
        return out or '(无输出)'
    except Exception as e:
        return f'<命令失败: {e}>'


def hr(t=''):
    print('\n' + '=' * 74)
    if t:
        print(t)
        print('=' * 74)


# ---------------------------------------------------------------- 0. 机器状态
hr('0. 机器与运行状态（有没有东西还在跑 = 还在计费）')
print('host   :', sh('hostname'), '|', sh('date "+%F %T"'),
      '| 已开机', sh('uptime -p'))
print('GPU    :', sh('nvidia-smi --query-gpu=name,memory.used,memory.total,'
                    'utilization.gpu --format=csv,noheader'))
print('python :', sh('python3 -c "import torch,sys;print(sys.version.split()[0],'
                    '\'torch\',torch.__version__,\'cuda\',torch.cuda.is_available())"'))
print('mamba  :', sh('python3 -c "import mamba_ssm;print(\'mamba_ssm OK\','
                    'mamba_ssm.__version__ if hasattr(mamba_ssm,\'__version__\') else \'\')"'))
print('\n[进程] 只看 python/paper_experiments/gnn_baselines：')
print(sh("ps -eo pid,etime,pcpu,pmem,args --sort=-pcpu "
         "| grep -E 'python|paper_exper|gnn_base' | grep -v grep | head -12"))
print('\n[磁盘]')
print(sh('df -h /root /root/autodl-tmp 2>/dev/null | grep -v Filesystem'))


# ---------------------------------------------------------------- 1. 找项目目录
hr('1. 找项目目录（含 paper_experiments.py 或 gnn_baselines.py）')
SKIP = ('site-packages', 'dist-packages', '.cache', 'conda', 'node_modules',
        'wheels', '.git', '.local', 'autodl-fs', '__pycache__')
MARK = ('paper_experiments.py', 'gnn_baselines.py')
cwd = os.getcwd()
found, t0 = [], time.time()
for root in ('/root', '/home', '/workspace', '/mnt'):
    if not os.path.isdir(root):
        continue
    base = root.rstrip('/').count('/')
    for dp, dn, fn in os.walk(root, topdown=True):
        if time.time() - t0 > 150:
            break
        if dp.count('/') - base > 5:
            dn[:] = []
            continue
        dn[:] = [d for d in dn
                 if not d.startswith('.')
                 and not any(s in d.lower() for s in SKIP)]
        if any(m in fn for m in MARK):
            found.append(dp)
if any(os.path.isfile(os.path.join(cwd, m)) for m in MARK) and cwd not in found:
    found.insert(0, cwd)
print(f'找到 {len(found)} 个：')
for d in found:
    print('   ', d)
if not found:
    print('   [!] 一个都没找到 —— 盒子可能已被释放，或项目在别的路径。')
    print('   当前目录 =', cwd, '  请把 `pwd` 和 `ls -la` 的输出贴给我。')


# ---------------------------------------------------------------- 2/3/4/5 逐目录
for d in found:
    hr(f'2. {d} —— 文件清单（按 mtime 倒序，前 40 个）')
    rows = []
    for pat in ('*.json', '*.log', '*.pt', '*.npy', '*.md', '*.sh', '*.bak'):
        for f in glob.glob(os.path.join(d, pat)):
            try:
                st = os.stat(f)
            except OSError:
                continue
            rows.append((st.st_mtime, os.path.basename(f), st.st_size))
    for mt, name, sz in sorted(rows, reverse=True)[:40]:
        print(f'   {time.strftime("%m-%d %H:%M", time.localtime(mt))}'
              f'  {sz / 1e6:9.2f} MB  {name}')
    if len(rows) > 40:
        print(f'   … 另有 {len(rows) - 40} 个（截断）')
    if not rows:
        print('   （空 —— 这个目录没有任何结果文件）')

    hr(f'3. {d} —— 结果 JSON 的实际内容')
    for jf in ('paper_results.json', 'gnn_results.json', 'baseline_results.json'):
        path = os.path.join(d, jf)
        if not os.path.isfile(path):
            print(f'\n[{jf}] 不存在')
            continue
        try:
            r = json.load(open(path, encoding='utf-8'))
        except Exception as e:
            print(f'\n[{jf}] 读不了：{e}')
            continue
        print(f'\n[{jf}]  {len(r)} 条记录'
              f'（{os.path.getsize(path) / 1024:.0f} KB）')
        if not isinstance(r, dict):
            print('   顶层不是 dict，原样打印：', str(r)[:400])
            continue
        for cid in sorted(r):
            m = r[cid]
            if not isinstance(m, dict):
                print(f'   {cid}: {m}')
                continue
            nums = '  '.join(
                f'{k}={m[k]:.4f}' if isinstance(m[k], float) else f'{k}={m[k]}'
                for k in NUMKEYS if k in m)
            print(f'   {cid}')
            if nums:
                print(f'      {nums}')
                continue
            # 平铺的键一个都没有 → 换递归展开（baseline_results.json 那种形状）
            kv = flat(m)
            if not kv:
                print('      (空的)')
                continue
            # 上限 12 行/条：baseline_results.json 里有 N × 基线数 × 时域 ×
            # full/subset 的笛卡尔积，不限长的话一个文件就能喷几百行，
            # 贴回来反而没法看。截断时明说截了多少。
            CAP = 12
            kv = [x for x in kv if not str(x[0]).endswith(('mape_raw_15',
                                                           'mape_raw_30',
                                                           'mape_raw_60'))]
            for i in range(0, min(len(kv), CAP * 4), 4):
                chunk = '  '.join(
                    f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                    for k, v in kv[i:i + 4])
                print('      ' + chunk)
            if len(kv) > CAP * 4:
                print(f'      … 另有 {len(kv) - CAP * 4} 项未打印（已截断）')

    # --- 预期 vs 实际：直接点出缺哪个（精确 cid，不是前缀）
    gpath = os.path.join(d, 'gnn_results.json')
    if os.path.isfile(gpath):
        print(f'\n[GNN 基线按 e{"/e".join(map(str, GNN_EPOCHS))} 逐格核]')
        try:
            g = json.load(open(gpath, encoding='utf-8'))
            if not isinstance(g, dict):
                print('   顶层不是 dict，跳过')
            else:
                for ds, mdl in GNN_EXPECT:
                    for ep in GNN_EPOCHS:
                        cid = f'{ds}|{mdl}|s42|c2|e{ep}'
                        print(f'   {"OK  " if cid in g else "缺!!"}  {cid}')
                extra = [k for k in g
                         if not any(k == f'{ds}|{m}|s42|c2|e{e}'
                                    for ds, m in GNN_EXPECT
                                    for e in GNN_EPOCHS)]
                if extra:
                    print(f'   （另有 {len(extra)} 条非预期记录，见上一节）')
        except Exception as e:
            print('   读不了：', e)

    hr(f'4. {d} —— 每个 log 的开头 5 行 + 结尾 8 行')
    for lf in sorted(glob.glob(os.path.join(d, '*.log'))):
        name = os.path.basename(lf)
        sz = os.path.getsize(lf)
        try:
            lines = open(lf, encoding='utf-8', errors='replace').read().splitlines()
        except Exception as e:
            print(f'\n[{name}] 读不了：{e}')
            continue
        print(f'\n[{name}]  {sz / 1024:.0f} KB  {len(lines)} 行')
        for ln in lines[:5]:
            print('   头|', ln[:160])
        if len(lines) > 13:
            print('   ...')
        for ln in lines[-8:]:
            print('   尾|', ln[:160])

    hr(f'5. {d} —— 代码是不是旧的（盒子 md5 vs 本地 md5）')
    for fn, want in LOCAL_MD5.items():
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            print(f'   [缺] {fn} 不存在')
            continue
        got = sh(f'md5sum "{p}"').split()[0]
        same = (got == want)
        print(f'   {"[一致]  " if same else "[不同]!!"}  {fn}')
        print(f'        盒子 {got}')
        print(f'        本地 {want}   大小 {os.path.getsize(p):,} B')
    p = os.path.join(d, 'gnn_baselines.py')
    n = sh(f'grep -c "MambaST" "{p}" 2>/dev/null || echo 0').strip()
    print(f'   gnn_baselines.py 里 "MambaST" 出现次数 = {n}'
          f'（本地是新增了 MambaST 的，盒子应该没有/较少）')

    hr(f'6. {d} —— 数据目录与 parquet')
    for sub in ('数据集', 'data', 'dataset', 'Dataset'):
        sd = os.path.join(d, sub)
        if not os.path.isdir(sd):
            continue
        print(f'[{sd}]')
        print(sh(f'ls -l "{sd}"'))
        print(sh(f'find "{sd}" -name "*.parquet" -printf "%TY-%Tm-%Td %TH:%TM  '
                 f'%10s  %p\\n" 2>/dev/null | head -30'))
