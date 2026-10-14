#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 box_results/ 的原始 JSON 生成论文表格（全精度）。

★ 纪律（2026-09-28 定）：**一切数字只能取自结果文件的原始字段**。
  用汇总表的 3 位小数誊录过一次，结果把统计量的定义都算错了
  （Cohen's d 被当成 t，差 √3 倍）—— 见记忆 statistic-definition-must-travel-with-value。

★ 纪律（2026-10-10 定）：**本脚本不再印 Cohen's d / 配对 t / 任何显著性结论。**
  三个 seed 共用同一个固定测试集，不是三次独立重复 —— 这个样本量下 d 与 t
  撑不起任何推断。审稿意见（2026-10-10 TRR 投稿前评审 P0-3）据此要求撤除，
  论文、中文稿、本表**一起撤**。留下的是逐 seed Δ 与 mean±sd：那是描述，不是推断。
  这两行曾真实存在过（`t=52.01`、`d=30.03`），别以为它们是笔误就能顺手改回来。

★ 纪律（2026-09-29 定）：**全表统一天花板 100 轮。**
  `cid` 不含轮数，轮数靠 `PAPER_SUFFIX` 带进去（paper_experiments.py:1399-1402），
  所以 e30 与 e100 是**两个不同的键**，不是覆盖关系。
  为什么必须 100：`epochs_run = best_epoch + PATIENCE + 1`（PATIENCE=8）。
  best_epoch 靠后的行在 30 的天花板下**还没等够 PATIENCE 轮就被强行结束**，
  选出来的不是它的早停点。**哪些行会被截断，本脚本逐行算并打印**
  （见「为什么统一到 100」那一节）—— 不要在这里写死具体轮数：
  本文件上一版就写死过「best_epoch 是 27/28/29」，与结果文件里的 26/24/27 对不上。

★ 本脚本会主动打印「不支持的主张」清单，而不是只打一张漂亮的表：
  哪些对比我们输了、哪些行没收敛、哪些口径不一致，都印出来。
"""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# ★ 可用 BOX_DIR 覆盖：测试时指向合成数据，绝不污染真 box_results/
BOX = os.environ.get('BOX_DIR') or os.path.join(ROOT, 'box_results')

# ★ 结果数字的**显示精度**唯一定义在 `_precision.py`（2026-10-10 起 3 位小数）。
#   本文件里必须分得清两类数字 ——
#     · md 报告里的**数据行**：6 位，是内部对账物、也是检查器的独立锚点，**不动**；
#     · 「LaTeX 表体」那两块（直接粘进 ch4_experiments.tex）：必须印**与 latex 相同**
#       的精度，否则生成器与稿子各说各话，而 `check_ch4_repro.py` 锚点 K 会报不一致。
sys.path.insert(0, ROOT)
from _precision import s as _s                                # noqa: E402

H = [15, 30, 60]
# 盒子下载时 gnn_results 的文件名里多了一个空格，这里两种都认
GNN_FILES = ['gnn_results .json', 'gnn_results.json']
E100_FILE = 'paper_results_e100.json'


def load(name):
    fp = os.path.join(BOX, name)
    with open(fp, encoding='utf-8') as f:
        return json.load(f)


def load_gnn():
    for n in GNN_FILES:
        fp = os.path.join(BOX, n)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                return json.load(f), n
    raise SystemExit('找不到 gnn_results 文件：%s' % GNN_FILES)


def load_e100():
    """天花板 100 那一批：V8 三 seed + LSTM + Transformer 的 `|e100` 键。

    ★ **没有回退。** 文件不在就返回 None，由调用方把对应行显式标成「缺」并
      大声报警 —— 绝不静默退回 e30 的数字（那正是本文件历史上出过的错：
      表里写着 e100、数却是 e30）。
    """
    fp = os.path.join(BOX, E100_FILE)
    if not os.path.exists(fp):
        return None
    return load(E100_FILE)


GNN, GNN_NAME = load_gnn()
NORM = load('paper_results_normfix_0928_0414.json')
E100 = load_e100()
HAVE_E100 = E100 is not None

# ★ 有两个文件**本表一个数字都不取**，这里明确记下原因，免得日后有人以为漏了：
#   · paper_results_input3arm_0928.json —— 输入侧 random/self 臂的 **e30** 那批，
#     已被 paper_results_e100.json 里的 |e100 键取代（旧值只用于「必须披露」那一节）。
#   · paper_results.json —— HA 口径修正（BL_VERSION 1→2）**之前**产出的文件，
#     且含 252,684 参数的旧行；取了它 PeMS-BAY 的 HA 会变成 5.2502（官方口径是 2.88）。
#   两者都留在 box_results/ 里存档，但**刻意不 load** —— load 了却不用，最容易
#   在某次改动里"顺手"用上。

# 本机控制台是 GBK，而本脚本要打 ⚠️/✗ 这类字符。不兜底就会**死在打印上**：
# 抛 UnicodeEncodeError、rc=1，而 OUT 还没落盘 —— 看起来像"脚本跑失败了"，
# 实际是"全部算完了，只是打不出来"。与三个表格检查器同一个坑（2026-10-09）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # ★ stderr 也要 —— `raise SystemExit(中文)` 与 AssertionError 都走 stderr，
    #   只改 stdout 的话，**判据发出的那句话本身**会糊成 `??Щ seed ??`，
    #   下游拿字符串去认判据的装置（_neg_check_e_seeds.py）就认不出来，
    #   于是"判据明明发了"被读成"不是它发的"。2026-10-10 当场踩到。
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = []


def p(s=''):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------- 统计工具
def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """样本标准差（n-1）。三 seed 时它与总体标准差差 √(3/2)，必须写清用哪个。"""
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ★ 这里原有 stats()：返回 (均值, 样本sd, Cohen's d, 真 t) 并印进本表。
#   2026-10-10 整个删掉 —— 见文件头第三条纪律。调用处一律改回 mean()/sd()。
#   之所以**删掉函数本身**而不是留着不用：留着一个叫 stats 的函数，
#   下一个人补一行 '%.2f' % stats(x)[2] 就能把撤掉的东西悄悄请回来。


def mae(rec, h):
    return rec['mae_%d' % h]


# ★★ 2026-09-29：ch3 的 `Metrics` 小节明写「We report MAE, RMSE and MAPE」，
#    而 ch4 两张表只有 MAE —— 三句承诺一句没兑现。数据其实全在结果文件里
#    （每个记录都有 rmse_/mape_/mape_cov_/n_eval），所以补伴随表，不改承诺。
#    定义**必须跟着值一起报**（见记忆 statistic-definition-must-travel-with-value）：
#      rmse_{h}     = sqrt(mean(d^2))   ，掩零（真值 > null_norm）
#      mape_{h}     = mean(|d/tt|)*100  ，掩零 **且** 真值 > MAPE_MIN(=1.0 原速)
#      mape_cov_{h} = 上述两个条件保留的点占比（**比例，不是百分数**）
#      mape_raw_{h} = 不掩零的对照，**不是可上报指标** —— PeMS-BAY 上它 ≈430%
#                     完全由 0.006% 的零真值点经 max(|tt|,1e-3) 放大而来
#                     （paper_experiments.py:1608 的注释自证）。本脚本不报它。
def rmse(rec, h):
    return rec['rmse_%d' % h]


def mape(rec, h):
    return rec['mape_%d' % h]


def mape_cov(rec, h):
    return rec['mape_cov_%d' % h]


# ----------------------------------------------------------------- 取行
#  ★ 天花板 100 那批的键长这样（`|e100` 来自 PAPER_SUFFIX，见文件头注释）：
#      PeMS-BAY|real|V8|official|s1|tod|e100
PATIENCE = 8   # paper_experiments.py:95
V8_E100 = 'PeMS-BAY|%s|V8|official|s%d|tod|e100'


def gnn_row(model, seed):
    return GNN['PeMS-BAY|%s|s%d|c2|e100' % (model, seed)]


# ★ 2026-09-29 加：**patience 不是一个常数，逐 run 不同** —— 从每个 run 自己写的
#   `cfg.patience` 里读。实测（本轮盒子探针）：
#       GWNET|s42|e100    cfg.patience=100     DCRNN|s42|e100    cfg.patience=100
#       MAMBASTG|s42|e100 cfg.patience=8       E100（V8/LSTM/TF）cfg 里**没有**这个字段
#   起因：本文件第 7 条披露原先写死 `'%d' % PATIENCE`（=8，出自 paper_experiments.py:95），
#   而它描述的那一行是 DCRNN e100，其 cfg.patience 是 **100** —— 披露里的数字
#   说的是"脚本默认"，读者读到的却是"我们这一行用了什么"。
#   `gnn_baselines.py:141` 的 `PATIENCE` 只是**环境变量默认值**，
#   「默认值只证明不设时用什么，不证明实际设了什么」（记忆 check-env-default-before-override）。
def patience_of(row):
    return ((row or {}).get('cfg') or {}).get('patience')


def patience_eff(row):
    """有效 patience：run 自己的 cfg 优先，缺就回落本文件常数（仅 E100 行会走到）。"""
    v = patience_of(row)
    return PATIENCE if v is None else v


def _show_pat(row):
    """给读者看的 patience：**只印 cfg 里的值**，没记就照实写"未记录"，不拿常数顶替。"""
    v = patience_of(row)
    return '%d' % v if v is not None else '（cfg 未记录）'


def _e100(name, key):
    """从 E100 里取一行。**缺就报错，不回退。**"""
    if E100 is None:
        raise SystemExit(
            '★ 缺 box_results/%s —— 表 1 的 Ours(V8) 行、LSTM/TF 行与表 3 的三臂\n'
            '  都必须是天花板 100 的那批数字。把盒子上的 res_json.tgz 放进\n'
            '  box_results/ 再跑 merge_e100.py。\n'
            '  **绝不退回 e30 的数字**：表里写 e100、数却是 e30，正是本文件出过的错。'
            % E100_FILE)
    if key not in E100:
        same_arm = sorted(k for k in E100 if k.split('|')[1:3] == key.split('|')[1:3])
        raise SystemExit('★ E100 里没有键 %s\n  同臂已有的键：%s\n  （缺的那条还没跑完？）'
                         % (key, same_arm or '一条都没有'))
    return E100[key]


def v8_row(seed):
    """表 1 的 Ours(V8) 行 = 表 3 的 `real` 臂（**同一次运行**，见 arm3_row）。"""
    return _e100('V8', V8_E100 % ('real', seed))


def arm3_row(arm, seed):
    """表 3 三臂。`real` 与 v8_row 是**同一个对象** —— 不是"数值相等"，是同一次运行。"""
    return _e100(arm, V8_E100 % (arm, seed))


def net_keys(src, arch):
    """按 arch 段取**全部** seed 的 |e100 键（有序）。src 是「键名 → 记录」的映射。

    按**谓词**找而不是写死键名：键里的 mode 段是本脚本没见过的值，
    写死就会"找不到"却看不出来是找错了还是真没跑。

    ★ 2026-10-10（E 批）：LSTM / Transformer 从单 seed 42 补齐到三 seed ⇒
      「取全部」是新常态。原实现只认一颗、>1 个键就 SystemExit；那一版是对的
      （当时多出来的键只可能是取错了），现在语义变了，改谓词取全。
    """
    return sorted(k for k in (src or {})
                  if k.split('|')[2:3] == [arch] and k.endswith('|e100'))


def net_label(bare, keys):
    """基线行的标签：**由实际拿到的种子集推出来**，不写死。

    单 seed 必须在标签里写明是哪一颗 —— 和三 seed 的行并排而不标注，
    读者会当成同一口径。多 seed 才是本表的默认口径，用裸名。
    """
    seeds = sorted({k.split('|')[4] for k in keys})
    if len(seeds) == 1:
        return '%s (%s)' % (bare, seeds[0])
    return bare


def net_tex_label(bare, keys):
    """LaTeX 行的标签。† 的含义是「单 seed」（表 caption 里写明）—— 三 seed 之后
    它没有指代对象，必须跟着消失，否则读者会去找一个不存在的脚注。"""
    return bare + r'$^{\dagger}$' if len(keys) == 1 else bare


SEEDS_ABL = [42, 123, 456]
SEEDS_V8 = [1, 2, 3]

p('=' * 78)
p('  论文表格（全精度，直接取自 box_results/ 的原始字段）')
p('  生成脚本：make_tables.py   数据集：PeMS-BAY（N=325）+ PeMS04（N=307）')
p('  天花板：统一 100 轮（epochs_run = best_epoch + %d 即真早停）' % PATIENCE)
p('=' * 78)
p()
if HAVE_E100:
    n_e100 = sum(1 for k in E100 if k.endswith('|e100'))
    p('  数据源 %s：%d 键，其中 |e100 共 %d 个' % (E100_FILE, len(E100), n_e100))
else:
    p('  ★★ 缺 %s ★★' % E100_FILE)
    p('     表 1 的 Ours(V8) 行、LSTM/TF 行、表 3 的三臂都取不到 —— 脚本会在')
    p('     需要它们的地方**直接报错停下**，不会拿 e30 的数字顶上。')
def latex_tab_numbers():
    """从 `latex/main.aux` 读 `\\newlabel{tab:...}{{N}}`，按出现顺序返回 [(label, num)]。

    ★ 2026-09-30：这一段**不再写死** latex 的表号。上一版写死了
      「表 2 tab:main / 表 4 tab:ablation / 表 5 tab:input」，ch3 插入 tab:trainproto
      之后整体差了一位 —— 写死的东西每次插表都会**静默过期**（同
      [[edit-the-generator-not-its-output]]：生成器该推导，不该断言）。
      读不到就返回 None，让调用方明说"读不到"，不猜、也不回落成旧数字。
    """
    aux = os.path.join(ROOT, 'latex', 'main.aux')
    try:
        txt = open(aux, encoding='utf-8', errors='replace').read()
    except OSError:
        return None
    # aux 里是 `\newlabel{tab:variants}{{1}{12}}` —— 编号后面紧跟的是 `}{`，不是 `}}`
    got = re.findall(r'\\newlabel\{(tab:[a-z_0-9]+)\}\{\{(\d+)\}', txt)
    return got or None


p('  BOX = %s' % BOX)
p()
p('  ⚠️ 两套表号，别混：本文件的「表 N」是**本文件自己的编号**（PeMS-BAY：表 1 主对比 /')
p('     伴随 RMSE·MAPE / 表 2 算子侧配对消融 / 表 3 输入侧三臂）。')
_nums = latex_tab_numbers()
if _nums:
    p('     latex 当前编号（从 latex/main.aux **读出**，不是写死的）：')
    for _k, _n in _nums:
        p('       表 %-2s %s' % (_n, _k))
    _m = dict(_nums)
    p('     → 本文件的表 2 = latex 表 %s（tab:ablation）、本文件的表 3 = latex 表 %s'
      % (_m.get('tab:ablation', '?'), _m.get('tab:input', '?')))
else:
    p('     ⚠️ latex/main.aux 没读到（还没编译过？）—— 这次表号印不出来。')
    p('        编译后跑 `grep newlabel{tab: latex/main.aux` 复核；')
    p('        **不要**照抄旧版本里写过的数字，那几个已经过期过一次了。')
p('     引用表号时点名是哪一套。')
p('     ★ 2026-09-30 起本文件第 4 部分还有 **PeMS04 自己的表 4（MAE）/ 4b·4c（RMSE·MAPE）/')
p('       表 5（算子侧消融）** —— 那三张是**另一个数据集的独立编号**，与上面的表 4/表 5')
p('       同名不同物。latex 里它们将是 tab:pems04 / tab:pems04_rmse / tab:pems04_abl。')

# ================================================================ 表 1
p()
p('─' * 78)
p('表 1  主对比表（PeMS-BAY，MAE；低者优）')
p('─' * 78)

# ★ 基线取自 0928 那两份（互相逐位一致），**不是** paper_results.json：
#   后者是 HA 口径修正（BL_VERSION 1→2）之前产出的文件，PeMS-BAY HA=5.2502、
#   METR-LA HA=7.9596，与官方口径（2.88 / 4.16）不符 → 不得使用。
#   依据：paper_experiments.py:886-914 记着官方口径的四个独立锚点。
base_persist = NORM['PeMS-BAY|基线-持久性']
base_ha = NORM['PeMS-BAY|基线-历史均值']

# ★★ 2026-09-29 重构：`rows` 现在存**原始记录**（每 seed 一条），不存已经取好的 MAE。
#    原因是表 1 与新增的伴随表（RMSE/MAPE）必须取同一份行清单 —— 若各建一份，
#    两边的标签、取行方式、参数列会各自漂开，而漂开**不会报错**（见记忆
#    narrative-span-can-drift-from-implementation）。现在一次构造、按指标多次提取。
rows = []          # (label, recs | None, params, ep)，recs = 每 seed 一条原始记录
rows.append(('Persistence', [base_persist], None, '—'))
rows.append(('Hist. average (HA)', [base_ha], None, '—'))

# ★★ 2026-10-10（评审 P0-2 / P0-4）：参考配方复现的两行**不再进表 1**。
#    它们走的是参考实现自己的优化器设置，与同表其余行不是一套协议，并排放就是
#    "把不可比的数并排"。两行（含 PeMS04 的两行）现集中在**新表 tab:repro**，
#    由 `check_ch4_repro.py` 对质。PeMS-BAY 的 DCRNN 那一条更被**整体撤回**
#    （那次运行没收敛，见 §4.6），所以这里只留 GWNet 一条。
#    下面仍建一份 `REPRO_BAY`：正文要在**跨协议**的前提下与它比（读到 MV 里的
#    一份拷贝），但它的数不占表 1 的行。
REPRO_BAY = []
for label, model in [('Graph WaveNet (repro.)', 'GWNET')]:
    r = gnn_row(model, 42)
    REPRO_BAY.append((label, [r], r['params'], str(r['epochs_run'])))

for label, model in [('MambaST (graph OFF)', 'MAMBAST'), ('MambaST (graph ON)', 'MAMBASTG')]:
    per = [gnn_row(model, s) for s in SEEDS_ABL]
    rows.append((label, per, per[0]['params'], '/'.join(str(r['epochs_run']) for r in per)))

per = [v8_row(s) for s in SEEDS_V8]
V8_ROWS = per                      # ★ 起个专名：下面几段都用它，不许再靠 `per`
rows.append(('**Ours (V8)**', per, per[0]['params'],
             '/'.join(str(r['epochs_run']) for r in per)))

# ★ LSTM / TF 只跑 seed 42（paper_experiments.py:58 原话：「跑一次即可，不用跟着消融跑三遍」）。
#   单 seed 就得在表里写明是单 seed —— 和三 seed 的 V8 并排而不标注，读者会当成同一口径。
#   取不到（文件里没有 |e100 键）就印「缺」，并在下面的限制清单里**据实**说明。
# ★★ 2026-10-10（E 批）：两个基线补齐到三 seed ⇒ 取行改成**取全部 seed**，
#    标签由**实际拿到的种子集**推出（net_label），不写死。写死的标签会在数据
#    到了之后继续印旧口径；PeMS04 侧那一处更糟 —— 它把键名写死，会**静默漏掉**
#    新种子（同一类病，那一边连错都不报）。
ARCH_BARE = {'LSTM': 'LSTM', 'TF': 'Transformer'}
NET_KEYS = {}
NET_ROWS = {}
for arch, hint in [('LSTM', 'res_LSTM_s42.json'), ('TF', 'res_TF_s42.json')]:
    hits = net_keys(E100, arch)
    NET_KEYS[arch] = hits
    per = [E100[k] for k in hits]
    NET_ROWS[arch] = per
    if not per:
        print('  [取行] ★ %s 在 %s 里没有 |e100 键 —— 该行印「缺」（%s）'
              % (arch, E100_FILE, hint))
        rows.append((ARCH_BARE[arch], None, None, None))
    else:
        print('  [取行] %s ← %s' % (arch, ', '.join(hits)))
        rows.append((net_label(ARCH_BARE[arch], hits), per, per[0]['params'],
                     '/'.join(str(r['epochs_run']) for r in per)))


def dump(metric):
    """按 `metric` 打印 `rows` 的三档值（seed 维取算术平均）。表 1 与伴随表共用。"""
    for label, recs, params, ep in rows:
        if recs is None:
            p('%-24s %10s %10s %10s %10s %10s' % (label, '缺', '缺', '缺', '—', '—'))
            continue
        vals = [mean([metric(r, h) for r in recs]) for h in H]
        p('%-24s %10.6f %10.6f %10.6f %10s %10s'
          % (label, vals[0], vals[1], vals[2],
             '{:,}'.format(params) if params else '—', ep))


p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump(mae)
p('  基线（Persistence / HA）取自 paper_results_normfix_0928_0414.json（BL_VERSION=2 官方口径）')
p('  ★ HA 三档近乎相同是**正确的**：官方 HA 的预测与预测步长无关，'
  'DCRNN Table 1 的 HA 行本就三档印同一个数。')

# ================================================================ 表 1b / 1c
# ★★ 为什么有这张表：ch3 的 `Metrics` 小节明写「We report MAE, RMSE and MAPE at
#    the 15/30/60-minute horizons」，并为 MAPE 写了阈值筛选与覆盖率说明，
#    而 ch4 原先两张表**只有 MAE** —— 承诺与兑现不符，是审稿人会直接抓的硬伤。
#    数据本来就在结果文件里，补表不改任何承诺。**与表 1 共用同一份 `rows`。**
p()
p('─' * 78)
p('表 1b/1c  表 1 的伴随表（同 7 行、同取行方式、同 seed 口径）')
p('─' * 78)
p('  表 1b  RMSE（低者优）')
p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump(rmse)
p()
p('  表 1c  MAPE（%；低者优。掩零，且真值 > MAPE_MIN）')
p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump(mape)

# ★ ch3:477-479 承诺「报告保留比例，使 MAPE 的口径可审」。逐行印出来，
#   并且**检查它是不是恒定** —— 恒定就必须在正文里说"恒定"，不能印一列
#   看上去在变、实则各格相同的数字（那会让读者以为有信息）。
covs = {label: [mean([mape_cov(r, h) for r in recs]) for h in H]
        for label, recs, _, _ in rows if recs is not None}
uniq_cov = sorted({round(v, 12) for vs in covs.values() for v in vs})
p()
p('  MAPE 覆盖率（保留点占比；**比例，不是百分数**）：')
for label, vs in covs.items():
    p('    %-24s %.6f / %.6f / %.6f' % (label, vs[0], vs[1], vs[2]))
p('  → %d 个格子只有 %d 个互异值：%s'
  % (sum(len(v) for v in covs.values()), len(uniq_cov),
     ', '.join('%.6f' % u for u in uniq_cov)))
if len(uniq_cov) == 1:
    p('     ⇒ **恒定**。正文写一句「覆盖率在所有行、所有时域上均为 %.6f」即可，'
      '不要为它在表里占一列。' % uniq_cov[0])

# ★ ch3:479 承诺「The number of evaluation points is reported alongside each result」。
#   实测 `n_eval = int(pred.shape[0])`（paper_experiments.py:881）数的是**窗口数**，
#   不是点数 —— 真实点数 = n_eval × 节点数 × 12 步。正文措辞必须跟着改，
#   否则「evaluation points」这个说法本身就是错的。
p()
p('  评估规模 n_eval（**这是"评估窗口数"，不是"点数"**）：')
for label, recs, _, _ in rows:
    if recs is None:
        continue
    p('    %-24s n_eval %s   n_train %s'
      % (label, sorted({r['n_eval'] for r in recs}),
         sorted({r['n_train'] for r in recs if 'n_train' in r}) or '—'))
p('  → 基线行没有 n_train 字段（它们不训练）。若各行的 n_eval 相同，正文报一次即可。')

# ★ 三指标是否同向 —— 正文要写"RMSE/MAPE 与 MAE 给出的是不是同一个结论"，
#   这些差值**必须由本脚本算**，不许手抄（文件头纪律：一切数字取自原始字段）。
#   ★★ 分母口径 = **`(a−b)/a`**，a 是**前一个**臂名 —— 与表 2 caption、
#      表 3 caption、`论文_表格_全精度.md` 的配对差**同一个口径**（记忆里明写
#      "分母口径必须连值一起记"，两套都"对"但数值差 0.02–0.9pp）。
#      正 = a 更差（a 的误差更大），负 = a 更好。
p()
p('  三指标同向性（Δ = (a−b)/a×100%，a = 前一个臂名；正 = a 更差）：')

METRICS = [('MAE', mae), ('RMSE', rmse), ('MAPE', mape)]
# ★ MV 是**取数表**，不是"表 1 有哪些行"。参考配方复现行必须留在这里，否则下面
#   cmp3('**Ours (V8)**', 'Graph WaveNet (repro.)') 会 KeyError —— 而正文确实要
#   做那一次比较（并注明它跨协议）。
MV = {label: {name: [mean([f(r, h) for r in recs]) for h in H] for name, f in METRICS}
      for label, recs, _, _ in rows + REPRO_BAY if recs is not None}


def cmp3(a, b):
    out = []
    for name, _ in METRICS:
        out.append('%s %s' % (name, '  '.join(
            '@%d %+.4f%%' % (h, (MV[a][name][i] - MV[b][name][i]) / MV[a][name][i] * 100.0)
            for i, h in enumerate(H))))
    return out


for a, b, why in [('**Ours (V8)**', 'Graph WaveNet (repro.)',
                   '★ **跨协议**（V8 走本文协议、GWNet 走参考配方，现在分属 tab:main 与 '
                   'tab:repro 两张表）—— 正文引用这个差时必须写明它跨协议'),
                  ('MambaST (graph OFF)', 'MambaST (graph ON)', '算子侧消融（图关 vs 图开）'),
                  ('**Ours (V8)**', 'Persistence', 'V8 vs 持久性（MAPE@15 可能反向）')]:
    p('    %s —— %s' % (a, why))
    for line in cmp3(a, b):
        p('       %s' % line)

p()
p('  ⚠️ 读法：MAPE 的 seed 噪声远大于 MAE（V8 三 seed 的 sd/均值 @15 为 '
  '%.2f%% 对 %.2f%%），所以 MAPE 上更小的差值不可判。' %
  (sd([mape(r, 15) for r in V8_ROWS]) / mean([mape(r, 15) for r in V8_ROWS]) * 100.0,
   sd([mae(r, 15) for r in V8_ROWS]) / mean([mae(r, 15) for r in V8_ROWS]) * 100.0))
p('     逐 seed 的 sd（V8）：')
for name, f in METRICS:
    p('       %-5s %s' % (name, '  '.join(
        '@%d %.6f' % (h, sd([f(r, h) for r in V8_ROWS])) for h in H)))

p()
p('  三 seed 的 V8 逐 seed 值（表 1 里那行是均值）：')
for s in SEEDS_V8:
    r = v8_row(s)
    p('    s%d  %.6f / %.6f / %.6f   best_ep %d  epochs_run %d  val_mae %.6f'
      % (s, mae(r, 15), mae(r, 30), mae(r, 60), r['best_epoch'], r['epochs_run'], r['val_mae']))

# ★★ 主动披露：e30 那批的 V8 数字**作废**，这里把差值印出来。
#    为什么作废：epochs_run = best_epoch + PATIENCE + 1，e30 那批 best_ep = 27/28/29
#    ⇒ 本该跑到 36~38 轮 ⇒ 是被 30 这个天花板截断的，不是收敛。
p()
p('  ⚠️ 与旧 e30 那批（paper_results_normfix_0928_0414.json）的差值 —— 必须披露：')
for s in SEEDS_V8:
    old = NORM['PeMS-BAY|real|V8|official|s%d|tod' % s]
    new = v8_row(s)
    p('    s%d  e30 %.6f/%.6f/%.6f（ep_run %d，best %d）→ e100 %.6f/%.6f/%.6f（ep_run %d，best %d）'
      % (s, mae(old, 15), mae(old, 30), mae(old, 60), old['epochs_run'], old['best_epoch'],
         mae(new, 15), mae(new, 30), mae(new, 60), new['epochs_run'], new['best_epoch']))
    p('        Δ = %s'
      % '  '.join('@%d %+.4f%%' % (h, (mae(new, h) - mae(old, h)) / mae(old, h) * 100.0)
                  for h in H))

# ================================================================ 表 2
p()
p('─' * 78)
p('表 2  算子侧配对消融：图开 vs 图关（唯一变量 = 支撑集）')
p('─' * 78)

off = [gnn_row('MAMBAST', s) for s in SEEDS_ABL]
on = [gnn_row('MAMBASTG', s) for s in SEEDS_ABL]

p('  臂均值：')
for lab, per in [('graph OFF', off), ('graph ON', on)]:
    p('    %-10s %.6f / %.6f / %.6f   params=%s'
      % (lab, mean([mae(r, 15) for r in per]), mean([mae(r, 30) for r in per]),
         mean([mae(r, 60) for r in per]), '{:,}'.format(per[0]['params'])))

p()
p('  逐 seed 配对 Δ = (MAE_OFF − MAE_ON) / MAE_OFF × 100%（正 = 开图更好）')
p('  %-6s %10s %10s %10s' % ('seed', '@15', '@30', '@60'))
deltas = {h: [] for h in H}
for i, s in enumerate(SEEDS_ABL):
    row = []
    for h in H:
        d = (mae(off[i], h) - mae(on[i], h)) / mae(off[i], h) * 100.0
        deltas[h].append(d)
        row.append(d)
    p('  %-6d %+9.4f%% %+9.4f%% %+9.4f%%' % (s, row[0], row[1], row[2]))

p()
p('  %-6s %14s' % ('h', 'mean±sd (%)'))
for h in H:
    m, s = mean(deltas[h]), sd(deltas[h])
    p('  %-6s %+8.4f±%.4f' % (h, m, s))

p()
p('  ★ 这里**不再印** Cohen\'s d / 配对 t / 显著性结论（2026-10-10 撤）。')
p('      三个 seed 共用同一个固定测试集，不是三次独立重复 —— 这个样本量下')
p('      d 与 t 撑不起推断，印出来只会被当成推断读。论文、中文稿、本表一起撤。')
p('      留在上面的是逐 seed Δ 与 mean±sd：描述，不是推断。')

p()
p('  支撑证据（独立于测试集，取自选 checkpoint 用的 val_mae）：')
for lab, per in [('off', off), ('on', on)]:
    p('    %-4s val_mae %s   best_ep %s   epochs_run %s'
      % (lab, ' / '.join('%.6f' % r['val_mae'] for r in per),
         ' / '.join(str(r['best_epoch']) for r in per),
         ' / '.join(str(r['epochs_run']) for r in per)))

# ================================================================ 表 3
p()
p('─' * 78)
p('表 3  输入侧三臂（全 7 通道容量对齐；e100，三 seed）')
p('─' * 78)

# ★★ 关键：`real` 臂与表 1 的 Ours(V8) 是**同一次运行**（同一个 cid、同一个 .pt），
#    不是"数值恰好相等"的两个东西。断言用的是 `is`（对象同一性），不是 ==。
#    这条以前没写下来，差点让表 1 与表 3 的同一个 run 用了**不同轮数**的两份数字。
for s in SEEDS_V8:
    assert arm3_row('real', s) is v8_row(s), \
        's%d：表 3 的 real 臂与表 1 的 Ours(V8) 不是同一次运行 —— 轮数会被拆成两套' % s
p('  ✅ real 臂 ≡ 表 1 的 Ours(V8)（同一 cid、同一 .pt；断言用 is 不用 ==）')
p()
p('  %-10s %10s %10s %10s' % ('arm', '@15', '@30', '@60'))
arm_rows = {}
for arm in ['real', 'random', 'self']:
    arm_rows[arm] = [arm3_row(arm, s) for s in SEEDS_V8]
    p('  %-10s %10.6f %10.6f %10.6f'
      % (arm, mean([mae(r, 15) for r in arm_rows[arm]]),
         mean([mae(r, 30) for r in arm_rows[arm]]),
         mean([mae(r, 60) for r in arm_rows[arm]])))

p()
p('  配对差（正 = 前者的 MAE 更大 = 更差）：')
p('  %-18s %12s %12s %12s' % ('pair', '@15', '@30', '@60'))
for a, b in [('real', 'self'), ('real', 'random'), ('self', 'random')]:
    cells = []
    for h in H:
        ds = [(mae(arm_rows[a][i], h) - mae(arm_rows[b][i], h)) / mae(arm_rows[a][i], h) * 100.0
              for i in range(3)]
        m, s = mean(ds), sd(ds)
        # 符号：这里算的是 (a-b)/a，所以负 = b 更好
        # ★ 不再附加 (t=…)：n=3 下的 t 不是可报的统计量（2026-10-10）。
        cells.append('%+.4f%%±%.4f' % (m, s))
    p('  %-18s %12s %12s %12s' % ('%s − %s' % (a, b), cells[0], cells[1], cells[2]))

p()
p('  val_mae（三 seed）：')
for arm in ['real', 'random', 'self']:
    p('    %-7s %s' % (arm, ' / '.join('%.6f' % r['val_mae'] for r in arm_rows[arm])))

# ================================================================ PeMS04
# ★★ 为什么有这一段：`SCI任务书.md` 的硬指标是「≥2 个数据集」。METR-LA 已于
#   2026-09-27 因**管线级不复现**被作者裁定弃用（HA 复现准，但所有训练模型 ≈2×
#   已发表值）→ 结果只剩 PeMS-BAY 一个，PeMS04 就是补这个位。
#
# ★★ 先验预期与判据**在看数字之前**写死在 `先验预期_PeMS04.md`；本文档只做
#   **追加注**，不回改那份预测。判据裁决也在下面逐条印出来。
#
# ★ 数字来源是两个**与上面几张表不同的文件**，绝不许交叉取用：
#     box_results/gnn_results.pems04_merged.json  21 键 = 13 PeMS-BAY + 8 PeMS04
#     box_results/paper_results_pems04.json        7 键 = 2 基线 + LSTM/TF/V8×3
#   两份都是盒子上 `work/` 的原件，sha256 已核。
P4_GNN_FILE = 'gnn_results.pems04_merged.json'
P4_PAPER_FILE = 'paper_results_pems04.json'
P4_ABL_SEEDS = [42, 123, 456]
P4_V8_SEEDS = [1, 2, 3]

G2 = load(P4_GNN_FILE)
P4 = load(P4_PAPER_FILE)

p()
p('=' * 78)
p('  第 4 部分  PeMS04（第二个数据集，N=307，流量 veh/5min）')
p('=' * 78)
p('  来源：%s（%d 键，其中 PeMS04 %d 条）'
  % (P4_GNN_FILE, len(G2), sum(1 for k in G2 if k.startswith('PeMS04|'))))
p('        %s（%d 键）' % (P4_PAPER_FILE, len(P4)))
p('  先验预期写死在 先验预期_PeMS04.md（2026-09-28，看数字之前）')


def _p4(key):
    """PeMS04 的 GNN 行。**缺就报错并列出同数据集已有的键，绝不回退到别的数据集。**"""
    if key not in G2:
        raise SystemExit('★ %s 里缺 %s\n  该数据集已有的键：\n    %s'
                         % (P4_GNN_FILE, key,
                            '\n    '.join(sorted(k for k in G2 if k.startswith('PeMS04|')))))
    return G2[key]


def _p4p(key):
    if key not in P4:
        raise SystemExit('★ %s 里缺 %s\n  已有键：\n    %s'
                         % (P4_PAPER_FILE, key, '\n    '.join(sorted(P4))))
    return P4[key]


# ★ 独立锚点：合并文件里那 13 条 PeMS-BAY 必须与 GNN（PeMS-BAY 专用文件）**逐位相同**。
#   两个文件是两次独立下载/合并的产物，拿它们互证才知道合并没动过 PeMS-BAY。
P4_BAY_IN_MERGED = {k: v for k, v in G2.items() if not k.startswith('PeMS04|')}
_bad = sorted(k for k in set(P4_BAY_IN_MERGED) | set(GNN)
              if P4_BAY_IN_MERGED.get(k) != GNN.get(k))
assert len(P4_BAY_IN_MERGED) == 13 and not _bad, \
    '合并文件里的 PeMS-BAY 子集与 PeMS-BAY 专用文件不一致（%s）—— 合并过程动过数据' % (_bad or '键数不符')
p('  ✅ 合并文件里的 PeMS-BAY 13 条与 %s **逐位相同**（合并没动 PeMS-BAY）' % GNN_NAME)
p('  ✅ PeMS04 共 %d 条：MAMBAST×3 + MAMBASTG×3 + GWNET + DCRNN'
  % sum(1 for k in G2 if k.startswith('PeMS04|')))

# ── 取行 ───────────────────────────────────────────────────────────
P4_ROWS = []
P4_ROWS.append(('Persistence', [_p4p('PeMS04|基线-持久性')], None, '—'))
P4_ROWS.append(('Historical average', [_p4p('PeMS04|基线-历史均值')], None, '—'))
# ★ 同 PeMS-BAY：参考配方复现的两行 2026-10-10 移入 tab:repro（评审 P0-2 / P0-4），
#   不再占表 4 的行。★ 注意两处**不对称**且必须保留：PeMS04 的 DCRNN 这次**收敛了**
#   （clip 5.0，best_ep 40，val_mae 平滑），所以它仍被报告；只有 PeMS-BAY 的那一条
#   被撤回。写正文时不要把"撤 DCRNN"说成对所有数据集都成立。
P4_REPRO = []
for lab, mdl in [('DCRNN (repro.)', 'DCRNN'), ('Graph WaveNet (repro.)', 'GWNET')]:
    r = _p4('PeMS04|%s|s42|c2|e100' % mdl)
    P4_REPRO.append((lab, [r], r['params'], str(r['epochs_run'])))
P4_OFF = [_p4('PeMS04|MAMBAST|s%d|c2|e100' % s) for s in P4_ABL_SEEDS]
P4_ON = [_p4('PeMS04|MAMBASTG|s%d|c2|e100' % s) for s in P4_ABL_SEEDS]
for lab, per in [('MambaST, graph off', P4_OFF), ('MambaST, graph on', P4_ON)]:
    P4_ROWS.append((lab, per, per[0]['params'],
                    '/'.join(str(r['epochs_run']) for r in per)))
P4_V8 = [_p4p('PeMS04|real|V8|official|s%d|tod|e100' % i) for i in P4_V8_SEEDS]
P4_ROWS.append(('**Ours (V8)**', P4_V8, P4_V8[0]['params'],
                '/'.join(str(r['epochs_run']) for r in P4_V8)))
# ★★ 2026-10-10（E 批）：这两行原来把 `s42` **写死在键名字面量里**。E 批补了 s2/s3
#    之后，写死的键会把新种子**静默漏掉** —— 表里继续印单 seed 的数、标签还挂着
#    "(s42)"，而同一轮 PeMS-BAY 侧会因多键报错。一半叫一半不叫，最坏的一种。
#    改谓词取全，标签与 PeMS-BAY 侧同一个函数（口径不许两边各写一套）。
P4_NET = {}          # arch -> {'bare','keys','label','tex'}
for arch, bare in [('LSTM', 'LSTM'), ('TF', 'Transformer')]:
    hits = net_keys(P4, arch)
    if not hits:
        raise SystemExit('★ %s 里找不到 %s 的 |e100 键 —— 该行无从取数。已有键：\n    %s'
                         % (P4_PAPER_FILE, arch, '\n    '.join(sorted(P4))))
    P4_NET[arch] = {'bare': bare, 'keys': hits,
                    'label': net_label(bare, hits),
                    'tex': net_tex_label(bare, hits)}
    per = [P4[k] for k in hits]
    P4_ROWS.append((P4_NET[arch]['label'], per, per[0]['params'],
                    '/'.join(str(r['epochs_run']) for r in per)))

# 参数量独立锚点：两臂必须相同（单变量前提），且 V8/LSTM/TF 的 252,940 / 794,124 / 795,276
# 必须与 PeMS-BAY 侧印出来的一致（同一份模型代码，不该因数据集而变）。
assert {r['params'] for r in P4_OFF} == {r['params'] for r in P4_ON} == {384140}, \
    'PeMS04 算子侧两臂参数量不等 —— 单变量前提破了'
assert {r['params'] for r in P4_V8} == {252940}, 'PeMS04 的 V8 参数量不是 252,940'
assert {r[2] for r in P4_ROWS if r[0].startswith('LSTM')} == {794124}, \
    'PeMS04 的 LSTM 参数量不是 794,124'
p('  ✅ 断言：两臂同参数 384,140、V8 252,940、LSTM 794,124（与 PeMS-BAY 侧同值）')
p('  ✅ n_eval 全为 %s、n_train 全为 %s'
  % (sorted({r['n_eval'] for _, rs, _, _ in P4_ROWS for r in rs}),
     sorted({r['n_train'] for _, rs, _, _ in P4_ROWS for r in rs if 'n_train' in r})))

# ★★ 2026-10-10（E 批）：**行内「只差一个 seed」断言**。
#    表里每一行都是「同一配置的几个 seed 取算术平均」；一旦某个 seed 的
#    params / n_train / n_eval 与同行其他 seed 不等，那就是把两种协议平均成
#    一个数 —— 而**均值不会喊疼**，它只会安静地错下去。逐字段相等是这一行
#    能成立的唯一前提，所以在这里硬卡。
#    ★ 盒子上 `queueE/check_E.sh` 已经先卡过一道；这里是**下载之后再卡一道**。
#      两道各自独立：盒子上的闸门管不了「传回来的文件被改过」，本地的闸门
#      管不了「盒子上的 job 用错配置」。谁坏了都不靠对方发现。
for _lab, _recs, _pa, _ep in list(rows) + list(P4_ROWS):
    if _recs is None or len(_recs) < 2:
        continue
    for _f in ('params', 'n_train', 'n_eval'):
        _vals = sorted({repr(r.get(_f)) for r in _recs})
        if len(_vals) > 1:
            raise SystemExit(
                '★ 行「%s」内 %s 不一致：%s\n'
                '  这些 seed 不是「只差一个 seed」，**不能平均**。'
                '先把不一致的那一格查清（是不是换了配置跑、或键取错了），'
                '再回来出表。' % (_lab, _f, _vals))


def dump4(metric, tab=None):
    for label, recs, params, ep in P4_ROWS:
        if recs is None:
            p('%-24s %10s %10s %10s %10s %10s' % (label, '缺', '缺', '缺', '—', '—'))
            continue
        v = [mean([metric(r, h) for r in recs]) for h in H]
        p('%-24s %10.6f %10.6f %10.6f %10s %10s'
          % (label, v[0], v[1], v[2],
             '{:,}'.format(params) if params else '—', ep))


p()
p('─' * 78)
p('表 4  主对比表（PeMS04，MAE；**量纲是 veh/5min，不可与 PeMS-BAY 的 mph 比大小**）')
p('─' * 78)
p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump4(mae)
p('  基线（Persistence / HA）取自 %s（BL_VERSION=2 官方口径）' % P4_PAPER_FILE)

p()
p('  表 4b  RMSE（同 7 行、同 seed 口径）')
p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump4(rmse)
p()
p('  表 4c  MAPE（%；低者优。掩零，且真值 > MAPE_MIN）')
p('%-24s %10s %10s %10s %10s %10s' % ('Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
dump4(mape)

# ── 跨行对比：正文里要引用的每一个百分数都必须由本脚本算，不许手打 ──────
P4_BY = {lab.strip('*'): recs for lab, recs, _, _ in P4_ROWS + P4_REPRO}


def p4v(lab, h):
    return mean([mae(r, h) for r in P4_BY[lab]])


p()
p('  三 seed 行的逐 seed 离散度（正文引用单 seed 基线时**必须**给这个数）：')
P4_SD = {}
for lab in ['Ours (V8)', 'MambaST, graph off', 'MambaST, graph on']:
    rs = P4_BY[lab]
    P4_SD[lab] = [sd([mae(r, h) for r in rs]) for h in H]
    p('    %-20s sd %s' % (lab, ' / '.join('%.6f' % v for v in P4_SD[lab])))

p()
p('  跨行对比 Δ = (a − b)/a × 100%（正 = a 更差；a 写在前）：')
for a, b in [('Ours (V8)', 'Graph WaveNet (repro.)'),
             ('Ours (V8)', 'DCRNN (repro.)'),
             ('Ours (V8)', 'MambaST, graph on'),
             (P4_NET['LSTM']['label'], 'Ours (V8)'),
             (P4_NET['TF']['label'], 'Ours (V8)'),
             ('MambaST, graph on', 'Graph WaveNet (repro.)'),
             ('MambaST, graph on', 'DCRNN (repro.)')]:
    p('    %-20s − %-24s %s' % (a, b, '  '.join(
        '@%d %+.4f%%' % (h, (p4v(a, h) - p4v(b, h)) / p4v(a, h) * 100.0) for h in H)))
# ★ 单 seed 基线 vs 三 seed 行：差值是「几个 sd」必须算出来（PeMS-BAY 那段就是用这个
#   判 LSTM 可判、TF 判不了的）。这里对 PeMS04 把同一件事逐行算一遍。
p()
p('  单 seed 基线相对 V8 的差值 = 若干倍 V8 的 seed-sd（判「可判 / 不可判」）：')
for lab in [P4_NET['LSTM']['label'], P4_NET['TF']['label'],
            'Graph WaveNet (repro.)', 'DCRNN (repro.)']:
    cells = []
    for i, h in enumerate(H):
        d = p4v(lab, h) - p4v('Ours (V8)', h)
        s = P4_SD['Ours (V8)'][i]
        cells.append('@%d %+.4f (=%.1f sd)' % (h, d, d / s))
    p('    %-24s %s' % (lab, '  '.join(cells)))

# ★★ LaTeX 表体直接由本脚本印出 —— 手工誊录是本项目出过多次的错的唯一来源
#    （记忆 statistic-definition-must-travel-with-value / md-table-append-run-checker）。
LATEX_LBL = [
    ('Persistence', 'Persistence'),
    ('Historical average', 'Historical average'),
    ('MambaST, graph off', 'MambaST, graph off'),
    ('MambaST, graph on', 'MambaST, graph on'),
    ('Ours (V8)', 'Ours (V8)'),
    (P4_NET['LSTM']['label'], P4_NET['LSTM']['tex']),
    (P4_NET['TF']['label'], P4_NET['TF']['tex']),
]


def latex_rows(metrics, with_pe=True):
    out = []
    for key, tex in LATEX_LBL:
        recs, params, ep = P4_BY[key], None, None
        for lab, rs, pa, e in P4_ROWS:
            if lab.strip('*') == key:
                params, ep = pa, e
        cells = [_s(mean([m(r, h) for r in recs])) for m in metrics for h in H]
        tail = (['---', '---'] if not params else ['{:,}'.format(params), ep]) if with_pe else []
        out.append((tex, cells + tail))
    return out


def pad(s, n):
    """按**显示宽度**补空格（中文字符按 2 列算）—— 只为可读，LaTeX 不在乎。"""
    w = sum(2 if ord(c) > 0x2000 else 1 for c in s)
    return s + ' ' * max(1, n - w)


for tag, mets in [('tab:pems04（MAE）', [mae]),
                  ('tab:pems04_rmse 左半（RMSE）', [rmse]),
                  ('tab:pems04_rmse 右半（MAPE）', [mape])]:
    p()
    p('  ── LaTeX 表体 · %s（**直接粘贴，不要手打**）' % tag)
    ncol = 3 * len(mets) + 2
    for tex, cells in latex_rows(mets):
        p('  ' + ' & '.join([pad(tex, 34 - 20)] + cells) + r' \\')

# ================================================================ 表 1d
# ★ 参考配方复现表（latex `tab:repro`，2026-10-10 新增，评审 P0-2 / P0-4）。
#   两个数据集各占一个块，被引的已发表值单独一行（Params/Epochs 留 `---`：
#   公开源不报参数量）。**两块绝不合并、也绝不与上面两张主表合并** —— 协议不同。
#   PeMS-BAY 的 DCRNN 复现行**不在**这里：那次运行没收敛，已整体撤回。
p()
p('─' * 78)
p('表 1d  参考配方复现表（latex tab:repro；两个数据集各一块，**协议与上面两张主表不同**）')
p('─' * 78)
p('  Panel A  MAE')
p('%-12s %-24s %10s %10s %10s %10s %8s'
  % ('dataset', 'Model', '@15', '@30', '@60', 'params', 'epochs_run'))
p('-' * 78)
# ★ 显式写死 [(数据集, 行名, 来源列表)]，**不要**先拿行名做字典再按键取：
#   `Graph WaveNet (repro.)` 在两个数据集里都叫这个名，按键取会**静默取到后一个**
#   （PeMS04 的那条）填进 PeMS-BAY 那一行 —— 数值上看起来完全正常。
#   本项目栽过同族的坑（记忆 model-arm-filter-needs-explicit-name）。
REPRO_TABLE = [
    ('PeMS-BAY', 'Graph WaveNet (repro.)', REPRO_BAY),
    ('PeMS04', 'DCRNN (repro.)', P4_REPRO),
    ('PeMS04', 'Graph WaveNet (repro.)', P4_REPRO),
]
REPRO_BY = {}
for ds, lab, pool in REPRO_TABLE:
    hit = [(recs, params, ep) for l, recs, params, ep in pool if l == lab]
    assert len(hit) == 1, '%s 的 %s 在来源列表里有 %d 条（应为 1）' % (ds, lab, len(hit))
    recs, params, ep = hit[0]
    REPRO_BY[(ds, lab)] = recs
    p('%-12s %-24s %10.6f %10.6f %10.6f %10s %8s'
      % (ds, lab, mean([mae(r, 15) for r in recs]), mean([mae(r, 30) for r in recs]),
         mean([mae(r, 60) for r in recs]), '{:,}'.format(params), ep))
# ★ 被引值的**源**：ST-Mamba（Information Fusion 2026）Table 5 的 GWNet 行，我们
#   转录在 知识库/文献清单-补充.md。★ 我们另一份底稿（论文_对比表_底稿.md:31）把
#   同一个模型记成 1.30/1.63/1.95（GWNet 原文 Table 2, p.5，@60 差 2%）。
#   **本条没有机器锚点**：本脚本只保证表↔正文自洽，不裁决该用哪个源；
#   若改用一手值，三处必须同时改（见 check_ch4_repro.py 的文件头）。
PUB_MAE = {15: 1.30, 30: 1.63, 60: 1.99}
p('%-12s %-24s %10.2f %10.2f %10.2f %10s %8s'
  % ('PeMS-BAY', 'Graph WaveNet (published)', PUB_MAE[15], PUB_MAE[30], PUB_MAE[60], '---', '---'))
p('  ⚠️ 上行的源是**转述**（ST-Mamba Table 5），公开源只印 MAE ⇒ panel B 不给它留格。')

p()
p('  Panel B  RMSE / MAPE（同 Panel A 里**我们自己跑的**那三行）')
p('%-12s %-24s %10s %10s %10s %10s %10s %10s'
  % ('dataset', 'Model', 'RMSE@15', 'RMSE@30', 'RMSE@60', 'MAPE@15', 'MAPE@30', 'MAPE@60'))
p('-' * 78)
for ds, lab, _pool in REPRO_TABLE:
    recs = REPRO_BY[(ds, lab)]
    p('%-12s %-24s %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f'
      % (ds, lab,
         mean([rmse(r, 15) for r in recs]), mean([rmse(r, 30) for r in recs]),
         mean([rmse(r, 60) for r in recs]), mean([mape(r, 15) for r in recs]),
         mean([mape(r, 30) for r in recs]), mean([mape(r, 60) for r in recs])))

p()
p('  复现 vs 已发表（口径 (ours − published)/published × 100%；正 = 我们更差）：')
_o = REPRO_BY[('PeMS-BAY', 'Graph WaveNet (repro.)')]
p('    %-24s %s' % ('Graph WaveNet (PeMS-BAY)', '  '.join(
    '@%d %+.1f%%' % (h, (mean([mae(r, h) for r in _o]) - PUB_MAE[h]) / PUB_MAE[h] * 100.0)
    for h in H)))
p('    ★ 这句话要与 latex §4.6 里那句**逐字一致**（`check_ch4_repro.py` 锚点 J 会查）。')
p('    ★ PeMS04 的两个复现行**没有**已发表对照：本文用统一 7:1:2 划分，'
  '而 PeMS04 文献惯例是 6:2:2 ⇒ 跨划分比绝对值的差没有意义。')
# ── 表 1d 的 LaTeX 表体（直接粘贴，不要手打）────────────────────────
# ★ 本文件内部的行名与 latex 里的行名**不是一套**，这块表体必须按 **latex 的写法**
#   出：md 里叫 `(repro.)`（描述"这条是复现的"），latex 里叫 `(ours)$^{\dagger}$`
#   （描述"这条是本管线跑的、单 seed"）—— 指同一条记录，但字面不同。若照 md 的内部
#   名粘贴，`check_ch4_repro.py` 的锚点 G 因为按**归一化名**（去括号）比集合，
#   会**静默放过**，而表上的行名就与正文/题注说的 `ours` 对不上了。
#   ⇒ 这里出一份**与 latex 逐行一致**的写法（含被引行的位置：在 PeMS-BAY 块内、
#   紧跟复现行），并由 `check_ch4_repro.py` 锚点 K 逐行对质，防止两边再漂。
TEX_LAB = {'Graph WaveNet (repro.)': 'Graph WaveNet (ours)',
           'DCRNN (repro.)': 'DCRNN (ours)'}
p('  ── LaTeX 表体 · tab:repro Panel A（MAE）')
# ★ 数据集块名要**连 `\emph{}` 的作用范围**一起照 latex 写：latex 里是
#   `\emph{PeMS-BAY} (speed, mph)`（只斜体数据集名，括号里的量纲是正体）。
#   写成 `\emph{PeMS-BAY (speed, mph)}` 排版上看不出差别 —— 但锚点 K 会报不一致，
#   而它报的正是这个：**"看不出来的差别"迟早会变成看得见的差别**。
_qual = {'PeMS-BAY': 'PeMS-BAY (speed, mph)',
         'PeMS04': 'PeMS04 (flow, vehicles per five minutes)'}
_cur = None
for ds, lab, _pool in REPRO_TABLE:
    if ds != _cur:
        _cur = ds
        _n, _q = _qual[ds].split(' ', 1)
        p('    \\multicolumn{6}{@{}l}{\\emph{%s} %s} \\\\' % (_n, _q))
    recs, params, ep = [(r, pa, e) for l, r, pa, e in _pool if l == lab][0]
    p('    %-34s & %s & %s & %s & %s & %s \\\\'
      % (TEX_LAB[lab] + '$^{\\dagger}$', _s(mean([mae(r, 15) for r in recs])),
         _s(mean([mae(r, 30) for r in recs])), _s(mean([mae(r, 60) for r in recs])),
         '{:,}'.format(params), ep))
    if ds == 'PeMS-BAY':      # ← 与 latex 里的位置一致：被引行在 PeMS-BAY 块内
        p('    %-34s & %.2f & %.2f & %.2f & --- & --- \\\\'
          % ('Graph WaveNet (published)', PUB_MAE[15], PUB_MAE[30], PUB_MAE[60]))
p('  ── LaTeX 表体 · tab:repro Panel B（RMSE / MAPE）')
_cur = None
for ds, lab, _pool in REPRO_TABLE:
    if ds != _cur:
        _cur = ds
        p('    \\multicolumn{7}{@{}l}{\\emph{%s}} \\\\' % ds)
    recs = REPRO_BY[(ds, lab)]
    p('    %-34s & %s & %s & %s & %s & %s & %s \\\\'
      % (TEX_LAB[lab] + '$^{\\dagger}$',
         _s(mean([rmse(r, 15) for r in recs])), _s(mean([rmse(r, 30) for r in recs])),
         _s(mean([rmse(r, 60) for r in recs])), _s(mean([mape(r, 15) for r in recs])),
         _s(mean([mape(r, 30) for r in recs])), _s(mean([mape(r, 60) for r in recs]))))

# ★ PeMS-BAY 侧覆盖率是**恒定**的（0.999936，正文只写一句）。PeMS04 不是 ——
#   基线行 0.9862568、模型行 0.9867567，差在掩零门槛上。**不检查就会在正文里
#   沿用"恒定"那句话**，而那句在第二个数据集上是假的。
cov4 = sorted({round(mean([mape_cov(r, h) for r in recs]), 12)
               for _, recs, _, _ in P4_ROWS if recs is not None for h in H})
cov4_all = sorted({round(mape_cov(r, h), 12)
                   for _, recs, _, _ in P4_ROWS if recs is not None for r in recs for h in H})
p()
p('  MAPE 覆盖率（逐行取三档均值）：%s' % ', '.join('%.6f' % c for c in cov4))
p('  → %d 个互异值。**PeMS04 上覆盖率不是恒定的**（逐 seed 也一样：%d 个互异值），'
  % (len(cov4), len(cov4_all)))
p('    正文不得沿用 PeMS-BAY 那句「所有行所有时域均为 0.999936」。')

# ── 算子侧配对消融（本段与 PeMS-BAY 那段是同一个量，只是换了数据集）────
p()
p('─' * 78)
p('表 5  算子侧配对消融（PeMS04）：图开 vs 图关，唯一变量 = 支撑集')
p('─' * 78)
p('  臂均值：')
for lab, per in [('graph OFF', P4_OFF), ('graph ON', P4_ON)]:
    p('    %-10s %.6f / %.6f / %.6f   params=%s'
      % (lab, mean([mae(r, 15) for r in per]), mean([mae(r, 30) for r in per]),
         mean([mae(r, 60) for r in per]), '{:,}'.format(per[0]['params'])))

p()
p('  逐 seed 配对 Δ = (MAE_OFF − MAE_ON) / MAE_OFF × 100%（正 = 开图更好）')
p('  %-6s %10s %10s %10s' % ('seed', '@15', '@30', '@60'))
p4deltas = {h: [] for h in H}
for i, s in enumerate(P4_ABL_SEEDS):
    row = []
    for h in H:
        d = (mae(P4_OFF[i], h) - mae(P4_ON[i], h)) / mae(P4_OFF[i], h) * 100.0
        p4deltas[h].append(d)
        row.append(d)
    p('  %-6d %+9.4f%% %+9.4f%% %+9.4f%%' % (s, row[0], row[1], row[2]))

p()
p('  %-6s %16s' % ('h', 'mean±sd (%)'))
for h in H:
    m, s = mean(p4deltas[h]), sd(p4deltas[h])
    p('  %-6s %+8.4f±%.4f' % (h, m, s))
p('  口径同表 2；**不印** Cohen\'s d / true t（2026-10-10 撤，n=3 下不成立）。')
p('  PeMS-BAY 对照：@15 %+.4f%%、@30 %+.4f%%、@60 %+.4f%%'
  % tuple(mean([(mae(off[i], h) - mae(on[i], h)) / mae(off[i], h) * 100.0
                for i in range(3)]) for h in H))

p()
p('  支撑证据（独立于测试集，取自选 checkpoint 用的 val_mae）：')
for lab, per in [('off', P4_OFF), ('on', P4_ON)]:
    p('    %-4s val_mae %s   best_ep %s   epochs_run %s'
      % (lab, ' / '.join('%.6f' % r['val_mae'] for r in per),
         ' / '.join(str(r['best_epoch']) for r in per),
         ' / '.join(str(r['epochs_run']) for r in per)))

# ── 先验判据的裁决（写死的那份判据，逐条对）────────────────────────
p()
p('─' * 78)
p('  先验预期（先验预期_PeMS04.md §6）的裁决 —— 逐条对，**不回改预测**')
p('─' * 78)
_h1 = mean(p4deltas[60])
BAY60 = mean([(mae(off[i], 60) - mae(on[i], 60)) / mae(off[i], 60) * 100.0
              for i in range(3)])
p('  H1  Δ@60 符号与 PeMS-BAY 一致（>0）          → %s  实测 %+.4f%%（三 seed 全正 %s）'
  % ('✅ 成立' if _h1 > 0 else '❌ 不成立', _h1,
     ' '.join('%+.2f' % x for x in p4deltas[60])))
p('  H2  0 < Δ@60(PeMS04) < Δ@60(PeMS-BAY)=%.4f%%，且量级估 1–4%%' % BAY60)
p('      → 前半（排序）%s：实测 %+.4f%% < %.4f%%'
  % ('✅ 成立' if 0 < _h1 < BAY60 else '❌ 不成立', _h1, BAY60))
p('      → 后半（量级）❌ **不成立**：预计缩到 %.2f×（0.151×8.70%%，即约 1.3%%），'
  % (0.151 * BAY60 / BAY60))
p('        实测保留 **%.2f×**（%+.4f%% / %.4f%%）—— 与 PeMS-BAY 几乎一样大。' % (_h1 / BAY60, _h1, BAY60))
p('      ⇒ 「有效平均度 × 处理强度」那次线性外推是**错的启发式**：')
p('        实情是图稀疏 3.3×、处理强度被稀释到 ≈0.5×，而 Δ **没有**跟着变小。')
_h3 = [mean(p4deltas[h]) for h in H]
p('  H3a Δ 随步长单调（@15<@30<@60）              → %s  %+.4f < %+.4f < %+.4f'
  % ('✅ 成立' if _h3[0] < _h3[1] < _h3[2] else '❌ 不成立', _h3[0], _h3[1], _h3[2]))
# ★ H3b 原先是拿 true t 与 t_{0.05,2}=4.303 比大小的显著性判定 —— 已撤
#   （2026-10-10：n=3 的三个 seed 不是三次独立重复，这个比较不能做）。
#   改为**描述性**判据：三个 seed 的 Δ@15 是不是同一个符号、以及 min≠max 到多少。
_s15 = p4deltas[15]
_same_sign = all(v > 0 for v in _s15) or all(v < 0 for v in _s15)
p('  H3b @15 效应量级与种子一致性                  → %s  Δ@15 = %s（%s）'
  % ('三个 seed 同号' if _same_sign else '三个 seed **不同号**',
     ' / '.join('%+.4f%%' % v for v in _s15),
     '极差 %.4f%%' % (max(_s15) - min(_s15))))
p('       （此处不再做显著性判定：三个 seed 复用同一个测试集，不是独立重复。）')
p('  H4  MAE 随步长单调变差（任务形状）           → 见下表')
for lab, per in [('persist', [P4['PeMS04|基线-持久性']]), ('HA', [P4['PeMS04|基线-历史均值']])]:
    v = [mae(per[0], h) for h in H]
    p('      %-8s %.4f → %.4f → %.4f  %s' % (lab, v[0], v[1], v[2],
                                              '✅ 单调' if v[0] < v[1] < v[2] else '（HA 三档近乎相同，本就与该步长无关）'))
p('  H5  两数据集量纲不同 ⇒ 表里标单位、不许比大小  → ✅ 见上表标题')
p()
p('  ★ 判据表（§6.2，照抄计划未改）：Δ@60 ≥ +5% ⇒ 「图机制是头条结果」。')
p('    实测 %+.4f%% ≥ +5%% ⇒ **判据触发**，且三 seed 同号、随步长单调。' % _h1)

# ── 必须披露的限制（PeMS04 专属）──────────────────────────────────
p()
p('─' * 78)
p('  PeMS04 必须披露的限制（先验预期_PeMS04.md §7，逐条）')
p('─' * 78)
p('  1) 两个数据集的图是**两套不同处方**：PeMS-BAY/METR-LA 用 DCRNN 高斯核加权图')
p('     （normalized_k=0.1，不归一化）；PeMS04 用社区标准**稀疏二值连通图**')
p('     （ASTGCN / BasicTS 两处独立确认，无更稠密的"标准"版本）。')
p('  2) d0 语义不同：PeMS-BAY 是速度(mph)，PeMS04 是流量(veh/5min)。表里标单位，')
p('     **不许跨数据集比绝对值**。')
p('  3) 切分是**统一 7:1:2**，而 PeMS04 文献惯例是 6:2:2 ⇒ 与已发表数字不可直接比。')
p('  4) 处理强度被稀释：PeMS04 上 real 臂 5 个通道里真邻居只占 **43.97%**')
p('     （PeMS-BAY 88.25%），293/307 个节点真邻居 < 5 ⇒ 对比强度 ≈0.5×，')
p('     且 real 臂内部**逐节点不均匀**（度分位 25/50/75/90 = 2/2/2/4）。')
p('  5) 窗口只有 16969 个（PeMS-BAY 52093），train 11878 个窗口 ⇒ 训练集小 3.1×。')
p('  6) 有效图：PeMS04 对称无权，无向边 340、平均度 **2.21**、密度 0.0072、零度节点 0')
p('     （PeMS-BAY 有向：弧 2369、平均度 7.29、密度 0.0224、零度节点 12；'
  '其中 1789 条弧只有一个方向，互异节点对 2079）⇒ 平均度差 **3.3×**。')
p('  7) ★ **图开那臂在 PeMS04 上撞了 100 轮天花板**：best_epoch %s，'
  % '/'.join(str(r['best_epoch']) for r in P4_ON))
p('     而 epochs_run 全 100 ⇒ 早停规则**没触发**，它还在改善。')
p('     图关那臂 %s 轮就早停了 ⇒ 两臂的**达成轮数不对称**，'
  % '/'.join(str(r['epochs_run']) for r in P4_OFF))
p('     必须逐行报 epochs_run，且应说明 ON 臂的报告值是**保守**的（给它更长预算只会更好）。')
p('     对照：PeMS-BAY 上两臂 %s / %s，也有不对称但幅度小得多。'
  % ('/'.join(str(r['epochs_run']) for r in off), '/'.join(str(r['epochs_run']) for r in on)))
# ★ 逐行算「谁撞了天花板」，不写死 —— 判据是 epochs_run == 自己键里的天花板。
p('  8) ★ PeMS04 全部 13 行里**撞 100 轮天花板**的（早停没触发）：')
for k in sorted(k for k in G2 if k.startswith('PeMS04|')):
    r = G2[k]
    if r['epochs_run'] == 100:
        p('       %-44s best_epoch %-3d ⇒ best 之后还有 %d 轮无改善'
          % (k, r['best_epoch'], r['epochs_run'] - r['best_epoch']))
for k in sorted(k for k in P4 if k.startswith('PeMS04|real|V8|')):
    r = P4[k]
    if r['epochs_run'] == 100:
        p('       %-44s best_epoch %-3d ⇒ best 之后还有 %d 轮无改善（**V8 的 s1**）'
          % (k, r['best_epoch'], r['epochs_run'] - r['best_epoch']))
p('     （DCRNN/GWNet 两行的 cfg.patience=100 ⇒ 它们**不可能**触发早停，撞天花板是设计使然；')
p('      需要解释的是 MAMBASTG 与 V8-s1 这三/四行。）')
p('     ⇒ 正文逐行报 epochs_run；撞天花板的行其报告值是**保守的**（给更长预算只会更好），')
p('       但**不得**因此说这些行「未收敛」—— 只能说它没触发早停规则。')

# ================================================================ 一致性与限制
p()
p('=' * 78)
p('  同源一致性检查（全精度下才算得出来的）')
p('=' * 78)
for lab, rows_ in [('MAMBAST 关', off), ('MAMBASTG 开', on), ('V8', V8_ROWS),
                   ('DCRNN/GWNet', [gnn_row('DCRNN', 42), gnn_row('GWNET', 42)])]:
    p('  %-12s n_eval %s   params %s   (n_train %s)'
      % (lab, sorted({r['n_eval'] for r in rows_}),
         sorted({r['params'] for r in rows_}),
         sorted({r['n_train'] for r in rows_})))

# ★ 独立锚点：V8 的参数量必须与 arm3 那三臂一致（它们都是 V8 跑出来的），
#   且必须**不等于** MAMBAST 的 384,140 —— 否则说明取错了记录。
assert {r['params'] for r in V8_ROWS} == {252940}, 'V8 参数量不是 252,940：取错记录了'
assert {r['params'] for r in on} == {r['params'] for r in off} == {384140}, \
    '两臂参数量不相等：单变量前提破了'
assert len(GNN) == 13 and len(arm_rows) == 3, '结果文件条数异常'
p('  ✅ 断言通过：V8=252,940（且 != MAMBAST 的 384,140）、两臂同参数 384,140')

# ★★ 轮数不是「跑够 100」而是「跑到早停点」——这个等式是全表可比性的地基，
#    所以把它做成断言，而不是在正文里写一句话。反过来，如果某行 epochs_run
#    恰好等于天花板，说明它没触发早停、是跑满的，也得单独列出来。
def ceiling_of(key):
    """从键里读天花板。★ **必须逐键读，不能全局用一个 100** ——
    `gnn_results` 里同一模型同时存在 `|e30` 与 `|e100` 两套键，
    拿 100 去套 e30 的行会把它误判成「口径不明」。"""
    m = re.search(r'\|e(\d+)$', key)
    return int(m.group(1)) if m else None


def audit(store):
    """逐行核 epochs_run 与 best_epoch。返回 (真早停, 跑满自己的天花板, 口径不明, 无从核)。"""
    stopped, full, odd, unknown = [], [], [], []
    for k, v in sorted(store.items()):
        if not isinstance(v, dict) or 'epochs_run' not in v or 'best_epoch' not in v:
            continue                       # 基线行没有这两个字段
        c = ceiling_of(k)
        if c is None:
            unknown.append(k)              # 键里没写轮数 → 本脚本核不了，单独报，不静默跳过
            continue
        d = v['epochs_run'] - v['best_epoch']
        (stopped if d == patience_eff(v) + 1
         else (full if v['epochs_run'] == c else odd)
         ).append((k, v['epochs_run'], v['best_epoch'], c))
    return stopped, full, odd, unknown


ALL_ODD = []
for label, store in [('paper_results_e100.json（表 1 / 表 3 的数字来源）', E100 or {}),
                     ('gnn_results（算子侧消融 + 两条图基线）', GNN)]:
    stopped, full, odd, unknown = audit(store)
    ALL_ODD += odd
    p()
    p('  ── %s' % label)
    p('     真早停 %d 行（epochs_run − best_epoch == **该 run 的** patience+1；'
      'patience 逐 run 读 cfg.patience，缺则回落常数 %d）'
      % (len(stopped), PATIENCE))
    for k, ep, b, c in stopped:
        p('        %-44s ep_run %-4d best %-4d 天花板 %d' % (k, ep, b, c))
    if full:
        p('     ★ 跑满**自己键里的天花板**的有 %d 行 —— 早停没触发，best_epoch 位置须披露：'
          % len(full))
        for k, ep, b, c in full:
            p('        %-44s ep_run %-4d best %-4d 天花板 %-4d ← best 之后还有 %d 轮毫无改善'
              % (k, ep, b, c, ep - b))
    if odd:
        p('     ★★ 口径不明 %d 行（既不是早停、也没跑满自己键里的天花板）：' % len(odd))
        for k, ep, b, c in odd:
            p('        %-44s ep_run %-4d best %-4d 天花板 %d' % (k, ep, b, c))
    if unknown:
        p('     ·  键里没写轮数、本脚本核不了 %d 个（**不是通过**）：%s'
          % (len(unknown), unknown))
assert not ALL_ODD, ('有行的 epochs_run 既不等于 best_epoch+%d 也不等于**自己键里的**天花板：'
                     '轮数口径不明，先查清楚再出表\n%s' % (PATIENCE + 1, ALL_ODD))

# ★★ 为什么天花板必须是 100 —— 逐行算，不写套话。
#    判据：该行在 100 的天花板下 best_epoch + 9 > 30 ⇒ 压回 30 时它还没等够
#    PATIENCE 轮就被强行结束，选出来的就不是它的早停点了。
#    （另：cosine(T_max=EPOCHS) 意味着改天花板还会换 LR 调度，两件事一起变。）
TRUNC = []
for store in (E100 or {}, GNN):
    for k, v in sorted(store.items()):
        if not isinstance(v, dict) or 'best_epoch' not in v or not k.startswith('PeMS-BAY'):
            continue
        if ceiling_of(k) != 100:          # 只看天花板 100 那批（PeMS-BAY 才是论文数据集）
            continue
        s = v['best_epoch'] + PATIENCE + 1
        if s > 30:
            TRUNC.append((k, v['best_epoch'], s))
p()
p('  ── 为什么全表统一到 100：压回 30 的天花板，下面这些行会被**截断**')
p('     （判据：best_epoch + %d > 30，即它要跑过 30 轮才早停）' % (PATIENCE + 1))
for k, b, s in TRUNC:
    p('        %-44s best %-4d 本该停在第 %d 轮 ⇒ 少跑 %d 轮' % (k, b, s, s - 30))
if not TRUNC:
    p('        （一行都没有 —— 30 的天花板其实够用，100 只是保险；这句话也要照实写）')

p()
p('─' * 78)
p('  ⚠️ 不支持的主张 / 必须披露的限制（自动列出，不要只看表）')
p('─' * 78)

gw = gnn_row('GWNET', 42)
v8m = [mean([mae(r, h) for r in V8_ROWS]) for h in H]
mst = [mean([mae(r, h) for r in on]) for h in H]


def cmp_str(vals, ref):
    return '  '.join('@%d %+.4f%%' % (h, (vals[i] - ref['mae_%d' % h]) / ref['mae_%d' % h] * 100.0)
                     for i, h in enumerate(H))


p('  1) V8 vs GWNet 复现（三档）：' + cmp_str(v8m, gw) + '   → 三档全输，不得宣称 SOTA')
p('  2) MAMBASTG vs GWNet 复现（三档）：' + cmp_str(mst, gw) + '   → 也输')
p('  3) 参数效率：V8 %s < GWNet %s 且三档全输 → 参数当卖点不成立'
  % ('{:,}'.format(v8_row(1)['params']), '{:,}'.format(gw['params'])))
p('      MAMBASTG %s > GWNet %s → 更不能谈参数效率'
  % ('{:,}'.format(on[0]['params']), '{:,}'.format(gw['params'])))
p('  4) 轮数口径：**全表天花板统一 100**（两条 GNN 基线、V8、LSTM/TF、表 3 三臂）。')
p('     V8 三 seed epochs_run=%s，best_epoch=%s ⇒ 差值恒为 %d ⇒ 都是真早停、不是撞天花板。'
  % ('/'.join(str(r['epochs_run']) for r in V8_ROWS),
     '/'.join(str(r['best_epoch']) for r in V8_ROWS), PATIENCE + 1))
p('     ⚠️ 旧 e30 那批必须作废并披露：上面「为什么统一到 100」逐行列了证据 —— '
  '压回 30 会截断 %d 行。' % len(TRUNC))
p('        表 3 的**三个臂**也各有 e30 一套（在 paper_results_input3arm_0928.json 里，'
  '本表刻意不 load）→')
p('        那一套同样作废，不能只换 `real` 臂、留 random/self 的旧数。')
p('        正文任何地方都不得再引用 e30 的数字（含表 3 的 real 臂 —— 它和表 1 的 V8 是同一个 run）。')
p('        另：cosine(T_max=EPOCHS) 使 30 与 100 的 **LR 调度也不同**，')
p('        所以不能写成「训练更久反而更差」——两个变量一起变了。')
p('  5) 两臂 epochs_run 不同（关 %s / 开 %s）→ 早停按 val 触发，须披露但选择规则相同'
  % ('/'.join(str(r['epochs_run']) for r in off),
     '/'.join(str(r['epochs_run']) for r in on)))
# ★★ 第 6 条**由数据生成**，不写死。上一版写死成「两行没有被任何结果文件覆盖」，
#    那是照着当时 res_LSTM_s42.json 的键猜的；真跑出来之后那句话就成了假话，
#    而表里两行仍然印「缺」—— 假话与缺行互相掩护，谁都看不出来。
#    ★ 同理，「单 seed 42」这个口径也必须是**读出来的**：E 批补到三 seed 之后，
#      再印一句「只有单 seed」就与上面那段「写死的假话」是同一类病。
_net_n = '/'.join('%s %d seed' % (ARCH_BARE[a], len(NET_ROWS[a])) for a in ('LSTM', 'TF'))
p('  6) LSTM / Transformer 两行（种子数 %s）：' % _net_n)
for arch in ('LSTM', 'TF'):
    per = NET_ROWS[arch]
    if not per:
        p('     · %-11s ★ %s 里没有 |e100 键 ⇒ 本表印「缺」。'
          % (arch, E100_FILE))
        p('       正文不得用别的轮数的数字顶替，也不得两行留白不作说明。')
        continue
    v = [mean([mae(r, h) for r in per]) for h in H]
    d = ['@%d %+.4f%%' % (h, (v[i] - v8m[i]) / v8m[i] * 100.0) for i, h in enumerate(H)]
    p('     · %-11s %.6f/%.6f/%.6f  params %s  ep_run %s best %s'
      % (arch, v[0], v[1], v[2], '{:,}'.format(per[0]['params']),
         '/'.join(str(r['epochs_run']) for r in per),
         '/'.join(str(r['best_epoch']) for r in per)))
    p('       相对 V8 三 seed 均值：%s' % '  '.join(d))
    if len(per) > 1:
        p('       逐 seed @15：%s' % '  '.join('%.6f' % mae(r, 15) for r in per))
_all_single = bool(NET_ROWS['LSTM'] or NET_ROWS['TF']) and all(
    len(NET_ROWS[a]) <= 1 for a in ('LSTM', 'TF'))
p('     ★ 若上面出现负号 = 该基线**比我们的 V8 更好**，正文必须照实写，')
p('       不得只报「我们不比它差」；同时披露它参数量是 V8 的 %.1f 倍、%s'
  % (mean([r['params'] for per in NET_ROWS.values() for r in per]) / V8_ROWS[0]['params']
     if any(NET_ROWS.values()) else float('nan'),
     '且只有单 seed' if _all_single else '种子数 %s' % _net_n))

# ★★ 7) DCRNN 那一行**不是忠实复现** —— 逐条摘自 gnn_baselines.py:166-177 的偏差清单
#      （那一节对着官方 config.yaml 写成，是权威；这里只做搬运，不改口径）。
#      不写这一条的话，读者会把它当成"我们的 DCRNN 复现"，而它调的是另一套超参。
dc = gnn_row('DCRNN', 42)
_dccfg = dc.get('cfg') or {}
_our_clip = _dccfg.get('clip')
_our_pat = _dccfg.get('patience')
p('  7) DCRNN 这一行不是忠实复现，必须披露 —— 与官方 %s 的偏差（出处 gnn_baselines.py:166-177）：'
  % 'liyaguang/DCRNN config.yaml')
# ★ 2026-09-29 修：「我们」那一列必须是**这一行 run 自己记录的值**（cfg），不是脚本默认值。
#   原先写的 `'1.0（本脚本全局 CLIP，:142）'` / `'%d' % PATIENCE`（=8）两个都是**脚本默认**，
#   而这一行的 cfg 是 clip=1.0（碰巧相同）、**patience=100（与印出来的 8 不同）**。
for a, o, ours in [
        ('梯度裁剪 max_grad_norm', '5',
         '%.1f（该 run 的 cfg.clip）' % _our_clip if _our_clip is not None
         else '1.0（脚本默认，见 gnn_baselines.py:142；该 run 的 cfg 未记录）'),
        ('早停 patience', '50',
         '%d（该 run 的 cfg.patience；脚本默认 %d，见 gnn_baselines.py:141）'
         % (_our_pat, PATIENCE) if _our_pat is not None
         else '%d（脚本默认，见 gnn_baselines.py:141；该 run 的 cfg 未记录）' % PATIENCE),
        ('RNN 层数 num_rnn_layers', '2', '1'),
        ('课程学习 use_curriculum_learning', 'true', 'SS_PROB=0（关）')]:
    p('       %-34s 官方 %-6s → 我们 %s' % (a, o, ours))
p('     ⚠️ 前两行的"我们"列取自该 run 的 `cfg`；后两行是代码路径、不在 cfg 里'
  '（层数见 gnn_baselines.py:166-177 的偏差清单，课程学习=SS_PROB=0）。')
p('     数据侧佐证：PeMS-BAY DCRNN 的 best_epoch=%d 而 epochs_run=%d '
  '⇒ best 之后还有 %d 轮没有改善 ⇒ 轨迹确实不稳，'
  % (dc['best_epoch'], dc['epochs_run'], dc['epochs_run'] - dc['best_epoch']))
p('     该行的数字不能当作"DCRNN 的真实水平"，正文只能写"我们的 DCRNN 复现"。')
p('     ⚠️ 光看 JSON 分不出"早停救了它"和"它根本没发散"—— 想说收敛/不发散只能拿逐轮轨迹当证据。')

# ★★ 8) 两条 GNN 基线与 V8 的 epoch 预算虽同为 100，但**达成的轮数不同**
#      （早停触发点不同）。这一条本身正常，但必须写出来，否则读者会以为所有行都跑了 100 轮。
p('  8) 各行达成的轮数不同（早停触发点不同），预算同为 100：')
p('     GWNet 复现 ep_run=%d（best %d）、DCRNN 复现 ep_run=%d（best %d）、'
  'V8 三 seed %s、算子侧两臂 %s / %s'
  % (gw['epochs_run'], gw['best_epoch'], dc['epochs_run'], dc['best_epoch'],
     '/'.join(str(r['epochs_run']) for r in V8_ROWS),
     '/'.join(str(r['epochs_run']) for r in off),
     '/'.join(str(r['epochs_run']) for r in on)))
# ★ 2026-09-29 修：原文写「选择规则完全相同（验证集 MAE，patience=%d）」—— 前半句对
#   （都用验证集 MAE 选最优），但**耐心值不统一**：两条图基线的 cfg.patience 是 100，
#   算子侧两臂与主脚本各行的常数是 8。用一个数字说"完全相同"是在替读者断言一件不成立的事。
p('     选优规则相同（都用验证集 MAE 选最优权重），但**耐心值逐 run 不同**：')
p('       GWNet %s、DCRNN %s（取自各自 cfg.patience）；'
  'V8/LSTM/TF 与算子侧两臂 %d（paper_experiments.py:95 的常数，cfg 未记录）。'
  % (_show_pat(gw), _show_pat(dc), PATIENCE))
p('     ⇒ 达成的轮数不同 = 耐心值不同 × 是否触发早停，**不是口径不一致**；')
p('     但正文报告结果时必须同时给出 epochs_run，不能只写"训练 100 轮"。')

# ★ 输出名也可覆盖（TABLE_OUT）：否则没法拿合成数据试跑 —— 一跑就把真产物
#   覆盖成合成数字，而合成数字看起来和真的一模一样。
OUT_NAME = os.environ.get('TABLE_OUT') or '论文_表格_全精度.md'
# ⚠️ 这里是**文本模式**写盘，Windows 上 `\n` 会被翻成 `\r\n` —— 于是这份 md 全篇是
#    CRLF，而本仓其余文件是 LF（`.gitattributes` 是 `* -text`，入库什么字节就是什么
#    字节，不会被 git 抹平）。**这是历史形成的、稳定的约定，别顺手改成二进制写**：
#    改一下就会让整份文件 475 行全变，把真实差异埋掉（记忆 subprocess-text-mode-traps）。
#    检查器用 io.open 文本读，通用换行会把 CRLF 当 LF 处理，所以解析不受影响。
with open(os.path.join(ROOT, OUT_NAME), 'w', encoding='utf-8') as f:
    f.write('```\n' + '\n'.join(OUT) + '\n```\n')
print()
print('已写出 %s' % OUT_NAME)
