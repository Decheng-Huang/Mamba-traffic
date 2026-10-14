#!/usr/bin/env bash
# box_smoke.sh —— 冒烟：证明「出身块在盒子上真的写得出来」。
#
# ⚠️ **不产出论文数字。** 5 轮、单配置、独立结果文件、后缀 |smoke ——
#    与论文的键（…|tod|e100）完全隔开，不可能被误当成正式结果。
#
# 为什么必须先跑它：出身块是本地产物，盒子上的代码此前没有它。
# 直接开跑 29 小时的正式重跑，万一出身块在盒子上写不出来（比如
# 路径权限、或新版根本没上传成功），那 29 小时得到的东西和旧结果一样
# 没有出生证明 —— 白烧。先花 ~13 分钟把这条路验通。
set -u
BASE=/root/autodl-tmp
PY=/root/miniconda3/envs/mamba2/bin/python
RESULT="$BASE/paper_results_smoke.json"
cd "$BASE" || { echo "FATAL: cd $BASE 失败"; exit 1; }

echo "== 解释器硬门（照抄 run_C.sh：谁跑它谁验它）=="
$PY -c "import mamba_ssm, causal_conv1d, torch; print('  torch', torch.__version__)" || {
  echo "★ FATAL: $PY 导不进官方核 —— 停。跑下去只会得到没有意义的结果。"; exit 1; }

echo
echo "== 先确认要跑的那份代码带着出身块 =="
if ! grep -q "_provenance" paper_experiments.py; then
  echo "★ FATAL: 盒子上的 paper_experiments.py 里没有 _provenance ——"
  echo "  说明新版没上传成功。重跑只会得到没有出生证明的结果，正是要避免的。停。"
  exit 1
fi
echo "  ✔ paper_experiments.py 里出现 _provenance"

echo
echo "== 冒烟：PeMS-BAY / V8 / real / s1 / 5 轮 =="
PAPER_MODE=matrix \
PAPER_DATASETS=PeMS-BAY \
PAPER_MODES=real \
PAPER_ARCH=V8 \
PAPER_SEEDS=1 \
PAPER_EPOCHS=5 \
PAPER_BLOCK=official \
PAPER_TOD=1 \
PAPER_SUFFIX='|smoke' \
PAPER_RESULT=paper_results_smoke.json \
$PY paper_experiments.py 2>&1 | tee smoke.log

echo
echo "== 断言 =="
[ -f "$RESULT" ] || { echo "★ FATAL: 结果文件没写出来"; exit 1; }
PROV="$BASE/_provenance/paper_results_smoke.json"
[ -f "$PROV" ] || {
  echo "★ FATAL: 出生证明没写出来（$PROV 不存在）—— 出身块没生效，别往下跑。"
  exit 1; }

$PY - "$RESULT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
want = 'PeMS-BAY|real|V8|official|s1|tod|smoke'
print('   结果文件共 %d 个键：' % len(d))
for k in sorted(d):
    v = d[k] if isinstance(d[k], dict) else {}
    print('     %-46s ep_run=%-4s mae15=%s'
          % (k, v.get('epochs_run', '-'), v.get('mae_15', '-')))
print()
print('   OK 冒烟键在场' if want in d else '   FATAL: 没找到 %s' % want)
sys.exit(0 if want in d else 1)
PYEOF
[ $? -eq 0 ] || exit 1

echo
echo "== 出生证明里写了什么 =="
$PY - "$PROV" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
for k in ('_schema', 'session_start_utc', 'written_utc', 'result_file',
          'result_sha256', 'n_entries', 'entry_keys'):
    if k in d:
        print('   %-18s %s' % (k, d[k]))
for k in ('host', 'python', 'packages', 'compute', 'git'):
    print('   %-18s %s' % (k, json.dumps(d.get(k), ensure_ascii=False)))
PYEOF

echo
echo "★ 冒烟结束。上面没有 FATAL 的话，再跑 bash $BASE/box_reprov.sh"
