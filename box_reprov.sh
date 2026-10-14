#!/usr/bin/env bash
# box_reprov.sh —— 在盒子上重跑论文用到的全部配置，让每一批结果都带上出生证明。
#
# 跑法（建议 nohup，断了也接着跑）：
#     cd /root/autodl-tmp
#     nohup bash box_reprov.sh > reprov_driver.log 2>&1 &
#     tail -f reprov_log/driver.log          # 看进度
#
# 设计（照抄 run_C.sh 那套已经被事故证明有效的纪律）
# ────────────────────────────────────────────────────────────
# 1. **写新文件，不碰旧结果。** 结果进 paper_results_reprov.json，
#    出生证明进 _provenance/paper_results_reprov.json。
#    论文现在用的 paper_results*.json 一个字节不动 —— 重跑万一不一致，
#    也不能让它在原地把旧数覆盖掉。
#
# 2. **后缀用 |e100，与论文的键逐字相同。** 这样新旧两份文件里的键名
#    完全一样（PeMS-BAY|real|V8|official|s1|tod|e100），比对时是
#    同一个字符串对同一个字符串，不是「差不多」的键。
#    （若改叫 |prov，得到的就是另一批键，永远没法直接对质。）
#
# 3. **每批跑完做正向断言：键数必须增加。** 事故签名是「跑完、rc=0、
#    日志只多一行、JSON 什么都没变」—— 只看返回码看不出来。
#
# 4. **串行。** 单进程就把卡打满，并发只乘内存不省时间。
#
# ⚠️ 代价：这四批合计约 **25–30 小时**（各配置早停，轮数见论文）。
#    中途想停就 Ctrl-C 或 kill —— 已完成的那几批结果与出生证明都有效，
#    可以只取已跑完的部分。
set -u
BASE=/root/autodl-tmp
PY=/root/miniconda3/envs/mamba2/bin/python
RESULT="$BASE/paper_results_reprov.json"
LOGDIR="$BASE/reprov_log"
cd "$BASE" || { echo "FATAL: cd $BASE 失败"; exit 1; }
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/driver.log"; }

count_keys() {
  [ -f "$1" ] || { echo 0; return; }
  "$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1],encoding="utf-8"))))' \
        "$1" 2>/dev/null || echo -1
}

# ── 0. 硬门：解释器 + 代码里必须有出身块 ──────────────────────────────
if ! $PY -c "import mamba_ssm, causal_conv1d, torch" >/dev/null 2>&1; then
  log "FATAL: $PY 导不进官方核 —— 停止。"
  exit 1
fi
log "解释器 OK: torch=$($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)"
if ! grep -q '_provenance' paper_experiments.py; then
  log "FATAL: paper_experiments.py 里没有 _provenance —— 新版没上传成功，先跑 _unpack_prov.sh"
  exit 1
fi

# ── 1. 等其它 job 结束（串行纪律）─────────────────────────────────────
waited=0
while [ "$(pgrep -af '[p]aper_experiments.py' | wc -l)" -gt 0 ]; do
  if [ "$waited" -ge 21600 ]; then log "已等 6 h，仍在跑 —— 不再等，直接开始"; break; fi
  [ $((waited % 1800)) -eq 0 ] && log "等其它 job 结束… 已等 $((waited / 1800)) 个半小时"
  sleep 300; waited=$((waited + 300))
done

# ── 2. 逐个配置组跑 + 正向断言 ────────────────────────────────────────
# 组名按「数据集_模型族」取，日志名即组名，出问题一眼知道在哪。
group() {
  name="$1"; shift
  before=$(count_keys "$RESULT")
  log "▶ $name 开始（结果文件现有 $before 键）"
  if env "$@" > "$LOGDIR/$name.log" 2>&1; then
    log "  $name 退出码 0"
  else
    log "  ✘ $name 退出码非 0 —— 看 $LOGDIR/$name.log"
  fi
  after=$(count_keys "$RESULT")

  if [ "$after" -le "$before" ]; then
    log "★★ FATAL: $name 跑完但结果文件没多出任何键（$before → $after）"
    log "   这就是空 job 的签名。立刻中止。现场："
    tail -30 "$LOGDIR/$name.log" | sed 's/^/      /' | tee -a "$LOGDIR/driver.log"
    exit 1
  fi
  log "  ✔ $name 真的写了结果（$before → $after 键）"

  if grep -q '官方 mamba_ssm: 不可用' "$LOGDIR/$name.log" 2>/dev/null; then
    log "★★ FATAL: $name 日志说官方核不可用 —— 那一批数字别信。中止。"
    exit 1
  fi
}

# PeMS-BAY  · 主表 4-1/4-2 + 输入侧三臂 4-4（V8，三 seed）
group bay_v8 PAPER_MODE=matrix PAPER_DATASETS=PeMS-BAY PAPER_MODES=real,random,self \
      PAPER_ARCH=V8 PAPER_SEEDS=1,2,3 PAPER_EPOCHS=100 PAPER_BLOCK=official \
      PAPER_TOD=1 PAPER_SUFFIX='|e100' PAPER_RESULT=paper_results_reprov.json \
      "$PY" paper_experiments.py

# PeMS-BAY  · 神网基线 LSTM / Transformer（s42）
group bay_nn PAPER_MODE=matrix PAPER_DATASETS=PeMS-BAY PAPER_MODES=real \
      PAPER_ARCH=LSTM,TF PAPER_SEEDS=42 PAPER_EPOCHS=100 PAPER_BLOCK=official \
      PAPER_TOD=1 PAPER_SUFFIX='|e100' PAPER_RESULT=paper_results_reprov.json \
      "$PY" paper_experiments.py

# PeMS04  · 第二基准主表 4-5/4-6（V8，三 seed）
group p04_v8 PAPER_MODE=matrix PAPER_DATASETS=PeMS04 PAPER_MODES=real \
      PAPER_ARCH=V8 PAPER_SEEDS=1,2,3 PAPER_EPOCHS=100 PAPER_BLOCK=official \
      PAPER_TOD=1 PAPER_SUFFIX='|e100' PAPER_RESULT=paper_results_reprov.json \
      "$PY" paper_experiments.py

# PeMS04  · 神网基线 LSTM / Transformer（s42）
group p04_nn PAPER_MODE=matrix PAPER_DATASETS=PeMS04 PAPER_MODES=real \
      PAPER_ARCH=LSTM,TF PAPER_SEEDS=42 PAPER_EPOCHS=100 PAPER_BLOCK=official \
      PAPER_TOD=1 PAPER_SUFFIX='|e100' PAPER_RESULT=paper_results_reprov.json \
      "$PY" paper_experiments.py

# ── 3. 收尾：列出全部键 + 出生证明 ────────────────────────────────────
log "──────── 重跑结束 ────────"
log "结果文件 $RESULT 现有 $(count_keys "$RESULT") 键："
"$PY" - "$RESULT" <<'PYEOF' 2>&1 | sed 's/^/    /' | tee -a "$LOGDIR/driver.log"
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
for k in sorted(d):
    v = d[k] if isinstance(d[k], dict) else {}
    print('%-50s ep_run=%-4s best=%-4s ep_time=%-8s mae15=%s'
          % (k, v.get('epochs_run', '-'), v.get('best_epoch', '-'),
             ('%.1f' % v['epoch_time']) if v.get('epoch_time') else '-',
             ('%.6f' % v['mae_15']) if v.get('mae_15') else '-'))
PYEOF

PROV="$BASE/_provenance/paper_results_reprov.json"
if [ -f "$PROV" ]; then
  log "✔ 出生证明已写：$PROV"
else
  log "★★ FATAL: 没找到出生证明 $PROV —— 结果没有来历，值不得信。"
  exit 1
fi
log "★ 取回用：bash $BASE/box_fetch.sh  （再在 JupyterLab 里下载 prov_out.tgz）"
log "★ 本脚本不关机。"
