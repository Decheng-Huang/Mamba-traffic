#!/bin/bash
# box_run_all.sh —— A/B 两臂跑完之后，一次性跑完剩下的全部实验，全程无人值守。
#
# 放在 paper_experiments.py / gnn_baselines.py 所在的目录，然后：
#     nohup bash box_run_all.sh &
#     tail -f chain_all.log
#
# 设计原则（用户要求：不浪费；花钱买效率可以）：
#   1. A/B 一结束立刻接上，不留一秒空闲计费。
#   2. **绝不盲跑**：主线用不用时段通道，由 A/B 的判决自动决定。规则先定死，
#      不事后挑口径。盲跑的代价是全部配置作废（¥48 级），这是本脚本存在的理由。
#   3. 矩阵模式会自动跳过 paper_results.json 里已有 cid 的配置 ——
#      A 臂（|notod）和 B 臂（|tod）都已完成，不会被重跑、也不会计两次钱。
#   4. 任何一步等不到结果就中止，不启动下游（避免崩了之后还继续烧）。
#
# ⚠️ 启动后必看抬头：
#      · 「A@60=...  B@60=...  Δ=...%  → 用/不用时段通道」
#      · 第一段「共 6 个配置」（含 B 臂，会被跳过 → 实跑 5 个）
#        第二段「共 4 个配置」
#      · 「时段通道(PAPER_TOD)=开/关」与判决一致
#    不对就 Ctrl-C，损失几秒。
#
# 预算（按实测 ¥2.08/h、153 s/轮）：9 个 Mamba 配置 ≈ 12.3 h ≈ ¥26；
#   GNN 4 个配置 ≈ 4–6 h ≈ ¥10；加上 A/B 剩余 ≈¥4 → **合计 ≈¥40、≈20 小时**。

LOG=chain_all.log
MKEY='"PeMS-BAY|real|V8|official|s1|tod"'
ts() { date +'%F %T'; }

{
  # ---------------------------------------------------------------- 1. 等 B 臂
  echo "[$(ts)] 等 Mamba B 臂落盘（最多 3 小时）"
  for _ in $(seq 1 180); do
    grep -q "$MKEY" paper_results.json && break
    sleep 60
  done
  if ! grep -q "$MKEY" paper_results.json; then
    echo "[$(ts)] ⚠️ 等满 3 小时没等到 B 臂结果 —— 中止，不启动下游（不白烧钱）"
    exit 1
  fi
  cp paper_results.json "paper_results.$(date +%m%d_%H%M).json"
  echo "[$(ts)] B 臂已落盘，结果已备份"

  # ---------------------------------------------------------------- 2. 定 tod
  cat > _pick_tod.py <<'PY'
import json, sys
try:
    r = json.load(open('paper_results.json', encoding='utf-8'))
except Exception as e:
    sys.exit(f'读不了 paper_results.json: {e}')
a = r.get('PeMS-BAY|real|V8|official|s1|notod', {}).get('mae_60')
b = r.get('PeMS-BAY|real|V8|official|s1|tod', {}).get('mae_60')
if a is None or b is None:
    sys.exit(f'两臂数字不齐：A={a} B={b}')
d = (a - b) / a * 100.0                     # 正 = 时段通道让 @60 误差变小
print(f'A@60={a:.3f}  B@60={b:.3f}  Δ={d:+.2f}%  →  '
      + ('用时段通道 |tod' if d >= 0 else '不用时段通道'), file=sys.stderr)
# 判据先定死：Δ@60 ≥ 0 就用，< 0 就不用。不事后挑。
print(1 if d >= 0 else 0)
PY
  USE=$(python _pick_tod.py) || { echo "[$(ts)] ⚠️ 判决脚本失败，中止"; exit 1; }
  case "$USE" in 0|1) ;; *) echo "[$(ts)] ⚠️ 判决结果异常（'$USE'），中止"; exit 1 ;; esac

  # ★★ 关键的第二个坑：PAPER_SUFFIX 必须跟着判决走 ★★
  #   USE=1 → 主线 cid 带 |tod   = B 臂的 cid → 复用它，不重跑 ✓
  #   USE=0 → 主线 cid **不带任何后缀** = 'PeMS-BAY|real|V8|official|s1'，
  #           那正是 **A5 的旧 cid**（改 tod 补丁之前跑的，钱还在盒子的
  #           paper_results.json 里）。续跑逻辑会认成「已有结果」直接跳过，
  #           于是**悄悄用旧代码的结果冒充新结果**。
  #           所以 USE=0 时必须加上 |notod，让它复用刚跑完的 A 臂。
  if [ "$USE" = "1" ]; then SUF=""; else SUF="|notod"; fi

  # ---------------------------------------------------------------- 3. Mamba 主实验
  # 「够发表」档 = 主线 real × 3 种子 + 消融 random/self × 1 种子。
  # 若要按任务书全量（消融也 3 种子），把第二段换成
  #     PAPER_SEEDS=1,2,3 PAPER_MODES=real,random,self  → 17 个配置，≈¥48
  echo "[$(ts)] 开始 Mamba 主实验（PAPER_TOD=$USE  PAPER_SUFFIX='$SUF'）"
  PAPER_MODE=matrix PAPER_DATASETS=PeMS-BAY,METR-LA \
  PAPER_ARCH=V8 PAPER_BLOCK=official PAPER_FORCE_PARQUET=1 \
  PAPER_TOD="$USE" PAPER_SUFFIX="$SUF" \
  PAPER_SEEDS=1,2,3 PAPER_MODES=real \
      python paper_experiments.py
  PAPER_MODE=matrix PAPER_DATASETS=PeMS-BAY,METR-LA \
  PAPER_ARCH=V8 PAPER_BLOCK=official PAPER_FORCE_PARQUET=1 \
  PAPER_TOD="$USE" PAPER_SUFFIX="$SUF" \
  PAPER_SEEDS=1 PAPER_MODES=random,self \
      python paper_experiments.py

  # ---------------------------------------------------------------- 4. GNN 基线
  # 官方输入处方：DCRNN/GWNet 2 通道（速度+tod），STGCN 1 通道。tod 不做归一化。
  # ★ 这里**故意不跑 STGCN**：任务书 §3.1 已查实它用的是 BJER4/PeMSD7，
  #   没有 METR-LA/PeMS-BAY 的数字，**不得进对比表**。为它花 ¥5 是纯浪费；
  #   真想要它的数字，本地免费跑（它只需要 numpy + torch，不用 mamba_ssm）。
  # ★ LSTM / Transformer 同理：不需要 mamba_ssm，本地跑，¥0。
  echo "[$(ts)] 开始 GNN 基线（官方处方，不含 STGCN）"
  PAPER_MODE=train PAPER_GNN_MODELS=DCRNN,GWNET \
  PAPER_GNN_DATASETS=PeMS-BAY,METR-LA PAPER_GNN_SEEDS=42 \
      python gnn_baselines.py

  echo "[$(ts)] ★★★ 全部结束 —— 立刻关机，别再计费 ★★★"
} 2>&1 | tee -a "$LOG"
