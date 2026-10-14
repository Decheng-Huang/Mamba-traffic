#!/bin/bash
# 在租的机器上跑：等 A 臂（|notod）结束 → 备份结果 → 立刻接 B 臂（|tod）。
#
# 为什么要链起来：机器按小时计费（实测 ¥2.08/h）。A 臂跑完到 B 臂启动之间
# 每一分钟都在烧钱，而且很容易"忘了"——空转 8 小时 = ¥16.6。
# 所以这个脚本自己做等待，不需要人守着，也不会留空档。
#
# 用法（放在 paper_experiments.py 所在的目录）：
#     nohup bash box_chain_b.sh &
#     tail -f chain.log
#
# 万一 _d1 读不到，load_dataset 会在【载入阶段】就抛 TodUnavailable，
# 两分钟内中止 —— 不会白白跑满 80 分钟。所以不需要额外的预检。

LOG=chain.log
ts() { date +'%F %T'; }

{
  echo "[$(ts)] 等待 A 臂结束…"
  # 注意：脚本自己是 `bash box_chain_b.sh`，不会被这个模式匹配到，
  # 所以不会自己等自己。
  while pgrep -f "paper_experiments" >/dev/null; do sleep 30; done
  echo "[$(ts)] A 臂已结束"

  # 备份。save_results() 是 open('w') 直接覆盖，中途断电就会把整个文件
  # 截断——包括已经跑完的 A 臂和两条基线。一次 cp 换掉这个风险，值得。
  cp paper_results.json "paper_results.$(date +%m%d_%H%M).json" \
    && echo "[$(ts)] 结果已备份"

  echo "[$(ts)] 开始 B 臂（PAPER_TOD=1，预计 80 分钟，约 ¥2.8）"
  # ★ 逐仓库照官方。Mamba 这一侧的时段通道由 PAPER_TOD 控制，
  #   与 GNN 基线那个 OFFICIAL_IN_DIM 是两个独立问题。
  # ★ 不设 PAPER_SUFFIX —— 配置 ID 会是 PeMS-BAY|real|V8|official|s1|tod，
  #   与 A 臂的 ...|notod 是两条独立记录。
  # ★ PAPER_FORCE_PARQUET=1 必须留：A 臂也开了，两臂必须读同一份数据源。
  PAPER_MODE=matrix PAPER_DATASETS=PeMS-BAY PAPER_MODES=real \
  PAPER_ARCH=V8 PAPER_BLOCK=official PAPER_SEEDS=1 \
  PAPER_TOD=1 PAPER_FORCE_PARQUET=1 \
      python paper_experiments.py 2>&1

  echo "[$(ts)] ★★★ B 臂结束 —— 立刻关机，别再计费 ★★★"
} 2>&1 | tee -a "$LOG"
