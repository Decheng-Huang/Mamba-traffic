# -*- coding: utf-8 -*-
# ============================================================================
#  GNN 基线：DCRNN / STGCN / Graph WaveNet
#
#  用途：补上 SCI 任务书 §1.4 的 Q2（"论文声称用了空间邻接信息，却不跟任何
#  做空间建模的模型比"）。三个模型各自按原论文实现：
#
#    DCRNN      Li et al., ICLR 2018 —— 双向随机游走扩散卷积 + DCGRU + seq2seq
#    STGCN      Yu  et al., IJCAI 2018 —— GLU 时间卷积 + Chebyshev 图卷积
#    GraphWaveNet Wu et al., IJCAI 2019 —— 自适应邻接 + 门控膨胀因果卷积
#
#  本文件**自包含**，不 import paper_experiments。原因：paper_experiments.py
#  没有 __main__ 守卫，一旦被 import 就会真跑整个实验矩阵（该项目已踩过一次，
#  见其文末注释）。为了让协议严格对齐，这里的 load_from_parquet / load_layout /
#  compute_metrics / MaskedMSELoss 是从主脚本**逐行抄过来**的（只改了张量
#  维度顺序，见各自注释），不是"参考着重写"。
#
# ---------------------------------------------------------------- 数据形态
#
#  ★ 本脚本最重要的设计决定 ★
#
#  主脚本的 Mamba 模型吃的是 (T, N, L, 1+k) 的**逐节点窗口**：每个节点只看
#  自己 + 最近邻 top-k（k=5，real 模式下是邻接矩阵权重最高的 5 个）的历史。
#  它的输入里**不含全图信息**。
#
#  DCRNN / STGCN / Graph WaveNet 是**全网图模型**，按文献惯例应吃
#  (B, L, N, C) 的**全图历史**。本脚本采用文献惯例：
#
#      GNN 基线：输入 (B, 12, N, C) + 邻接矩阵 (N, N) → 输出 (B, 12, N)
#                C = 官方输入处方：DCRNN/GWNET 2（速度 + tod），STGCN 1
#      Mamba  ：输入 (T, N, 12, 1+5)（每节点只看到 top-5 邻居）
#
#  即：**基线能看到全部 325 / 207 个节点，Mamba 只能看到 top-5 邻居。**
#  这是对基线有利的不对等设置，是有意为之的保守选择 ——
#  如果 Mamba 在这种不对等条件下仍不落下风，"不需要显式空间建模"的结论才
#  站得住；如果输了，也必须如实报告，不能事后改口径说"我们本来就不公平"。
#
#  ⚠️ 写论文时必须把这条不对等明确写进"实验设置"，否则审稿人会当成隐瞒。
#
# ---------------------------------------------------------------- 协议
#
#  数据：数据集/{train,val,test}.parquet（PeMS-BAY，325 节点）
#        数据集/METR-LA/{train,val,test}.parquet（207 节点）
#  5 分钟粒度，输入 12 步（过去 1 小时），输出 12 步（未来 1 小时）。
#  HORIZONS = {15: 3, 30: 6, 60: 12} —— 分钟映射到步数。
#
#  ★ 布局是**时间优先**（node_id 循环 0..N-1）。列顺序 / 布局错了不会报错，
#    只会静默算出垃圾。所以两条自检必须保留：
#      1) parquet 第一块 node_id 必须等于 arange(N)
#      2) 窗口重叠 x_t[1:12] == x_{t+1}[0:11] 的平均绝对误差 < 1e-3
#
#  ★ 指标口径 = DCRNN 官方 null_val=0（掩零）：
#      真值为 0 的点视为缺失，不计入 MAE/RMSE/MAPE。
#      MAPE 在掩零基础上再要求真值 > MAPE_MIN（默认 1.0）——
#      真值 0.1 时单点比值能到几百，能把整体 MAPE 从 9% 抬到 400%。
#    PeMS-BAY 真值非零 100.0%（掩零几乎无影响），
#    METR-LA 真值非零 87.9%（12.1% 是缺失填的 0，差异巨大）。
#
#  ★ 标准化：按**训练集**的 per-node 均值 / 标准差做 z-score。
#    注意这与 DCRNN/GWNet 开源实现常用的"全数据集单一标量均值方差"不同。
#    这里跟主脚本对齐（per-node），代价是反标准化要用 (N,1,1) 广播；
#    好处是掩零阈值 (-mean_y/std_y) 也变成 per-node，缺失点判定更准。
#
# ---------------------------------------------------------------- 环境变量
#
#   PAPER_MODE           checkdata | train（默认）| selftest
#   PAPER_GNN_MODELS     DCRNN,STGCN,GWNET
#   PAPER_GNN_DATASETS   PeMS-BAY,METR-LA
#   PAPER_GNN_SEEDS      42,123,456（任务书 §3：≥3 个种子，报均值±标准差）
#   PAPER_GNN_EPOCHS     100
#   PAPER_GNN_BS         64
#   PAPER_STRICT_N       1（节点数不符时默认报错，同主脚本）
#   PAPER_GNN_SOURCE     parquet（默认）| npy | auto
#   PAPER_GNN_DEVICE     cuda | cpu（默认自动）
#   学习率 / 权重衰减 / 调度：**逐模型，全部取自上列三个官方仓库**，见下面
#   TRAIN_CFG 表（每行都写了出处文件和字段名）。可覆盖：
#     PAPER_GNN_LR_DCRNN   1e-2     PAPER_GNN_WD_DCRNN   0
#     PAPER_GNN_LR_STGCN   1e-3     PAPER_GNN_WD_STGCN   0
#     PAPER_GNN_LR_GWNET   1e-3     PAPER_GNN_WD_GWNET   1e-4
#   调度器（默认逐模型用文献值，可用 PAPER_GNN_SCHED 全局强制）：
#     PAPER_GNN_SCHED      auto（默认，按 TRAIN_CFG）| step | multistep | cosine | none
#     PAPER_GNN_LR_STEP    5      仅 SCHED=step 时用
#     PAPER_GNN_LR_GAMMA   0.7    仅 SCHED=step 时用
#   其余：
#     PAPER_GNN_PATIENCE   8      验证集 MAE 早停耐心（同主脚本）
#     PAPER_GNN_CLIP       1.0    梯度裁剪（同主脚本）
#     PAPER_GNN_DCRNN_SS   0      调度采样概率，0=关（默认）
#     PAPER_GNN_DCRNN_CHEB 0      0=论文的 K 跳扩散（默认），1=官方代码的 Chebyshev 递推
#     PAPER_GNN_DCRNN_LAYERS 2    DCGRU 堆叠层数（官方 num_rnn_layers=2）
#     PAPER_GNN_DCRNN_UNITS  64   隐状态维度（官方 rnn_units=64）
#     PAPER_GNN_DCRNN_K      2    扩散跳数（官方 max_diffusion_step=2）
#     PAPER_GNN_STGCN_KS / _KT    图卷积阶数 / 时间卷积核（官方默认均为 3）
#     PAPER_GNN_GW_ADAPT   1      1=启用自适应邻接（默认），0=只用固定转移矩阵
#
#  命令行也可以：python gnn_baselines.py --selftest / --checkdata / --train
#
# ---------------------------------------------------------------- 训练协议
#
#  与主脚本 train_one 一致的部分：
#    Adam + clip_grad_norm_(1.0) + 验证集 MAE 早停（PATIENCE=8）+ 恢复最优权重
#    随机种子：torch.manual_seed(seed) + np.random.seed(seed)
#    MaskedMSELoss 掩零损失，训练/评价口径统一
#    训练/验证/测试划分直接用同一份 parquet，**不重新划分**
#  与主脚本不同的部分（都是有意的，写论文要说明）：
#    学习率 / 调度器用各模型文献值（见上），不是主脚本的 5e-4 + cosine
#    验证集用**全量**（主脚本用前 1024 条）—— 图模型一次前向很便宜，
#      全量验证的早停信号噪声更小；可用 PAPER_GNN_VAL_N 改回
#
# ============================================================================

import os, sys, json, time, math, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认 GBK，打不出 ✓ ⚠️ 这类字符会直接抛 UnicodeEncodeError，
# 把真正的报错盖掉。改成 UTF-8（Linux / Kaggle 上本来就是 UTF-8，这行是空操作）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- 配置
MODE = os.environ.get('PAPER_MODE', 'train').strip()
WANT_MODELS = [s.strip().upper() for s in
               os.environ.get('PAPER_GNN_MODELS', 'DCRNN,STGCN,GWNET').split(',')
               if s.strip()]
WANT_DS = [s.strip() for s in
           os.environ.get('PAPER_GNN_DATASETS', 'PeMS-BAY,METR-LA').split(',')
           if s.strip()]
# 默认 3 个种子：任务书 §3 硬指标「≥3 个（42 / 123 / 456），报均值±标准差」。
# 必须与主脚本 paper_experiments 用同一批种子，否则基线表和主表不可比。
SEEDS = [int(s) for s in os.environ.get('PAPER_GNN_SEEDS', '42,123,456').split(',')
         if s.strip()]
EPOCHS = int(os.environ.get('PAPER_GNN_EPOCHS', 100))
BS = int(os.environ.get('PAPER_GNN_BS', 64))
PATIENCE = int(os.environ.get('PAPER_GNN_PATIENCE', 8))
CLIP = float(os.environ.get('PAPER_GNN_CLIP', 1.0))
VAL_N = int(os.environ.get('PAPER_GNN_VAL_N', 0))     # 0 = 全量验证集
SOURCE = os.environ.get('PAPER_GNN_SOURCE', 'parquet').strip().lower()
RESULT_FILE = os.environ.get('PAPER_GNN_RESULT', 'gnn_results.json')

# ★ 2026-09-26 新增：只重算指标、不训练（与 paper_experiments.PAPER_RECOMPUTE 同口径）。
#   背景：本文件的 compute_metrics 掩码原来判在**反标准化之后**，真值 0 经往返后
#   不再精确为 0，METR-LA 有 8.67% 的零值逃过掩码被当成有效点评分（任务书 §0.3）。
#   掩码已修；早停/选点走 quick_eval，其口径**本来就是对的** → 权重不必重训，
#   载入现成的 `_best.pt` 重算一遍即可。重算路径不碰任何训练语句、不覆写 .pt。
RECOMPUTE = os.environ.get('PAPER_GNN_RECOMPUTE', '0').strip() == '1'

# 节点数不符时是报错还是只警告。默认报错：2026-09-23 有一次 Kaggle 数据集里
# 只有 50 节点的文件，脚本静默回退到 N=50 把整个实验跑完了（2.8 小时白烧），
# 而且是在那段"前 50 个传感器几乎不连通"的图上跑的，结论根本不能用。
STRICT_N = bool(int(os.environ.get('PAPER_STRICT_N', 1)))

# ============================================================================
#  ★ 逐模型训练超参 —— 2026-09-24 逐条对着官方仓库核实过 ★
#
#  这张表是本次审查的产物。之前三个模型共用一个 StepLR(5, 0.7) + 统一 lr，
#  其中 GWNet 的 lr 是错的（写 1e-2，官方默认 1e-3，差 10 倍）。下面每一行
#  都标了出处文件与字段，改动前请先回原始仓库确认，不要凭印象改。
#
#  DCRNN  —— liyaguang/DCRNN, data/model/pretrained/METR-LA/config.yaml
#      train.base_lr: 0.01                → lr = 1e-2
#      train.lr_decay_ratio: 0.1
#      train.steps: [20, 30, 40, 50]      → ×0.1 于第 20/30/40/50 轮（MultiStep）
#      train.optimizer: adam
#      无 weight_decay 字段                     → wd = 0
#      train.max_grad_norm: 5             → 注意：本脚本全局 CLIP=1.0，
#                                            与官方 5 不同，见偏差清单
#      train.patience: 50, train.epoch: 100     → 本脚本 PATIENCE=8，见偏差清单
#      model.max_diffusion_step: 2        → K=2 ✓（已对齐）
#      model.num_rnn_layers: 2            → 本脚本默认 layers=1，见偏差清单
#      model.use_curriculum_learning: true      → 本脚本 SS_PROB=0，见偏差清单
#
#  STGCN  —— VeritasYin/STGCN_IJCAI-18, main.py + models/trainer.py
#      main.py  --lr   default=1e-3             → lr = 1e-3
#      main.py  --opt  default='RMSProp'！
#             官方默认优化器其实是 RMSProp，Adam 是同一分支里的备选
#             （trainer.py: if opt=='RMSProp' ... else AdamOptimizer）。
#             本脚本统一用 Adam（与另外两个模型一致），见偏差清单。
#      trainer.py: lr = exponential_decay(args.lr, global_steps,
#                     decay_steps=5*epoch_step, decay_rate=0.7, staircase=True)
#             → 每 5 轮 ×0.7，正是 StepLR(step_size=5, gamma=0.7) ✓（已对齐）
#      无 weight_decay 字段                     → wd = 0
#
#  GWNET  —— nnzhan/Graph-WaveNet, train.py
#      --learning_rate  default=0.001      → lr = 1e-3  ★本次修正（原为 1e-2）
#      --weight_decay   default=0.0001     → wd = 1e-4  ★本次修正（原为 1e-5）
#      学习率衰减：train.py 里那段
#          #lr = max(0.000002, args.learning_rate * (0.1 ** (i // 10)))
#      是**注释掉的**                        → 官方实际不衰减，sched = none
#      --dropout 0.3  ✓（已对齐）  --epochs 100 ✓  --batch_size 64 ✓
#      --in_dim  default=2：官方输入 = 速度 + **tod**（时刻），dow 默认关
#      ★ 2026-09-24 已按官方处方实现，逐仓库的代码证据见 ALL_COLS 后面那段注释
#        （原先这里写"本脚本 in_dim=1"，把三家一起砍成 1 通道，已作废）
#
#  另：三个官方仓库的**损失函数都是 masked MAE**（GWNet engine.py:
#  self.loss = util.masked_mae），本脚本统一用 masked MSE 以与主脚本口径
#  一致。这是有意的取舍（内部可比 > 逐篇复刻），但必须写进论文，见偏差清单。
# ============================================================================
# MambaST 三臂**共用同一份**优化器协议。写在同一个 dict 里再展开，是为了让
# "两臂只差支撑集"这件事在代码里也是结构性成立的，而不是靠人记得同步改两处。
_MAMBAST_PROTO = dict(lr_env='PAPER_GNN_LR_MAMBAST', lr=5e-4,
                      wd_env='PAPER_GNN_WD_MAMBAST', wd=1e-5, sched='cosine')
_MAMBAST_SRC = ('paper_experiments.py train_one: LR=5e-4, '
                'Adam(lr=5e-4, weight_decay=1e-5), '
                'CosineAnnealingLR(T_max=EPOCHS) —— 与主模型 V8 逐字同协议')

TRAIN_CFG = {
    # ★ 2026-09-25：MambaST 三臂。src 只是**说明文字**，优化器参数三者完全相同
    #  （selftest 里有断言：去掉 src 后三个 dict 必须逐键相等）。
    'MAMBAST': dict(_MAMBAST_PROTO,
                    src=_MAMBAST_SRC + '；本臂 = 图关（支撑集 I_N），'
                                       '与 MAMBASTG/MAMBASTN 参数量对照见 MAMBAST_ARMS'),
    'MAMBASTG': dict(_MAMBAST_PROTO,
                     src=_MAMBAST_SRC + '；本臂 = 图开（支撑集 A_f/A_b），'
                                        '与 MAMBAST 逐字节同协议、参数量逐位相同'),
    'MAMBASTN': dict(_MAMBAST_PROTO,
                     src=_MAMBAST_SRC + '；本臂 = 不建图模块（对照恒等臂是否够用）'),
    'DCRNN': dict(lr_env='PAPER_GNN_LR_DCRNN', lr=1e-2,
                  wd_env='PAPER_GNN_WD_DCRNN', wd=0.0,
                  sched='multistep', milestones=[20, 30, 40, 50], gamma=0.1,
                  src='DCRNN config.yaml: base_lr=0.01, lr_decay_ratio=0.1, '
                      'steps=[20,30,40,50], optimizer=adam'),
    'STGCN': dict(lr_env='PAPER_GNN_LR_STGCN', lr=1e-3,
                  wd_env='PAPER_GNN_WD_STGCN', wd=0.0,
                  sched='step', step=5, gamma=0.7,
                  src='STGCN main.py: --lr 1e-3; trainer.py: '
                      'exponential_decay(rate=0.7, every 5 epochs)'),
    'GWNET': dict(lr_env='PAPER_GNN_LR_GWNET', lr=1e-3,
                  wd_env='PAPER_GNN_WD_GWNET', wd=1e-4,
                  sched='none',
                  src='GWNet train.py: --learning_rate 0.001, '
                      '--weight_decay 0.0001, lr decay commented out'),
}
# PAPER_GNN_SCHED：auto = 用 TRAIN_CFG 的逐模型调度；其余值全局强制
SCHED = os.environ.get('PAPER_GNN_SCHED', 'auto').strip().lower()
LR_STEP = int(os.environ.get('PAPER_GNN_LR_STEP', 5))
LR_GAMMA = float(os.environ.get('PAPER_GNN_LR_GAMMA', 0.7))
SS_PROB = float(os.environ.get('PAPER_GNN_DCRNN_SS', 0))       # 调度采样，默认关
DCRNN_CHEB = bool(int(os.environ.get('PAPER_GNN_DCRNN_CHEB', 0)))
GW_ADAPT = bool(int(os.environ.get('PAPER_GNN_GW_ADAPT', 1)))

# ============================================================================
#  ★ 2026-09-25：MambaST —— Mamba 时序 + 图卷积空间的混合模型 ★
#
#  为什么加它：主模型 V8（paper_experiments.py 的 Predictor）在**结构上**用不了
#  邻居信息，不是调参问题。它的 forward 第一行就是 `x.reshape(B*N, L, C)`，
#  把节点维拍进 batch，之后每个节点是完全独立的序列，全文件没有一次跨节点运算；
#  邻居通道唯一的混合处是 `nn.Linear(7, 128)`（:730），而这个权重在 **325 个节点
#  间共享** —— 它能学"第 3 个输入通道整体多重要"，学不到"节点 47 的邻居 12
#  比邻居 31 更可信"。参数在结构上就不支持这件事。
#  实测印证：★ 2026-09-30 订正 —— 「real ≈ self、三臂打平」是**修块前**那一批（1.471 / 1.459）。
#  修好后（e100、全 7 通道容量对齐）real−self = +2.6922 / +0.9254 / −1.6774%（t=5.4/3.1/−1.9），
#  **符号三档翻转**、@60 反而是负的；real−random 三档都不显著。现行值见 `论文_表格_全精度.md` 表 3，
#  由 `cn_thesis_data.py --check` 对质。**上面那条机制论证不受影响**（它靠的是 forward 的形状，
#  不是这两个数），但**不许再拿这两个数当它的证据**。
#
#  这个模型要回答的**只有一个问题**：
#    PeMS-BAY @60 上相对 GWNet 的 +17.9% 差距，是不是因为没有空间算子？
#    ★ 2026-09-30：这一问当时（09-25）的输入是修块前的 +17.9%；现行是 **+13.3776%**（e100）。
#
#  ⚠️ 定位（务必读）：MambaST 是**空间机制的分析实验，不是新的主模型**。
#     它比 V8 多 131,328 个图卷积参数（口径见下），总量 ≈383k，
#     **比 GWNet 的 371,152 还大**。所以它不能顶替 V8 ——
#     论文"参数效率"的定位（SCI任务书.md §0.4：252,684 = GWNet 的 68%）
#     一个字不改，主模型仍是 V8。
#
#     让 MambaST 的结论与参数量无关的是这个设计：
#       MAMBAST（图关，支撑集 = I_N）与 MAMBASTG（图开，支撑集 = A_f/A_b）
#       **参数量逐位相同**，差异只有支撑集取值。所以 Δ 不可能来自容量，
#       只可能是"邻居信息 + 多跳加权"。审稿人"你参数更多所以更好"的质疑
#       在这对臂内部不成立。
# ============================================================================
MAMBA_DMODEL = int(os.environ.get('PAPER_GNN_MAMBA_DMODEL', 128))
MAMBA_DSTATE = int(os.environ.get('PAPER_GNN_MAMBA_DSTATE', 16))
MAMBA_LAYERS = int(os.environ.get('PAPER_GNN_MAMBA_LAYERS', 2))
MAMBA_ORDER = int(os.environ.get('PAPER_GNN_MAMBA_ORDER', 2))
# auto / official / naive。**不要指望它悄悄退回手写块** —— 显式要 official
# 而没装 mamba_ssm 时直接报错，否则本地与卡上跑的不是同一个网络。
MAMBA_BLOCK = os.environ.get('PAPER_GNN_MAMBA_BLOCK', 'auto').strip().lower()
# naive 扫描的显存护栏：deltaA 形状是 (B*N, L, d_inner, d_state)，
# bs=64 / 325 节点 / d_inner=256 / d_state=16 时**单个 deltaA 就是 4.09 GB**。
# 所以"按节点分块"不是优化，是必需项。0 = 不分块（跑官方核时用）。
MAMBA_SEQ_BUDGET = int(os.environ.get('PAPER_GNN_MAMBA_SEQ_BUDGET', 1600))

# 三臂 = 一个模型 + 三种支撑集取值。用来把"空间机制"这一个变量单独拎出来。
MAMBAST_ARMS = {
    'MAMBAST': 'identity',    # 图关：支撑集 I_N，逐节点通道混合，不跨节点
    'MAMBASTG': 'real',       # 图开：A_f = D^-1 A 与 A_b = A_f^T
    'MAMBASTN': 'none',       # 连图模块都不建（防"恒等对照很怪"的质疑），参数少 131,328
}

HORIZONS = {15: 3, 30: 6, 60: 12}     # 分钟 -> 步长（数据是 5 分钟粒度）
MAPE_MIN = float(os.environ.get('PAPER_MAPE_MIN', 1.0))  # MAPE 只统计真值 > 此值的点

DEVICE = os.environ.get('PAPER_GNN_DEVICE', '').strip() or \
    ('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集登记表。dirs 是目录候选，pats 是 npy 文件名优先级。
DATASETS = {
    'PeMS-BAY': dict(
        dirs=['数据集', '/kaggle/input/datasets/huangdecheng/ffff12',
              '/kaggle/input/ffff12'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=325, step_min=5),
    'METR-LA': dict(
        dirs=['数据集/METR-LA', '/kaggle/input/metrla'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=207, step_min=5),
    # 2026-09-28 加：第二个数据集（METR-LA 因管线级不复现弃用后补位）。
    # 数据由 build_pemsd4.py 从 ASTGCN 的 pems04.npz 生成，切分统一 7:1:2。
    # adj_mx.npy 是社区标准稀疏二值图（340 无向边 / 有效平均度 2.21 / 0 孤立节点），
    # 与 PeMS-BAY 的高斯核加权图（1184 边 / 7.29）是**两套不同处方**，论文必须披露。
    'PeMS04': dict(
        dirs=['数据集/PeMS04'],
        pats=['{s}_aligned.npy', '{s}_50.npy', '{s}.npy'],
        N=307, step_min=5),
}


def log(*a):
    print(*a, flush=True)


# ============================================================================
#  数据（从 paper_experiments.py 逐行抄来，语义必须完全一致）
# ============================================================================
# parquet 的列顺序必须与 rebuild_data.py 一致，否则 x/y 通道会错位。
X_COLS = [f'x_t-{11 - i}_d0' for i in range(11)] + ['x_t+0_d0']
Y_COLS = [f'y_t+{i + 1}_d0' for i in range(12)]
ALL_COLS = X_COLS + Y_COLS

# ============================================================================
#  ★ 2026-09-24：官方输入到底含不含时间协变量 —— 已回官方仓库查证 ★
#
#  之前这里写的是"官方输入含时间协变量（时刻/星期），本脚本 in_dim=1"，
#  把三家一起砍成 1 通道。查证结果**不是三家都一样**，而且连 tod/dow 都混了：
#
#   Graph WaveNet (nnzhan/Graph-WaveNet)
#     · generate_training_data.py L12-13 签名 add_time_in_day=True, add_day_in_week=False
#     · L28-40  把 tod 拼成第 1 个通道；L61-67 调用处 tod **硬编码 True**，dow 来自 --dow
#     · train.py L20  --in_dim  default=2   test.py L20 同
#     · util.py L144-158  只有第 0 通道做 z-score，**tod 保持 [0,1) 原值**
#     · 结论：官方默认 **2 通道（速度 + tod）**，dow 默认关
#   DCRNN (liyaguang/DCRNN)
#     · scripts/generate_training_data.py L29-40  同款 tod 拼接，L67-73 调用处
#       add_time_in_day=True, add_day_in_week=False **两个都写死**
#     · data/model/*.yaml 全部 input_dim: 2（含 bay / la / test / pretrained 四份）
#     · lib/utils.py L184-188  同样只标准化第 0 通道
#     · 结论：官方默认 **2 通道（速度 + tod）**，dow 关
#   STGCN (VeritasYin/STGCN_IJCAI-18)
#     · main.py L65  data_gen(...) 不传 C_0 → 用默认 1；全文无任何 tod/dow 代码
#     · models/trainer.py L30  placeholder [None, n_his+1, n, **1**] 通道维写死 1
#     · 结论：官方 **1 通道（只有速度）**，**不要给它加 tod** —— 加了就是非官方配置
#
#  tod 的算法两家逐字相同：`(t - 当天零点) / 86400` → 一天内的小数比例 [0,1)，
#  5 分钟粒度即 {0, 1/288, ..., 287/288}。**不是 sin/cos、不是 one-hot、不再归一化。**
#  ★ 这与 parquet 里的 `_d1` 列是同一个定义（实测 288 个唯一值、步进恰 1/288）
#    → 我们手上的 `_d1` 就是官方那个 tod 协变量，可以直接用，不需要自己造。
#
#  为什么这件事必须现在改：官方 SOTA 数字（GWNet 1.95 / DCRNN 2.07 等）都是
#  **带 tod 训出来的**。把基线砍成 1 通道再跟它们比，等于让基线替我们背锅，
#  偏差方向恰好朝有利于本文模型的一侧 —— 这正是"不能靠着错误硬编"要挡的。
# ============================================================================
TOD_COLS = [f'x_t-{11 - i}_d1' for i in range(11)] + ['x_t+0_d1']
# 1 = 按官方处方给（GWNet/DCRNN 带 tod，STGCN 不带）；0 = 三家统统砍成 1 通道
GNN_TOD = bool(int(os.environ.get('PAPER_GNN_TOD', 1)))

# 每个模型的输入通道数 = 官方处方。这是**基线保真度**的定义，不是可调超参。
OFFICIAL_IN_DIM = {
    'DCRNN': 2 if GNN_TOD else 1,     # config input_dim: 2
    'GWNET': 2 if GNN_TOD else 1,     # train.py --in_dim default=2
    'STGCN': 1,                       # trainer.py 通道维写死 1（与 GNN_TOD 无关）
    # ★ 2026-09-25：MambaST 三臂。注意这里的口径与主模型 V8 **不同**（V8 是
    #   1 + 5 邻居 + 1 tod = 7 通道），这是有意为之：本实验要隔离的变量是
    #   "有没有图算子"，支撑集是唯一允许变的量，所以输入里**不能再塞邻居通道**
    #   （塞了就变成第二个空间通路，ON/OFF 的差异就不干净了）。
    #   tod 通道照官方处方保留，与 GWNet/DCRNN 一致。
    'MAMBAST': 2 if GNN_TOD else 1,
    'MAMBASTG': 2 if GNN_TOD else 1,
    'MAMBASTN': 2 if GNN_TOD else 1,
}
# 到底要不要读 `_d1`：只要有一个模型需要它，就得读
USE_TOD = GNN_TOD


def find_split_file(d, split, pats):
    """按优先级找 npy 数据文件"""
    for pat in pats:
        p = os.path.join(d, pat.format(s=split))
        if os.path.exists(p):
            return p
    return None


def load_from_parquet(path, N, tag):
    """直接从 parquet 读成 (N, T, 24)，等价于 rebuild_data.py 的产物。

    分批读（iter_batches），峰值内存只有一个 batch，不是整张表。
    注意：返回前的 reshape/transpose 会再复制一份，所以单个划分的峰值
    约为文件解压后大小的 2 倍（PeMS-BAY train 约 2.3 GB），读下一个划分前
    调用方必须 del 掉上一个。
    """
    import pyarrow.parquet as pq          # 只在真要走这条路时才依赖 pyarrow
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    if n_total % N != 0:
        raise ValueError(f'{tag}: {n_total} 行不能被 N={N} 整除')
    T = n_total // N

    a = np.empty((n_total, 24), dtype=np.float32)
    # 时段通道与节点无关（与主脚本同一实测结论），所以每个时刻只留 node_id==0
    # 那一行 —— 存 (T, 12) 而不是 (N, T, 12)。全量 train 后者要 569 MB。
    tod = np.empty((T, 12), dtype=np.float32) if USE_TOD else None
    cols = ['node_id'] + ALL_COLS + (TOD_COLS if USE_TOD else [])
    off, seen, g1 = 0, False, 0
    for b in pf.iter_batches(batch_size=1 << 20, columns=cols):
        m = b.num_rows
        # 只抽查第一块：node_id 必须从 0 开始按 0..N-1 循环（= 时间优先布局）
        if not seen and m >= N:
            if not np.array_equal(np.asarray(b.column('node_id'))[:N], np.arange(N)):
                raise ValueError(
                    f'{tag}: node_id 开头不是 0..{N - 1}，parquet 不是时间优先布局，'
                    f'按 N={N} 解读会静默算出垃圾。')
            seen = True
        for k, c in enumerate(ALL_COLS):
            a[off:off + m, k] = np.asarray(b.column(c), dtype=np.float32)
        if USE_TOD:
            sel = np.nonzero(np.asarray(b.column('node_id')) == 0)[0]
            if len(sel):
                for j, c in enumerate(TOD_COLS):
                    tod[g1:g1 + len(sel), j] = np.asarray(
                        b.column(c), dtype=np.float32)[sel]
                g1 += len(sel)
        off += m
    if off != n_total:
        raise ValueError(f'{tag}: 只读到 {off} 行，metadata 说 {n_total}')
    if not seen:
        raise ValueError(f'{tag}: 没读到完整的一个时刻，N={N} 可能不对')
    if USE_TOD:
        # 自检：数量对不上、或取值跑出 [0,1)，说明读错列/读错行 —— 立刻报错，
        # 别让它变成"以为喂了官方 tod、其实喂的是垃圾"。
        if g1 != T:
            raise ValueError(f'{tag}: 时段通道取到 {g1} 个时刻，应为 {T}')
        if not (float(tod.min()) >= 0.0 and float(tod.max()) < 1.0):
            raise ValueError(
                f'{tag}: 时段通道取值 {float(tod.min()):.4f}~{float(tod.max()):.4f} '
                f'越界，应落在 [0,1)（官方 tod 就是这个区间）。可能读错了列。')

    d = np.ascontiguousarray(a.reshape(T, N, 24).transpose(1, 0, 2))
    del a
    # 同一把尺子：与 load_layout 完全相同的重叠自检
    ov = float(np.abs(d[:, :-1, 1:12] - d[:, 1:, 0:11]).mean())
    if not (ov < 1e-3):
        raise ValueError(f'{tag}: parquet 布局自检失败（重叠误差={ov:.4f}，应为 0）')
    return d, ov, tod


class TodUnavailable(Exception):
    """要用时段通道（官方 tod 协变量），但这份数据里没有 `_d1` 列。

    故意不继承 ValueError：load_dataset / _load_one_split 里有
    `except ValueError: continue`，继承的话会被吞掉，然后给出一个
    "所有候选都没通过布局自检"的错误结论 —— 那是误导。
    """


def load_layout(a, N, tag):
    """
    自适应两种布局，并强制自检。

    已知两种：
        (24, N*T)  时间优先，列号 = t*N + s
        (N*T, 24)  节点优先，行号 = s*T + t

    判据是滑动窗口的相邻重叠必须为 0：x_t[1:12] == x_{t+1}[0:11]
    用错朝向不会报错，只会静默算出垃圾，所以这里必须硬性拦下。
    """
    # ★ 2026-09-24：.npy 里只有 24 个 _d0 通道，没有 _d1 时段列。
    #   开着 tod 却走 npy 路径 = 基线被砍成 1 通道，但日志上看起来像
    #   "按官方处方跑了" —— 这种假实验比报错坏得多。必须拦住。
    if USE_TOD:
        raise TodUnavailable(
            f'{tag}: 已开启官方 tod 协变量（PAPER_GNN_TOD=1，默认），'
            f'而 .npy 里没有 _d1 列。\n'
            f'    时段列只能从 parquet 读。请设 PAPER_GNN_SOURCE=parquet，'
            f'或设 PAPER_GNN_TOD=0 明确放弃官方处方（须写进偏差清单）。')
    if a.shape[0] == 24:
        T = a.shape[1] // N
        d = a.T.reshape(T, N, 24).transpose(1, 0, 2).copy()
    elif a.shape[1] == 24:
        T = a.shape[0] // N
        d = a.reshape(N, T, 24).copy()
    else:
        raise ValueError(f'{tag}: shape {a.shape} 既不是 24 行也不是 24 列')
    ov = float(np.abs(d[:, :-1, 1:12] - d[:, 1:, 0:11]).mean())
    if not (ov < 1e-3):
        raise ValueError(
            f'{tag}: 布局自检失败（重叠误差={ov:.4f}，应为 0）。\n'
            f'  文件形状 {a.shape}，按 N={N} 解读。\n'
            f'  可能原因：(1) N 猜错了；(2) 这个 npy 的样本顺序被打乱过。\n'
            f'  对策：跑 rebuild_data.py 从 parquet 重建 *_aligned.npy，'
            f'或设 PAPER_GNN_SOURCE=parquet 直读 parquet。')
    # 第三个返回值是时段通道。npy 路径没有 → None（上面已拦下 USE_TOD 的情况）
    return d, ov, None


class NodeCountMismatch(Exception):
    """声明的节点数在数据里读不通，且 STRICT_N 要求报错。

    故意不继承 ValueError：load_dataset 内部用 `except ValueError: continue`
    来跳过读不通的候选 N，继承的话会被吞掉。
    """


def _has_any_train(d, cfg):
    return bool(find_split_file(d, 'train', cfg['pats'])
                or os.path.exists(os.path.join(d, 'train.parquet')))


def _discover_kaggle_dirs(cfg):
    """在 /kaggle/input 下自己找可能的数据集目录（目录名由上传者定，写死容易对不上）。"""
    root = '/kaggle/input'
    if not os.path.isdir(root):
        return []
    aligned, parq, fallback = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if not x.startswith('.')]
        if not _has_any_train(dirpath, cfg):
            continue
        if 'train.parquet' in filenames:
            parq.append(dirpath)
        elif 'train_aligned.npy' in filenames:
            aligned.append(dirpath)
        else:
            fallback.append(dirpath)
    # parquet 就是 ground truth，排最前
    return parq + aligned + fallback


def _load_one_split(d, s, N, pats, source):
    """按 source 策略读一个划分，返回 (d_array, ov, 用的文件名, tod)。

    tod 是 (T,12) 的时段通道或 None。开着官方 tod 处方时只能从 parquet 读，
    所以这里把 parquet 排到最前 —— 否则 source=auto 会先拿 npy，然后
    load_layout 抛 TodUnavailable，一份本来读得通的 parquet 白白浪费。
    """
    fpq = os.path.join(d, f'{s}.parquet')
    fnp = find_split_file(d, s, pats)

    if source == 'parquet' or (USE_TOD and os.path.exists(fpq)):
        order = [('pq', fpq), ('npy', fnp)]
    elif source == 'npy':
        order = [('npy', fnp)]
    else:                                   # auto = 主脚本的顺序：npy 优先
        order = [('npy', fnp), ('pq', fpq)]

    err = None
    for kind, path in order:
        if not path or not os.path.exists(path):
            continue
        try:
            if kind == 'pq':
                dd, ov, td = load_from_parquet(path, N, f'{s}.parquet')
            else:
                # astype(copy=False)：已经是 float32 时不再复制一份（npy 有 1.1 GB）
                arr = np.load(path).astype(np.float32, copy=False)
                # USE_TOD 时这里抛 TodUnavailable —— 不在 except 的捕获列表里，
                # 会直接冒到用户面前（要的就是这个）。
                dd, ov, td = load_layout(arr, N, os.path.basename(path))
                del arr
            return dd, ov, os.path.basename(path), td
        except (ValueError, OSError, FileNotFoundError) as e:
            err = e
            continue
    raise FileNotFoundError(err if err is not None else f'{d}/{s} 找不到')


def load_dataset(name):
    """返回 (train, val, test, adj, N, tods)。每份形状 (N, T, 24)。语义同主脚本。

    tods 是 [train_tod, val_tod, test_tod]（各 (T,12)）或 None。
    """
    cfg = DATASETS[name]

    cands = [c for c in cfg['dirs']
             if os.path.exists(c) and _has_any_train(c, cfg)]
    for extra in _discover_kaggle_dirs(cfg):
        if extra not in cands:
            cands.append(extra)
    if not cands:
        raise FileNotFoundError(
            f'{name}: 找不到数据。找过 {cfg["dirs"]}，文件名优先级 {cfg["pats"]}，'
            f'/kaggle/input 下也没扫到。')

    # 两轮：先用【声明的 N】试遍所有候选目录；全试完都不行，再依次试别的 N。
    cand_N = [cfg['N']] + [n for n in (50, 207, 325) if n != cfg['N']]
    out = None
    for k, N in enumerate(cand_N):
        for d in cands:
            try:
                splits, ovs, used, tods = [], [], [], []
                for s in ['train', 'val', 'test']:
                    dd, ov, nm, td = _load_one_split(d, s, N, cfg['pats'], SOURCE)
                    splits.append(dd)
                    ovs.append(ov)
                    used.append(nm)
                    tods.append(td)
                if any(t is not None for t in tods) and any(t is None for t in tods):
                    raise TodUnavailable(
                        f'{name}: train/val/test 里只有一部分能取到时段通道 '
                        f'({[None if t is None else t.shape for t in tods]})。\n'
                        f'    多半是有的划分走 parquet、有的走 npy —— 混合来源不可信，'
                        f'先统一 PAPER_GNN_SOURCE。')
            except (ValueError, OSError, FileNotFoundError):
                continue          # 这个目录按这个 N 读不通，换下一个
            log(f'  {name}: 目录 {d}')
            log(f'    文件 {used[0]}  N={N}  ' + '  '.join(
                f'{s}={t.shape[1]:,}(自检{ov:.1e})' for s, t, ov in
                zip(['train', 'val', 'test'], splits, ovs)))
            if k > 0:
                # 节点数不符是配置错误，默认直接报错，别让它悄悄降级。
                # 注意这句 raise 在 try 之外，否则会被上面的 except 吞掉。
                msg = (f'{name}: 声明的是 N={cfg["N"]}，实际只有 N={N} 的文件读得通 '
                       f'（用的是 {used[0]}）。\n'
                       f'    节点数不对，"全量节点"的结论不成立。'
                       f'确认要按 N={N} 跑就设 PAPER_STRICT_N=0。')
                if STRICT_N:
                    raise NodeCountMismatch(msg)
                log(f'    ⚠️  {msg}')
            out = (splits, N, d, tods if any(t is not None for t in tods) else None)
            break
        if out is not None:
            break
    if out is None:
        raise ValueError(
            f'{name}: 候选目录 {cands} × 候选 N={cand_N} 都没通过布局自检。')

    splits, N, d, tods = out
    # 邻接矩阵
    adj = None
    for cand in [os.path.join(d, 'adj_mx.npy'),
                 os.path.join(d, 'sensor_graph', 'adj_mx.npy'),
                 os.path.join(d, 'adj_50.npy')]:
        if os.path.exists(cand):
            adj = np.load(cand).astype(np.float64)
            break
    if adj is None:
        raise FileNotFoundError(f'{name}: 找不到 adj_mx.npy')
    if adj.shape[0] != N:
        adj = adj[:N, :N]          # adj 是全量节点、数据是前 N 个，取子阵
    np.fill_diagonal(adj, 0)
    return splits[0], splits[1], splits[2], adj, N, tods


def assert_window_overlap(X, tag, n_probe=512):
    """在**最终喂进模型的那张张量**上再验一次窗口重叠不变式。

    同一把尺子（load_layout 用的是 x_t[1:12] == x_{t+1}[0:11]，误差 < 1e-3），
    只是换了位置：这一步能抓到 prepare_split / transpose 把 (N,T,12,1) 拼错的
    情况 —— 那种错误在形状上完全合法，只有数值不变式能发现。
    只抽前 n_probe 个窗口：全量比较会额外造一个 1.3 GB 的临时数组。
    """
    m = min(n_probe, X.shape[0] - 1)
    if m <= 0:
        return 0.0
    # X 是 (T, L, N, 1)：时间步在 dim1、节点在 dim2
    ov = float(np.abs(X[:m, 1:12, :, 0] - X[1:m + 1, 0:11, :, 0]).mean())
    if not (ov < 1e-3):
        raise ValueError(
            f'{tag}: 采样得到的输入张量窗口重叠误差={ov:.4f}（应为 ~1e-7）。\n'
            f'   形状 {X.shape}。说明 prepare_split 的维度变换错了 —— '
            f'形状对不代表语义对，喂进模型的就是垃圾。')
    return ov


def prepare_split(d24, mean_x, std_x, mean_y, std_y, tag, tod=None, in_dim=1):
    """(N, T, 24) -> ((T,12,N,C) 输入, (T,12,N) 目标)，原地标准化省一份内存。

    in_dim / tod：**官方输入处方**。C = in_dim：
        in_dim=1 → 只有速度（STGCN 官方如此）
        in_dim=2 → 通道 0 = 标准化速度，通道 1 = **原值 tod**（GWNet/DCRNN 官方如此）
    官方两个仓库的 tod 都**不做任何归一化**（`lib/utils.py` / `util.py` 只对
    第 0 通道做 z-score），所以这里绝不能顺手把 tod 也标准化 —— 那就不是
    官方那个协变量了，基线的输入分布会跟文献里的对不上。

    ★ 维度顺序就是这里定的：GNN 基线吃 (B, L, N, C) 全图历史。
      主脚本的 make_samples 产出的是 (T, N, L, 1+k) 逐节点窗口，两者的
      语义完全不同，不能混用（见文件头）。
    ★ 原地（-=, /=）而不是新开数组：全量 train 的 (N,T,24) 有 1.1 GB，
      再复制两份就是 3.4 GB，这台机器 16 GB 内存会很难受。
      代价是调用方不能再复用 d24 —— 所以 main 里每个划分都是"建完就 del"。
    """
    N, T, _ = d24.shape
    x = d24[:, :, :12]
    y = d24[:, :, 12:]
    # ★ 2026-09-24：与主脚本同一口径 —— x 里的缺测（原始值 0）先填成该节点均值再标准化，
    #   填完归一化恰好是 0（「无信息」），而不是「该路段瞬间瘫痪」。
    #   **y 不填**：它的 0 是掩零口径的锚点（compute_metrics 里 m_null = tt > 0），
    #   填成均值后归一化是 0，会被判成有效点，等于把缺失当真值评。
    #   np.copyto 而非 np.where：x 是 d24 的视图，要原地写回，不能新开数组
    #   （全量 train 的 (N,T,24) 有 1.1 GB，复制两份这台 16 GB 机器会很难受）。
    np.copyto(x, mean_x, where=(x <= 0))
    x -= mean_x
    x /= std_x
    y -= mean_y
    y /= std_y
    # (N, T, 12) -> (T, 12, N) -> 补通道维 -> (T, 12, N, 1)
    # ★ 必须是 (T, L, N, C)，不能顺手写成主脚本那种 (T, N, L, C)：
    #   三个模型的内部都按 dim1 = 时间、dim2 = 节点来切分/换位，写反了
    #   形状照样合法（N 和 L 都是整数），但时间轴和节点轴被整体对调，
    #   模型学的是"同一时刻的 12 个邻居"而不是"同一节点的 12 个历史步"。
    #   这种错不会报错，只会得到一条躺平的损失曲线。
    #   本脚本第一版就是错的，靠 assert_window_overlap 上的重叠不变式抓出来。
    Xs = np.ascontiguousarray(x.transpose(1, 2, 0))      # (T, 12, N)
    if in_dim == 1:
        X = Xs[..., None]                                # 视图，不额外复制
    else:
        if tod is None:
            raise TodUnavailable(
                f'{tag}: 要 {in_dim} 通道（官方处方含 tod），但没拿到时段通道。\n'
                f'    检查 PAPER_GNN_TOD 与数据是否走 parquet。')
        if tod.shape[0] != Xs.shape[0]:
            raise ValueError(
                f'{tag}: 时段通道有 {tod.shape[0]} 个样本，速度有 {Xs.shape[0]} 个 —— '
                f'两者不同源，通道会错位（错位比不加特征更坏）。')
        if tod.shape[1] != Xs.shape[1]:
            raise ValueError(f'{tag}: 时段通道宽 {tod.shape[1]}，应为 {Xs.shape[1]}')
        X = np.empty(Xs.shape + (in_dim,), dtype=np.float32)
        X[..., 0] = Xs
        # ★ 布局是 (T, L, N, C)：dim1 是时刻、dim2 是节点（见本函数上面那段警告）。
        #   所以 tod 要取 (T, 12, 1) 沿 N 广播 —— 写成主脚本那种 (T,1,12) 会被
        #   当成"沿 12 个时刻广播"，要么报错要么（N 恰好等于 12 时）静默喂错。
        X[..., 1] = tod[:, :, None]                      # (T,12,1) 广播到 N 个节点
        del Xs
    Y = np.ascontiguousarray(y.transpose(1, 2, 0))
    ov = assert_window_overlap(X, tag)
    log(f'  [{tag}] 输入张量窗口重叠自检={ov:.2e}（应≈0）  '
        f'X{X.shape} Y{Y.shape}')
    return X, Y


# ============================================================================
#  指标（从主脚本逐行抄来，只改了张量维度顺序）
# ============================================================================
def compute_metrics(pred, true, std_y, mean_y, N):
    """
    ★ 与 paper_experiments.compute_metrics 逐行等价，唯一区别：
      主脚本的 pred/true 是 (T, N, 12)（逐节点窗口），本脚本是 (T, 12, N)
      （全图窗口，B 和 L 在前）。索引从 [:, :, idx-1] 改成 [:, idx-1, :]，
      广播 reshape 从 (1,N,1) 改成 (1,1,N)。数学完全一样。

      掩零口径 = DCRNN 官方的 null_val=0：真值为 0 的点不计入指标。
      MAPE 在掩零基础上再要求真值 > MAPE_MIN（真值 0.1 时单点比值能到几百）。
      未掩零的值保留为 *_all 供审稿人核对，论文表里用掩零值。
    """
    sd = std_y.reshape(1, 1, N)
    mn = mean_y.reshape(1, 1, N)
    p = pred * sd + mn
    t = true * sd + mn
    # ★★ 2026-09-26 修：与 paper_experiments.compute_metrics 同步修 ★★
    #   原来 `m_null = tt > 0`（反标准化后的真值）。真值 0 往返后不精确为 0：
    #   METR-LA 有 8.67% 的零值变成 +1e-15 而逃过掩码、被当成有效点评（单点误差
    #   ~56 mph），持久性 @60 被抬 5.3634 → 5.9624，且随浮点舍入浮动不可复现。
    #   PeMS-BAY 逃逸 0.0006%，可忽略。改判在标准化空间（= MaskedMSELoss 的口径）。
    null_norm = ((-mean_y) / std_y).reshape(1, N)
    res = {}
    for hmin, idx in HORIZONS.items():
        pp, tt = p[:, idx - 1, :], t[:, idx - 1, :]
        d = pp - tt
        key = str(hmin)
        m_all = np.ones_like(tt, dtype=bool)
        m_null = true[:, idx - 1, :] > null_norm   # ← 标准化空间，精确
        if not m_null.any():
            m_null = m_all
        res[f'mae_{key}'] = float(np.abs(d[m_null]).mean())
        res[f'rmse_{key}'] = float(np.sqrt((d[m_null] ** 2).mean()))
        mm = m_null & (tt > MAPE_MIN)
        res[f'mape_{key}'] = float(np.abs(d[mm] / tt[mm]).mean() * 100) \
            if mm.any() else float('nan')
        res[f'mape_raw_{key}'] = float(np.abs(d / np.maximum(np.abs(tt), 1e-3)).mean() * 100)
        res[f'mape_cov_{key}'] = float(mm.mean())
        res[f'mae_all_{key}'] = float(np.abs(d[m_all]).mean())
        res[f'rmse_all_{key}'] = float(np.sqrt((d[m_all] ** 2).mean()))
        res[f'null_frac_{key}'] = float(1.0 - m_null.mean())
    res['n_eval'] = int(pred.shape[0])
    return res


class MaskedMSELoss(nn.Module):
    """DCRNN 口径的掩码 MSE：真值为 0（缺失）的点不计入损失。

    ★ 与主脚本 MaskedMSELoss 的区别（这是个容易静默搞错的地方）：
      主脚本的 pred/target 是 (B, N, 12)，null 阈值广播成 (1, N, 1)；
      本脚本是 (B, 12, N)，必须广播成 (1, 1, N)。
      搞错了不会报错 —— (1, N, 1) 与 (B, 12, N) 在 N=B 时也能广播成功，
      但掩的是完全错的点。所以 selftest 里专门做了掩码损失的数值自检。
    """

    def __init__(self, null_norm):
        super().__init__()
        v = np.asarray(null_norm, dtype=np.float32).reshape(-1)
        self.register_buffer('null_norm', torch.as_tensor(v).reshape(1, 1, -1))

    def forward(self, pred, target):
        m = (target > self.null_norm).float()
        n = m.sum()
        if n < 1:
            return F.mse_loss(pred, target)
        return ((pred - target) ** 2 * m).sum() / n


@torch.no_grad()
def infer(model, X, bs):
    model.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def quick_eval(model, X, Y, std_y, mean_y, N, bs):
    """验证集 MAE（掩零口径），早停用。口径与 compute_metrics 一致。"""
    if X.shape[0] == 0:
        return float('inf'), None
    pred = infer(model, X, bs)
    true = Y[:pred.shape[0]]
    if not np.isfinite(pred).all():
        return float('inf'), pred
    d = (pred - true) * std_y.reshape(1, 1, N)     # 均值在差分里抵消
    m = true > ((-mean_y) / std_y).reshape(1, 1, N)
    if not m.any():
        m = np.ones_like(true, dtype=bool)
    return float(np.abs(d[m]).mean()), pred


# ============================================================================
#  模型 1：DCRNN（Li et al., ICLR 2018）
# ============================================================================
def build_dcrnn_supports(adj):
    """双向随机游走转移矩阵，与 DCRNN 官方 _calculate_random_walk_matrix 一致。

    官方写法：
        supp[0] = rw(adj).T
        supp[1] = rw(adj.T).T
    度数 0 的孤立节点：d_inv 置 0（不是 1），否则那一行全是 inf。
    PeMS-BAY 邻接矩阵稀疏度 97.8%，孤立节点是真实存在的。
    """
    def rw(m):
        d = m.sum(axis=1)
        d_inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0)
        return d_inv[:, None] * m
    return [rw(adj).T, rw(adj.T).T]


class DiffusionConv(nn.Module):
    """扩散图卷积：sum_{s in 双向} sum_{k<K} (S_s^k X) W_{s,k}

    ★ 一个必须知道的文献出入 ★
      DCRNN 论文（式 1-2）是 K 跳幂次扩散：x_k = S @ x_{k-1}。
      但官方**放出的代码**（dcrnn_cell.py 的 _gconv）里写的是
          x2 = 2 * S @ x1 - x0
      这是 Chebyshev 递推，不是幂次扩散 —— 即每一项不是 S^k x 而是 T_k(S) x。
      两者在 K=2 时不同（项数 2 vs 3，函数形式也不同）。
      默认按**论文**实现（PAPER_GNN_DCRNN_CHEB=0）；想复现官方代码的数字
      就设 PAPER_GNN_DCRNN_CHEB=1。写论文时必须注明用了哪一个。
    """

    def __init__(self, supports, in_dim, out_dim, K, cheb=False, bias=True):
        super().__init__()
        self.K = max(1, int(K))
        self.cheb = bool(cheb)
        n_sup = len(supports)
        # 论文版：K 项（x, Sx, ..., S^{K-1}x）；官方版：1 + K 项（x, Sx, 2S(Sx)-x, ...）
        self.n_terms = (1 + self.K) if self.cheb else self.K
        self.n_sup = n_sup
        self.weight = nn.Parameter(
            torch.empty(n_sup, self.n_terms, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        # supports 不参与训练，但必须跟着 .to(device) 走
        for i, s in enumerate(supports):
            self.register_buffer(f'S{i}', torch.as_tensor(s, dtype=torch.float32))

    def _terms(self, x, S):
        """返回 [T_0 x, T_1 x, ...]，长度 = self.n_terms"""
        if self.cheb:
            x0 = x
            x1 = torch.einsum('nm,bmc->bnc', S, x0)
            out = [x0, x1]
            for _ in range(2, self.K + 1):
                x2 = 2.0 * torch.einsum('nm,bmc->bnc', S, x1) - x0
                out.append(x2)
                x0, x1 = x1, x2
            return out
        out, cur = [x], x
        for _ in range(1, self.K):
            cur = torch.einsum('nm,bmc->bnc', S, cur)
            out.append(cur)
        return out

    def forward(self, x):
        """x: (B, N, C_in) -> (B, N, C_out)"""
        out = None
        for i in range(self.n_sup):
            S = getattr(self, f'S{i}')
            for j, term in enumerate(self._terms(x, S)):
                w = self.weight[i, j]       # (C_in, C_out)
                y = torch.einsum('bnc,cd->bnd', term, w)
                out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """DCRNN 的 DCGRU。门控写法照官方代码：

        v = sigmoid(conv([x, h]))       # bias_start=1.0，初始偏向"保留状态"
        r, u = split(v, 2)
        c = tanh(conv([x, r*h]))
        h' = u * h + (1-u) * c          # ← 官方是 u*state + (1-u)*candidate

    注意最后一行与 PyTorch nn.GRU 的 h' = (1-u)*h + u*c 方向相反，
    这里按 DCRNN 官方写，不要"顺手改成标准 GRU"。
    """

    def __init__(self, supports, num_units, in_dim, K, cheb=False):
        super().__init__()
        self.num_units = num_units
        self.conv_ru = DiffusionConv(supports, in_dim + num_units, 2 * num_units,
                                     K, cheb=cheb, bias=True)
        self.conv_c = DiffusionConv(supports, in_dim + num_units, num_units,
                                    K, cheb=cheb, bias=True)
        # bias_start=1.0（官方）—— 让初始 update gate 偏向"保留状态"
        with torch.no_grad():
            self.conv_ru.bias.fill_(1.0)

    def forward(self, inputs, state):
        v = torch.sigmoid(self.conv_ru(torch.cat([inputs, state], dim=-1)))
        r, u = torch.split(v, self.num_units, dim=-1)
        c = torch.tanh(self.conv_c(torch.cat([inputs, r * state], dim=-1)))
        return u * state + (1.0 - u) * c


class DCRNN(nn.Module):
    """seq2seq：编码器吃 12 步历史，解码器自回归吐 12 步未来。

    编码器 steps = 12（全部历史步）；解码器第一步的输入是**零向量**
    （官方代码里的 go symbol，不是"最后一个观测值"）。
    训练时解码器逐步喂真值（teacher forcing）；ss_prob > 0 时以该概率改喂
    上一步自己的预测（detach），即 DCRNN 原版的 scheduled sampling，
    默认关（ss_prob=0 = 全程 teacher forcing）。
    """

    def __init__(self, adj, N, num_units=64, K=2, layers=2, out_len=12,
                 cheb=False, ss_prob=0.0, in_dim=1):
        """layers 默认 2：官方 METR-LA config 的 model.num_rnn_layers=2。

        ⚠️ 2026-09-24 修正：此前 forward 里写死 enc_cells[0]，
           layers>1 会被**静默忽略**（参数建了、前向不用、也不报错）。
           现在按官方结构真正堆叠，编码器/解码器各 layers 层。
        """
        super().__init__()
        self.N = N
        self.out_len = out_len
        self.layers = layers
        self.ss_prob = ss_prob
        supports = build_dcrnn_supports(adj)
        self.enc_cells = nn.ModuleList()
        self.dec_cells = nn.ModuleList()
        for i in range(layers):
            # 第 0 层吃原始输入，其余层吃上一层的隐状态
            # ★ 2026-09-24：编码器第 0 层吃 in_dim 维（官方 input_dim: 2 =
            #   速度 + tod）。**解码器第 0 层恒为 1**，因为官方解码器每步吃的是
            #   "上一时刻的目标值 / go symbol"，output_dim=1，时间协变量只进编码器。
            #   这里跟着官方走，不把 tod 也塞进解码器。
            in_enc = in_dim if i == 0 else num_units
            in_dec = 1 if i == 0 else num_units
            self.enc_cells.append(DCGRUCell(supports, num_units, in_enc, K, cheb=cheb))
            self.dec_cells.append(DCGRUCell(supports, num_units, in_dec, K, cheb=cheb))
        self.out_proj = nn.Linear(num_units, 1)

    def forward(self, x, teacher=None):
        """x: (B, 12, N, C)  ->  (B, 12, N)   C = in_dim（官方 2 = 速度 + tod）

        ★ tod 只进编码器（第 0 层 in_enc=in_dim）；解码器恒吃 1 维目标值，
          与官方 output_dim=1 一致。见 __init__ 里的注释。

        teacher（= 标准化空间的真值序列 (B,12,N)）只在训练时给，用于 teacher
        forcing / scheduled sampling；推理时传 None，解码器完全自回归。
        """
        B, L, N, C = x.shape
        seq = [x[:, t] for t in range(L)]           # 12 个 (B, N, 1)

        # ---- 编码器：全部 12 步历史，逐层堆叠
        h = [torch.zeros(B, N, c.num_units, device=x.device, dtype=x.dtype)
             for c in self.enc_cells]
        for xt in seq:
            inp = xt
            for l, cell in enumerate(self.enc_cells):
                h[l] = cell(inp, h[l])
                inp = h[l]
        state = h                                  # 每层一个隐状态

        # ---- 解码器：第一步输入是零（go symbol），其后自回归
        outs = []
        for t in range(self.out_len):
            if t == 0:
                dec_in = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
            else:
                if teacher is not None and self.training:
                    # teacher forcing：解码器第 t 步的输入是第 t-1 步的**真值**。
                    # ss_prob = 0（默认，= 不做 curriculum learning）时全程 teacher
                    # forcing，这正是 DCRNN 原版的训练方式；
                    # ss_prob = p 时以概率 p 改喂上一步自己的预测（detach，不让
                    # 梯度穿回上一步）。
                    # ★ 必须 unsqueeze(-1)：teacher 是 (B,12,N)（和输出同一布局），
                    #   而 cell 要的是 (B,N,1)。少了这一步，(B,N) 会在 GRU 内部
                    #   被当成 (B,N) 参与 concat 而 shape 报错 —— 或者更糟，
                    #   在 N 恰好等于某个维度时静默广播。
                    dec_in = teacher[:, t - 1].unsqueeze(-1)
                    if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                        dec_in = outs[-1].detach()
                else:
                    dec_in = outs[-1].detach() if self.training else outs[-1]
            inp = dec_in
            for l, cell in enumerate(self.dec_cells):
                state[l] = cell(inp, state[l])
                inp = state[l]
            outs.append(self.out_proj(state[-1]))    # 取最上层
        return torch.stack(outs, dim=1).squeeze(-1)  # (B, 12, N)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
#  模型 2：STGCN（Yu et al., IJCAI 2018）
# ============================================================================
def build_cheb_supports(adj, K):
    """Chebyshev 多项式基 T_k(L_tilde)，L_tilde = 2L/λmax - I。

    L = I - D^{-1/2} (A+I) D^{-1/2}（带自环的归一化拉普拉斯）。
    λmax 用 numpy 精确算（325 个节点，eigvalsh 毫秒级），
    不猜"归一化拉普拉斯 λmax ≈ 2" —— 那条界只在无自环时成立，
    加了自环后 λmax 会变，猜错等于换了滤波器的频率刻度。
    """
    N = adj.shape[0]
    A = adj + np.eye(N, dtype=np.float64)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = np.eye(N) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    lam_max = float(np.linalg.eigvalsh(L).max())
    lam_max = max(lam_max, 1e-6)
    Lt = 2.0 * L / lam_max - np.eye(N)
    supports = [np.eye(N), Lt]
    for k in range(2, K):
        supports.append(2.0 * Lt.dot(supports[k - 1]) - supports[k - 2])
    return [s.astype(np.float32) for s in supports[:K]]


class ChebConv(nn.Module):
    """Chebyshev 图卷积（K=3）。x: (B, C_in, T, N) -> (B, C_out, T, N)"""

    def __init__(self, K, c_in, c_out, supports):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, c_in, c_out))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(c_out))
        for i, s in enumerate(supports):
            self.register_buffer(f'T{i}', torch.as_tensor(s, dtype=torch.float32))

    def forward(self, x):
        out = None
        for k in range(self.K):
            Tk = getattr(self, f'T{k}')
            xk = torch.einsum('nm,bctm->bctn', Tk, x)      # 沿节点维做图卷积
            y = torch.einsum('bctn,co->botn', xk, self.weight[k])
            out = y if out is None else out + y
        return out + self.bias.view(1, -1, 1, 1)


class TemporalConvLayer(nn.Module):
    """GLU 时间卷积（原论文 Kt=3）。

    原版还带一条恒等/对齐支路（c_in != c_out 时先 1x1 卷积对齐通道），
    这里照搬。⚠️ 官方对 c_in > c_out 的情况用的是"因果卷积降维"而不是 1x1，
    本实现统一成 1x1 + BN（见报告"不确定的地方"）。
    """

    def __init__(self, Kt, c_in, c_out, N):
        super().__init__()
        self.Kt = Kt
        self.c_in, self.c_out = c_in, c_out
        # 两侧各 pad (Kt-1)//2：时间维不变（Kt=3 时 pad 1）
        self.conv = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), padding=((Kt - 1) // 2, 0))
        if c_in != c_out:
            self.align = nn.Sequential(nn.Conv2d(c_in, c_out, (1, 1)),
                                       nn.BatchNorm2d(c_out))
        else:
            self.align = None

    def forward(self, x):
        res = self.align(x) if self.align is not None else x
        h = self.conv(x)
        a, b = torch.split(h, self.c_out, dim=1)
        return a * torch.sigmoid(b) + res


class STConvBlock(nn.Module):
    """一个 ST-Conv block：GLU 时间卷积 → Chebyshev 图卷积 → GLU 时间卷积 → LN"""

    def __init__(self, Ks, Kt, N, c_in, c_hid, c_out, supports, dropout=0.0):
        super().__init__()
        self.tmp1 = TemporalConvLayer(Kt, c_in, c_hid, N)
        self.gconv = ChebConv(Ks, c_hid, c_hid, supports)
        self.tmp2 = TemporalConvLayer(Kt, c_hid, c_out, N)
        self.ln = nn.LayerNorm([N, c_out])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tmp1(x)
        x = self.gconv(x)
        x = self.tmp2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.drop(x)


class STGCN(nn.Module):
    """两个 ST-Conv block + 输出层。

    ★ 必须写进注释的改动 ★
      原论文是为**单步**预测设计的：最后的输出层是 Conv2d(c, 1, (T, 1))，
      把整段历史时间维压成 1 个数（T 是输入步数 12）。
      这里要做 12 步预测，于是把输出层改成 Conv2d(c_out, 12, (12, 1))：
      输出通道变成 12 个 horizon，时间维同样被压成 1。
      也就是说 STGCN 的"多步"不是真的逐步预测，而是让最后这层卷积
      在时间维上一次性学出 12 个 horizon 的线性组合 ——
      这正是"STGCN 不适合多步预测"这一已知局限的直接体现，
      也是它在 @60 上大概率吃亏的结构性原因。

    ⚠️ 参数量与 N 有关：STConvBlock 里的 LayerNorm 作用在 [N, c_out] 上
      （原论文如此），所以它带 2*N*c_out 个可学参数。N=20 → 38,460；
      N=325 → 116,540。跨数据集比参数量时要意识到这一点，
      别把"STGCN 参数量随数据集变大"当成 bug。
    """

    def __init__(self, adj, N, Ks=3, Kt=3, out_len=12,
                 blocks=((1, 16, 64), (64, 16, 64)), dropout=0.0):
        super().__init__()
        self.N = N
        self.out_len = out_len
        supports = build_cheb_supports(adj, Ks)
        self.st_blocks = nn.ModuleList()
        c_last = blocks[0][0]
        for (c_in, c_hid, c_out) in blocks:
            self.st_blocks.append(
                STConvBlock(Ks, Kt, N, c_in, c_hid, c_out, supports, dropout))
            c_last = c_out
        # 时间维 12 -> 1，通道维 c_last -> out_len（=12 个 horizon）
        self.output = nn.Sequential(
            nn.BatchNorm2d(c_last),
            nn.ReLU(),
            nn.Conv2d(c_last, out_len, (12, 1)),
        )

    def forward(self, x):
        # (B, L, N, C) -> (B, C, L, N)
        h = x.permute(0, 3, 1, 2)
        for blk in self.st_blocks:
            h = blk(h)
        h = self.output(h)                 # (B, 12, 1, N)
        return h[:, :, 0, :]               # (B, 12, N)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
#  模型 3：Graph WaveNet（Wu et al., IJCAI 2019）
# ============================================================================
class GWNConv1d(nn.Module):
    """因果膨胀 1D 时间卷积：只在**时间维左侧** pad (k-1)*d，时间维长度不变。

    ★ 这里是两个连在一起、都很容易搞错的点：
      1) kernel 必须是 (1, k)：输入布局是 (B, C, N, T)，kernel 的第一位作用在
         节点维、第二位作用在时间维。写成 (k, 1) 就变成"沿节点下标做卷积"了 ——
         既没有时间建模，还把节点维长度改掉（N 从 20 缩到 19，报错还算走运，
         不报错的话就是静默算垃圾）。
      2) pad 必须落在最后一维（T）的左侧，且长度恰好 (k-1)*d。
         不 pad 的话，8 层膨胀卷积（dilation 1,2,1,2,1,2,1,2）会把 T=12
         一路砍到 0；两侧对称 pad 又会看到未来（不再是因果的）。
    """

    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.conv = nn.Conv2d(c_in, c_out, (1, k), dilation=(1, dilation))

    def forward(self, x):
        x = F.pad(x, ((self.k - 1) * self.dilation, 0))   # 只 pad 最后一维(T)的左侧
        return self.conv(x)


class GWNLinear(nn.Module):
    """官方 linear = 沿通道维的 1x1 Conv2d（作用在 (B,C,N,T) 上）"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, (1, 1))

    def forward(self, x):
        return self.conv(x)


class GWNGraphConv(nn.Module):
    """图卷积：对每个 support 做 order 阶（x, Sx）拼接后再 1x1 降维。

    x: (B, C, N, T)，support: (N, N) —— 输出 (B, c_out, N, T)。
    """

    def __init__(self, c_in, c_out, support_len, order=2, dropout=0.0):
        super().__init__()
        self.order = order
        self.mlp = GWNLinear(order * support_len * c_in, c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, supports):
        out = []
        for A in supports:
            x1 = torch.einsum('nm,bcmt->bcnt', A, x)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum('nm,bcmt->bcnt', A, x1)
                out.append(x1)
        h = torch.cat(out, dim=1)
        return self.drop(self.mlp(h))


class GWNet(nn.Module):
    """Graph WaveNet。

    blocks=4 × layers=2 ⇒ 8 层门控因果卷积，dilation 序列 1,2,1,2,1,2,1,2
    （每进入一个新 block，dilation 重置为 1 再翻倍 —— 与官方代码一致，
    也正是任务书里说的"dilation 1,2,1,2 两轮"）。

    support = [前向转移 A_f, 后向转移 A_b] + [自适应邻接]
        A_f = D^{-1} A,  A_b = A_f^T
        A_adp = softmax(ReLU(E1 @ E2^T)),  E1: (N, 10), E2: (10, N)
    官方用 E2 形状 (10, N) 而不是 (N, 10)，等价于把 E2^T 写死进参数，
    这里照搬官方（数学上是同一个东西）。

    ★ 输出层改动：官方把 skip 累加后过 (1,1) 卷积，输出 (B, 12, N, T)；
      本脚本要把时间维压掉，于是最后一层 kernel 用 (1, 12) → (B, 12, N, 1)，
      squeeze 成 (B, 12, N)，与另外两个模型统一接口。

    ★ 层内顺序 —— 2026-09-24 对着官方 master/model.py:157-197 逐行核对后改正 ★
      官方 forward 的真实顺序是：
          x = tanh(filter_conv(residual)) * sigmoid(gate_conv(residual))
          skip += skip_conv(x)                # ← skip 取**图卷积之前**的门控输出
          x = gconv(x, supports)              # ← 图卷积在 skip 之后
          x = x + residual[: , :, :, -T:]     # 残差
          x = bn(x)
      本脚本此前把 gconv 放在 skip **之前**，并在注释里断言"否则输出里一点
      图信息都没有"。那句话是**反的**：x 经 gconv 后进入下一层的门控，
      下一层的 skip 就带上了空间信息，8 层下来 skip 里图信息很充分。
      已按官方顺序改正（DCRNN/STGCN 的实现不受影响）。

    ★ 残差支路是"图卷积 **或** 1×1"，不是两者串联（官方 :193-197）：
          if gcn_bool and supports:  x = gconv[i](x, supports)
          else:                      x = residual_convs[i](x)
      即启用图卷积时 residual_convs **整条是死支路**，参数照建（官方也照建，
      为了结构一致）但拿不到任何梯度。本脚本此前写成 gconv → residual_conv
      串联，等于给 GWNet 多加了一层 1×1，已改正。

    已知死支路（与官方一致，不是 bug）：启用图卷积时全部 residual_convs
    无梯度；另外最后一层的 gconv 输出在循环结束后即被丢弃（官方同样如此）。
      看梯度诊断时不要慌。
    """

    def __init__(self, adj, N, out_len=12, in_dim=2, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, node_emb_dim=10,
                 dropout=0.3, adaptive=True):
        # ★ 2026-09-24：in_dim 默认从 1 改成 **2** —— 官方 train.py L20
        #   `--in_dim default=2`（速度 + tod）。原来默认 1 是把基线砍了一只手。
        super().__init__()
        self.N = N
        self.out_len = out_len
        self.dropout = dropout
        self.blocks, self.layers = blocks, layers
        self.kernel_size = kernel_size
        self.adaptive = adaptive

        # ---- 固定转移矩阵 + 自适应邻接
        d = adj.sum(axis=1)
        d_inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0)
        A_f = d_inv[:, None] * adj
        self.register_buffer('A_f', torch.as_tensor(A_f, dtype=torch.float32))
        self.register_buffer('A_b', torch.as_tensor(A_f.T.copy(), dtype=torch.float32))
        self.support_len = 2 + (1 if adaptive else 0)
        if adaptive:
            self.nodevec1 = nn.Parameter(torch.randn(N, node_emb_dim) * 0.01)
            self.nodevec2 = nn.Parameter(torch.randn(node_emb_dim, N) * 0.01)

        self.start_conv = nn.Conv2d(in_dim, residual_channels, (1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        new_dilation = 1
        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    GWNConv1d(residual_channels, dilation_channels,
                              kernel_size, new_dilation))
                self.gate_convs.append(
                    GWNConv1d(residual_channels, dilation_channels,
                              kernel_size, new_dilation))
                self.residual_convs.append(GWNLinear(dilation_channels, residual_channels))
                self.skip_convs.append(GWNLinear(dilation_channels, skip_channels))
                self.bns.append(nn.BatchNorm2d(residual_channels))
                self.gconvs.append(GWNGraphConv(dilation_channels, residual_channels,
                                                self.support_len, order=2,
                                                dropout=dropout))
                new_dilation *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_len, (1, out_len))

    def _supports(self):
        out = [self.A_f, self.A_b]
        if self.adaptive:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            out.append(adp)
        return out

    def forward(self, x):
        # ★ 布局必须是 (B, C, N, T)，不是 (B, C, T, N) —— 这是官方约定，
        #   搞反了不会立刻报错：因果卷积会去 pad 节点维（kernel=1 时"看似没事"），
        #   而图卷积的 einsum 'nm,bcmt->bcnt' 会把**时间维当成节点维**去乘邻接矩阵。
        #   本脚本第一版就是这么错的，selftest 抓到的是
        #   "einsum(): subscript m has size 11 ... does not broadcast"（T=11 vs N=20）。
        h = x.permute(0, 3, 2, 1)          # (B, L, N, C) -> (B, C, N, L)
        h = self.start_conv(h)
        supports = self._supports()
        skip = None
        n = self.blocks * self.layers
        for i in range(n):
            residual = h
            f = torch.tanh(self.filter_convs[i](residual))
            g = torch.sigmoid(self.gate_convs[i](residual))
            h = f * g
            # ★ 官方顺序：skip 取图卷积**之前**的门控输出
            s = self.skip_convs[i](h)
            skip = s if skip is None else skip + s
            # ★ 残差支路二选一：启用图卷积时走 gconv，residual_convs 是死支路
            h = self.gconvs[i](h, supports)
            h = h + residual[:, :, :, -h.size(3):]
            h = self.bns[i](h)
        s = F.relu(skip)
        s = F.relu(self.end_conv_1(s))
        s = self.end_conv_2(s)             # (B, out_len, N, 1)
        return s[:, :, :, 0]               # (B, out_len, N)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
#  Mamba 原语 —— 从 paper_experiments.py **逐行抄来，未改一字**
# ============================================================================
# 出处（按 paper_experiments.py 的行号）：
#   selective_scan_fn           :576-588
#   RMSNorm                     :591-604
#   MambaBlockNaive             :607-673
#   HAS_OFFICIAL / Official     :676-694
#
# ★ 抄而不 import：paper_experiments.py **没有 __main__ 守卫**（§7 变更记录里
#   记着已经踩过一次），import 它会直接真跑整张实验矩阵。
# ★ 抄的代价是**必须同步**：那边改了块的定义，这里不改，就会出现
#   "手写 vs 官方"或"本地 vs 卡上"拿两个不同网络对比的情况。改一处务必改两处。
def selective_scan_fn(u, delta, A, B, C, D):
    """手写 Python 循环扫描 —— 与 kaggle_ffff12.py 逐字一致，保留用于效率对照"""
    Bs, L, D_in = u.shape
    D_state = A.shape[1]
    delta = torch.clamp(delta, -4.0, 4.0)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    x = torch.zeros(Bs, D_in, D_state, device=u.device, dtype=u.dtype)
    ys = torch.zeros(Bs, L, D_in, device=u.device, dtype=u.dtype)
    for i in range(L):
        x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
        ys[:, i] = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
    return ys + D.unsqueeze(0).unsqueeze(0) * u


class RMSNorm(nn.Module):
    """和 mamba_ssm 内部的 RMSNorm 同款，用来把手写块的结构和官方块对齐"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 先转 float32 再算，避免输入本身已在 float32 边界时平方溢出
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class MambaBlockNaive(nn.Module):
    """
    手写 Python 循环扫描。

    ★ 2026-09-23 修的一个真 bug ★
      原写法没有块内归一化，也没有块内残差，而官方 mamba_ssm.Mamba 的
      forward 是「先 RMSNorm → 混合器 → 加残差」。后果有两个：
        (1) 数值爆炸。实测（全量 325 节点、真实数据、初始权重、未训练）：
            各层 out_proj 输出 absmax = 1.3e4 → 2.3e12 → 3.4e38，
            第三层正好撞上 float32 上限 3.4e38 → nan。
        (2) 对比不公平。官方块带 pre-norm+残差、手写块不带，
            两者根本不是同一个网络，「手写 vs 官方」的效率数字不可比。
      补上 RMSNorm 和块内残差后：三层输出稳定在 8.8e2 → 2.6e3 → 3.0e3。
      注意：调用方要相应用 `x = l(x)`，**不能再补 `+ x`** —— 重复加残差
      就是上面那条逐层放大的成因。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        # 官方 mamba_ssm 的 dt_rank = ceil(d_model/16)，d_model=128 时是 8。
        self.dt_rank = int(math.ceil(d_model / 16))
        self.norm = RMSNorm(d_model)          # ← 与官方块对齐的 pre-norm
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2,
                                bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.linspace(0.1, 1.0, d_state)
        self.A_log = nn.Parameter(A_init.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for m in [self.in_proj, self.x_proj, self.out_proj, self.dt_proj]:
            if m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.dt_proj.bias)

    @property
    def A(self):
        return torch.exp(torch.clamp(self.A_log, max=0.0))

    def forward(self, x):
        residual = x                       # ← 与官方块一致：块内残差
        x = self.norm(x)                   # ← 与官方块一致：pre-norm
        B, L, _ = x.shape
        xg, res = self.in_proj(x).split(self.d_inner, dim=-1)
        xc = self.conv1d(xg.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        dt, Bp, Cp = torch.split(self.x_proj(xc),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan_fn(xc, delta, self.A, Bp, Cp, self.D)
        return self.out_proj(y * F.silu(res)) + residual


HAS_OFFICIAL = False
MambaBlockOfficial = None          # 没装 mamba_ssm 时保持 None
try:
    from mamba_ssm import Mamba

    class MambaBlockOfficial(nn.Module):
        """
        官方 mamba_ssm 的 Mamba **混合器** + 与手写块逐项对应的 pre-norm / 残差。

        ★ 2026-09-26 修的一个真 bug ★
          原写法是 `return self.mamba(x)`，注释声称「Mamba 内部已含 RMSNorm +
          MambaBlock + 残差」。**这条注释是错的**：mamba_ssm 的 `Mamba`
          （`modules/mamba_simple.py`）只含混合器，pre-norm 与残差在
          `modules/block.py` 的 `Block` 里 —— `Block` 包住 `Mamba` 才有。
          实测（4090 / mamba_ssm 2.2.2）逐键对比 state_dict：
            MambaBlockOfficial 每块 116,480 参数，**没有任何 norm 相关的键**；
            MambaBlockNaive    每块 116,608 参数，多一个 `norm.weight (128,)`。
          即 official 支路当时跑的是**没有归一化、也没有残差的网络** —— 正是
          2026-09-23 在手写支路修掉的那次数值爆炸（1.3e4 → 2.3e12 → 3.4e38）
          的成因。后果不只是数值：`论文_实验设置_EN.md` §4.3 声称两条支路
          「compute the same function; they differ only in how the scan is
          executed」，而当时它们差着归一化和残差 —— 是两个不同的网络，
          「official vs naive」的差里捆着 kernel 与架构两样东西。
          补上 norm + 残差后两边**参数量与结构逐项相同**（各 116,608），
          差的只剩扫描实现，§4.3 那句话才成立。

        ★ 两边共用本文件的 `RMSNorm`（不换成 mamba_ssm 自己那份融合核版本）：
          目的是让「唯一变量是扫描实现」这件事严格成立。
        """

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.norm = RMSNorm(d_model)           # ← 与手写块同一个实现
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

        def forward(self, x):
            residual = x                           # ← 与手写块一致：块内残差
            return self.mamba(self.norm(x)) + residual

    HAS_OFFICIAL = True
except Exception as _e:
    log(f'[提示] 官方 mamba_ssm 不可用（{type(_e).__name__}），MambaST 只能跑手写块。')


def flatten_node_windows(x):
    """(B, L, N, C) -> (B*N, L, C)，行号 b*N+n 恰是节点 n 的 L 步窗口。

    ★ 这一步的 permute 与 paper_experiments.py:706-713 修掉的那个 bug 长得极像、
      方向**相反**，所以单独抽成一个函数，让自检和模型调的是同一份实现：
        · 那边：x 已经是 (B,N,L,C)，节点轴在 dim1，本来直接 reshape 就对，
          多写的 permute 反而把 (批次, 节点, 时刻) 三者搅在一起 → 真 bug。
        · 这边：x 是 (B,L,N,C)，节点轴在 dim2，**不 permute 就是错的**。
      实测断言（见 _selftest_mambast）：z[b*N+n] 逐位等于 x[b,:,n,:]。
    """
    B, L, N, C = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B * N, L, C)


def build_gwn_supports(adj):
    """GWNet 的固定转移矩阵，与 GWNet.__init__:1209-1215 的构造**逐位相同**。

    抽成函数是为了让 MambaST 用同一份支撑集，同时**不改动 GWNet 本身**；
    自检里断言它等于 GWNet.A_f / GWNet.A_b，将来只改一处会立刻被抓到。
    返回 [A_f, A_b]，A_f = D^-1 A（随机游走），A_b = A_f^T，均为 float32。
    """
    d = adj.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0)
    A_f = d_inv[:, None] * adj
    return [torch.as_tensor(A_f, dtype=torch.float32),
            torch.as_tensor(A_f.T.copy(), dtype=torch.float32)]


# ============================================================================
#  模型 4：MambaST —— 论文的空间机制分析实验（不是主模型，理由见文件头注释）
# ============================================================================
class MambaST(nn.Module):
    """Mamba 扫时间轴（每节点独立），图卷积扫节点轴。

        x: (B, 12, N, C)  ->  (B, 12, N)

    三臂由 graph 决定：
      'identity' → 建图模块，但支撑集取 I_N：逐节点通道混合，**不跨节点**。
                   与 'real' **参数量逐位相同** —— 这是本实验的关键。
      'real'     → 支撑集取 A_f/A_b（与 GWNet 同一套）。
      'none'     → 根本不建图模块（少 131,328 参数），用来检验恒等对照够不够用。

    ★ 布局纪律（本仓库最容易静默出错的地方，逐步写清）：
        步 0  x                        (B, L, N, C)
        步 1  flatten_node_windows(x)  (B*N, L, C)   节点轴从 dim2 换进 batch
        步 2  proj                     (B*N, L, D)
        步 3  Mamba 块 ×n_layers       (B*N, L, D)   块内自带 pre-norm + 残差
        步 4  reshape + permute        (B, D, N, L)  = GWNet 的 (B, C, N, T)
        步 5  GWNGraphConv             (B, D, N, L)  缩并的是**节点轴**
        步 6  permute 回 + LayerNorm   (B*N, L, D)
        步 7  h = h + 图支路            (B*N, L, D)
        步 8  norm + head              (B*N, 12)
        步 9  reshape + permute        (B, 12, N)

    ★ 图支路为什么用 pre-norm 残差 `h = h + LN(gconv(h))`，而不是 GWNet 的
      `gconv(h) + residual → BatchNorm`：GWNGraphConv 只拼接 A·x 与 A²·x（没有
      x 自身那一项），所以在**两个支撑的整行都为零**的节点上它的输出与输入无关。
      ★ 2026-10-10 更正：PeMS-BAY 上 A_f 整行为零的节点有 **12** 个、A_b 整行为零
      的有 17 个，但**两个方向同时为零的只有 6 个**（下标 0/1/39/77/92/282）。原先
      这里写「12 个孤立节点，它们的 A_f/A_b 整行为零」——只对前一半成立，图卷积
      在另外 6 个节点上仍能读到后向邻居。没有残差的话，图支路会把那 6 个节点的
      特征**替换成一个与输入无关的常数**，等于对本来还能处理的节点主动破坏。
    """

    def __init__(self, adj, N, out_len=12, in_dim=2, graph='real',
                 d_model=None, d_state=None, n_layers=None, order=None,
                 block_cls=None, budget=None):
        super().__init__()
        self.d_model = MAMBA_DMODEL if d_model is None else int(d_model)
        d_state = MAMBA_DSTATE if d_state is None else int(d_state)
        self.n_layers = MAMBA_LAYERS if n_layers is None else int(n_layers)
        order = MAMBA_ORDER if order is None else int(order)
        self.N, self.out_len, self.graph = N, out_len, graph
        # 与 GWNet/DCRNN 的属性口径对齐：_selftest_forward 会取
        # getattr(model,'blocks',1)*getattr(model,'layers',1)。注意**不能**把
        # self.layers 写成 ModuleList —— 那样这个乘就会 TypeError。
        self.blocks, self.layers = 1, self.n_layers
        self.budget = MAMBA_SEQ_BUDGET if budget is None else int(budget)
        if graph not in ('identity', 'real', 'none'):
            raise ValueError(f'MambaST 的 graph 只能是 identity/real/none，收到 {graph}')

        if block_cls is None:
            if MAMBA_BLOCK == 'official':
                if not HAS_OFFICIAL:
                    raise RuntimeError(
                        'PAPER_GNN_MAMBA_BLOCK=official 但本机没有 mamba_ssm。'
                        '要么装上（pip install mamba-ssm causal-conv1d），'
                        '要么显式设 =naive。**不要让它悄悄退回手写块** —— '
                        '那样本地与卡上跑的就不是同一个网络，两边的数不可比。')
                block_cls = MambaBlockOfficial
            elif MAMBA_BLOCK == 'naive':
                block_cls = MambaBlockNaive
            elif MAMBA_BLOCK == 'auto':
                # ★ 必须看设备：官方块走的是 mamba_ssm 的 CUDA 核，喂 CPU 张量会
                #   直接 RuntimeError。所以 auto = "有官方块 **且** 在 GPU 上"。
                #   这条选择的后果要说清：本地(CPU)拿到的必然是手写块，卡上(GPU)
                #   拿到的是官方块，两者只是**结构**等价、不是数值等价 ——
                #   所以本地微跑的绝对数字不能和卡上的比（计划 §七已写明）。
                use_official = HAS_OFFICIAL and str(DEVICE).startswith('cuda')
                block_cls = MambaBlockOfficial if use_official else MambaBlockNaive
                log(f'    MambaST 块选择: auto → '
                    f'{"官方 mamba_ssm" if use_official else "手写 Python 扫描"}'
                    f'（设备={DEVICE}，官方块可用={HAS_OFFICIAL}）')
            else:
                raise ValueError(f'PAPER_GNN_MAMBA_BLOCK 只能是 auto/official/naive，'
                                 f'收到 {MAMBA_BLOCK}')
        self.official = (block_cls is MambaBlockOfficial)

        # 支撑集：两臂都注册同名的 buffer（**state_dict 的键与形状必须逐位相同**，
        # 否则"两臂同权重"的自检做不了）。开关只影响 _supports() 的返回值。
        sup = build_gwn_supports(adj)
        self.register_buffer('A_f', sup[0])
        self.register_buffer('A_b', sup[1])
        self.support_len = len(sup)
        # persistent=False：不进 state_dict，免得两臂的键不一致
        self.register_buffer('I_N', torch.eye(N, dtype=torch.float32),
                             persistent=False)

        self.proj = nn.Linear(in_dim, self.d_model)
        self.mambas = nn.ModuleList([block_cls(self.d_model, d_state=d_state)
                                     for _ in range(self.n_layers)])
        if graph == 'none':
            self.gconvs = None
            self.gnorms = None
        else:
            self.gconvs = nn.ModuleList([
                GWNGraphConv(self.d_model, self.d_model, self.support_len,
                             order=order, dropout=0.0)
                for _ in range(self.n_layers)])
            self.gnorms = nn.ModuleList([nn.LayerNorm(self.d_model)
                                         for _ in range(self.n_layers)])
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(12 * self.d_model, out_len)

        for m in (self.proj, self.head):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _supports(self):
        """图关时返回 [I_N, I_N]：形状、个数、参数量都与 [A_f, A_b] 完全一致，
        唯一区别是不跨节点。"""
        if self.graph == 'identity':
            return [self.I_N, self.I_N]
        return [self.A_f, self.A_b]

    def _temporal(self, h, blk):
        """按**节点**分块跑一个 Mamba 块。块内每个算子（Linear、末维 depthwise
        Conv1d、scan、RMSNorm）都逐行独立、无跨行归约，所以分块与不分块
        **必须逐位相等** —— 自检 _selftest_mambast 里有这条断言。

        为什么必须分块：naive 扫描的 deltaA 形状是 (B*N, L, d_inner, d_state)，
        bs=64/325 节点/256/16 时单个 deltaA 就是 4.09 GB。
        """
        # ★ 进来的是 (B*N, L, D) —— dim0 是**行数** B*N，不是 B。
        #   第一版写成 `B, N = h.shape[0], self.N`，于是下面
        #   `out.reshape(B, N, L, D)` 变成 reshape(96, 24, 12, 128) 直接炸。
        #   行号 b*N+n 与 (b, n) 一一对应，所以 B 只能由 rows // N 反推。
        rows = h.shape[0]
        B, N = rows // self.N, self.N
        L, D = h.shape[1], h.shape[2]
        if self.budget <= 0 or rows <= self.budget:
            return blk(h)
        nc = max(1, self.budget // max(1, B))      # 每块的节点数
        out = torch.empty_like(h)
        out_v = out.reshape(B, N, L, D)
        h_v = h.reshape(B, N, L, D)
        for n0 in range(0, N, nc):
            n1 = min(N, n0 + nc)
            # blk 出的是 3D (B*nc, L, D)，而 out_v 的槽位是 4D (B, nc, L, D) ——
            # 不 reshape 回去就会被当成广播，报"维度 1 上 1 vs nc 不匹配"。
            out_v[:, n0:n1] = blk(h_v[:, n0:n1].reshape(-1, L, D)).reshape(
                B, n1 - n0, L, D)
        return out

    def _graph_branch(self, h, B, supports, i):
        """(B*N, L, D) -> 图卷积 -> (B*N, L, D)。

        中间那步 permute 是必需的：GWNGraphConv 的 einsum 是
        'nm,bcmt->bcnt'，它把 dim=2 当**节点轴**。喂错布局不会报错
        （只要 N 和 L 恰好不冲突），只会静默算出垃圾 —— 这个坑本仓库
        GWNet 第一版踩过。
        """
        L, D = h.shape[1], h.shape[2]
        z = h.reshape(B, self.N, L, D).permute(0, 3, 1, 2).contiguous()
        z = self.gconvs[i](z, supports)                    # (B, D, N, L)
        return z.permute(0, 2, 3, 1).reshape(B * self.N, L, D)

    def forward(self, x):
        B = x.shape[0]
        L = x.shape[1]
        h = self.proj(flatten_node_windows(x))        # (B*N, L, C) -> (B*N, L, D)
        supports = self._supports()
        for i, blk in enumerate(self.mambas):
            # ★ 块内已含 pre-norm 与残差，这里**不要**再补 + h
            h = self._temporal(h, blk)
            if self.gconvs is not None:
                h = h + self.gnorms[i](self._graph_branch(h, B, supports, i))
        h = self.norm(h)
        out = self.head(h.reshape(B * self.N, L * self.d_model))   # (B*N, 12)
        return out.reshape(B, self.N, L).permute(0, 2, 1).contiguous()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
#  模型工厂
# ============================================================================
def build_model(name, adj, N, in_dim=None):
    """按名字建模型。三个模型的接口完全一致：(B,12,N,C) -> (B,12,N)

    in_dim 缺省取 OFFICIAL_IN_DIM[name]（官方输入处方）。显式传只用在一处：
    自检里想对比"官方处方 vs 砍成 1 通道"的参数量差异。
    """
    if in_dim is None:
        in_dim = OFFICIAL_IN_DIM[name]
    if name == 'DCRNN':
        return DCRNN(adj, N, num_units=int(os.environ.get('PAPER_GNN_DCRNN_UNITS', 64)),
                     K=int(os.environ.get('PAPER_GNN_DCRNN_K', 2)),
                     layers=int(os.environ.get('PAPER_GNN_DCRNN_LAYERS', 2)),
                     cheb=DCRNN_CHEB, ss_prob=SS_PROB, in_dim=in_dim)
    if name == 'STGCN':
        # STGCN 官方通道维写死 1（trainer.py L30）。**显式忽略 in_dim**，
        # 免得有人从外面传个 2 进来就悄悄变成非官方配置。
        return STGCN(adj, N, Ks=int(os.environ.get('PAPER_GNN_STGCN_KS', 3)),
                     Kt=int(os.environ.get('PAPER_GNN_STGCN_KT', 3)))
    if name == 'GWNET':
        return GWNet(adj, N, adaptive=GW_ADAPT, in_dim=in_dim)
    if name in MAMBAST_ARMS:
        # 三臂只有 graph 这一个参数不同（MAMBASTN 连模块都不建）。
        # d_model/d_state/layers/order/block 全部走同一份默认值，
        # 保证两臂除了支撑集之外**逐字节同构**。
        return MambaST(adj, N, out_len=12, in_dim=in_dim,
                       graph=MAMBAST_ARMS[name])
    raise ValueError(f'未知模型 {name}，可选 DCRNN / STGCN / GWNET / '
                     f'{" / ".join(MAMBAST_ARMS)}')


def make_optimizer(name, model):
    """按 TRAIN_CFG 表构造优化器与调度器（逐模型，出处见该表）。

    lr / wd 可用环境变量覆盖；PAPER_GNN_SCHED=auto 之外的取值会全局强制调度。
    """
    cfg = TRAIN_CFG[name]
    lr = float(os.environ.get(cfg['lr_env'], cfg['lr']))
    wd = float(os.environ.get(cfg['wd_env'], cfg['wd']))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    kind = SCHED if SCHED != 'auto' else cfg['sched']
    if kind == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        sch_desc = f'CosineAnnealingLR(T_max={EPOCHS})'
    elif kind == 'multistep':
        ms = cfg.get('milestones', [20, 30, 40, 50])
        gam = cfg.get('gamma', 0.1)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=gam)
        sch_desc = f'MultiStepLR(milestones={ms}, gamma={gam})'
    elif kind == 'step':
        st = cfg.get('step', LR_STEP)
        gam = cfg.get('gamma', LR_GAMMA)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=st, gamma=gam)
        sch_desc = f'StepLR(step={st}, gamma={gam})'
    elif kind == 'none':
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _e: 1.0)
        sch_desc = '无衰减（官方 schedule 即恒定）'
    else:
        raise ValueError(f'未知调度 {kind}（auto/step/multistep/cosine/none）')
    forced = '' if SCHED == 'auto' else f'  ← PAPER_GNN_SCHED={SCHED} 强制'
    log(f'  优化器 Adam(lr={lr:g}, weight_decay={wd:g}) / {sch_desc}{forced}')
    log(f'    出处：{cfg["src"]}')
    return opt, sched


# ============================================================================
#  训练（协议对齐 paper_experiments.train_one）
# ============================================================================
def train_one(name, model, data, seed, epochs, bs, tag):
    """data = ((Xtr,Ytr),(Xva,Yva),(Xte,Yte), std_y, mean_y, N)

    tag 用作权重文件名前缀（配置 ID，已把 '|' 换成 '_'）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), std_y, mean_y, N = data
    model = model.to(DEVICE)
    n_param = model.num_params
    opt, sched = make_optimizer(name, model)
    # 掩码 MSE：真值 0 = 缺失，不计入。null 在标准化空间的位置是 -mean_y/std_y。
    # ★ 必须 .to(DEVICE)：crit 不是 model 的子模块，model.to(cuda) 带不上它。
    crit = MaskedMSELoss((-mean_y) / std_y).to(DEVICE)

    # ★★ 2026-09-26 新增：PAPER_GNN_RECOMPUTE=1 时**跳过训练**，只载入已有检查点重算 ★★
    #   与 paper_experiments.train_one 的同名分支逐条同构，理由见那里的注释。
    #   DCRNN 这里用 infer()（= 纯模型前向），与训练循环 :1804 的 teacher forcing
    #   不是同一件事：teacher forcing 只喂损失，最终指标 :1845 也是走 infer()。
    #   **重算必须与 :1845 完全同路**，否则重算出来的数就不是原来那个量的修正版。
    if RECOMPUTE:
        ck = f'{tag}_best.pt'
        if not os.path.exists(ck):
            log(f'  [重算] 缺 {ck} —— 跳过（没有检查点就无从重算）')
            return None
        try:
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
        except Exception as e:
            log(f'  [重算] {ck} 与配置 {tag} 不符 —— 跳过：'
                f'{type(e).__name__}: {e}')
            return None
        log(f'  [重算] 载入 {ck}，只推理、不训练、不覆写任何 .pt')
        pred = infer(model, Xte, bs)
        if not np.isfinite(pred).all():
            log(f'  [重算] {tag} 预测含 nan/inf —— 作废，不写入结果')
            return None
        m = compute_metrics(pred, Yte, std_y, mean_y, N)
        m.update(params=n_param, best_epoch=-1, val_mae=float('nan'),
                 epoch_time=0.0, epochs_run=0, n_train=0, recomputed=True)
        log(f'  [重算] {tag}  MAE@15={m["mae_15"]:.4f}  '
            f'MAE@30={m["mae_30"]:.4f}  MAE@60={m["mae_60"]:.4f}')
        return m

    S = Xtr.shape[0]
    n_b = max(1, (S + bs - 1) // bs)
    best_val, best_state, best_ep, bad = float('inf'), None, -1, 0
    log(f'  参数 {n_param:,}   训练样本 {S:,}   每 epoch {n_b:,} 步   bs={bs}')
    log(f'  验证样本 {Xva.shape[0]:,}（{"全量" if VAL_N == 0 else f"前 {VAL_N} 条"}）'
        f'   测试样本 {Xte.shape[0]:,}')
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(S)
        tot = 0.0
        for bi in range(n_b):
            idx = perm[bi * bs:(bi + 1) * bs]
            if len(idx) == 0:
                continue
            opt.zero_grad()
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE)
            if name == 'DCRNN':
                pred = model(xb, yb)      # DCRNN 需要真值序列做 teacher forcing
            else:
                pred = model(xb)
            loss = crit(pred, yb)
            if not torch.isfinite(loss):
                log(f'  [中止] ep{ep} 第 {bi} 步 loss 非有限 —— 权重已存 {tag}_last.pt')
                torch.save(model.state_dict(), f'{tag}_last.pt')
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += loss.item()
        sched.step()
        tr_loss = tot / n_b

        Xva_use = Xva if VAL_N == 0 else Xva[:VAL_N]
        Yva_use = Yva if VAL_N == 0 else Yva[:VAL_N]
        v_mae, _ = quick_eval(model, Xva_use, Yva_use, std_y, mean_y, N, bs)
        flag = ''
        if v_mae < best_val:
            best_val, best_ep, bad = v_mae, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            flag = '  ★'
        else:
            bad += 1
        log(f'  ep {ep:3d}/{epochs}  train_loss={tr_loss:.6f}  '
            f'val_MAE={v_mae:.4f}  lr={opt.param_groups[0]["lr"]:.2e}  '
            f'({time.time() - t0:.0f}s){flag}')

        if bad >= PATIENCE:
            log(f'  早停：验证集 {PATIENCE} 轮没改善，最优在 ep{best_ep}')
            break

    # 两个权重都存（同主脚本）：只留 _best 的话，一旦 _best 在测试集上出 nan
    # 就查不了"好轮次 vs 坏轮次"的对比；会话重启又会把 _best 一起清掉。
    torch.save(model.state_dict(), f'{tag}_last.pt')
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f'{tag}_best.pt')

    pred = infer(model, Xte, bs)
    if not np.isfinite(pred).all():
        log('  [警告] 测试集预测含 nan —— 该配置结果作废')
        return None
    m = compute_metrics(pred, Yte, std_y, mean_y, N)
    m.update(params=n_param, best_epoch=best_ep, val_mae=best_val,
             epoch_time=(time.time() - t0) / max(1, ep + 1),
             epochs_run=ep + 1, n_train=S)
    return m


# ============================================================================
#  selftest：不碰任何真实数据文件，全部用随机张量 + 手算期望值
# ============================================================================
def _selftest_metrics():
    """指标自检：手算期望值，断言 compute_metrics 一致（容差 1e-9）。

    ---- 构造 ----
    N=2 个节点，T=1 个窗口，std_y=[2,2]，mean_y=[10,10]（per-node 标准化）。
    compute_metrics 内部先反标准化：real = norm*std + mean，
    所以下面直接给"真实单位"的真值/预测，再反推标准化空间的输入。

    真实真值 true_real (1,12,2)，只列出会被用到的三个 horizon 下标
    （HORIZONS = {15:3, 30:6, 60:12} → 下标 2 / 5 / 11），其余填 10.0 无所谓：
        true_real[0, 2, :] = [ 0.0,  4.0]      # h=3
        true_real[0, 5, :] = [ 1.0,  0.0]      # h=6
        true_real[0,11, :] = [ 2.0, 21.0]      # h=12
    真实预测 pred_real，使误差 d = pred - true 为：
        d[0, 2, :] = [ 2.0,  1.0]              # 2.0 那一项会被掩零排除
        d[0, 5, :] = [-0.5,  3.0]              # 3.0 那一项被掩零排除
        d[0,11, :] = [ 4.0, -2.0]

    ---- 手算 ----
    MAPE_MIN = 1.0（默认）。掩码：m_null = (真值 > 0)，MAPE 再 & (真值 > 1.0)。

    h=3（下标 2）：
        真值 = [0, 4] → 掩零只剩 node1。
        MAE  = |1.0| / 1                  = 1.0
        RMSE = sqrt(1.0^2 / 1)            = 1.0
        MAPE : 真值 > 1 的只有 node1（4.0），|1.0/4.0| = 0.25 → 25.0%
        mape_cov = 1/2 = 0.5      null_frac = 1 - 1/2 = 0.5
        mae_all（不掩零）= (|2.0| + |1.0|)/2 = 1.5

    h=6（下标 5）：
        真值 = [1, 0] → 掩零只剩 node0，误差 -0.5。
        MAE  = 0.5      RMSE = sqrt(0.25) = 0.5
        MAPE : 需要真值 > 1.0，而 node0 真值恰好 = 1.0 → 不满足（边界）
               于是 MAPE 的样本集为空 → 代码返回 nan
        mape_cov = 0/2 = 0.0      null_frac = 0.5

    h=12（下标 11）：
        真值 = [2, 21]，两个都 > 1，全部保留。
        MAE  = (|4.0| + |-2.0|)/2         = 3.0
        RMSE = sqrt((16 + 4)/2) = sqrt(10) ≈ 3.1622776601683795
        MAPE = (|4/2| + |2/21|)/2 = (2 + 0.09523809523809523)/2
             = 1.0476190476190476 → ×100 = 104.76190476190476 %
        mape_cov = 2/2 = 1.0      null_frac = 0.0
    """
    std_y = np.array([2.0, 2.0], dtype=np.float64).reshape(2, 1, 1)
    mean_y = np.array([10.0, 10.0], dtype=np.float64).reshape(2, 1, 1)

    true_real = np.full((1, 12, 2), 10.0)
    true_real[0, 2, :] = [0.0, 4.0]
    true_real[0, 5, :] = [1.0, 0.0]
    true_real[0, 11, :] = [2.0, 21.0]

    pred_real = true_real + 1.0
    pred_real[0, 2, :] = [0.0 + 2.0, 4.0 + 1.0]
    pred_real[0, 5, :] = [1.0 - 0.5, 0.0 + 3.0]
    pred_real[0, 11, :] = [2.0 + 4.0, 21.0 - 2.0]

    # 转成 compute_metrics 期望的标准化空间
    true_n = (true_real - mean_y.reshape(1, 1, 2)) / std_y.reshape(1, 1, 2)
    pred_n = (pred_real - mean_y.reshape(1, 1, 2)) / std_y.reshape(1, 1, 2)

    m = compute_metrics(pred_n, true_n, std_y, mean_y, 2)
    exp = {
        'mae_15': 1.0, 'rmse_15': 1.0, 'mape_15': 25.0,
        'mape_cov_15': 0.5, 'null_frac_15': 0.5, 'mae_all_15': 1.5,
        'mae_30': 0.5, 'rmse_30': 0.5, 'mape_cov_30': 0.0, 'null_frac_30': 0.5,
        'mae_60': 3.0, 'rmse_60': math.sqrt(10.0),
        'mape_60': 104.76190476190476, 'mape_cov_60': 1.0, 'null_frac_60': 0.0,
        'n_eval': 1,
    }
    bad = []
    for k, v in exp.items():
        got = m[k]
        if not (abs(float(got) - float(v)) <= 1e-9):
            bad.append(f'{k}: 期望 {v} 实得 {got}')
    # MAPE 空集必须是 nan，不能是 0 或报错
    if not math.isnan(m['mape_30']):
        bad.append(f'mape_30 期望 nan（真值集合为空），实得 {m["mape_30"]}')

    # ---- 掩码损失自检（专门抓 (1,1,N) / (1,N,1) 广播搞反这个坑）----
    # mean_y=10, std_y=2 → 真实 0 在标准化空间是 -10/2 = -5.0
    crit = MaskedMSELoss(np.array([-5.0]))
    p = torch.tensor([[[-4.0, -4.0]]])      # 真实 [2.0, 2.0]
    t = torch.tensor([[[-5.0, -4.0]]])      # 真实 [0.0, 2.0] ← 第一个是缺失
    lv = float(crit(p, t))
    # 手算：第一个点被掩掉，只剩第二个点，误差 0 → 损失 0.0
    # 若掩码广播写反（(1,N,1) 用到 (B,12,N) 上），掩的位置会错，损失 ≠ 0
    if abs(lv - 0.0) > 1e-9:
        bad.append(f'掩码损失：期望 0.0（真值 0 的点应被排除），实得 {lv}')
    return bad


def _selftest_forward(N=20, B=2):
    """前向 shape / 反向梯度自检。邻接矩阵用随机对称非负矩阵。"""
    report = []
    rng = np.random.RandomState(0)
    A = rng.rand(N, N).astype(np.float64)
    A = (A + A.T) / 2.0
    A[A < 0.5] = 0.0                       # 稀疏化，更接近真实邻接矩阵
    np.fill_diagonal(A, 0.0)

    # ★ 2026-09-24：输入通道数按各模型的官方处方给（DCRNN/GWNET 为 2，STGCN 为 1）。
    #   统一喂 1 通道会在 DCRNN/GWNET 上直接 shape 报错 —— 这正是"官方处方变了"
    #   的信号，但也容易被人顺手改成 1 通道让自检"通过"，那就白测了。
    # 假阈值，只测梯度是否有限。★ 必须 .to(DEVICE)：crit 不是 model 的子模块，
    # model.to(cuda) 带不上它。train_one 里一直是这么写的，selftest 漏了这一步，
    # 于是在有 GPU 的机器上跑 selftest 会直接 RuntimeError（device 不一致）。
    crit = MaskedMSELoss(np.full(N, -1.0)).to(DEVICE)

    for name in ['DCRNN', 'STGCN', 'GWNET'] + list(MAMBAST_ARMS):
        torch.manual_seed(0)
        # ★ 2026-09-25：三个输入张量从循环**外**挪到这里。原来建在循环外、
        #   建在任何 manual_seed **之前** —— 于是每次运行拿到的输入都不同，
        #   自检打印的 loss / absmax / 梯度签名全都不可复现，做不了
        #   "改前 vs 改后逐字节相同" 的指纹（规划的指纹脚本正是栽在这上面：
        #   报告值 DCRNN -54.777428 与我复现的 -54.791961 对不上，参数量却对得上，
        #   就是因为随机数流没被固定）。挪进来之后每个模型拿到**同一组**输入，
        #   既复现又可比。
        x1 = torch.randn(B, 12, N, 1)
        x2 = torch.randn(B, 12, N, 2)
        target = torch.randn(B, 12, N)
        model = build_model(name, A, N).to(DEVICE)
        np_ = model.num_params
        n_layers = getattr(model, 'blocks', 1) * getattr(model, 'layers', 1)
        # 1) shape + 有限性
        model.train()
        xb = (x2 if OFFICIAL_IN_DIM[name] > 1 else x1).to(DEVICE)
        # DCRNN 在 train() 下还会走 teacher forcing 支路（吃真值序列），
        # 这条支路在推理时根本不会被执行 —— 不专门喂一次就永远测不到。
        # 本脚本第一版就是这里把 (B,N) 当成 (B,N,1) 用了。
        yb = torch.randn(B, 12, N, device=DEVICE)
        pred = model(xb, yb) if name == 'DCRNN' else model(xb)
        shape_ok = (tuple(pred.shape) == (B, 12, N))
        finite_ok = bool(torch.isfinite(pred).all())
        # 2) 反向：所有参数梯度有限（能抓到扩散卷积的数值问题），
        #    并且所有参数都该拿到梯度 —— "某些参数梯度为 None" 正是
        #    "这一支根本没接到输出上"的信号（GWNet 的层内顺序写反、
        #    或者某条支路被接错，都会这样表现，而且前向不会报错）。
        #
        #    GWNet 例外：官方结构下**必然**有一批参数无梯度（逐条对
        #    master/model.py:157-197 推过）：
        #      · 全部 residual_convs —— 启用图卷积时走的是 gconv 分支，
        #        1×1 那条残差支路整个是死的（官方 :193-197 的 if/else）
        #      · gconvs[n-1] —— 最后一层的图卷积输出在循环结束后被丢弃
        #      · bns[n-1]    —— 同上
        #    这里做**双向**核对：既不允许"该活的死了"（unexpected_none），
        #    也不允许"该死的活了"（expected_alive）—— 后者说明结构又被改动，
        #    那时这三条注释也该重新推一遍，不能默默放行。
        model.zero_grad()
        loss = crit(pred, target.to(DEVICE))
        loss.backward()
        g_bad = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
        g_none = [n for n, p in model.named_parameters() if p.grad is None]

        if name == 'GWNET':
            last = n_layers - 1

            def expected_dead(nm):
                return (nm.startswith('residual_convs.')
                        or nm.startswith(f'gconvs.{last}.')
                        or nm.startswith(f'bns.{last}.'))
        else:
            def expected_dead(nm):
                return False

        dead = [n for n in g_none if expected_dead(n)]
        unexpected_none = [n for n in g_none if not expected_dead(n)]
        # 该死不死的：预期无梯度却有梯度
        expected_alive = [n for n, p in model.named_parameters()
                          if expected_dead(n) and p.grad is not None]
        report.append(dict(
            name=name, params=np_, shape=tuple(pred.shape),
            # ★ 记真实输入形状，不要在上面那条日志里把 (2,12,20,C) 写死 ——
            #   写死后改了通道数也看不出来（这次 in_dim 1→2 就撞上了）。
            in_shape=tuple(xb.shape),
            shape_ok=shape_ok, finite=finite_ok,
            grad_ok=(len(g_bad) == 0 and len(unexpected_none) == 0
                     and len(expected_alive) == 0),
            grad_bad=g_bad[:5], grad_none=unexpected_none[:5],
            dead=len(dead), dead_alive=expected_alive[:5],
            loss=float(loss),
            output_absmax=float(pred.abs().max()),
        ))
        del model, pred, loss, xb
    return report


def _selftest_mambast():
    """MambaST 三臂的空间机制自检（计划 §3.1–3.3）。

    本仓库的失败模式是「不出异常、只出错数」，所以下面每一条都做成
    **数值不变式**，而不是「能跑通」：

      A. 参数量 / state_dict 逐位相同 —— 恒等对照的前提。不等就没有单变量可言。
      B. 支撑集敏感性 —— 图通路是活的。
         ★ 这一条**不能靠梯度测**：恒等臂里图卷积的那个 Linear 也是活的，
           `p.grad is not None` 在两臂上都成立，测不出"支路被接死"。
      C. 节点局域性 —— einsum 缩并的确实是节点轴、A 没被转置。
      D. 图关时无跨节点串味 / 图开时有 —— 恒等臂真的是逐节点的（exactly 0，不是"很小"）。
      E. 节点置换等变性 —— 节点身份真的被用上了。
      F. 分块 vs 不分块 —— 显存省下来了，数不能变。
      G. flatten_node_windows 的布局 —— (B,L,N,C) 的节点轴没被搅乱。
    """
    bad, info = [], []
    nb, N = 4, 24
    rng = np.random.RandomState(3)
    A = rng.rand(N, N)
    A = (A + A.T) / 2.0
    A[A < 0.6] = 0.0                       # 稀疏化，更接近真实邻接矩阵
    np.fill_diagonal(A, 0.0)
    A[0, :] = 0.0
    A[:, 0] = 0.0                          # 节点 0 两个方向都清空（PeMS-BAY 有 6 个这样的节点；
                                           # 只清空行则 A_f 为零而 A_b 仍非零，那是另一回事）

    # ---- A0. build_gwn_supports 必须与 GWNet.__init__ 的构造逐位相同。
    #     抄一份支撑集出来是为了不改动 GWNet 本身；一旦有人只改了一边，
    #     两臂比的就成了两种不同的图，这里会立刻报出来。
    sup = build_gwn_supports(A)
    gw = GWNet(A, N, adaptive=False, in_dim=2)
    for i, nm in enumerate(('A_f', 'A_b')):
        ref = getattr(gw, nm)
        if tuple(sup[i].shape) != tuple(ref.shape):
            bad.append(f'{nm} 形状 {tuple(sup[i].shape)} != GWNet 的 {tuple(ref.shape)}')
        elif not torch.equal(sup[i], ref):
            bad.append(f'{nm} 与 GWNet.__init__ 不逐位相等（最大差 '
                       f'{float((sup[i] - ref).abs().max()):.3e}）')
    if float(sup[0][0].abs().sum()) != 0.0:
        bad.append('孤立节点 0 的 A_f 行不为零 —— d_inv 的除零保护失效')
    del gw

    def mk(arm, seed=0):
        torch.manual_seed(seed)
        m = build_model(arm, A, N).eval().to(DEVICE)
        return m

    c_in = OFFICIAL_IN_DIM['MAMBAST']
    x = torch.randn(nb, 12, N, c_in, device=DEVICE)
    m_i, m_g, m_n = mk('MAMBAST'), mk('MAMBASTG'), mk('MAMBASTN')
    pi, pg, pn = m_i.num_params, m_g.num_params, m_n.num_params
    info.append(f'参数量 OFF={pi:,} ON={pg:,} 无图模块={pn:,}')

    # ---- A1. 参数量与 state_dict 的键/形状必须完全相同
    if pi != pg:
        bad.append(f'★ MAMBAST 与 MAMBASTG 参数量不等（{pi:,} vs {pg:,}）—— '
                   f'恒等对照的前提被破坏，两臂差值不再可归因于图机制')
    si, sg = m_i.state_dict(), m_g.state_dict()
    if list(si.keys()) != list(sg.keys()):
        bad.append(f'两臂 state_dict 的键不同：只在 OFF {sorted(set(si) - set(sg))[:5]}，'
                   f'只在 ON {sorted(set(sg) - set(si))[:5]}')
    else:
        sh = [k for k in si if tuple(si[k].shape) != tuple(sg[k].shape)]
        if sh:
            bad.append(f'两臂 state_dict 形状不同：{sh[:5]}')
    if not pi > pn:
        bad.append(f'MAMBASTN 本该比 MAMBAST 少参数（少一个图模块），'
                   f'实测 {pn:,} vs {pi:,}')

    with torch.no_grad():
        # ---- B. 支撑集敏感性（必须用非零输入：零输入经扩散后还是零，bias 抵消不掉）
        z = torch.randn(nb, m_g.d_model, N, 12, device=DEVICE)
        o0 = m_g.gconvs[0](z, [m_g.A_f, m_g.A_b])
        deg = A.sum(axis=1)
        i_, j_ = int(deg.argmax()), int(np.flatnonzero(A[int(deg.argmax())])[0])
        S2 = m_g.A_f.clone()
        S2[i_, j_] += 0.5
        o1 = m_g.gconvs[0](z, [S2, m_g.A_b])
        dsup = float((o1 - o0).abs().max())
        if not dsup > 1e-6:
            bad.append(f'扰动 A_f[{i_},{j_}] 后 gconvs[0] 输出不变（最大差 {dsup:.3e}）'
                       f'—— 图通路是死的（GWNet 的 residual_convs 那类死支路）')
        info.append(f'支撑集扰动响应={dsup:.3e}')

        # ---- C. 节点局域性：支撑集只有一个非零元 S[i,j]=1，只扰动节点 j 的输入，
        #         变化只能落在节点 i 上。bias 在两个前向里相同，相减自动抵消。
        S = torch.zeros(N, N, device=DEVICE)
        S[i_, j_] = 1.0
        zb = torch.zeros(nb, m_g.d_model, N, 12, device=DEVICE)
        zj = zb.clone()
        zj[:, :, j_, :] = 1.0
        d = (m_g.gconvs[0](zj, [S, S]) - m_g.gconvs[0](zb, [S, S])).abs().amax(dim=(0, 1, 3))
        if not float(d[i_]) > 1e-6:
            bad.append(f'扰动节点 {j_} 未影响节点 {i_}（支撑集 S[{i_},{j_}]=1）—— '
                       f'图卷积没有在缩并节点轴（或 A 没被用上）')
        leak = [int(n) for n in range(N) if n != i_ and float(d[n]) > 0.0]
        if leak:
            bad.append(f'支撑集只有一个非零元，扰动却泄漏到节点 {leak[:6]} —— '
                       f'einsum 的轴选错了')
        if float(d[j_]) > 0.0:
            bad.append(f'变化落在节点 {j_} 而不是 {i_} —— A 被转置了'
                       f'（einsum 的 n/m 取反）')

        # ---- D. 关图无串味 / 开图有串味
        xb = x.clone()
        xb[:, :, j_, :] += 3.0
        # ★ 模型输出是 (B, L, N)，逐节点要压掉的是 (0,1) 而不是 (0,2)。
        #   写 (0,2) 会留下 (L,)=12 这一维，后面按节点号索引就越界 ——
        #   这正是本仓库"不报错、只出错数"的典型形态，别顺手改回去。
        pn_i = (m_i(xb) - m_i(x)).abs().amax(dim=(0, 1))       # (N,) 逐节点
        if not float(pn_i[j_]) > 1e-6:
            bad.append(f'图关臂：扰动节点 {j_} 后它自己的输出没变 —— 测试本身失效')
        leak = [int(n) for n in range(N) if n != j_ and float(pn_i[n]) != 0.0]
        if leak:
            bad.append(f'★ 图关臂：扰动节点 {j_} 泄漏到节点 {leak[:6]} —— '
                       f'恒等支撑集下不该存在任何跨节点通路')
        pn_g = (m_g(xb) - m_g(x)).abs().amax(dim=(0, 1))       # (N,) 逐节点
        nbr = np.flatnonzero(A[j_])
        hit = [int(n) for n in nbr if float(pn_g[n]) > 0.0]
        if not hit:
            bad.append(f'★ 图开臂：扰动节点 {j_} 没有影响它的任何邻居 —— '
                       f'开图与关图行为一致，图机制实际没接上')
        info.append(f'图开臂邻居命中 {len(hit)}/{len(nbr)}')

        # ---- E. 节点置换等变性。★ 必须**同时**置换支撑集：只换输入不换支撑集，
        #         测的就不再是等变性，而是"支撑集与输入不一致"引起的另一个错误。
        perm = np.random.RandomState(7).permutation(N)
        pt = torch.as_tensor(perm, dtype=torch.long, device=DEVICE)
        inv = torch.argsort(pt)                        # inv[perm[k]] == k
        for nm_, need in (('MAMBAST', False), ('MAMBASTG', True)):
            # ★ 必须用**两个**模型实例：一个是原始支撑集、一个是置换后的支撑集。
            #   第一版图省事只用一个：先把支撑集换成置换版，再拿它去跑**未置换**的
            #   输入当基准 a，于是 a 本身就是个 (输入, 支撑集) 配错的量 ——
            #   报出来的 2.13 是测试自己的错，不是模型的。手工逐层复算 d=6.6e-7。
            m_a = mk(nm_)                              # 原始支撑集
            m_b = mk(nm_)                              # 同一份权重（mk 里重新 seed 0）
            if need:
                m_b.A_f = m_b.A_f[pt][:, pt].contiguous()
                m_b.A_b = m_b.A_b[pt][:, pt].contiguous()
            ka = [k for k in m_a.state_dict() if k not in ('A_f', 'A_b')]
            same = all(torch.equal(m_a.state_dict()[k], m_b.state_dict()[k])
                       for k in ka)
            if not same:
                bad.append(f'{nm_} 等变性自检失效：两个实例的权重不一致，'
                           f'比的就不只是支撑集了')
            # ★ 输入是 4D (B,L,N,C)，输出是 3D (B,L,N)。节点轴都在 dim=2，
            #   但下标个数不同：输入 x[:, :, pt, :]，输出 b[:, :, inv]。
            #   多写一个 ':' 就 IndexError；少写一个会更糟（静默切错轴）。
            a = m_a(x)                                 # (B, L, N) 原始节点序
            b = m_b(x[:, :, pt, :])                    # (B, L, N) 节点已置换
            d_ok = float((b[:, :, inv] - a).abs().max())
            d_bad = float((b - a).abs().max())
            if not d_ok < 1e-4:
                bad.append(f'{nm_} 置换等变性不成立：置换输入并还原后最大差 {d_ok:.3e}')
            if not d_bad > 1e-6:
                bad.append(f'{nm_} 置换节点顺序后输出居然不变 —— 模型根本没在看节点身份')
            del m_a, m_b

        # ---- F. 分块 vs 不分块。省显存不能改数：块内算子（Linear、末维
        #         depthwise Conv1d、scan、RMSNorm）逐行独立、无跨行归约。
        #   ★ 2026-09-26 修订（4090 上实测后）。本检查要抓的是「块边界把节点写错
        #     槽位」这类逻辑 bug，其签名是受影响节点错到 O(1e-1)~O(1) 相对量级。
        #     · 手写扫描：CPU float32 **逐位相等**；CUDA float32 **不等**，相对差
        #       3.4e-5（MAMBAST）/ 2.1e-5（MAMBASTG）—— 块内喂给 cuBLAS 的批形状
        #       不同（nc 行 vs B*N 行），kernel 选择随之改变，float32 求和顺序差异
        #       被 12 步递推放大；逐节点差弥散（图开臂 2.3× 跨度，图关臂 415× 但
        #       最大仅 3.4e-5），无任何 O(1e-2) 以上离群点。故该分支的**判定改用
        #       float64 逐位相等**：索引类 bug 不会因升精度而消失，所以它在设备
        #       无关意义上就是「分块没有改数」的证明，强于任何容差。
        #     · 官方 CUDA 核：selective_scan_cuda **只吃 float/half/bfloat16**，
        #       升 float64 会直接 RuntimeError（实测踩到过），所以上面那条判据在
        #       这条分支上用不了。官方 scan 逐行独立，分块喂进去的仍是同一批行，
        #       故期望逐位相等；这里只留 1e-3 的宽松红线，并把实测值打进信息行
        #       —— 若实测确为 0，再把这个分支也收紧成逐位相等。
        def _chunk_pair(mod, xin):
            mod.budget = 10 ** 9
            a = mod(xin)
            mod.budget = 5
            b = mod(xin)
            mod.budget = MAMBA_SEQ_BUDGET
            return a, b

        y_w, y_c = _chunk_pair(m_g, x)
        bit32 = bool(torch.equal(y_w, y_c))
        dmax = float((y_w - y_c).abs().max())
        scale = max(1.0, float(y_w.abs().max()))

        if getattr(m_g, 'official', False):
            if dmax > 1e-3 * scale:
                bad.append(f'官方核下分块与不分块结果差 {dmax:.3e}，超出浮点噪声量级'
                           f'（输出量级 {scale:.3e}）')
            info.append(f'分块一致性=官方核 '
                        f'{"逐位相等" if bit32 else f"差 {dmax:.2e}"}')
        else:
            m64 = m_g.double()
            y64_w, y64_c = _chunk_pair(m64, x.double())
            bit64 = bool(torch.equal(y64_w, y64_c))
            d64 = float((y64_w - y64_c).abs().max())
            del m64
            m_g.float()                                # 还原：后面的检查还要用 m_g
            if not bit64 and d64 > 1e-8 * scale:
                bad.append(f'分块与不分块在 float64 下不逐位相等（差 {d64:.3e}）'
                           f' —— 这不是浮点噪声，是块逻辑改了数')
            if dmax > 1e-3 * scale:
                bad.append(f'分块与不分块结果差 {dmax:.3e}，超出浮点噪声量级'
                           f'（输出量级 {scale:.3e}）')
            info.append(f'分块一致性={"逐位相等" if bit32 else f"float32 差 {dmax:.2e}"}'
                        f' / float64 {"逐位相等" if bit64 else f"差 {d64:.2e}"}')

        # ---- 输出量级（§3.3 的 nan 排查口径：O(1e0)~O(1e3)，不是 1e4+）
        am = float(m_i(x).abs().max())
        if not am < 1e4:
            bad.append(f'图关臂输出 absmax={am:.3e} —— 数量级爆了（块内残差被重复加过？）')
        info.append(f'输出 absmax={am:.3g}')

    # ---- G. flatten_node_windows 的布局
    zt = torch.randn(2, 12, N, 3)
    fz = flatten_node_windows(zt)
    ok = all(torch.equal(fz[b * N + n], zt[b, :, n, :])
             for b in range(2) for n in (0, 5, N - 1))
    if not ok:
        bad.append('flatten_node_windows 布局错误：z[b*N+n] ≠ x[b,:,n,:]')
    else:
        wrong = zt.reshape(2 * N, 12, 3)      # "不多写 permute" 的那个写法
        if torch.equal(fz, wrong):
            bad.append('flatten 结果与不 permute 的写法相同 —— 这个测试没有鉴别力'
                       '（形状恰好退化），换一个 N/L 让它有意义')
        else:
            nd = sum(1 for r in range(2 * N) if not torch.equal(fz[r], wrong[r]))
            info.append(f'flatten 与错误写法有 {nd}/{2 * N} 行不同')
    del m_i, m_g, m_n
    return bad, '；'.join(info)


def _selftest_causal():
    """§3.4 因果泄漏：模型绝不能看到 y。

    分两层，缺一层都不算测到：
      结构层 —— 从 train_one 的源码里数 model 调用点，确认吃真值的那一条被守卫。
      数值层 —— 把 d24 的未来 12 列换成垃圾，prepare_split 出的 X 必须**逐位不变**。
                同时 Y 必须变 —— 否则说明根本没改到目标列，测试没有鉴别力。
    """
    import inspect
    bad = []

    # ---- 结构层
    src = inspect.getsource(train_one).split('\n')
    calls = [i for i, l in enumerate(src) if 'model(xb' in l]
    if len(calls) != 2:
        bad.append(f'train_one 里 model(xb...) 有 {len(calls)} 处，应为 2 处'
                   f'（DCRNN 的 teacher forcing + 其余模型的单参调用）')
    else:
        head = src[calls[0]].strip()
        guard = '\n'.join(src[max(0, calls[0] - 2):calls[0]])
        if 'yb' not in head:
            bad.append(f'第一个调用点 {head!r} 不吃真值序列，与预期不符')
        elif "name == 'DCRNN'" not in guard:
            bad.append(f'第一个调用点 {head!r} 没有被 if name == \'DCRNN\' 守着 —— '
                       f'真值序列可能进了非 DCRNN 模型')
    if 'teacher' in inspect.signature(MambaST.forward).parameters:
        bad.append('MambaST.forward 的签名里有 teacher —— 结构上就能看到 y')

    # ---- 数值层：合成窗口张量，不碰任何真实数据文件。
    #      必须在 12 列之后放**足够远**的值：只改 y 那 12 列，x 那 12 列不动。
    N, T = 5, 40
    rng = np.random.RandomState(11)
    S = (rng.rand(N, T + 23).astype(np.float32) + 0.5)
    d24 = np.empty((N, T, 24), dtype=np.float32)
    for t in range(T):
        d24[:, t, :12] = S[:, t:t + 12]
        d24[:, t, 12:] = S[:, t + 12:t + 24]
    d0, d1 = d24.copy(), d24.copy()
    d1[:, :, 12:] = -999.0                      # 未来 12 列换成垃圾
    z = np.zeros((N, 1, 1), dtype=np.float32)
    on = np.ones((N, 1, 1), dtype=np.float32)
    X0, Y0 = prepare_split(d0, z, on, z, on, '泄漏自检-原样', in_dim=1)
    X1, Y1 = prepare_split(d1, z, on, z, on, '泄漏自检-改未来', in_dim=1)
    if not np.array_equal(X0, X1):
        bad.append(f'★ 因果泄漏：改掉未来 12 列后 X 变了（最大差 '
                   f'{float(np.abs(X0 - X1).max()):.3e}）—— 输入里混进了未来信息')
    if np.array_equal(Y0, Y1):
        bad.append('测试失效：改掉未来 12 列后 Y 没变 —— 没改到目标列，'
                   '上面的"X 不变"就不说明任何问题')
    return bad


def selftest():
    log('=' * 78)
    log(f'selftest（不训练、不碰真实数据）   设备={DEVICE}')
    log('=' * 78)
    checks = []

    # --- 1. 指标自检
    try:
        bad = _selftest_metrics()
        checks.append(('指标数值自检（掩零 / MAPE_MIN / 掩码损失）', not bad,
                       '; '.join(bad) if bad else '全部与手算值一致（容差 1e-9）'))
    except Exception as e:
        checks.append(('指标数值自检', False, f'{type(e).__name__}: {e}'))

    # --- 2&3. 前向 shape + 反向梯度
    try:
        rows = _selftest_forward()
        for r in rows:
            checks.append((
                f'{r["name"]} 前向 shape',
                r['shape_ok'] and r['finite'],
                f'入 {r["in_shape"]}（官方处方 {OFFICIAL_IN_DIM[r["name"]]} 通道）'
                f' → 出 {r["shape"]}  有限={r["finite"]}  '
                f'absmax={r["output_absmax"]:.3g}'))
            checks.append((
                f'{r["name"]} 反向梯度有限且参数都接到图上',
                r['grad_ok'],
                ('全部参数梯度有限' if r['grad_ok'] else
                 f'非有限梯度 {r["grad_bad"]}  意外无梯度 {r["grad_none"]}  '
                 f'预期该无梯度却有 {r["dead_alive"]}') +
                (f'；死支路 {r["dead"]} 个参数（GWNet 全部 residual_convs + '
                 f'末层 gconv/bn，官方结构同样如此）' if r['dead'] else '') +
                f'   loss={r["loss"]:.4g}'))
        log('\n模型参数量（论文里谈参数效率要用）：')
        for r in rows:
            log(f'  {r["name"]:<8} {r["params"]:>10,} 参数   '
                f'输出 {r["shape"]}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        checks.append(('前向/反向自检', False, f'{type(e).__name__}: {e}'))

    # --- 4. MambaST 空间机制（计划 §3.1–3.3）
    try:
        bad_, info = _selftest_mambast()
        checks.append((
            'MambaST 空间机制（支撑集同源 / 图通路活性 / 节点局域性 / '
            '置换等变 / 分块一致）',
            not bad_,
            '；'.join(bad_) if bad_ else info))
    except Exception as e:
        import traceback
        traceback.print_exc()
        checks.append(('MambaST 空间机制自检', False, f'{type(e).__name__}: {e}'))

    # --- 5. 因果泄漏（计划 §3.4）
    try:
        bad_ = _selftest_causal()
        checks.append((
            '因果泄漏（结构 + 数值）',
            not bad_,
            '；'.join(bad_) if bad_ else
            'train_one 里真值只进 DCRNN 支路；改掉未来 12 列后 X 逐位不变、Y 必变'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        checks.append(('因果泄漏自检', False, f'{type(e).__name__}: {e}'))

    log('\n' + '-' * 78)
    n_bad = 0
    for nm, ok, detail in checks:
        log(f'  [{"通过" if ok else "失败"}] {nm}: {detail}')
        n_bad += (0 if ok else 1)
    log('-' * 78)
    log(f'selftest 汇总：{len(checks) - n_bad}/{len(checks)} 项通过'
        f'{"，全部通过" if n_bad == 0 else f"，{n_bad} 项失败"}')
    log('注意：selftest 只验证「形状对、数值不炸、指标口径与手算一致」，')
    log('      不验证模型精度 —— 精度只能真训一轮才知道。')
    log('      这三个模型都很小（0.04–0.37 M 参数，输入 (B,12,N,C)），')
    log('      6 GB 显存足够，**本机就能跑**，不必为基线租卡；')
    log('      租卡预算留给 paper_experiments 的 verify 协议。')
    if n_bad:
        sys.exit(1)


# ============================================================================
#  checkdata：只读数据 + 布局自检 + 打印时刻数，不训练、不写文件
# ============================================================================
def checkdata():
    log('数据自检（不训练、不写文件）')
    for ds in WANT_DS:
        tr, va, te, adj, N, tds = load_dataset(ds)
        deg = (adj > 0).sum(axis=1)
        log(f'    ✓ 形状 train{tr.shape} val{va.shape} test{te.shape}')
        if tds is None:
            log(f'      时段通道: 无（PAPER_GNN_TOD=0，或数据里没有 _d1 列）')
        else:
            log(f'      时段通道: {[t.shape for t in tds]}  '
                f'取值范围 {float(tds[0].min()):.4f}~{float(tds[0].max()):.4f}')
        log(f'      官方输入处方: ' + '  '.join(
            f'{m}={OFFICIAL_IN_DIM[m]}通道' for m in ('DCRNN', 'GWNET', 'STGCN')))
        log(f'      train 时刻数={tr.shape[1]:,}  val={va.shape[1]:,}  test={te.shape[1]:,}'
            f'   节点数={N}')
        log(f'      邻接非零 {int((adj > 0).sum()):,}  '
            f'邻居数 中位{int(np.median(deg))}/最大{int(deg.max())}  '
            f'孤立节点 {int((deg == 0).sum())}  '
            f'稀疏度 {100 * (adj == 0).mean():.1f}%')
        # 跨划分连续性：train 末尾与 val 开头、val 末尾与 test 开头应当首尾相接
        for a, b, nm in ((tr, va, 'train→val'), (va, te, 'val→test')):
            gap = float(np.abs(a[:, -1, 1:12] - b[:, 0, 0:11]).mean())
            log(f'      {nm} 接缝重叠误差={gap:.6f}（应≈0）')
        # 标准化前的原始值范围（掩零比例也顺便报一下，METR-LA 有 12.1% 的 0）
        for nm, arr in (('train', tr), ('val', va), ('test', te)):
            y = arr[:, :, 12:]
            log(f'      {nm} 目标 真值为0占比={100 * (y == 0).mean():.1f}%  '
                f'min={y.min():.2f} max={y.max():.2f}')
        del tr, va, te, adj, tds
    log('')
    log('（checkdata 模式：不训练、不写文件）')


# ============================================================================
#  主流程
# ============================================================================
def main():
    global MODE
    ap = argparse.ArgumentParser(
        description='DCRNN / STGCN / Graph WaveNet 基线（协议对齐 paper_experiments.py）')
    ap.add_argument('--selftest', action='store_true',
                    help='只做 shape 与指标数值自检，不训练、不碰数据文件')
    ap.add_argument('--checkdata', action='store_true',
                    help='只验数据读得通 + 布局自检，不训练')
    ap.add_argument('--train', action='store_true', help='跑训练（默认）')
    args = ap.parse_args()
    if args.selftest:
        MODE = 'selftest'
    elif args.checkdata:
        MODE = 'checkdata'
    elif args.train:
        MODE = 'train'

    if MODE == 'selftest':
        selftest()
        return
    if MODE == 'checkdata':
        log('=' * 78)
        log(f'GNN 基线   模式=checkdata   设备={DEVICE}')
        log('=' * 78)
        checkdata()
        return
    if MODE != 'train':
        raise SystemExit(f'未知 PAPER_MODE={MODE}，可选 checkdata / train / selftest')

    log('=' * 78)
    log(f'GNN 基线   模式=train   设备={DEVICE}')
    log(f'  模型 {WANT_MODELS}   数据集 {WANT_DS}   种子 {SEEDS}')
    log(f'  轮数 {EPOCHS}   bs {BS}   早停 patience {PATIENCE}   梯度裁剪 {CLIP}')
    log(f'  数据源策略 PAPER_GNN_SOURCE={SOURCE}')
    if DEVICE == 'cuda':
        p = torch.cuda.get_device_properties(0)
        log(f'  {torch.cuda.get_device_name(0)}  {p.total_memory / 1024 ** 3:.1f}GB  '
            f'torch {torch.__version__}')
    log('=' * 78)

    results = {}
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, encoding='utf-8') as f:
            results = json.load(f)
        log(f'续跑：已有 {len(results)} 条结果')

    def save_results():
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    jobs = [(ds, mdl, sd) for ds in WANT_DS for mdl in WANT_MODELS for sd in SEEDS]
    log(f'\n共 {len(jobs)} 个配置')
    if not jobs:
        log('没有配置可跑（PAPER_GNN_MODELS / PAPER_GNN_DATASETS 是不是设空了？）')
        return

    for ds_name, model_name, seed in jobs:
        # ★ 配置 ID 必须包含模型名 + 数据集 + 种子。
        #   主脚本的 ID 形如 `PeMS-BAY|real|V8|official|s1`，不含"要跑哪个实验"，
        #   曾经出现过不同配置撞同一个 ID 而被静默跳过的事故。这里带上模型名，
        #   且模型名在 ID 里的位置固定，不会与别的配置前缀相同。
        # ★ 2026-09-24：输入通道数写进 ID。in_dim 从 1 改成官方处方（GWNet/DCRNN
        #   为 2）之后，旧 ID 下的结果就作废了 —— 但 ID 不变的话续跑逻辑会认成
        #   「已有结果」直接跳过，报出来的还是 1 通道的旧数字，看起来却像新跑的。
        # ★ 2026-09-24：轮数也写进 ID。理由与上面 in_dim 那条完全一样 ——
        #   cid 里不含 EPOCHS 时，先用 30 轮跑一批、日后改成 100 轮（或反过来），
        #   续跑逻辑会把旧协议的结认成「已有结果」直接跳过，表格里看着是新跑的、
        #   实际是旧轮数的数。本仓库已经因为同一类问题踩过一次（in_dim），
        #   所以这里不再留口子。已确认全脚本只有 cid.replace('|','_') 一处用到
        #   cid，无位置解析，加字段安全。
        #   注意：这会让本次之前产出的 cid 全部失效（旧结果不会被复用），
        #   这正是想要的 —— 旧结果没有轮数标注，无法确认它是几轮跑的。
        in_dim = OFFICIAL_IN_DIM[model_name]
        cid = f'{ds_name}|{model_name}|s{seed}|c{in_dim}|e{EPOCHS}'
        if cid in results:
            log(f'\n[跳过] {cid}（已有结果）')
            continue
        log(f'\n{"=" * 78}\n[{len(results) + 1}/{len(jobs)}] {cid}\n{"=" * 78}')
        try:
            tr3, va3, te3, adj, N, tods = load_dataset(ds_name)

            # --- 反标准化参数：只用训练集算（per-node），与主脚本一致
            # ★ 2026-09-24 改：逐节点统计量只按**有效值（>0）**算 —— 与主脚本同一口径。
            #   原先把缺测的 0 也算进 mean/std，两者都被拉偏；更要紧的是输入侧没填，
            #   一个 66.7 mph 的节点收到 0 就是 -55σ（主脚本那场 nan 的根因）。
            #   这里必须同改：只改主脚本等于让基线和 Mamba 吃不同的输入，
            #   偏差方向恰好朝有利于本文模型的一侧，属于「不能靠着错误硬编」要挡的。
            ty = tr3[:, :, 12:]
            tx = tr3[:, :, :12]

            def _stat(a):
                """逐节点 mean/std，缺测（0）不计入。整节点全缺测时退化为 0/1。"""
                ok = (a > 0).any(axis=(1, 2), keepdims=True)
                m = a > 0
                n = np.maximum(m.sum(axis=(1, 2), keepdims=True), 1)
                mu = (a * m).sum(axis=(1, 2), keepdims=True) / n
                var = (((a - mu) ** 2) * m).sum(axis=(1, 2), keepdims=True) / n
                return (np.where(ok, mu, 0.0),
                        np.where(ok, np.sqrt(var), 1.0) + 1e-8)

            mean_x, std_x = _stat(tx)
            mean_y, std_y = _stat(ty)
            del ty

            # 分布漂移仪表：主脚本那场 nan 就是靠这条线索定位的
            # （train 极差 4.8，测试期偏差到 65.8σ）。这里每个配置留一行。
            for _tag, _a in (('train', tx), ('val', va3[:, :, :12]),
                             ('test', te3[:, :, :12])):
                _mx = max(abs(float(_a.min())), abs(float(_a.max())))
                _sig = float(np.abs((_a - mean_x) / std_x).max())
                log(f'  [分布] {_tag}: 原始 absmax={_mx:.4g}  '
                    f'标准化后 absmax={_sig:.2f}σ')

            # 输入处方：in_dim>1 才需要 tod（STGCN 恒为 1，不给它）
            _td = tods if in_dim > 1 else None
            if in_dim > 1 and _td is None:
                raise TodUnavailable(
                    f'{cid}: 官方处方要 {in_dim} 通道（含 tod），但 load_dataset '
                    f'没给出时段通道。检查 PAPER_GNN_TOD 与 PAPER_GNN_SOURCE。')
            if _td is not None:
                log(f'  输入处方=官方：速度 + tod（{in_dim} 通道，tod 保持 [0,1) 原值，'
                    f'仅第 0 通道做 z-score）  时段范围 '
                    f'{float(_td[0].min()):.4f}~{float(_td[0].max()):.4f}')
            else:
                log(f'  输入处方：仅速度（{in_dim} 通道）')

            Xtr, Ytr = prepare_split(tr3, mean_x, std_x, mean_y, std_y, 'train',
                                     tod=None if _td is None else _td[0],
                                     in_dim=in_dim)
            del tr3
            Xva, Yva = prepare_split(va3, mean_x, std_x, mean_y, std_y, 'val',
                                     tod=None if _td is None else _td[1],
                                     in_dim=in_dim)
            del va3
            Xte, Yte = prepare_split(te3, mean_x, std_x, mean_y, std_y, 'test',
                                     tod=None if _td is None else _td[2],
                                     in_dim=in_dim)
            del te3, tods
            log(f'  张量 X{Xtr.shape} Y{Ytr.shape}   '
                f'（GNN 基线吃全图历史 (T,12,N,{in_dim})，不是逐节点窗口）')

            model = build_model(model_name, adj, N)
            del adj
            data = ((Xtr, Ytr), (Xva, Yva), (Xte, Yte), std_y, mean_y, N)
            tag = cid.replace('|', '_')
            m = train_one(model_name, model, data, seed, EPOCHS, BS, tag)
            del data, model
            if m is None:
                continue
            _tc = TRAIN_CFG[model_name]
            m['cfg'] = dict(
                dataset=ds_name, model=model_name, seed=seed, N=N,
                bs=BS, epochs=EPOCHS, source=SOURCE, device=DEVICE,
                lr=float(os.environ.get(_tc['lr_env'], _tc['lr'])),
                weight_decay=float(os.environ.get(_tc['wd_env'], _tc['wd'])),
                sched=SCHED if SCHED != 'auto' else _tc['sched'],
                sched_src=_tc['src'],
                patience=PATIENCE, clip=CLIP, val_n=VAL_N,
                loss='masked MSE（主脚本口径；三个官方实现均为 masked MAE）',
                dcrnn_cheb=DCRNN_CHEB, dcrnn_layers=int(
                    os.environ.get('PAPER_GNN_DCRNN_LAYERS', 2)),
                gw_adaptive=GW_ADAPT,
                in_dim=in_dim,
                gnn_tod=GNN_TOD,
                in_dim_src=(
                    'DCRNN: data/model/*.yaml input_dim=2；'
                    'GWNet: train.py --in_dim default=2；'
                    'STGCN: trainer.py 通道维写死 1'
                    ) if GNN_TOD else 'PAPER_GNN_TOD=0，三家统统砍成 1 通道（非官方）',
                input_layout=f'(B,12,N,{in_dim}) 全图历史',
                ckpt_best=f'{tag}_best.pt', ckpt_last=f'{tag}_last.pt')
            results[cid] = m
            save_results()
            log(f'  ✓ MAE@15={m["mae_15"]:.3f} @30={m["mae_30"]:.3f} '
                f'@60={m["mae_60"]:.3f}  RMSE@60={m["rmse_60"]:.3f}  '
                f'MAPE@60={m["mape_60"]:.2f}%（覆盖率 {m["mape_cov_60"]:.3f}）')
        except Exception as e:
            log(f'  [失败] {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()

    if not results:
        log('\n没有结果。')
        return

    log('\n' + '=' * 100)
    log('GNN 基线汇总（MAPE 为过滤近零真值后的值）')
    log('=' * 100)
    hdr = (f'{"配置":<26} {"参数":>9} '
           f'{"MAE@15":>7} {"MAE@30":>7} {"MAE@60":>7} '
           f'{"RMSE@60":>8} {"MAPE@60":>9} {"s/ep":>7}')
    log(hdr)
    log('-' * len(hdr))
    for cid, m in sorted(results.items()):
        log(f'{cid:<26} {m["params"]:>9,} '
            f'{m["mae_15"]:>7.3f} {m["mae_30"]:>7.3f} {m["mae_60"]:>7.3f} '
            f'{m["rmse_60"]:>8.3f} '
            f'{m["mape_60"]:>8.2f}% {m["epoch_time"]:>7.1f}')
    log('=' * 100)
    log(f'结果已写入 {RESULT_FILE}')
    log("""
    这张表怎么用（写论文前先看这三条）：

      1. 三个模型是在**同一份 parquet 划分、同一套掩零指标**下跑的，
         可以直接和 paper_results.json 里的持久性基线 / Mamba 主模型并表。

      2. ⚠️ 输入不对等：GNN 基线看全部节点（325 / 207），Mamba 只看 top-5 邻居。
         这是刻意偏袒基线的保守设置，必须在"实验设置"里写明。

      3. ⚠️ STGCN 的多步输出是"最后一层时间卷积压成 12 个 horizon"，
         不是逐步预测（见 STGCN 类注释）。若它在 @60 上明显吃亏，
         要区分"模型不行"和"多步适配方式不行"这两件事。
    """)


if __name__ == '__main__':
    main()
