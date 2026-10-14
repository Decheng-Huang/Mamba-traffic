# 基于 Mamba 的短时交通流预测：空间信息消融（工程源码）

上海杉达学院 本科毕业设计 · 工程实现部分

本仓库只含**工程源码**与结果原始文件：数据预处理 → 模型与基线 → 训练 / 评估 → 结果 JSON → 论文表格。
**不含**论文正文（LaTeX / Word）、模型权重与数据集。

---

## 一、代码地图

### 模型与基线

| 文件 | 内容 |
| --- | --- |
| `mamba_official.py` | 主干模型。基于官方 `mamba_ssm` 的 selective-scan 核 |
| `gnn_baselines.py` | 图基线：Graph WaveNet、DCRNN |
| `baseline_models.py` | LSTM、Transformer |
| `ha_official.py` | 历史均值（Historical average） |
| `baseline_only.py` | 持久性（Persistence）基线 —— 零成本，先于租 GPU 跑完 |

输入侧固定取 **5 个邻居通道**（`paper_experiments.py:63` `NUM_NEIGHBORS = 5`），
另有一个可开关的时段通道（`PAPER_TOD`）。

### 数据

| 文件 | 内容 |
| --- | --- |
| `preprocess_local.py` | Caltrans PeMS 的 parquet → 张量。数据目录见 §三 |
| `build_pemsd4.py` | 把 ASTGCN 的 `pems04.npz` 转成本仓库的 parquet 口径（第二个基准） |
| `rebuild_data.py` | 从 parquet 重建「时刻对齐」的数据文件 |
| `download_metrla.py` | METR-LA 的下载与出处（该数据集最终未被论文采用，见文件头注） |

### 训练与实验

| 文件 | 内容 |
| --- | --- |
| `paper_experiments.py` | 主实验驱动。逐配置训练 → 写结果 JSON；含 `_provenance` 出身块 |
| `box_paper_experiments.py` | 在租用实例上批跑的驱动 |
| `compute_metrics.py` | MAE / RMSE / MAPE，并按缺测覆盖率给出 `mape_cov` |
| `repeat_experiments.py` | 多种子重复 |
| `input3arm_analysis.py` | **论文核心消融**：输入侧三臂（real / random / self）的配对分析 |
| `ab_verdict.py` | 时段通道 A/B 判决 —— 判据在看到数字之前写死 |
| `gate_gnn_graph.py` / `gate_gen_screen.py` | 合成图依赖门：先证明「这个任务真的需要图」，再决定要不要租卡 |
| `fingerprint_gnn.py` | `gnn_baselines.py` 的结构指纹，任何改动后跑一遍对质 |
| `merge_e100.py` | 把 `res_*.json` 里天花板 100 的键合并成 `paper_results_e100.json` |
| `recompute_baseline.py` | 用当前生产代码重算持久性基线，把它的定义钉死 |

### 结果与表格

| 文件 | 内容 |
| --- | --- |
| `make_tables.py` | `box_results/*.json` → 论文表格 |
| `_precision.py` | 数字显示精度的**唯一**定义处（`make_tables.py` 依赖它） |
| `box_results/` | 结果原始 JSON —— 上面那些表的输入 |
| `check_box.py` | 只读盘点实例上已经跑出了什么 |
| `check_parquet.py` | 时间戳是否连续递增、train / val / test 是否前后相接 |

### 在实例上批跑

`box_run_all.sh`、`box_reprov.sh`、`box_smoke.sh`、`box_chain_b.sh`、`box_fetch.sh`、`_unpack_*.sh`
—— 无人值守批跑、取回结果、带出生证明重跑。
脚本里的路径按当时租用的实例写死（`/root/autodl-tmp`），换机器要改。

---

## 二、环境

依赖由代码里**实际的 `import`** 扫出：

```
numpy  pandas  pyarrow  torch  mamba-ssm (+ causal-conv1d)
```

`mamba_ssm` 是官方 CUDA 核，必须与 `torch` / CUDA 版本匹配才能装上。

**本仓库刻意不锁定版本号**：当时的运行环境没有留下可核对的版本记录，
与其填一组猜出来的数字，不如在这里说明这一点。

---

## 三、数据

两个基准：**PeMS-BAY**（325 传感器）与 **PeMS04**（307 传感器），
同出 Caltrans PeMS 的 District 4。数据由原始提供方分发，**本仓库不再分发**。

`preprocess_local.py` 按这个顺序找数据目录：

1. 环境变量 `TRAFFIC_DATA_DIR`
2. 脚本同级的 `数据集/` 目录

该目录下应有 `train.parquet` / `val.parquet` / `test.parquet` / `adj_mx.npy`。

---

## 四、跑法

实验在租用的 NVIDIA GPU 实例上完成，流程是：

1. 本地预处理出张量 → 上传到实例
2. `bash box_smoke.sh` —— 冒烟。5 轮、单配置、独立结果文件，**不产出论文数字**
3. `nohup bash box_run_all.sh &` / `nohup bash box_reprov.sh &` —— 无人值守批跑
4. `bash box_fetch.sh` —— 把结果、出生证明、日志打包，供下载

结果取回本地之后：`merge_e100.py` → `make_tables.py` → 论文表格。

---

## 五、几点说明

- **出身块**：`paper_experiments.py` 会把解释器、CUDA、代码指纹写进结果文件。
  `box_results/` 里的文件多数**早于**这个机制，因此不含该块。
- **METR-LA**：论文原计划用 PeMS-BAY + METR-LA 两个数据集；METR-LA 在 2026-09-27
  被判定「管线级不复现」而弃用。相关脚本保留，文件头注写明了原因与判据。
- 数据与模型权重均未随包分发。

---

## 六、许可与引用

本仓库以 **MIT 许可**发布（见 `LICENSE`）：可以自由使用、修改、分发，**但必须保留版权声明**。

引用格式见 `CITATION.cff`；GitHub 会在仓库页右侧显示「Cite this repository」。

