# 三任务 BO 评估与结构搜索

流程为共享策略 → 每个候选的三任务联合 PPO 微调 → 固定工况独立测量 → BO 选下一个结构。按 [专家训练与运行操作](专家训练与运行操作.md) 和 [共享策略蒸馏详细说明](共享策略蒸馏详细说明.md) 完成前置步骤。入口自动读取默认实验目录中的共同结构表和选定的共享模型。

使用一次性迁移的结构库时，在 `evaluate` 和 `search` 命令末尾加 `--root robolab/logs/multitask_full2500_physical_migrated_v1`。

## 1. 测一个候选

进入服务器上的仓库根目录。把 `真实结构ID` 换成共同库中的 Rough ID。命令会完成候选的专属微调，再做 BO 测量。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py evaluate --candidate 真实结构ID --gpu 0
```

候选默认微调 100 次。已有同一共享模型、同一候选和同一预算的完整微调目录时，附加 `--finetuned 该目录`，直接开展测量。普通故障后用原命令继续。人工暂停后加 `--resume-paused`。默认输出位于实验根的 `bo/<结构ID>/`。

### 预试验工况

工况来自 `configs/bo_pilot_conditions.json`。可用 `--conditions` 指定另一个冻结工况文件。

| 任务 | 指令 | 固定场地与负载 |
|---|---:|---|
| 跑步 | 2.0 m/s | 水平面，30 m 路线 |
| 负重 | 0.5 m/s | 水平面，8 m 路线，背包实装 10 kg，沿用原有安装位置 |
| 上楼 | 0.5 m/s | 1 m 进场，12 级，级高 0.075 m、踏面深 0.25 m，标记到达上平台 |

预试验每个任务 20 秒、2 个种子、每种子 2 个副本。每个种子与副本序号对应一组三任务重复。仿真在原有环境的摔倒触发条件下运行。固定工况禁用命令重采样、站立和转向指令。正式实验前需确认这些工况、测量窗口和权重。

### 指标与成败

| 任务 | 三项代价，均为越小越好 |
|---|---|
| 跑步、负重 | 整机机械 CoT，前向速度 RMSE，六个膝踝电机中最大的持续额定扭矩占比 |
| 楼梯 | 整机机械 CoT，前向速度 RMSE，机身横滚与俯仰角速度的合成 RMS |

- **CoT** 为 `正机械功 / (有效整机质量 × 9.81 × 实际前向位移)`。以 50 Hz 控制步末采样，累加全身各关节的 `max(施加扭矩 × 关节角速度, 0) × 0.02 s`。负重质量读取实际仿真整机质量，已包含背包。楼梯位移取水平路径方向，功继续按全身关节积分。不可用位移对应空 CoT。
- **速度 RMSE** 以共同路径方向的速度和固定指令比较。机身晃动按 `sqrt(mean(ωx² + ωy²))`，角速度用机身坐标系，单位 rad/s。
- **持续负荷** 用实际关节输出扭矩的 RMS 除以该型号持续额定输出扭矩，取双膝、双踝俯仰和双踝横滚六者最大值。全身 23 台逐关节保存额定占比、扭矩 RMS 和实测绝对峰值。额定分母缺失时对应占比留空。
- **试次失败** 由摔倒或平均有符号前向速度严格低于指令的 10% 触发。0.5 m/s 工况的速度边界为 0.05 m/s，恰好达到边界时只检查摔倒。路线完成、是否上到平台和保护中止独立记录。采集故障记为无效，重跑同一命令可复用已验收的其他任务测量。

`bo_samples.npz` 保留世界路径速度、根位置、机身角速度、实际扭矩、关节速度和有效段掩码。终止步使用重置前的状态。`bo_trials.json` 包含单次成败、实际载荷和安装位置、原始指标及逐关节数据。`bo_evaluation.json` 汇总三个任务的配对结果。原有 `candidate_result.json` 仍提供微调前后流程所用的命令跟踪评价。

## 2. 从基准得到归一化配置

先在同一工况作重复基准测量，再由基准报告生成配置。下面文件名是示例，可把基准结果保存为 `baseline.json` 后直接运行。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py calibrate baseline.json --benchmark-name RPO仿真基准 --output bo_scoring.json
```

至少需要两个三个任务同时成功且指标齐全的重复。每项用这些重复的均值作为基准，记录标准差。生成的是**等权预试验**配置。可靠正基准使用 `x / baseline`。近零基准可在相应项填写正的物理 `scale`，改用 `1 + (x - baseline) / scale`。三个任务的权重和、各任务内三项权重和分别为 1。确认后冻结工况、基准和权重，再开始正式搜索。仿真和实机分别校准。

基准可以来自同工况的原始 RPO 独立测量，也可以是命名明确的工程参照。若使用 `evaluate` 跑出的库内候选作参照，应按该候选命名。原始 RPO 各任务的同工况基准需实测采集。

### 导入原始 RPO 或实机测量

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py ingest --spec baseline_measurements.json --output baseline.json
```

测量清单采用 JSON。顶层需要 `domain`（`simulation` 或 `hardware`）、`measurement_convention`（`rpo_three_task_mechanical_positive_v1`）、`dt_s`（0.02）、`candidate` 和 `trials`。每条试次填写 `task`（`stairs`、`backpack` 或 `run`）、`seed`、从 0 开始的 `replica`、`trace`、实测 `total_mass_kg`、`fell` 及 `motors`。可同时标记 `reached_top`、`route_completed` 和 `protection_stop`。负重加 `payload.mass_kg` 和 `payload.mount_position_torso_m`。

每条 `trace` 是与清单同目录或相对路径的 NPZ，包含 `initial_position[3]`、`position[N,3]`、`vx[N]`、`omega[N,2]`、`torque[N,23]`、`joint_vel[N,23]`。23 列按 `motors` 顺序填写，每台记录 `joint`、`motor_id`、`torque_side: "joint_output"`、`rated_torque_nm` 和 `source`。请先把位置、速度和 IMU 轴统一到上面的约定，把驱动器输出校核为关节侧 N·m，并对齐到 50 Hz 采样。程序验证完整的种子、副本和任务覆盖。实机采集由测量系统执行。

## 3. 启动 BO

搜索使用训练环境中的 NumPy、SciPy 和 scikit-learn。缺少 scikit-learn 时可单独安装仓库已有的 BO 依赖范围 `scikit-learn>=1.4,<2`。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py search --scoring bo_scoring.json --trials 5 --gpu 0
```

前四个结构按腿长距离和电机差异做分散初始化。随后用联合可行概率加权期望改进，搜索范围为冻结的 5＋5 共同结构库。可行性模型读取全部有效配对标签。性能模型只读取三任务同时成功且九项指标齐全的配对。失败仍参与可行性，综合分为空。没有可行性能数据时继续探索联合可行性。

`--trials` 是本次搜索的总候选预算。增大该值并重跑可延长同一搜索。工况、归一化、共享模型或每候选预算变更时，显式用 `--output` 开新目录。候选记录含每组可行性及综合分，预算报告同时给出联合成功次数和有效次数。最小的已观测均分用于筛选后续复测对象，最终可靠性门槛由正式实验确定。

本地验证覆盖公式、配对、导入、约束 GP 和搜索恢复，并运行了真实负重采样以及固定楼梯网格生成。完整三任务 BO 搜索需本轮正式专家和共享模型。
