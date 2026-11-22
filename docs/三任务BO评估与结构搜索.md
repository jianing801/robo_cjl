# 微调后的三任务评估与结构搜索

先完成上楼、负重和跑步均采用水平航向速度的鲁棒选模，确认 `shared_policy/selection.json` 和 `shared_policy/completed.json` 记录了获选模型及其身份，后者的 `protocol.evaluation_velocity_frame` 为 `horizontal_heading`。下列命令在仓库根目录执行，使用迁移的 2500 结构库。使用从头生成的结构库时省去 `--migrated`。[共享策略蒸馏及产物](共享策略蒸馏详细说明.md#查看产物和下一步)

## 第一步　运行固定的 50 个结构并冻结尺度

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py initialize --migrated --gpu 0 --updates 200
```

程序按原第 21 组的既定顺序评估 50 个结构。每个结构从同一胜出共享模型独立开始，做 200 次三任务联合 PPO 更新。确认最后一次更新和冻结模型的身份后，直接进入固定工况评估。`initialize`、`search` 和 `evaluate` 的默认微调预算均为 200 次。所有正式候选使用相同预算。

每个结构做 3 条正式试次。三项任务统一使用固定种子 `20261118`，每项任务跑 20 秒。初始机身位置、姿态和关节状态固定，关闭观测噪声、推力、动力学随机化和动作采样。50 个结构共 150 条正式试次。名义和鲁棒步态诊断可通过独立微调入口运行。[独立跟踪评价](候选结构微调与续训.md#独立跟踪评价与固定工况-bo)

| 任务 | 工况 | 三项代价及对应权重 |
|---|---|---|
| Rough 上楼 | 0.5 m/s，从底部平地起步，测量 20 秒。级高 7.5 cm，踏面深 30 cm | 整机正机械功 CoT、前向速度 RMSE、横滚与俯仰角速度合成 RMS，各 1/3 |
| 负重 | 平地直行 0.5 m/s，实装 10 kg | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |
| AMP 跑步 | 平地直线 1.5 m/s，使用本结构参考 | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |

固定工况的指令沿世界水平 +X 前进。三项任务均以该方向的实际速度计算前向速度 RMSE，并按表中权重计入连续硬件代价。蒸馏选模按三项等权任务成功率排序，成功要求存活并达到指令行程门槛，默认 20%。[共享策略的选模判据](共享策略蒸馏详细说明.md#如何从-abc-中选一个)

机器人从 2 m 长的底部平地中央起步，向前 1 m 到第一级，20 秒从起步开始计时。前方按评估时长 × 最高指令速度 × 2 配置连续上升的台阶。固定工况的目标行程为 10 m，前方有 67 级，登顶不作为成功条件。轨迹触及场地边界时，该次测量无效，需要延长场地后复评。旧专家的 12 级试验单独归档。

**三项任务都是代价，越低越好。** 正式试次只要摔倒，或终点相对起点沿世界水平 +X 的净前进距离低于指令行程的 **20%**，就记为策略失败，对应任务代价留空。指令行程为指令速度乘以 20 秒，上楼和负重至少净前进 **2 m**，AMP 跑步至少 **6 m**。结构必须三项任务均成功才有综合代价和最终排名资格。仿真中断、场地越界或测量无效时保留诊断记录，修复后用原命令补跑该试次。

净前进行程门槛属于定标和搜索的固定工况。已有按平均速度判定的定标时，用 `initialize --output` 指定新目录，后续搜索通过 `--scoring` 使用新目录中的 `scoring.json`。

定标按任务分别进行。每个结构在某项任务上成功时，用该项正式试次的指标参与该任务定标。在该任务的合格结构中，按结构等权，通过线性插值计算每项指标的 P10 和 P90。某任务的有效结构不足两种，或某项指标的分位差为零时，程序中止定标。

先按 `z = (x − P10) / (P90 − P10)` 把指标统一到同一尺度，再按表中权重加总。每项的 P10 对应 0、P90 对应 1，参照样本中间约 80% 落在这两端之间。有效观测超出参照区间时，按同一公式保留小于 0 或大于 1 的值。综合代价默认对三个任务等权。定标产物 `bo/calibration50_single_seed/scoring.json` 冻结九项指标的端点、权重和可用结构数，后续候选沿用同一套参数。

Moharir 等的第 III.B 节式（5）给出了相同的分位数缩放公式，并保留区间外数值。《综合指标构建手册》第 5.3 节说明如何使用固定参照尺度，第 6.10 节式（31）给出了线性加权。历史专家图的原始权重和分数见[专家评估记录](Rough与AMP专家评估及评分记录.md#归一化方法与文献)。[Moharir 等，AIAA AVIATION 2026](https://doi.org/10.2514/6.2026-4497) · [作者公开全文](https://www.researchgate.net/publication/406033991_A_Unified_Objective_Layer_for_Multidisciplinary_Optimization_of_Aircraft_Arrival_Trajectories) · [OECD/JRC 手册](https://doi.org/10.1787/9789264043466-en)

### 暂停、恢复与保存产物

同一轮定标中断后重复原命令，人工暂停后加 `--resume-paused`。正式 BO 启动的 200 次微调启用精简留存。每个结构保存初始和最近两次已提交更新的完整学习状态、每次更新的 JSON 提交与指标、最终模型及原始评估数据。各任务的 PPO 微调 reward 位于候选的 `tuning/training_reward.csv`，包括每步均值、完成回合数及回合累计回报。`tuning/training_completed.json` 记录最终 PPO 更新、训练清单和冻结模型的身份。正式结果保存在 `bo_evaluation.json`，逐条保留已提交的正式试次，恢复时继续未完成的试次。训练更新提交后释放当次的采样批次张量和临时 actor，保留扭矩审计摘要。指定 `--output` 时，恢复须使用相同目录，后续搜索用 `--scoring` 指向该目录的 `scoring.json`。

选用已完成追加蒸馏后的胜出模型时，给 `initialize`、`search` 和 `evaluate` 加同一个 `--shared-continuation <追加轮目录>`。程序核验历史回合，并按任务成功率选择共享模型。切换起始模型时，为定标和搜索各指定新的 `--output`，搜索通过 `--scoring` 指向该定标目录。已有清单只接受其绑定的模型与工况。

旧三种子协议保留在 `robolab/experiments/multitask_morphology/configs/bo_conditions.json`。复现该协议时显式传 `--conditions`，并使用原实验目录。新单种子定标默认写入 `bo/calibration50_single_seed`。

### 归档的 100 次定标

续跑或复现历史 100 次实验时，在原冻结版本中显式传 `--updates 100`，并沿用原工况、起始模型和输出目录。旧公式定标的完整测量如需按锚定分位数重算，可用支持该定标版本的 `initialize --updates 100` 核验 50 份原始报告并生成 `percentile_scoring/scoring.json`，保留原报告和评分。200 次正式流程使用自身的定标测量、分位端点和搜索目录。

## 第二步　在冻结尺度下运行 BO

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py search --migrated --trials 60 --gpu 0 --updates 200
```

示例总预算 60 个结构，包含已经测过的 50 个，只新增 10 个候选。搜索先载入定标报告及其哈希，校验起始学生、结构键、微调预算和工况。每个新候选从同一个胜出模型微调后实测。约束 GP 用已有试次学习可行性及代价，选择下一结构；默认三任务之间等权。最终报告在三项任务均成功的结构中给出最低代价候选。增加总预算可继续同一搜索。中断后重复原命令，人工暂停加 `--resume-paused`。

蒸馏选模的 200 个结构均已排除这 50 个定标结构。历史教师参考成绩保留在独立归档中；新的尺度来自本次学生微调后的测量。[固定结构与原始记录](reference/rough_amp_expert_20261121/rough/selection.json)

## 单独评估一个候选

把 `真实结构ID` 换成共同结构库里该候选的 `stairs` ID。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py evaluate --migrated --gpu 0 --updates 200 --candidate 真实结构ID
```

`evaluate` 输出单结构的固定工况测量。已有预算、初始模型和结构身份均匹配的完整微调时，可以用 `--finetuned` 复用其输出目录。使用同预算的冻结定标文件计算分数时，运行 `bo.py score <评估目录>/bo_evaluation.json --scoring <定标目录>/scoring.json`。

## 附录　导入原始 RPO 或实机测量

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py ingest --spec baseline_measurements.json --output baseline.json
```

测量清单采用 JSON。顶层需要 `domain`（`simulation` 或 `hardware`）、`measurement_convention`（`rpo_three_task_mechanical_positive_v1`）、`dt_s`（0.02）、`candidate` 和 `trials`。每条试次填写 `task`（`stairs`、`backpack` 或 `run`）、`seed`、从 0 开始的 `replica`、`trace`、实测 `total_mass_kg`、`fell` 及 `motors`。可同时标记 `reached_top`、`route_completed` 和 `protection_stop`。负重加 `payload.mass_kg` 和 `payload.mount_position_torso_m`。

每条 `trace` 是与清单同目录或相对路径的 NPZ，包含 `initial_position[3]`、`position[N,3]`、`vx[N]`、`omega[N,2]`、`torque[N,23]`、`joint_vel[N,23]`。23 列按 `motors` 顺序填写，每台记录 `joint`、`motor_id`、`torque_side: "joint_output"`、`rated_torque_nm` 和 `source`。请先把位置、速度和 IMU 轴统一到上面的约定，把驱动器输出校核为关节侧 N·m，并对齐到 50 Hz 采样。程序验证完整的种子、副本和任务覆盖。实机采集由测量系统执行。
