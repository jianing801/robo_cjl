# 微调后的三任务评估与结构搜索

先完成上楼、负重和跑步均采用水平航向速度的鲁棒选模，确认 `shared_policy/selection.json` 和 `shared_policy/completed.json` 记录了获选模型及其身份，后者的 `protocol.evaluation_velocity_frame` 为 `horizontal_heading`。下列命令在仓库根目录执行，使用迁移的 2500 结构库。使用从头生成的结构库时省去 `--migrated`。[共享策略蒸馏及产物](共享策略蒸馏详细说明.md#查看产物和下一步)

## 第一步　运行固定的 50 个结构并冻结尺度

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py initialize --migrated --gpu 0
```

程序按原第 21 组的既定顺序评估 50 个结构。每个结构从同一胜出共享模型独立开始，在同一仿真进程内并行运行 4096 个环境，上楼、负重、跑步分别占 1366、1365、1365 个。默认做 100 次三任务联合 PPO 更新。每次每环境采 24 步，同一批数据训练 5 个 epoch，每个 epoch 分 4 个小批。

训练启动时按普通 RSL-RL 的方式，在每个环境的有效步数范围内随机初始化一次回合计数，随后直接进入 PPO。过程保留总体和三项各自的 Mean reward。随机初始计数产生的首个短回合也按普通 PPO 日志口径记录。前 50 个定标结构和后 50 个搜索候选同时冻结回合计时方式、环境分配、更新预算和优化配置。

每个结构做 3 条正式试次。三项任务统一使用固定种子 `20261118`，每项任务跑 20 秒。初始机身位置、姿态和关节状态固定，关闭观测噪声、推力、动力学随机化和动作采样。50 个结构共 150 条正式试次。名义和鲁棒步态诊断可通过独立微调入口运行。[独立跟踪评价](候选结构微调与续训.md#独立跟踪评价与固定工况-bo)

| 任务 | 工况 | 三项代价及对应权重 |
|---|---|---|
| Rough 上楼 | 0.5 m/s，从底部平地起步，测量 20 秒。级高 7.5 cm，踏面深 30 cm | 整机正机械功 CoT、前向速度 RMSE、横滚与俯仰角速度合成 RMS，各 1/3 |
| 负重 | 平地直行 0.5 m/s，实装 10 kg | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |
| AMP 跑步 | 平地直线 1.5 m/s，使用本结构参考 | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |

固定工况的指令沿世界水平 +X 前进。三项任务均以该方向的实际速度计算前向速度 RMSE，并按表中权重计入连续硬件代价。蒸馏选模按三项等权任务成功率排序，成功要求存活并达到指令行程门槛，默认 20%。[共享策略的选模判据](共享策略蒸馏详细说明.md#如何从-abc-中选一个)

机器人从 2 m 长的底部平地中央起步，向前 1 m 到第一级，20 秒从起步开始计时。前方按评估时长 × 最高指令速度 × 2 配置连续上升的台阶。固定工况的目标行程为 10 m，前方有 67 级，登顶不作为成功条件。轨迹触及场地边界时，该次测量无效，需要延长场地后复评。旧专家的 12 级试验单独归档。

**正式 BO 和蒸馏选模都以未摔倒、且沿指令方向的水平行程达到指令行程的 20% 为成功判据。** 正式 BO 每次测量完整的 20 秒，指令固定沿世界水平 +X，实际行程取该方向的净前进距离。上楼和负重至少前进 **2 m**，AMP 跑步至少前进 **6 m**。任务失败时对应代价留空，三个任务均成功才计算综合代价并参与最终排名。三项任务代价越低越好。仿真中断、场地越界或测量无效时保留诊断记录，修复后用原命令补跑该试次。

净前进行程门槛属于定标和搜索的固定工况。已有按平均速度判定的定标时，用 `initialize --output` 指定新目录，后续搜索通过 `--scoring` 使用新目录中的 `scoring.json`。

定标按任务分别进行。每个结构在某项任务上成功时，用该项正式试次的指标参与该任务定标。在该任务的合格结构中，按结构等权，通过线性插值计算每项指标的 P10 和 P90。某任务的有效结构不足两种，或某项指标的分位差为零时，程序中止定标。

先按 `z = (x − P10) / (P90 − P10)` 把指标统一到同一尺度，再按表中权重加总。每项的 P10 对应 0、P90 对应 1，参照样本中间约 80% 落在这两端之间。有效观测超出参照区间时，按同一公式保留小于 0 或大于 1 的值。综合代价默认对三个任务等权。定标产物 `bo/calibration50_single_seed_ppo100_random/scoring.json` 冻结九项指标的端点、权重和可用结构数，后续候选沿用同一套参数。

Moharir 等的第 III.B 节式（5）给出了相同的分位数缩放公式，并保留区间外数值。《综合指标构建手册》第 5.3 节说明如何使用固定参照尺度，第 6.10 节式（31）给出了线性加权。历史专家图的原始权重和分数见[专家评估记录](Rough与AMP专家评估及评分记录.md#归一化方法与文献)。[Moharir 等，AIAA AVIATION 2026](https://doi.org/10.2514/6.2026-4497) · [作者公开全文](https://www.researchgate.net/publication/406033991_A_Unified_Objective_Layer_for_Multidisciplinary_Optimization_of_Aircraft_Arrival_Trajectories) · [OECD/JRC 手册](https://doi.org/10.1787/9789264043466-en)

### 暂停、恢复与保存产物

同一轮定标中断后重复原命令，人工暂停后加 `--resume-paused`。正式 BO 使用精简留存。每个结构保存初始和最近两次已提交更新的完整学习状态、每次更新的 JSON 提交与指标、最终模型及原始评估数据。`tuning/mean_reward.csv` 每代保存总体和三个任务各自的 Mean reward，`tuning/training_reward.csv` 还保存每步均值、完成回合数及回合累计回报。`tuning/training_completed.json` 记录最终 PPO 更新、训练清单和冻结模型的身份。正式结果保存在 `bo_evaluation.json`，逐条保留已提交的正式试次，恢复时继续未完成的试次。训练更新提交后释放当次的采样批次张量和临时 actor，保留扭矩审计摘要。指定 `--output` 时，恢复须使用相同目录，后续搜索用 `--scoring` 指向该目录的 `scoring.json`。

选用已完成追加蒸馏后的胜出模型时，给 `initialize`、`search` 和 `evaluate` 加同一个 `--shared-continuation <追加轮目录>`。程序核验历史回合，并按任务成功率选择共享模型。切换起始模型时，为定标和搜索各指定新的 `--output`，搜索通过 `--scoring` 指向该定标目录。已有清单只接受其绑定的模型与工况。

新单种子定标默认写入 `bo/calibration50_single_seed_ppo100_random`。启动时会核验已有模型和定标结果的回合计时方式及环境分配。

### 训练中断

恢复时从最后一次已提交更新加载 Actor、Critic 和 Adam。仿真重新创建，按相同的随机计时规则开始新的回合，奖励曲线由更新记录接续。

## 第二步　在冻结尺度下运行 BO

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py search --migrated --trials 100 --gpu 0
```

正式总预算为 **100 个结构**。前 50 个是原第 21 组的定标结构，后 50 个由 BO 逐个提议。搜索先载入定标报告及其哈希，校验起始学生、结构键、微调预算和工况。每个新候选从同一个胜出模型微调后实测。约束 GP 用已有试次学习可行性及代价，选择下一结构。默认三任务之间等权。最终报告在三项任务均成功的结构中给出最低代价候选。增加总预算可继续同一搜索。中断后重复原命令，人工暂停加 `--resume-paused`。

蒸馏选模的 200 个结构均已排除这 50 个定标结构。历史教师参考成绩保留在独立归档中；新的尺度来自本次学生微调后的测量。[固定结构与原始记录](reference/rough_amp_expert_20261121/rough/selection.json)

## 单独评估一个候选

把 `真实结构ID` 换成共同结构库里该候选的 `stairs` ID。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py evaluate --migrated --gpu 0 --candidate 真实结构ID
```

`evaluate` 输出单结构的固定工况测量。已有预算、初始模型和结构身份均匹配的完整微调时，可以用 `--finetuned` 复用其输出目录。使用同预算的冻结定标文件计算分数时，运行 `bo.py score <评估目录>/bo_evaluation.json --scoring <定标目录>/scoring.json`。

## 附录　导入原始 RPO 或实机测量

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py ingest --spec baseline_measurements.json --output baseline.json
```

导入默认按上文的正式单种子工况和 20% 水平行程门槛判定。使用 `calibrate` 做独立定标时，至少需要两组各任务均成功的试次。可用 `--conditions` 指定实际种子或副本数，并设置 `min_forward_progress_ratio` 为 0.20。复现旧的三种子测量时，显式传入 `--conditions robolab/experiments/multitask_morphology/configs/bo_conditions.json`。

测量清单采用 JSON。顶层需要 `domain`（`simulation` 或 `hardware`）、`measurement_convention`（`rpo_three_task_mechanical_positive_v1`）、`dt_s`（0.02）、`candidate` 和 `trials`。每条试次填写 `task`（`stairs`、`backpack` 或 `run`）、`seed`、从 0 开始的 `replica`、`trace`、实测 `total_mass_kg`、`fell` 及 `motors`。可同时标记 `reached_top`、`route_completed` 和 `protection_stop`。负重加 `payload.mass_kg` 和 `payload.mount_position_torso_m`。

每条 `trace` 是与清单同目录或相对路径的 NPZ，包含 `initial_position[3]`、`position[N,3]`、`vx[N]`、`omega[N,2]`、`torque[N,23]`、`joint_vel[N,23]`。23 列按 `motors` 顺序填写，每台记录 `joint`、`motor_id`、`torque_side: "joint_output"`、`rated_torque_nm` 和 `source`。请先把位置、速度和 IMU 轴统一到上面的约定，把驱动器输出校核为关节侧 N·m，并对齐到 50 Hz 采样。程序验证完整的种子、副本和任务覆盖。实机采集由测量系统执行。
