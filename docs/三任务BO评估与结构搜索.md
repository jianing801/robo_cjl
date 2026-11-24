# 微调后的三任务评估与结构搜索

先完成上楼、负重和跑步均采用水平航向速度的鲁棒选模，确认 `shared_policy/selection.json` 和 `shared_policy/completed.json` 记录了获选模型及其身份，后者的 `protocol.evaluation_velocity_frame` 为 `horizontal_heading`。本轮使用第 5 轮获选模型、迁移的 2500 结构库，以及已评估的[原始 RPO 配置](原始配置基准.md)。[共享策略蒸馏及产物](共享策略蒸馏详细说明.md#查看产物和下一步)

<a name="第一步运行固定的-50-个结构并冻结尺度"></a>

## 第一步　准备 50 个初始结构的测量

下列命令在代码仓库根目录执行。变量使用本轮租卡上的目录。

```bash
entry="$PWD/robolab/experiments/multitask_morphology"
data=/root/sj-tmp/bo-ppo100-round5-data
experiment=/data/chenjiale/robo_cjl/robolab/logs/multitask_full2500_physical_migrated
continuation=/root/sj-tmp/shared-distill-20261121/shared_policy_round5
panel="$data/calibration50_single_seed_ppo100_random_200hz"
reference="$data/rpo_original_025_030_10010l_4340p_ppo100_200hz/evaluation"
scoring="$data/scoring_original_rpo_200hz"
search="$data/search_single_seed_ppo100_random_200hz_original_rpo"
```

已有与原机评估一致的 `measurements_completed.json` 时，直接进入第二步。首次准备这 50 个结构时运行以下命令。已完成同协议微调的结构可通过 `--reuse-finetuned-panel <已完成的微调目录>` 只做复评。

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" initialize \
  --root "$experiment" --shared-continuation "$continuation" \
  --output "$panel" --measure-only --gpu 0
```

程序按原第 21 组的既定顺序评估 50 个结构。每个结构从同一胜出共享模型独立开始，在同一仿真进程内并行运行 4096 个环境，上楼、负重、跑步分别占 1366、1365、1365 个。默认做 100 次三任务联合 PPO 更新。每次每环境采 24 步，同一批数据训练 5 个 epoch，每个 epoch 分 4 个小批。

训练启动时按普通 RSL-RL 的方式，在每个环境的有效步数范围内随机初始化一次回合计数，随后直接进入 PPO。过程保留总体和三项各自的 Mean reward。随机初始计数产生的首个短回合也按普通 PPO 日志口径记录。前 50 个初始结构和后 50 个搜索候选采用相同的回合计时方式、环境分配、更新预算和优化配置。

每个结构做 3 条正式试次。三项任务统一使用固定种子 `20261118`，每项任务跑 20 秒。初始机身位置、姿态和关节状态固定，关闭观测噪声、推力、动力学随机化和动作采样。50 个结构共 150 条正式试次。名义和鲁棒步态诊断可通过独立微调入口运行。[独立跟踪评价](候选结构微调与续训.md#独立跟踪评价与固定工况-bo)

| 任务 | 工况 | 三项代价及对应权重 |
|---|---|---|
| Rough 上楼 | 0.5 m/s，从底部平地起步，测量 20 秒。级高 7.5 cm，踏面深 30 cm | 绝对机械 CoT、前向速度 RMSE、横滚与俯仰角速度合成 RMS，各 1/3 |
| 负重 | 平地直行 0.5 m/s，实装 10 kg | 绝对机械 CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |
| AMP 跑步 | 平地直线 1.5 m/s，使用本结构参考 | 绝对机械 CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |

固定工况的指令沿世界水平 +X 前进。三项任务均以该方向的实际速度计算前向速度 RMSE，并按表中权重计入连续硬件代价。蒸馏选模按三项等权任务成功率排序，成功要求存活并达到指令行程门槛，默认 20%。[共享策略的选模判据](共享策略蒸馏详细说明.md#如何从-abc-中选一个)

机器人从 2 m 长的底部平地中央起步，向前 1 m 到第一级，20 秒从起步开始计时。前方按评估时长 × 最高指令速度 × 2 配置连续上升的台阶。固定工况的目标行程为 10 m，前方有 67 级，登顶不作为成功条件。轨迹触及场地边界时，该次测量无效，需要延长场地后复评。旧专家的 12 级试验单独归档。

**正式 BO 和蒸馏选模都以未摔倒、且沿指令方向的水平行程达到指令行程的 20% 为成功判据。** 正式 BO 每次测量完整的 20 秒，指令固定沿世界水平 +X，实际行程取该方向的净前进距离。上楼和负重至少前进 **2 m**，AMP 跑步至少前进 **6 m**。任务失败时对应代价留空，三个任务均成功才计算综合代价并参与最终排名。三项任务代价越低越好。仿真中断、场地越界或测量无效时保留诊断记录，修复后用原命令补跑该试次。

正式评估默认每 5 ms 采样一次（200 Hz），20 秒每项有 4000 个物理步。策略每 20 ms 输出一次动作。正式评分读取 `observed.absolute_mechanical_cot`，将每个物理步、每个关节的机械功率 `|τ × q̇|` 累加再积分，以整机重量（负重时含背包）乘净水平行程为分母。`observed.cot` 保存正机械 CoT，供对照。绝对机械 CoT 描述关节的驱动与制动机械功，电耗需另行测量。[TALOS 评估，式（10）](https://pmc.ncbi.nlm.nih.gov/articles/PMC8934419/)

测量完成时，`measurements_completed.json` 记录 50 份结构报告及其哈希。`evaluation_timing.json` 只累计本轮最终接受的测量进程用时。

## 第二步　用原机基准冻结评分

每项指标按 `候选指标 ÷ 原机同一任务的对应指标` 归一化，再按表中权重相加。**综合代价 = 0.4 × 上楼代价 + 0.3 × 负重代价 + 0.3 × 跑步代价。** 原机三项任务和综合代价均为 1。候选的综合代价为 0.8 时，按这套权重比原机低 20%。50 个初始结构的原始测量用于提供 BO 观测，九项分母始终取自原机实测。固定参照和线性加权的定义见《综合指标构建手册》。[OECD/JRC 手册](https://doi.org/10.1787/9789264043466-en)

原机的 `evaluation` 目录包含 `reference_protocol.json` 和 `candidate/bo_evaluation.json`。以下命令在 CPU 上核对原机和 50 份初始报告的模型、工况和微调预算，复用其原始测量。

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" freeze-scoring \
  --measurements-root "$panel" --original-rpo-reference "$reference" --output "$scoring"
```

`$scoring` 中的 `scoring.json` 冻结九项分母、权重和数据来源，`initial_observations.json` 记录 50 个结构的可行性和新评分，`completed.json` 记录评分校验信息。初始结构的旧评分产物留在原目录。

## 第三步　搜索后 50 个结构

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" search \
  --root "$experiment" --shared-continuation "$continuation" \
  --scoring "$scoring/scoring.json" --output "$search" --trials 100 --gpu 0
```

正式总预算为 **100 个结构**。前 50 个是原第 21 组的初始观测，后 50 个由 BO 逐个提议。搜索在独立目录从 `BO_TRIAL 51 / 100` 开始。旧目标下曾被选中的结构仍可以被提议，届时按本轮协议重新微调和评估。约束 GP 用已有试次学习可行性和代价，通过约束期望改进选择下一结构。候选在任一任务失败时记录为不可行，最终报告在三项任务均成功的结构中给出最低代价候选。蒸馏选模的 200 个结构已排除这 50 个初始结构。[固定结构与原始记录](reference/rough_amp_expert_20261121/rough/selection.json)

### 暂停、恢复与保存产物

中断后重复原命令，沿用同一输出目录和评分文件。人工暂停后加 `--resume-paused`。恢复微调时从最后一次已提交更新加载 Actor、Critic 和 Adam。仿真重新创建，按相同的随机计时规则开始新的回合，奖励曲线由更新记录接续。评估从尚未完成的试次继续。

每个结构保存初始和最近两次已提交更新的完整学习状态、每次更新的记录和指标、最终模型及原始评估数据。`tuning/mean_reward.csv` 保存每代总体和分任务 Mean reward，`tuning/training_reward.csv` 还保存每步均值、完成回合数及回合累计回报。`tuning/training_completed.json` 记录最终 PPO 更新和模型身份，`bo_evaluation.json` 汇总正式评估。

## 单独评估一个候选

把 `真实结构ID` 换成共同结构库里该候选的 `stairs` ID。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py evaluate --migrated --gpu 0 --candidate 真实结构ID
```

`evaluate` 输出单结构的固定工况测量。复用本轮获选模型时加上 `--shared-continuation "$continuation"`。已有预算、初始模型和结构身份均匹配的完整微调时，可以用 `--finetuned` 复用其输出目录。用 `bo.py score <评估目录>/bo_evaluation.json --scoring "$scoring/scoring.json"` 计算同协议下的原机基准评分。

## 附录　复现旧版 P10/P90 评分

旧版评分将候选指标按 `(x − P10) / (P90 − P10)` 缩放，再按任务内权重和任务权重相加。使用原有报告和同目录的 `scoring.json` 可复算旧结果。缺少旧版评分文件时，运行 `bo.py freeze-scoring --measurements-root <旧版测量面板目录>` 恢复该口径。历史专家成绩见独立归档。[专家评估记录](Rough与AMP专家评估及评分记录.md#归一化方法与文献) · [Moharir 等，AIAA AVIATION 2026](https://doi.org/10.2514/6.2026-4497)

## 附录　导入原始 RPO 或实机测量

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py ingest --spec baseline_measurements.json --output baseline.json
```

导入默认使用历史 50 Hz 单种子工况和 20% 水平行程门槛。使用 `calibrate` 做独立定标时，至少需要两组各任务均成功的试次。可用 `--conditions` 指定实际种子或副本数，并设置 `min_forward_progress_ratio` 为 0.20。复现旧的三种子测量时，显式传入 `--conditions robolab/experiments/multitask_morphology/configs/bo_conditions.json`。

测量清单采用 JSON。顶层需要 `domain`（`simulation` 或 `hardware`）、`measurement_convention`（`rpo_three_task_mechanical_positive_v1`）、`dt_s`（0.02）、`candidate` 和 `trials`。每条试次填写 `task`（`stairs`、`backpack` 或 `run`）、`seed`、从 0 开始的 `replica`、`trace`、实测 `total_mass_kg`、`fell` 及 `motors`。可同时标记 `reached_top`、`route_completed` 和 `protection_stop`。负重加 `payload.mass_kg` 和 `payload.mount_position_torso_m`。

每条 `trace` 是与清单同目录或相对路径的 NPZ，包含 `initial_position[3]`、`position[N,3]`、`vx[N]`、`omega[N,2]`、`torque[N,23]`、`joint_vel[N,23]`。23 列按 `motors` 顺序填写，每台记录 `joint`、`motor_id`、`torque_side: "joint_output"`、`rated_torque_nm` 和 `source`。请先把位置、速度和 IMU 轴统一到上面的约定，把驱动器输出校核为关节侧 N·m，并对齐到 50 Hz 采样。程序验证完整的种子、副本和任务覆盖。实机采集由测量系统执行。
