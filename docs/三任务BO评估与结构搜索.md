# 微调后的三任务评估与结构搜索

先完成[共享策略蒸馏](共享策略蒸馏详细说明.md)。下列命令在仓库根目录执行，使用迁移的 2500 结构库。使用从头生成的结构库时省去 `--migrated`。

## 第一步　运行固定的 50 个结构并冻结尺度

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py initialize --migrated --gpu 0
```

程序在原第 21 组按既定序号逐个取结构。每个结构从同一胜出共享模型独立开始，默认做 100 次三任务联合 PPO 更新，然后记录固定工况下的实际表现。后续 BO 新候选使用相同微调预算。每项任务跑 20 秒，关闭随机扰动，使用 20261118、20261119、20261120 三个种子，各一个环境试次。50 个结构共取得 450 条任务试次。原始模型、成功与失败、测量条件和检查点身份均可追溯；完全确定的试次可能重复相同结果。

| 任务 | 工况 | 三项代价及对应权重 |
|---|---|---|
| Rough 上楼 | 0.5 m/s，从底部平地起步，测量 20 秒。级高 7.5 cm，踏面深 30 cm | 整机正机械功 CoT、前向速度 RMSE、横滚与俯仰角速度合成 RMS，各 1/3 |
| 负重 | 平地直行 0.5 m/s，实装 10 kg | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |
| AMP 跑步 | 平地直线 1.5 m/s，使用本结构参考 | CoT 0.4、前向速度 RMSE 0.4、六个膝踝关节最大持续额定扭矩占比 0.2 |

机器人从 2 m 长的底部平地中央起步，向前 1 m 到第一级，20 秒从起步开始计时。前方按评估时长 × 最高指令速度 × 2 配置连续上升的台阶。固定工况的目标行程为 10 m，前方有 67 级，登顶不作为成功条件。轨迹触及场地边界时，该次测量无效，需要延长场地后复评。旧专家的 12 级试验单独归档。

**三项任务都是代价，越低越好。** 每个结构先在同一任务上取三次成功试验的指标均值，上楼按摔倒和前向均速判定。在该任务的合格结构均值中，按结构等权，通过线性插值计算每项指标的 P10 和 P90。缺少有效测量时先排查原因。任务失败保留为可行性记录，对应代价留空。某任务的有效结构不足两种，或某项指标的分位差为零时，程序中止定标。

先按 `z = (x − P10) / (P90 − P10)` 把指标统一到同一尺度，再按表中权重加总。每项的 P10 对应 0、P90 对应 1，参照样本中间约 80% 落在这两端之间。有效观测超出参照区间时，按同一公式保留小于 0 或大于 1 的值。综合代价默认对三个任务等权。定标产物 `bo/calibration50/scoring.json` 冻结九项指标的端点、权重和可用结构数，后续候选沿用同一套参数。

Moharir 等的第 III.B 节式（5）给出了相同的分位数缩放公式，并保留区间外数值。《综合指标构建手册》第 5.3 节说明如何使用固定参照尺度，第 6.10 节式（31）给出了线性加权。历史专家图的原始权重和分数见[专家评估记录](Rough与AMP专家评估及评分记录.md#归一化方法与文献)。[Moharir 等，AIAA AVIATION 2026](https://doi.org/10.2514/6.2026-4497) · [作者公开全文](https://www.researchgate.net/publication/406033991_A_Unified_Objective_Layer_for_Multidisciplinary_Optimization_of_Aircraft_Arrival_Trajectories) · [OECD/JRC 手册](https://doi.org/10.1787/9789264043466-en)

### 复用已有的完整定标测量

如果 `bo/calibration50/` 中已有按旧公式定标的完整记录，按原工况和预算重新运行 `initialize`。程序校验旧定标记录和 50 份原始评估报告，直接用这些测量生成新版 `bo/calibration50/percentile_scoring/scoring.json`，无需重新微调或仿真。旧评分文件和报告仍可用于复现此前的分数。新版 `search` 默认选用已完成的分位数评分文件。自定定标目录时，可用 `--scoring` 指明新版评分文件。

中断的旧版定标先用原提交完成，然后用新版复用测量。已有旧版搜索记录时，可用原提交继续该搜索，或在新版用 `--output` 指定新搜索目录。

## 第二步　在冻结尺度下运行 BO

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py search --migrated --trials 60 --gpu 0
```

示例总预算 60 个结构，包含已经测过的 50 个，只新增 10 个候选。搜索先载入定标报告及其哈希，校验起始学生、结构键、微调预算和工况。每个新候选从同一个胜出模型微调后实测。约束 GP 用已有试次学习可行性及代价，选择下一结构；默认三任务之间等权。最终报告在全部三个重复均取得完整综合代价的结构中给出最低代价候选，可用独立种子复核。增加总预算可继续同一搜索。中断后重复原命令，人工暂停加 `--resume-paused`。

蒸馏选模的 200 个结构均已排除这 50 个定标结构。历史教师参考成绩保留在独立归档中；新的尺度来自本次学生微调后的测量。[固定结构与原始记录](reference/rough_amp_expert_20261121/rough/selection.json)

## 附录　导入原始 RPO 或实机测量

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/bo.py ingest --spec baseline_measurements.json --output baseline.json
```

测量清单采用 JSON。顶层需要 `domain`（`simulation` 或 `hardware`）、`measurement_convention`（`rpo_three_task_mechanical_positive_v1`）、`dt_s`（0.02）、`candidate` 和 `trials`。每条试次填写 `task`（`stairs`、`backpack` 或 `run`）、`seed`、从 0 开始的 `replica`、`trace`、实测 `total_mass_kg`、`fell` 及 `motors`。可同时标记 `reached_top`、`route_completed` 和 `protection_stop`。负重加 `payload.mass_kg` 和 `payload.mount_position_torso_m`。

每条 `trace` 是与清单同目录或相对路径的 NPZ，包含 `initial_position[3]`、`position[N,3]`、`vx[N]`、`omega[N,2]`、`torque[N,23]`、`joint_vel[N,23]`。23 列按 `motors` 顺序填写，每台记录 `joint`、`motor_id`、`torque_side: "joint_output"`、`rated_torque_nm` 和 `source`。请先把位置、速度和 IMU 轴统一到上面的约定，把驱动器输出校核为关节侧 N·m，并对齐到 50 Hz 采样。程序验证完整的种子、副本和任务覆盖。实机采集由测量系统执行。
