# 三任务 BO 评估与连续结构搜索

更新：2026-11-27。本流程从第五轮最终共享 Actor 出发，给每个结构相同的 200 次三任务联合 PPO 微调预算，再用固定工况实测指标比较硬件。正式搜索分为前 50 个已测结构和后 50 个连续 BO 候选；2500 个格点是专家训练结构库，**不是 BO 的搜索边界**。

正式候选微调采用已验证的方案 B，算法身份 `shared_native_ppoamp_expert_prior_pcgrad_fixed_all_std`（三个任务分别固定各自专家的23维动作标准差，Actor 采用 PCGrad）。前 50 个和后 50 个候选均采用 B；以前的 V4 评估及分母保留为历史对照，不可混合。新 B 的第 200 代原机基准已实测并核验，归一化系数见[原机实测归档](reference/rpo_original_round5_ppo200/normalization.json)。前50个结构测量完成后才冻结正式评分并启动搜索。详见[原生 PPO 微调](共享策略原生PPO微调说明.md)。

**2026-11-27 工况更新：** 新方案 B 微调继承三个专家的完整原生训练工况；200 代预算、固定种子、20 秒正式评估工况及四项指标不变。`native_ppo_settings.training_conditions=native_expert` 区分新旧训练数据，旧工况的校准和原机分母均保留为历史，不能用作新协议的正式 BO 数据。详见[原生 PPO 微调说明](共享策略原生PPO微调说明.md)。

[总流程](跨形态多任务共享策略训练流程.md) · [共享蒸馏与选模](共享策略蒸馏详细说明.md) · [原机实测基准](原始配置基准.md)

<a name="第一步运行固定的-50-个结构并冻结尺度"></a>

正式训练使用三任务原生专家工况（完整随机化），不是固定评估工况。此前1000代共享加训仅为备用，不作为候选起点。

## 第一步　准备或复用前 50 个结构

在仓库根目录执行。以下为目录示例；已有实验时将变量指向实际的数据根、已完成的原第 21 组测量和同一第五轮 Actor。不要移动或覆盖旧报告中已经冻结的来源路径及校验记录。

```bash
entry="$PWD/robolab/experiments/multitask_morphology"
experiment="$PWD/robolab/logs/multitask_full2500_physical_migrated"
continuation="$experiment/shared_policy_round5"
data="$experiment/bo"
panel="$data/calibration50_pcgrad_ppo200_fixed_all_std_200hz"
reference="$data/original_rpo_reference_pcgrad_ppo200_fixed_all_std_200hz"
scoring="$data/scoring_original_rpo_four_metrics_pcgrad_ppo200_fixed_all_std"
search="$data/search_continuous_four_pcgrad_200"
```

已有 `measurements_completed.json` 且与原机基准的共享起点、原生 PPO 算法身份、微调、工况一致时，直接使用原始测量进入第二步，**无需重做微调或评估**。旧算法的 50 个观测虽保留，但不是新算法的数据。否则准备前 50 个结构：

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" initialize \
  --root "$experiment" --shared-continuation "$continuation" \
  --output "$panel" --measure-only --gpu 0
```

程序按原第 21 组固定顺序得到 50 个观测（与 200 个蒸馏选模结构无重叠）。已有同协议微调产物可用 `--reuse-finetuned-panel <目录>` 复用模型，只补缺失的固定工况测量。方案 B 的原机基准先用已验证的原机单结构预测资产及参考动作计划重新实测；V4 档案不作为 B 的分母。原机基准包含 `reference_protocol.json`、`candidate/bo_evaluation.json`、源训练与检查点身份和三项原始轨迹。[历史 V4 基准](原始配置基准.md) 前 50 个结构和后续新候选均从同一第五轮 Actor 独立开始。

若租卡仅迁移了完整结构库与专家、第五轮最终Actor和完成凭据，而没有迁移原始蒸馏训练回放，可在 `initialize` 和 `search` 两条命令中附加 `--round5-reference "$reference"`。该入口逐项校验第五轮完成、模型与源计划、三个专家及原机200代检查点和原始测量；仍从同一精确第五轮Actor独立微调，不宣称重新读取了原始蒸馏回放。

### 准备或复用方案 B 的原机基准

5090实例已存在经真实测量的原机200代参考目录 `/root/sj-tmp/rpo_native_conditions_all_fixed_std200_reference_20261127`。将上面的 `reference` 变量指向该目录（原始数据必须仍可读取），直接使用后面的 `freeze-scoring`，不重复训练或评估原机。其他实例缺少完整原始证据时，可按以下方式重新准备。

`reference_plan.json` 必须是已验证的原机预测模型单结构计划（不是原厂 URDF，也不是全结构库）。例如租卡已经保存的 `/root/sj-tmp/rpo_native_conditions_all_fixed_std1000_20261127/plan.json`。

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" reference \
  --root "$experiment" --shared-continuation "$continuation" \
  --reference-plan /root/sj-tmp/rpo_native_conditions_all_fixed_std1000_20261127/plan.json \
  --output "$reference" --gpu 0
```

原机和候选均用同一个第五轮共享 Actor，200 次 B 更新和三项固定 200 Hz 测量。只有实测成功并通过全部身份校验后才能冻结新分母。

### 微调与固定工况

- 单仿真进程 4096 个环境，上楼、负重、跑步分别 1366、1365、1365 个；三任务共享 Actor，各自复制专家原生 Critic、实际归一化及 23 维动作标准差，按任务门控选择独立 Critic/探索参数；三个任务的探索向量分别继承各自专家并冻结，三任务 Actor 梯度采用 PCGrad；AMP 判别器在独立副本上随训练同步更新，老师和物理参数不改。
- 普通 PPO 随机初始化回合步数计数，先做 10 次只更新 Critic 的预热（不计入正式预算），每个候选做 **200 次原生 PPO 更新**，每次每环境采 24 步，训练 5 个 epoch、4 个小批/epoch；禁止用近邻结构替代实际提议。
- 微调后冻结 Actor，固定种子 `20261118`，每任务一条 20 秒试次（50 个结构共 150 条）。评估关闭观测噪声、推力、动力学和初态随机化及动作采样，50 Hz 控制、200 Hz 物理采样（步长 5 ms，每任务 4000 个物理步）。

| 任务 | 20 秒工况 | 成功所需最小净行程 |
|---|---|---:|
| 上楼 | 从底部平地中央起步，距第一级 1 m，0.5 m/s；级高 7.5 cm，踏面深 30 cm | 2 m |
| 负重 | 平地直行，实装 10 kg，0.5 m/s | 2 m |
| AMP 跑步 | 平地直跑，1.5 m/s；使用与实际结构运动学匹配的参考动作 | 6 m |

固定工况沿世界水平 +X 方向计速度和净行程。楼梯覆盖至少两倍目标行程（默认 67 级），含起步段，不要求登顶。任务成功需要完整评估、没有摔倒或失败终止，且前向水平净行程达到指令行程的 20%。任一任务失败即为不可行观测，不生成综合代价、不参与最终可行排名；中断或场地越界等无效试次记录诊断原因，补跑直至有效。程序以原始评估和任务成功判据确定可行性，不以 Mean reward 或机械指标阈值代替。

<a name="第二步用原机基准冻结评分"></a>

## 第二步　用原机基准冻结评分

每个任务的原始报告在 `observed` 中同时记录四项性能指标，以及正机械 CoT、机械功和全身关节原始轨迹。评价指标统一如下：

| 指标 | 定义（越低越好） | 原始字段 |
|---|---|---|
| 绝对机械 CoT | 全身各关节绝对机械功率之和积分，除以整机重量与水平净前进位移的乘积；负重含背包 | `absolute_mechanical_cot` |
| 速度 RMSE（m/s） | 世界水平 +X 实际速度与指令速度的均方根误差 | `forward_rmse_mps` |
| 横滚俯仰角速度 RMS（rad/s） | 横滚与俯仰角速度合成 RMS，衡量姿态波动 | `roll_pitch_rms_rad_s` |
| 膝踝最大额定扭矩占比 | 六个膝踝关节各自实际扭矩 RMS 与持续额定扭矩之比的最大值 | `knee_ankle_max_rated_ratio` |

正机械 CoT 单独保留；绝对机械 CoT 同时计入驱动和制动功，不能等同实际电耗。[TALOS 评估，式（10）](https://pmc.ncbi.nlm.nih.gov/articles/PMC8934419/)

### 默认四指标方案 `four`（本轮正式权重）

各指标以**该任务原机实测值为固定分母**，即 `z = 候选原始指标 / 原机对应原始指标`，不截断。原机四项分项均为 1。任务代价为分项按权重线性加和，综合代价为 `0.4 × 上楼 + 0.3 × 负重 + 0.3 × 跑步`，原机综合代价为 1。

| 指标 | 上楼权重 | 负重权重 | 跑步权重 |
|---|---:|---:|---:|
| 绝对机械 CoT | 0.25 | 0.40 | 0.25 |
| 速度 RMSE | 0.25 | 0.25 | 0.40 |
| 横滚俯仰角速度 RMS | 0.40 | 0.20 | 0.15 |
| 膝踝最大额定扭矩占比 | 0.10 | 0.15 | 0.20 |
| 合计 | 1.00 | 1.00 | 1.00 |

上表是本轮正式搜索采用的默认权重。上楼优先考虑机身平稳性，负重优先考虑运输成本，跑步优先考虑速度跟踪。权重是工程偏好，保留完整原始指标以便做敏感性分析。B 的十二项分母由本轮原机第200代真实测量确定；[既有 V4 分母](reference/rpo_original_configuration/normalization.json)仅为历史对照。固定参照和线性加权可参考 [OECD/JRC 综合指标构建手册](https://doi.org/10.1787/9789264043466-en)。

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" freeze-scoring \
  --measurements-root "$panel" --original-rpo-reference "$reference" \
  --metric-profile four --output "$scoring"
```

默认权重从仓库的 `robolab/experiments/multitask_morphology/configs/bo_metric_weights_four.json` 读取。后续确定新权重时，复制该 JSON、调整四项任务内权重（非负、每个任务和为 1），在上述命令追加 `--metric-weights <调整后的JSON路径>`，并指定新的 `--output`。三任务总权重仍为 0.4/0.3/0.3。

程序在 CPU 上核验原机、50 份原始测量的模型和协议身份，生成独立的 `scoring.json`（十二项分母、**本次冻结权重**和来源）、`initial_observations.json`（50 个重新计分的初始观测）和 `completed.json`。以后修改配置不影响已冻结的评分或历史实验；每套新权重必须单独建评分和搜索目录。旧测量、旧评分和旧搜索目录不改动。

## 第三步　从前 50 个观测开始连续 BO

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" search \
  --root "$experiment" --shared-continuation "$continuation" \
  --scoring "$scoring/scoring.json" --output "$search" --trials 100 --gpu 0
```

也可通过根目录的统一入口启动同一流程（`--trials` 默认 100）：

```bash
conda run --no-capture-output -n robo_cjl python start.py bo --gpu 0 \
  --root "$experiment" --shared-continuation "$continuation" \
  --scoring "$scoring/scoring.json" --bo-output "$search"
```

不传 `--scoring` 时，搜索默认读取实验根目录下 `bo/scoring_original_rpo_four_metrics_pcgrad_ppo200_fixed_all_std/scoring.json`。这个文件必须先冻结成功；其他四指标权重配置的评分目录显式传 `--scoring`。库外 AMP 参考动作使用独立 HHTools Python；默认自动定位，必要时传 `--reference-python <HHTools Python 路径>`。

- 前 50 个结构只加载冻结的原始观测和本次评分；首个新提议是 `BO_TRIAL 51 / 100`。新设计在大腿 0.20–0.40 m、小腿 0.25–0.45 m 的连续范围内搜索，默认五款膝电机和五款踝电机全部参与（25 种离散电机组合）。
- 约束 GP 对全部已完成的历史观测拟合，提议来自 Sobol 点与 L-BFGS-B 有界局部优化，最大化约束期望改进（EI × 可行概率）。改进阈值为所有已实测可行结构中的全局最优综合代价。不能把有限格点采样作为连续优化，也不因某款电机额外加罚。
- 正式新增 50 个结构采用踝电机配额，五款踝电机均保留，膝电机不设此配额。前 50 个固定初始观测不计入配额，仅统计这轮已提议的新结构；同一提议因仿真故障重试仍只占一个名额。每次提议都使用同一个基于全部已完成观测拟合的 GP，配额仅限制可参与本次 cEI 最大化的踝电机，不改变综合代价、可行性定义、GP 训练数据或全局最优阈值。

| 正式搜索阶段 | 本轮新增结构数 | 踝电机提议条件（此前新增提议计数） |
|---|---:|---|
| 第 51–75 个结构 | 25 | 仅允许计数小于 5 的踝电机；结束时五款各评估 5 个 |
| 第 76–100 个结构 | 25 | 仅允许计数小于 20 的踝电机；按持续更新的 cEI 自由提议 |

因此，结束时每款踝电机在新增 50 个结构中出现 5–20 次，合计 50 次。配额保证各候选电机获得最低探索量，同时避免单一型号占据绝大多数搜索预算；它不是五个独立 BO 实验，也不是保证每款电机有相同最优性能的结论。

- 若新提议恰与旧库内的未评估结构完整物理键相同，才复用经核验的资产；库外结构按冻结 SW-GP 模型生成实际 URDF，并核验三个任务的物性与身份。按实际运动学重新生成和验证 AMP 八条参考动作，再进行原生 URDF → USD 转换。共享 Actor 保留原训练库物理归一化尺度。
- 新提议逐个保存确切腿长、电机、获取函数审计，随后完成微调和评估，再把真实观测加入 GP，直至总预算完成。预测代价不是实测结果。

如实际装配只允许紧凑的三个踝电机，可显式指定允许名单。新开的 50＋50 搜索仍对允许名单内每款电机执行新增 5–20 次配额（三款时前 15 次完成最低覆盖，后 35 次继续按 cEI 搜索并遵守上限）；其他预算或导入已完成历史的续搜保留原规则：

```bash
--allowed-ankles DM-J4340P-2EC DM-J4340P-2EC-V1.1-48V DM-J6216P-2EC
```

此参数只限制新增提议，不删除已测的 RS06/8009P 观测，也不修改评分。对在同一评分和协议下已完成的搜索，使用 `--history <已完成搜索目录>` 和新的独立 `--output`，若历史目录包含多个完成预算，应以 `--history-budget <编号>` 明确选择；总 `--trials` 设为希望达到的完整预算。调整四指标任务内权重时使用新的评分文件，并从前 50 个 B 观测开始独立搜索。

## 暂停恢复与保存产物

普通故障后重复原命令，沿用相同预算、算法、评分、配额、允许电机名单、输入身份和输出目录。人工暂停后明确加 `--resume-paused`。候选原生 PPO 每代记录奖励、Mean reward 和 PPO 诊断；仅在第 0 代、每 10 代、指定里程碑以及最终代写入完整学习状态，保留初始、指定里程碑和最近两份完整状态。该保存间隔不改变正常不中断的策略更新、采样或评价协议。候选从最近完整提交恢复 Actor、Critic、归一化器、固定的三任务动作标准差、两个 Adam、判别器、判别器归一化与采样缓冲区、实际学习率和随机数状态。故障或暂停后，如已有随后至多 9 代仅记录指标的更新，将保留旧尝试记录并从完整断点重新执行这些代数；尚未完成的更新也需重做。正式奖励 CSV 将按实际恢复后训练记录接续，避免重复代数。旧版检查点仅用于历史研究，不能接入当前微调。PhysX 场景及回合状态不恢复，重启仿真仍使用相同的随机回合计时方式。预热中断而尚未提交第 0 代则从原始 Actor 重做预热。训练结束但测量未完成时仅补测；原始历史和已提交数据不可修改。

| 产物 | 用途 |
|---|---|
| `proposal.json`、`observation.json` | 候选参数、获取函数记录、原始报告索引、分项与综合代价、可行性 |
| `candidate/tuning/checkpoints/`、`training_completed.json` | 每 10 代及初始、里程碑、最终代保存完整学习状态；保留初始、指定里程碑和最近两份完整状态。最终提交绑定 Actor 校验身份，各代均保留轻量指标记录 |
| `candidate/tuning/mean_reward.csv`、`training_reward.csv`、`candidate/training_timing.json` | 总体与分任务 Mean reward；按各任务自己的尺度看趋势，结合回合和终止计数。Mean reward 用最近结束的最多 100 个回合，重启接续已保存的窗口；训练计时累计实际子进程运行阶段，不计等待 |
| `candidate/bo_evaluation.json`、原始 200 Hz 采样及 `native_asset_evidence.json`（库外） | 三任务固定工况实测、原始物理量、模型与 USD 身份 |
| `budget_*.json` | 完成预算汇总、可行性和最优设计 |

## 单独微调、评价与复算

已有库内结构可通过共同 `shared_bank` 查询 `key`（大腿米、小腿米、膝踝完整型号）与任务 `design_ids`，再使用对应真实结构 ID：

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" evaluate \
  --root "$experiment" --shared-continuation "$continuation" --gpu 0 --candidate 真实结构ID
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" score \
  <评估目录>/bo_evaluation.json --scoring "$scoring/scoring.json"
```

统一入口也可直接微调库内候选，内部映射共同结构身份并调用同一个 BO `evaluate`。在已设置上文变量的终端执行：

```bash
conda run --no-capture-output -n robo_cjl python start.py 微调 \
  --root "$experiment" --shared-continuation "$continuation" \
  --candidate-task stairs --candidate-design-id 真实结构ID --gpu 0
```

`--candidate-task` 可选 `stairs`、`backpack` 和 `run`，以所在任务库查询结构 ID。默认微调 200 代、4096 环境、独立 10 代 Critic 预热，使用同一方案 B PPO 及三项各一条 20 秒、200 Hz 固定工况评估。输出正式 `bo_evaluation.json`。通过 `--finetune-output` 指定单候选输出目录，人工暂停后用相同命令加 `--resume-paused` 恢复。已有相同候选、起始模型、预算和 PPO 配置的完整微调可用 `bo.py evaluate --finetuned` 复用。

硬件性能 GP 实现也用于独立 [Rough BO](../rough_bo/粗糙地形优化说明.md)；其旧单任务教师和评价协议不等于三任务正式 BO。

旧版实机或原始测量导入的接口、格式和核验约定可在 Git 历史中的本说明查阅。历史评分数据与冻结代码保留其原始身份；新版四指标运行不得把 50 Hz 旧报告作为 200 Hz 实测报告。

## 两台独立 GPU 的固定50结构分段计算

两个实例必须使用同一冻结共享计划、第五轮 Actor 和原机 PPO200 基准。分段只影响初始结构索引，不改变每个结构的微调、固定评估或后续 cEI 算法。使用相同绝对实验路径（数据盘上的兼容链接保持已冻结路径可解析），不同实例上的磁盘相互独立：

- 原卡：原 `initialize` 命令增加 `--first-index 1 --last-index 27`。
- 新卡：同一命令增加 `--first-index 28 --last-index 50`。
- 每段完成写入 `shard_01_27_completed.json` 或 `shard_28_50_completed.json`；已有通过校验的报告直接复用，不重复 PPO 或评估。
- 新卡完成后按原路径同步 `design_28` 至 `design_50` 和对应分段完成记录。保留文件哈希和原始轨迹，不覆盖原卡第1–27号。
- 原卡使用同一 `initialize` 命令、去掉范围参数并加 `--finalize-only`。这一步不执行仿真，只验证两段身份、索引、报告、检查点及原始轨迹。完整覆盖50个结构后才写 `measurements_completed.json`，然后按原流程冻结评分、在原卡顺序搜索第51–100号。

禁止同一结构同时由两卡执行；人工暂停后仅在明确恢复时追加 `--resume-paused`。不同微调协议或不完整的分段不能合并。
