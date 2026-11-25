# 三任务 BO 评估与连续结构搜索

更新：2026-11-25。本流程从同一份获选共享 Actor 出发，给每个结构相同的 100 次三任务联合 PPO 微调预算，再用固定工况实测指标比较硬件。正式搜索分为前 50 个已测结构和后 50 个连续 BO 候选；2500 个格点是专家训练结构库，**不是 BO 的搜索边界**。

**原生 PPO 微调修复：** 新方法的算法身份为 `shared_native_ppoamp_expert_prior_v4`，旧版手写 PPO V2 和原生共享 Critic V3 的观测只能用于历史复算和对照；不得直接用于新方法的代理模型或归一化分母。在原机 1000 代对照实验通过验证之前，不重启 BO。详见[共享策略原生 PPO 微调说明](共享策略原生PPO微调说明.md)。

[总流程](跨形态多任务共享策略训练流程.md) · [共享蒸馏与选模](共享策略蒸馏详细说明.md) · [原机实测基准](原始配置基准.md)

<a name="第一步运行固定的-50-个结构并冻结尺度"></a>

## 第一步　准备或复用前 50 个结构

在仓库根目录执行。以下为目录示例；已有实验时将变量指向实际的数据根、已完成的原第 21 组测量和同一获选 Actor。不要移动或覆盖旧报告中已经冻结的来源路径及校验记录。

```bash
entry="$PWD/robolab/experiments/multitask_morphology"
experiment="$PWD/robolab/logs/multitask_full2500_physical_migrated"
continuation="$experiment/shared_policy_round5"
data="$experiment/bo"
panel="$data/calibration50_native_expert_prior_ppo100_200hz"
reference="$data/rpo_original_025_030_10010l_4340p_native_expert_prior_ppo100_200hz/evaluation"
scoring="$data/scoring_original_rpo_four_metrics_native_expert_prior"
search="$data/search_continuous_four_native_expert_prior_100"
```

已有 `measurements_completed.json` 且与原机基准的共享起点、原生 PPO 算法身份、微调、工况一致时，直接使用原始测量进入第二步，**无需重做微调或评估**。旧算法的 50 个观测虽保留，但不是新算法的数据。否则准备前 50 个结构：

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" initialize \
  --root "$experiment" --shared-continuation "$continuation" \
  --output "$panel" --measure-only --gpu 0
```

程序按原第 21 组固定顺序得到 50 个观测（与 200 个蒸馏选模结构无重叠）。已有同协议微调产物可用 `--reuse-finetuned-panel <目录>` 复用模型，只补缺失的固定工况测量。原机基准需包含 `reference_protocol.json` 和 `candidate/bo_evaluation.json`，缺少时按[原机复现流程](原始配置基准.md#复现单结构实验)生成。前 50 个结构和后续新候选均从同一获选 Actor 独立开始。

### 微调与固定工况

- 单仿真进程 4096 个环境，上楼、负重、跑步分别 1366、1365、1365 个；三任务共享 Actor，各自复制专家原生 Critic、实际归一化及 23 维动作标准差，按任务门控选择独立 Critic/探索参数；AMP 判别器在独立副本上随训练同步更新，老师和物理参数不改。
- 普通 PPO 随机初始化回合步数计数，先做 10 次只更新 Critic 的预热（不计入正式预算），每个候选做 **100 次原生 PPO 更新**，每次每环境采 24 步，训练 5 个 epoch、4 个小批/epoch；禁止用近邻结构替代实际提议。
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

上表是本轮正式搜索采用的默认权重。上楼优先考虑机身平稳性，负重优先考虑运输成本，跑步优先考虑速度跟踪。权重是工程偏好，保留完整原始指标以便做敏感性分析。原机十二项分母见[实测基准](原始配置基准.md#2026-11-24-实测基准)，计算使用完整精度，不使用表中四舍五入的数值。固定参照和线性加权可参考 [OECD/JRC 综合指标构建手册](https://doi.org/10.1787/9789264043466-en)。

```bash
conda run --no-capture-output -n robo_cjl python "$entry/bo.py" freeze-scoring \
  --measurements-root "$panel" --original-rpo-reference "$reference" \
  --metric-profile four --output "$scoring"
```

默认权重从仓库的 `robolab/experiments/multitask_morphology/configs/bo_metric_weights_four.json` 读取。后续确定新权重时，复制该 JSON、调整四项任务内权重（非负、每个任务和为 1），在上述命令追加 `--metric-weights <调整后的JSON路径>`，并指定新的 `--output`。三任务总权重仍为 0.4/0.3/0.3。`--metric-weights` 仅用于 `four`，旧三指标权重保持不变。

程序在 CPU 上核验原机、50 份原始测量的模型和协议身份，生成独立的 `scoring.json`（十二项分母、**本次冻结权重**和来源）、`initial_observations.json`（50 个重新计分的初始观测）和 `completed.json`。以后修改配置不影响已冻结的评分或历史实验；每套新权重必须单独建评分和搜索目录。旧测量、旧评分和旧搜索目录不改动。

### 旧三指标对照 `three`

旧原机比率评分继续使用：上楼绝对 CoT、速度 RMSE、横滚俯仰角速度 RMS 各 1/3；负重和跑步绝对 CoT、速度 RMSE、膝踝最大额定扭矩占比分别 0.4、0.4、0.2。任务总权重仍为 0.4/0.3/0.3。使用已有冻结评分文件，或者用同一 `freeze-scoring` 命令改为 `--metric-profile three` 并指定**独立**输出路径。两种评分使用同样的连续 BO 算法，不能在同一次搜索中混用。早期 P10/P90 与专家专用评分只用于历史对照，仍以各自的旧评分和原始报告复现，不作为当前正式搜索的新定标尺度。

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

不传 `--scoring` 时，搜索默认读取实验根目录下 `bo/scoring_original_rpo_four_metrics/scoring.json`。这个文件必须先冻结成功；其他评分目录（包括旧三指标、调整后权重）显式传 `--scoring`。库外 AMP 参考动作使用独立 HHTools Python；默认自动定位，必要时传 `--reference-python <HHTools Python 路径>`。

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

此参数只限制新增提议，不删除已测的 RS06/8009P 观测，也不修改评分。对在同一评分和协议下已完成的历史搜索，使用 `--history <已完成搜索目录>` 和新的独立 `--output`，若历史目录包含多个完成预算，应以 `--history-budget <编号>` 明确选择；总 `--trials` 设为希望达到的完整预算。需要冻结源码的历史实验用其原样代码树和数据，不能修改旧清单来使校验通过。切换三/四指标或调整四指标任务内权重时使用新的评分文件，并从前 50 个观测开始独立搜索，不将旧代价直接混入。

## 暂停恢复与保存产物

普通故障后重复原命令，沿用相同预算、算法、评分、配额、允许电机名单、输入身份和输出目录。人工暂停后明确加 `--resume-paused`。候选原生 PPO 从最近完整提交恢复 Actor、Critic、归一化器、可学习动作标准差、两个 Adam、判别器、判别器归一化与采样缓冲区、实际学习率和随机数状态；半次更新重做。PhysX 场景及回合状态不恢复，重启仿真仍使用相同的随机回合计时方式。预热中断而尚未提交第 0 代则从原始 Actor 重做预热。训练结束但测量未完成时仅补测；原始历史和已提交数据不可修改。

| 产物 | 用途 |
|---|---|
| `proposal.json`、`observation.json` | 候选参数、获取函数记录、原始报告索引、分项与综合代价、可行性 |
| `candidate/tuning/checkpoints/`、`training_completed.json` | 完整学习状态、最后一次 PPO 提交和最终 Actor 校验身份；保留初始、指定里程碑和最近两次完整状态，每代轻量提交记录都保留 |
| `candidate/tuning/mean_reward.csv`、`training_reward.csv` | 总体与分任务 Mean reward；按各任务自己的尺度看趋势，结合回合和终止计数。Mean reward 用最近结束的最多 100 个回合，重启接续已保存的窗口 |
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

已有同候选、同起始模型、同预算和 PPO 配置的完整微调用 `--finetuned` 复用。独立 `start.py 微调` 可做库内结构的名义/鲁棒步态诊断，但其训练规模、任务报告和行程门槛与正式 BO 不同，不作为正式评分数据。诊断仍支持 `--candidate-task`、`--candidate-design-id`、`--finetune-output`、`--finetune-evaluation-seeds`、`--evaluation-profile` 和 `--resume-paused`；默认名义与鲁棒均测，20 秒，实际指令行程达到 50% 为诊断成功。单独诊断的 `candidate_result.json` 不是正式 `bo_evaluation.json`。

硬件性能 GP 实现也用于独立 [Rough BO](../rough_bo/粗糙地形优化说明.md)；其旧单任务教师和评价协议不等于三任务正式 BO。

旧版实机或原始测量导入的接口、格式和核验约定可在 Git 历史中的本说明查阅。历史评分数据与冻结代码保留其原始身份；新版四指标运行不得把 50 Hz 旧报告作为 200 Hz 实测报告。
