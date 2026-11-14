# 三任务跨形态专家与共享控制策略

这是三任务的独立整合目录。**新的完整训练从这里启动**，不再把负重、AMP 或蒸馏入口挂在 `rough_hybrid` 下面。旧目录、旧模型和旧实验保留兼容用途，不在本次整理中迁移或覆盖。

整个流程分为两步：先分别训练三个跨形态专家，再用冻结的三个教师，通过行为克隆（BC）和 DAgger 训练**同一个、由任务与形态共同条件化的共享 Actor**。三个专家不是一个共享策略；蒸馏得到的共享 Actor 也不等于已经完成共享 PPO 微调。共享 Critic/PPO 代码保留在 `shared_policy` 中，不由下面的蒸馏命令自动启动。

## 目录

```text
multitask_morphology/
  prepare.py              三任务结构库一键生成与联合校验
  train.py                --task rough / backpack / amp，专家统一入口
  configs/                三个正式完整配置
  engine/                 中立的轮换、场景、PPO、负重、全身电机公共引擎
  amp_engine/             AMP 原生环境、PPO-AMP、参考动作生成及独立依赖
  audit_banks.py          三任务结构、实际 URDF 与电机一致性检查
  validate_teacher.py     只读检查专家完成记录、完整检查点和源码身份
  prepare_shared.py       由三个已完成专家生成共同结构库及蒸馏计划
  distill.py              共享 Actor 的 BC → DAgger → 逐任务配对评估
  shared_policy/          共享网络、观测适配、采集、训练与录像工具
```

`engine` 是公共引擎，不是 Rough 实验目录。三任务各有独立 bank、训练输出、Actor/Critic、Adam 和完成记录；默认串行启动，不会自动抢占其他任务的 GPU。基础机器人、GP、候选电机曲线、原生任务与 RSL-RL 继续使用仓库公共依赖。

## 正式参数

| 项目 | 三任务共同设置 |
|---|---|
| 结构库 | 10 × 10 × 5 × 6 = 3000 种完整结构 |
| 大腿 / 小腿 | 0.20–0.40 m / 0.25–0.45 m，各 10 个等间距点 |
| 膝 / 踝电机 | 完整 5 款 / 6 款；不含 4310P |
| 并行环境 / 每组 | 4096 个副本 / 60 种结构 |
| 换组 | 每 50 次 PPO 更新；每 50 组全库一遍（2500 次更新） |
| 正式分组 | 冻结均衡排表：每组30种电机组合各2种；每档大腿/小腿各6种 |
| 专家预算 | Rough 12000；负重 12000；AMP 5000 次更新 |
| 全库轮数 | Rough / 负重各 4 轮＋第 5 轮 40 组；AMP 完整 2 轮 |
| 全身电机 | 23 关节按对应型号应用实际扭矩—转速硬包络 |
| 软约束 | 全身统一 −12，动态可用力矩利用率超过 80% 的部分平方求和；控制步长只乘一次 |
| GP | 按用户授权允许上述范围外推；保留外推审计 |

膝候选：RS04、DM-J10010L-2EC、DM-J10010-2EC、DM-J6248P-2EC、DM-J10422P-2EC。

踝候选：RS06、DM-J4340P-2EC、DM-J8009P-2EC-V1.1、DM-J4340P-2EC-V1.1-48V、DM-J8006-2EC-V1.1、DM-J6216P-2EC。普通 4340P 与 V1.1-48V 是不同候选，不能混用其曲线。

固定腿部电机使用 10010L，肩和臂使用普通 4340P。六关节旧惩罚关闭，避免与全身惩罚重复。动态可用力矩接近零时不进行除法，该项暂不计惩罚并单独计数；这保留此前确定的边界处理，不代表超速区已受到额外惩罚。

Rough 与 AMP 复用同一批基础机器人 URDF；负重只在同一基础机器人上增加独立背包体，0.30 kg 是模板初始化质量，训练时由原生负载课程覆盖。负重课程按绝对更新 0–999、1000–2999、3000+ 推进，不在换组后重新开始。AMP 根据真实运动学生成各结构参考动作，保留原生三帧、按观测项排列的教师输入与训练得到的归一化器；共享学生使用十帧流式适配。

## 1. 环境与结构库

以下 Linux 命令均从仓库根目录运行，Python/Isaac Sim/Isaac Lab **沿用已验证环境，不升级版本**。Windows 可使用相同 Python 入口，将环境变量与换行改为 PowerShell 写法。

```bash
cd /data/chenjiale/robo_cjl
conda activate robo_cjl
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENTRY=robolab/experiments/multitask_morphology
export ROOT=/data/chenjiale/robo_cjl/robolab/logs/multitask_full3000_balanced_v2

# 必须填写本机真实存在的原始 RPO URDF 及显示网格目录。
export TEMPLATE=/absolute/path/to/original_robot.urdf
export MESH_DIR=/absolute/path/to/original_meshes

# 去掉 --generate-assets 为 CPU 配置预览，不生成结构或训练。
python -u "$ENTRY/prepare.py" --output-root "$ROOT" \
  --template "$TEMPLATE" --mesh-dir "$MESH_DIR" --generate-assets
```

`ROOT` 必须是新目录。程序只生成一次基础 3000 种 URDF，再绑定 AMP 与负重结构库，最后写 `bank_audit.json`；任何联合校验失败都不会启动训练。显示网格仍是原始模型，不冒充参数化 CAD 外观。不要把原始 URDF 换成某个已生成的候选 URDF。

AMP 的参考动作生成器需要独立 Python 3.12 环境。此前已安装可跳过；新安装使用本目录自带快照，不改变训练环境：

```bash
python "$ENTRY/amp_engine/setup_hhtools.py" --env-name amp_hhtools
```

## 2. 分别启动三个跨形态专家

下面三条是独立任务，按空闲显卡安排，**不要在同一张卡上同时运行**。`--device cuda:1` 表示请求这张物理卡；内部子进程受可见卡映射影响可能显示 `cuda:0`，这不代表跑错卡。按实际卡号修改。

```bash
python -u "$ENTRY/train.py" --task rough \
  --bank "$ROOT/banks/rough/bank.json" --output-root "$ROOT/experts/rough" \
  --device cuda:1 --gpu-min-free-gb 12.5 --gpu-max-utilization 100 --execute

python -u "$ENTRY/train.py" --task backpack \
  --bank "$ROOT/banks/backpack/bank.json" --output-root "$ROOT/experts/backpack" \
  --device cuda:1 --gpu-min-free-gb 12.5 --gpu-max-utilization 100 --execute

python -u "$ENTRY/train.py" --task amp \
  --bank "$ROOT/banks/amp/bank.json" --output-root "$ROOT/experts/amp" \
  --reference-library "$ROOT/references/library.json" \
  --reference-device cuda --hhtools-env amp_hhtools \
  --device cuda:1 --gpu-min-free-gb 12.5 --gpu-max-utilization 100 --execute
```

AMP 会先用 GPU/CUDA 生成并校验缺失的参考动作，再启动训练；有效参考缓存可以复用。首次 CUDA 编译会增加耗时。训练启动前独立转换并校验原生资产；同进程跨组保留策略、优化器及训练缓冲区，重建仿真场景，清空旧轨迹索引。转换隔离、超时与有限重试保留，不保证原生导出器永不挂起。

去掉 `--execute` 只创建/检查计划，不执行训练。需要详细引擎参数时：

```bash
python "$ENTRY/engine/train.py" --help
python "$ENTRY/amp_engine/train.py" --help
```

### 中断续训与暂停

确认没有同任务残留 worker 后，在**同一条原启动命令**末尾加 `--resume`，保持 bank、配置、输出目录和预算不变。只接受已提交完整阶段的同链检查点，保留 Adam、学习率与更新编号；不声称物理轨迹逐位连续。未提交的阶段可能重做。

Ctrl-C 会写 `USER_PAUSED`。明确决定恢复时，先确认进程已停，再由操作者移除**对应任务输出目录**中的该标记，然后执行原命令加 `--resume`；程序不会自动清除此标记。不要恢复历史背包专家，不要用新目录代码直接续旧目录训练。目录整理改变源码身份，旧训练仍使用其原冻结代码；本目录用于新完整实验。

## 3. 三专家蒸馏为一个共享策略

三个专家都完成后，先绑定其最终检查点。该步骤校验 bank、源码身份、绝对更新数、模型/优化器及 AMP 参考库；未完成或不同任务会拒绝。专家训练完成记录只证明预算完成，不等于步态已验证，后续保留教师与学生的配对评估。

```bash
python -u "$ENTRY/prepare_shared.py" \
  --rough-root "$ROOT/experts/rough" \
  --backpack-root "$ROOT/experts/backpack" \
  --amp-root "$ROOT/experts/amp" \
  --output "$ROOT/shared_plan"

python -u "$ENTRY/distill.py" --plan "$ROOT/shared_plan/plan.json" \
  --output "$ROOT/shared_actor" --device cuda:1 \
  --batch-designs 10 --collect-steps 1000 --stride 5 \
  --epochs 20 --dagger-rounds 3 --timeout-seconds 1800 \
  --gpu-min-free-gb 12.5 --gpu-max-utilization 100
```

这一步会立即执行：先教师演示采集和 BC，再让学生行走、教师对学生真实遇到的状态给动作标签，累计全部旧数据做 DAgger。教师权重冻结，不重训专家。各任务等权采样，任务内均匀采样结构。三个 bank 必须有完全相同的腿长与膝踝型号集合；负重附件差异由任务标识区分。

全库真实轨迹数据量较大，需给输出盘预留相应空间。为控制拟合时的内存，每个“结构 × 数据批次 × 训练/验证分流”默认最多均匀保留 32 行参与拟合，原始采集文件不删除；可显式设置 `--max-rows-per-design-shard`，设为 0 才会全部载入内存。这个参数不是任务权重，也不改变教师动作或仿真物理。

结束时对**全共同库、三个任务**分别执行教师/学生配对评估，逐环境保存原始结果。通过要求：完整 20 秒、不摔、前进速度至少 0.1 m/s；楼梯另要求上升至少 0.1 m。按用户要求忽略横移，不把 MSE 小或进程结束当作通过。该门槛是名义步态门槛，不代表所有扰动与所有新结构均成功。查看 `shared_actor/completed.json` 中 `nominal_passed` 与各批次 `summary.json`。续蒸馏在原命令末尾加 `--resume`，不变更参数预算。

单结构录像：从 `shared_plan/plan.json` 对应任务的 `design_ids` 中复制一个 ID，`STUDENT` 填 `shared_actor/completed.json` 中的 `student` 路径。负重示例固定为 10 kg。录像输出目录必须新建。

```bash
export DESIGN=tr_replace_with_one_real_id
export STUDENT=/absolute/path/from/shared_actor/completed.json
python "$ENTRY/shared_policy/rollout.py" --plan "$ROOT/shared_plan/plan.json" \
  --task backpack --design-ids "$DESIGN" --student "$STUDENT" --payload-kg 10 \
  --mode evaluate --steps 1000 --output "$ROOT/video_backpack_one" --video --headless --device cuda:0
```

每种结构默认有 8 个用于训练/验证及负载覆盖的仿真副本；录像摄像机跟踪其中一个机器人。若需要严格只创建一台机器人，不能直接改 `copies` 破坏训练/验证分流，应使用专用单机评估计划，和正式蒸馏计划分开记录。

## 输出与适用边界

- `banks/`、`bank_audit.json`：三任务一致结构与实际 URDF 审计。
- `experts/TASK/plan.json`、`commits/`、`completed.json`：专家身份、阶段提交和最终模型位置；全身动态利用率、零能力区和终止等健康日志保留。
- `references/library.json`：AMP 每结构参考动作身份。真正新结构必须重新生成匹配运动学的参考，不能按近似腿长借用。
- `shared_plan/plan.json`：共同结构库、三个教师及参考库哈希。
- `shared_actor/`：真实采集数据、BC/DAgger 学生、配对原始步态证据、完成报告。

本目录是代码整合与启动规范，不声称新的 3000 结构三专家已经训好。历史 200 结构共享策略结果继续属于其历史实验，不能用来替代本次全库验证。旧入口与旧实验未删除、未搬动。

## 已启用的正式均衡分组

三个正式配置现在都绑定 `scheduling/formal_schedule.json`，使用离线验证过的种子43排表。每组60种、每轮50组；每款膝电机每组12种、每款踝电机每组10种。详见 [构造、搜索与覆盖证据](scheduling/README.md)。上述启动命令不需要另加参数。

每轮重复同一套组成员与组顺序，以保留已验证的连续窗口和跨轮覆盖。组内已独立打散；第 `cycle` 轮再旋转 `cycle×16` 个位置，将4096除以60多出的16个环境副本轮流分配给不同结构，组成员和组顺序不变。训练启动不执行退火，也不依赖额外求解器。

排表SHA256、公共加载器SHA256及绑定配置进入实验身份；每次调度校验完整性、容量和精确边际。排表变化、网格变化或哈希不符都会拒绝启动/续训。旧配置未声明 `group_schedule` 时保留旧调度分支，旧实验仍必须使用其原冻结源码；不能在旧run里替换配置后直接续训。

本次只改变新实验的排表。3000种结构、物理、电机曲线、奖励、负载课程、4096环境、每组50代以及各任务总预算不变。结构库网格未改变，已有bank和AMP参考若通过原生身份校验，可继续作为新实验的输入，不必为了排表重做运动学参考；训练输出必须使用新目录。本次未启动GPU训练。
