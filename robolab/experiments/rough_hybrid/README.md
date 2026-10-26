# Rough时空混合多形态训练

独立入口，不修改原 `train.py --multi_morph`、原空间Rough代码或历史结果。不含AMP、背包、蒸馏、自动评估或遗忘指标。

## 方法与当前配置

完整结构库 = 大腿长度数 × 小腿长度数 × 膝型号数 × 踝型号数。每阶段只加载一组真实URDF，多形态同时采样联合更新。默认正式入口采用同一个worker和Kit应用跨组运行：仅重建环境/场景，Actor、Critic、Adam、学习率、归一化及rollout存储分配持续保留。新组清空旧轨迹索引，不把旧组样本用于新组PPO。

当前 `config.local900.json`：

| 项目 | 设置 |
|---|---|
| 环境/全库 | 4096 / 10×10×3×3 = 900结构 |
| 大腿/小腿 | 0.25–0.40m / 0.30–0.45m，各10个等间距值 |
| 每组/轮换 | 30结构，每50次正式PPO更新换组 |
| 总预算 | 12000更新，240阶段，完整覆盖8轮 |
| 膝电机 | DM-J10010L-2EC、RS04、DM-J10010-2EC |
| 踝电机 | DM-J4340P-2EC、DM-J6216P-2EC、RS06 |

电机是seed42抽样后的固定结果，保证含10010L与普通4340P。每组每个大腿/小腿取值各3次；9种电机组合分别出现3或4次；每轮900结构不重不漏。4096环境在30结构间分为136或137个。无额外“复习组”或缺席间隔约束。

结构输入始终按**完整库**编码，当前24维（18连续特征＋3膝型号＋3踝型号），不会按当前组重编号。改变电机型号即改变bank身份，即使输入维数不变也不能直接续用旧模型。除下述显式扭矩软约束外，原RPO-Rough奖励、命令、随机化、镜像和PPO参数不变；楼梯5–10cm。沿用已验证副本的基体质量随机化目标torso_link，在独立适配器内设置，不改原任务。

### 6216P与跑酷式扭矩软约束（2026-10-26）

`config.local900.json`明确将4310P替换为6216P，并增加：

```json
"torque_soft_constraint": {"weight": -4.0, "limit_ratio": 0.8}
```

只对两膝与四踝的实际输出力矩施加
`sum_valid_j relu(abs(applied_torque_j) / available_j - 0.8)^2`。
此版本改为无量纲动态利用率，不再是跑酷原版的绝对超额扭矩平方。分母直接读取硬限幅同一物理子步使用的曲线可用力矩（含各环境峰值上限），不使用第一个URDF的共用上限。当前对称包络保持原样，不引入新的驱动/制动曲线。奖励采样最后一个物理子步，不冒充整个控制周期的平均饱和率。

初始权重−4.0由原权重−0.01乘参考力矩20 N·m的平方得到；在动态可用力矩20 N·m时新旧等强，不是所有电机工况都等强，也未经长训调优。六关节都恰好满载时成本0.24、奖励率−0.96；控制步长0.02秒时单步−0.0192。原生RewardManager负责且仅负责一次`weight * dt`，并记录该奖励项。

可用力矩不大于峰值的1e−6时不作除法，该关节从利用率成本中排除，并在`env._design_torque_soft_diagnostics`保留unavailable掩码，不能解释为健康。该运行时诊断另含利用率及裁剪前请求超额N·m，目前未接入持久化日志。输出裁成零时本奖励不会提供超速惩罚；此为明确边界，不用任意极小分母制造巨额奖励。首次奖励前必须已有执行器物理步，否则明确报缺失动态上限。

这是**输出扭矩软约束**，不是削平前请求惩罚，不保证消除超大原始动作。速度相关硬包络、PD、0.25动作尺度、动作裁剪和其余控制语义均不改。配置设weight=0或省略该字段时不安装新项。参数及新增源码均进入实验身份；修改后不能沿用旧run或旧checkpoint冒充原链续训。

6216P使用冻结目录现有45 N·m模型及对应SW-GP；260 rpm端点是已注明的用户批准48V假设，不是厂家确认值。必须使用新输出目录重新生成bank，不手改旧bank或旧模型；600个相同型号设计与300个新型号设计均由当前生成器独立生成和校验，原数据保持不变。仅完成工程验证不代表已解决长期收敛问题。

## 启动

从仓库根执行，需要已安装原RPO依赖及本机机器人URDF/mesh资源。以下路径换成服务器真实路径；不可复制Windows绝对路径。物理GP/电机曲线取仓库冻结SW快照，不使用旧策略。

先预览配置（CPU，不启动Isaac）：

```bash
python robolab/experiments/rough_hybrid/prepare_bank.py --config robolab/experiments/rough_hybrid/config.local900.json
```

在新目录生成完整结构库：

```bash
python robolab/experiments/rough_hybrid/prepare_bank.py \
  --config robolab/experiments/rough_hybrid/config.local900.json \
  --template /actual/rpo/urdf/rpo.urdf --mesh-dir /actual/rpo/meshes \
  --output-dir /data/hybrid900/bank --generate-assets
```

默认仅创建CPU计划；加 `--execute` 才训练：

```bash
python robolab/experiments/rough_hybrid/train.py \
  --config robolab/experiments/rough_hybrid/config.local900.json \
  --bank /data/hybrid900/bank/bank.json \
  --output-root /data/hybrid900/run --execute
```

已有CPU计划或中断续训时，使用同一命令加 `--resume`。全新目录不可加resume，已存在目录不可隐式覆盖。默认自动选空闲显卡，支持 `--device cuda:0`、`--python`、`--kit-args`；GPU锁不能约束其他用户。默认要求至少12GiB空闲显存，不代表显存充足保证。可用 `--gpu-min-free-gb` 显式调整检查门槛，不会改变环境数。

新版不进行每组预检。第一组从零初始化，后续组保留同一个策略、优化器、存储对象，并重绑新环境与结构采样分组，不通过重复加载模型模拟连续运行。每次更新输出实时显示在主终端，并同步保存在attempts/session_*/train/stdout.log；分母显示绝对总预算12000。各组独立manifest、geometry_audit和summary保留在attempts/group_*/result；同一次worker的模型统一保存在该会话第一组的models目录，commits记录精确路径。`--execution-mode per-group`仅用于显式选择旧的逐组进程模式。

默认自动搜索本实验此前各组的原生USD缓存，只复用转换哈希完全相同的资源，首次遇到的设计正常转换。显式传入`--asset-cache`时仍要求所需设计精确匹配，不用别的设计替代。Windows建议短输出路径。persistent模式只重建场景，不退出worker；仍存在场景重建开销，未证明具体提速幅度。

## 暂停、恢复与输出

共享服务器进程检查使用独立hybrid_processes：不通过process_iter预读所有进程的cwd；仅在匹配Rough命令且bank为相对路径时读取。进程退出竞争安全跳过，仍活跃但无法解析相对bank的进程则报告并停手。无须sudo，不杀其他用户进程。原rough_gpu及原空间Rough入口保持不变。

- Ctrl+C中断controller，停止同链worker并创建本实验 `USER_PAUSED`。确认恢复前由用户移除该标记；不操作背包的暂停标记。
- 只从最后完整提交阶段恢复，中断阶段最多重做50次；初始阶段无完整提交时拒绝隐式从头重跑。不选较大块内模型跳过调度。
- `plan.json` 冻结配置、bank和代码身份；修改后拒绝原链恢复。新同进程链中断后用同一命令加resume，仅重做最后未提交组，不新增预算。升级已有b313646逐组链须使用下方显式升级工具，不修改旧plan或旧检查点。

## 旧逐组训练升级为正式同进程训练

只支持完整源哈希匹配已审核b313646版本（LF或CRLF），原生训练源、bank、物理资产、上下文和全部训练配置必须与当前一致。只取最后完整提交组，30秒稳定检查、完整CPU加载验证后，在全新输出中生成bootstrap.pt及upgrade_audit.json。只更新两处身份元数据；策略/归一化张量、Adam、学习率、迭代和Torch随机状态逐项严格比对，原目录只读。未知版本或差异拒绝升级，不从头兜底。

服务器示例（原bank及旧训练路径保持原样）：

```bash
OLD_RUN=/data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/20261024_134517/train_20261024_142000
BANK=/data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/20261024_134517/bank/bank.json
NEW_RUN=/data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/persistent_$(date +%Y%m%d_%H%M%S)
python robolab/experiments/rough_hybrid/upgrade_run.py --old-root "$OLD_RUN" --bank "$BANK" --output-root "$NEW_RUN"
# 上一条成功后再执行。升级本身不启动GPU或训练。
python -u robolab/experiments/rough_hybrid/train.py --config robolab/experiments/rough_hybrid/config.local900.json --bank "$BANK" --output-root "$NEW_RUN" --resume --execute
```

默认自动选GPU；显式指定可附加`--device cuda:3`。共享卡无显存预留保证。不要把服务器示例直接用到本机，也不要恢复已弃用的950代旧链。当前600代的故障链如最后完整组确为stage11，升级后从600接着训练到绝对12000，而不是再追加12000。
- `commits/stage_*.json`：阶段提交记录。
- `attempts/session_*/train/`：会话日志、launch和有效/原始退出码；不生成preflight目录。
- `attempts/group_*/result/`：每组场景、几何审核、manifest和摘要。同一会话模型在首组models中，跨会话通过commits查精确路径；全局0起始编号，最终model_11999.pt。
- `completed.json`：仅预算完成，不代表步态合格。失败停止，不自动评估、重训或蒸馏。

配置模板为 `config.template.json`；环境数、长度数、电机数量/池、组大小、轮换更新数、总预算均可在新实验前调整。精确900配置有上述严格均衡保证；其他配置走分层分组，不承诺相同边际计数。

## 文件与验证

`prepare_bank.py`生成库；`train.py`调度并持有GPU锁；`persistent_worker.py`在同一个应用内持续调度；`train_stage.py`负责每组场景/更新；`upgrade_run.py`负责显式旧版本升级；`hybrid_bank/layout/policy/runtime/contract/support.py`为独立实现。共享必要依赖仅包括原生RPO任务/RSL-RL、物理快照、几何审计、GPU选择和原生混合资产spawner。复制历史见 `COPY_PROVENANCE.json`，其中哈希是复制时记录，不要求服务器安装旧的未提交实验文件。

```bash
python -m pytest robolab/experiments/rough_hybrid/test_hybrid.py robolab/experiments/rough_hybrid/test_persistent.py -q
```

已完成正式同进程入口的4096环境、两组各30真实结构、每组2次更新验证；不是旧逐组恢复测试。另有CPU中断续训、末组绝对预算和旧版本精确升级回归，见 `FORMAL_PERSISTENT.md`。历史逐组测试仍见 `VALIDATION.md`，不混为当前证据。未启动50更新切换/12000长训，不能声称收敛、防遗忘或长期无泄漏已验证。训练资源、模型、录像和本地日志不提交Git。
