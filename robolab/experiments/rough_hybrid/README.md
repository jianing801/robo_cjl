# Rough时空混合多形态训练

独立入口，不修改原 `train.py --multi_morph`、原空间Rough代码或历史结果。不含AMP、背包、蒸馏、自动评估或遗忘指标。

## 方法与当前配置

完整结构库 = 大腿长度数 × 小腿长度数 × 膝型号数 × 踝型号数。每阶段只加载一组真实URDF，多形态同时采样联合更新；阶段结束后释放仿真，换下一组。Actor、Critic、Adam、学习率、归一化和Torch随机状态连续保留；环境与rollout重新建立。

当前 `config.local900.json`：

| 项目 | 设置 |
|---|---|
| 环境/全库 | 4096 / 10×10×3×3 = 900结构 |
| 大腿/小腿 | 0.25–0.40m / 0.30–0.45m，各10个等间距值 |
| 每组/轮换 | 30结构，每50次正式PPO更新换组 |
| 总预算 | 12000更新，240阶段，完整覆盖8轮 |
| 膝电机 | DM-J10010L-2EC、RS04、DM-J10010-2EC |
| 踝电机 | DM-J4340P-2EC、DM-J4310P-2EC-48V、RS06 |

电机是seed42抽样后的固定结果，保证含10010L与普通4340P。每组每个大腿/小腿取值各3次；9种电机组合分别出现3或4次；每轮900结构不重不漏。4096环境在30结构间分为136或137个。无额外“复习组”或缺席间隔约束。

结构输入始终按**完整库**编码，当前24维（18连续特征＋3膝型号＋3踝型号），不会按当前组重编号。改变电机池数量会改变网络维数，不能直接续用旧模型。原RPO-Rough奖励、命令、随机化、镜像和PPO参数不作调优；楼梯5–10cm。沿用已验证副本的基体质量随机化目标torso_link，在独立适配器内设置，不改原任务。

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

`--asset-cache` 仅复用原生转换哈希严格匹配的USD缓存。Windows建议短输出路径。每阶段先2次真实PPO预检、丢弃参数，再正式训练；预检不计入12000，但必须计入计算成本。没有预检复用或自动扩预算。

## 暂停、恢复与输出

- Ctrl+C中断controller，停止同链worker并创建本实验 `USER_PAUSED`。确认恢复前由用户移除该标记；不操作背包的暂停标记。
- 只从最后完整提交阶段恢复，中断阶段最多重做50次；初始阶段无完整提交时拒绝隐式从头重跑。不选较大块内模型跳过调度。
- `plan.json` 冻结配置、bank和代码身份；修改后拒绝原链恢复。整理版独立身份依赖与先前短测版不同，旧短测bank/plan不能直接复用，需生成新库；历史结果不改。
- `commits/stage_*.json`：阶段提交记录。
- `attempts/stage_*/preflight/`、`train/`：独立日志及launch记录。
- `train/result/models/model_N.pt`：全局0起始更新编号；最终model_11999.pt。
- `completed.json`：仅预算完成，不代表步态合格。失败停止，不自动评估、重训或蒸馏。

配置模板为 `config.template.json`；环境数、长度数、电机数量/池、组大小、轮换更新数、总预算均可在新实验前调整。精确900配置有上述严格均衡保证；其他配置走分层分组，不承诺相同边际计数。

## 文件与验证

`prepare_bank.py`生成库；`train.py`调度；`train_stage.py`单组训练；`hybrid_bank/layout/policy/runtime/contract/support.py`为独立实现。共享必要依赖仅包括原生RPO任务/RSL-RL、物理快照、几何审计、GPU选择和原生混合资产spawner。复制历史见 `COPY_PROVENANCE.json`，其中哈希是复制时记录，不要求服务器安装旧的未提交实验文件。

```bash
python -m pytest robolab/experiments/rough_hybrid/test_hybrid.py -q
```

已完成本机4096环境、两组各30真实结构、每组2次正式更新的短验证，跨组参数/归一化/Adam精确恢复；见 `VALIDATION.md`。短测版本并非整理后版本的新GPU复测。未启动50更新切换/12000长训，不能声称收敛、防遗忘或提速已验证。训练资源、模型、录像和本地日志不提交Git。
