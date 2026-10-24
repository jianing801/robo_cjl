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

新版不进行每组预检：每组仅启动一次仿真并直接正式训练，第一组从零初始化，后续组完整恢复上一组Actor、Critic、Adam、学习率、归一化和随机状态。原生PPO每次更新的输出实时显示在主终端，并同步保存在该组stdout.log；分母显示绝对总预算12000，不是当前组50。

默认自动搜索本实验此前各组的原生USD缓存，只复用转换哈希完全相同的资源，首次遇到的设计正常转换。显式传入`--asset-cache`时仍要求所需设计精确匹配，不用别的设计替代。Windows建议短输出路径。每次换组仍退出旧worker并重建一次新场景，因此不是完全消除换组开销，也未证明具体提速幅度。

## 暂停、恢复与输出

- Ctrl+C中断controller，停止同链worker并创建本实验 `USER_PAUSED`。确认恢复前由用户移除该标记；不操作背包的暂停标记。
- 只从最后完整提交阶段恢复，中断阶段最多重做50次；初始阶段无完整提交时拒绝隐式从头重跑。不选较大块内模型跳过调度。
- `plan.json` 冻结配置、bank和代码身份；修改后拒绝原链恢复。用户已明确要求新版从头训练：使用全新实验目录，不带resume或旧模型，不升级旧的950代训练链；历史结果保留。新版自身未来中断后可按本节规则resume。
- `commits/stage_*.json`：阶段提交记录。
- `attempts/stage_*/train/`：独立日志及launch记录；新版不生成preflight目录。
- `train/result/models/model_N.pt`：全局0起始更新编号；最终model_11999.pt。
- `completed.json`：仅预算完成，不代表步态合格。失败停止，不自动评估、重训或蒸馏。

配置模板为 `config.template.json`；环境数、长度数、电机数量/池、组大小、轮换更新数、总预算均可在新实验前调整。精确900配置有上述严格均衡保证；其他配置走分层分组，不承诺相同边际计数。

## 文件与验证

`prepare_bank.py`生成库；`train.py`调度；`train_stage.py`单组训练；`hybrid_bank/layout/policy/runtime/contract/support.py`为独立实现。共享必要依赖仅包括原生RPO任务/RSL-RL、物理快照、几何审计、GPU选择和原生混合资产spawner。复制历史见 `COPY_PROVENANCE.json`，其中哈希是复制时记录，不要求服务器安装旧的未提交实验文件。

```bash
python -m pytest robolab/experiments/rough_hybrid/test_hybrid.py -q
```

已完成本机4096环境、两组各30真实结构、每组2次正式更新的短验证，跨组参数/归一化/Adam精确恢复；见 `VALIDATION.md`。短测版本并非整理后版本的新GPU复测。未启动50更新切换/12000长训，不能声称收敛、防遗忘或提速已验证。训练资源、模型、录像和本地日志不提交Git。
