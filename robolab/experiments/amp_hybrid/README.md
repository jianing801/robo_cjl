# AMP时空混合多形态训练（独立版本）

本目录仅训练一个AMP跑步策略。完整库按腿长×膝电机×踝电机生成，每阶段仅驻留一个小组，组内多结构共同采样、联合更新；组间连续继承同一策略。不是每组从头训练一个专家，也不融合Rough/背包。

## 配置与边界

示例config.local900.json：4096环境、10×10×3×3=900结构、每组30、50次正式更新换组。腿长0.25–0.40m / 0.30–0.45m；电机为Rough已确定的三膝/三踝池。当前24维结构输入（18连续特征＋3膝ID＋3踝ID）按完整库编码，仅追加Actor/Critic，不追加判别器。

2026-10-28更新：踝池明确为普通DM-J4340P-2EC、DM-J6216P-2EC、RS06，删除旧草稿的4310P；膝池为DM-J10010L-2EC、RS04、DM-J10010-2EC。与新版Rough的grid逐项一致，分组设计ID和各轮排表也一致。

为后续共享策略蒸馏，推荐**直接绑定Rough已有bank**，而不是再独立生成一批看起来相同的机器人。该入口只写一个AMP任务清单，不复制/重生成/改写URDF，900个design_id、URDF/report哈希、质量/惯量、曲线/armature、默认23关节姿态和24维context均完全继承。每次加载严格检查与原Rough bank的一致性，清理磁盘时必须保留共享资产和原bank。任务身份、原生AMP依赖哈希和动作参考身份仍单独冻结，不能加载Rough模型冒充AMP恢复。

同结构只是蒸馏准备的一部分：AMP与Rough原生观测、历史长度、命令及任务语义仍不同，后续学生策略还需统一观测适配、动作顺序/尺度、归一化及任务标记。当前不实现蒸馏、不合并判别器。AMP保留原0.25位置动作尺度；不把Rough楼梯奖励、−4扭矩奖励或健康奖励分项强行复制为AMP任务奖励。优化关节的真实速度—扭矩硬约束和对应物理参数保持一致。

本次用户没有另定AMP总预算，示例先保留原RPO-AMP的5000次更新，可在新实验前修改，不直接套用Rough的12000。5000/50=100组，等于3轮完整900库＋下一轮10组，不代表4轮都均等完成。模板所有训练数量仍可自行指定；改变环境数、全库、分组、轮换或预算需新建实验。

精确900配置每轮30组不重不漏，每组每个大腿/小腿值各3次、9个电机组合各3或4次。每组30结构的环境数量136/137。其他配置走分层分组，不承诺相同精确边际计数。没有复习组、遗忘指标或额外损失。

原RPO-AMP任务奖励、命令、DR、镜像、MLP、PPO和AMP判别器参数保持。学习率沿用原1e-4自适应，镜像开启，归一化保留原采样上限。仅适配新物理/电机、形态输入、初始化高度、组调度和对应参考。关节初始角不另调，出生高度按同一中立姿态FK计算。显示用本地平面替代远程视觉资源，平面物理参数保持。

当URDF启用固定关节合并时，仅将原事件中已合并的base_link选择器映射为torso_link（基体质量和质心随机化），范围、操作和其他事件不改；未合并时不映射，未知选择器拒绝。此为物理实体名称适配，不是新增随机化。每组native_contract记录具体映射。

## 换组继承什么

保存/严格恢复Actor、Critic及观测归一化、PPO Adam/学习率、AMP判别器、判别器归一化和独立Adam、Torch随机状态及全局更新计数。

默认同一Kit/AMP训练进程跨组运行，保留策略、PPO Adam、rollout分配、判别器、判别器归一化及判别器Adam对象。换组只重建仿真场景、动画与当前组参考路由；rollout索引、判别器负样本及示范缓存原地清空，不携带上一组旧样本；恢复场景初始化前的学习随机流。PPO更新不放在inference_mode中。不声称物理轨迹连续，也不保证显存完全不波动。

没有每组2代预检；每次PPO更新实时显示在主终端并写入日志，显示全局N/5000。只有完整组的模型、摘要与哈希全部通过才提交进度；中断仅从最后完整组恢复。默认协议amp_persistent_v1；旧逐组模式必须显式传--execution-mode per-group，禁止跨协议隐式续训。异常、failure.json或缺失摘要都拒绝当作成功，即使Kit返回0。

同步使用最新版Rough的本地URDF导入适配（不升级软件）、换组前未使用缓存释放、scene_memory.jsonl和失败后120秒清理保护；控制器另监视本次会话失败信号，防止worker卡在关闭阶段。实际returncode与有效失败码分别记录，不再把原始0码当作成功。共享的两个Rough辅助源码哈希纳入AMP实验身份，改变后不能静默续训。此处仍没有完整自动重启/续训循环。

本机已完成18环境、9种中点腿长/电机结构的真实Isaac三组/六次更新短测，以及96项AMP+Rough CPU回归；两套Adam连续、参考路由和几何审计通过。4096环境显存、长训和步态质量仍待验证。详见VALIDATION.md的最新记录。

## 参考动作

必须先用每个目标URDF生成HHTools R2R参考库，没有参考库时拒绝训练，不偷偷回退原动作。原AMP八段动作名称、混合权重和时间轴不改。每种目标统一六关键点：左右踝roll、左右膝、左右肘yaw，与当前原AMP判别器一致；不添加肩部。

按完整运动学/视觉碰撞几何签名缓存参考，只在确实几何相同时复用；不同电机造成几何变化时重新生成。不能仅凭腿长相同就假定同一示范。每个环境的实际design_id、对应参考目录和动画motion_id范围在启动时核查；只加载当前组需要的运动库。

沿用现有HHTools几何R2R方案；不额外做逐帧贴地、接触修复、平滑、重采样或动力学可行性修饰。脚底离地诊断会记录，不把参考可生成当作能跑好。

## 使用

在仓库根执行。先复制配置模板或使用示例；所有输出必须新目录。这里只给命令模板，不表示已经准备900组参考或启动训练。

CPU预览：
```bash
python robolab/experiments/amp_hybrid/prepare_bank.py --config robolab/experiments/amp_hybrid/config.local900.json
```

推荐：直接绑定与本机/服务器当前路径一致的Rough结构库，只创建新的AMP bank元数据：

```bash
python robolab/experiments/amp_hybrid/prepare_bank.py \
  --config robolab/experiments/amp_hybrid/config.local900.json \
  --rough-bank /actual/rough/bank/bank.json \
  --output-dir /data/amp_hybrid/bank --generate-assets
```

该命令不启动仿真或训练，不加载Rough策略；不要同时传template/mesh-dir。原Rough bank及资产保持只读。

兼容旧工作流：独立生成物理结构库（不自动保证与另一个Rough bank逐字节一致，因此不是本次对齐推荐路径）：
```bash
python robolab/experiments/amp_hybrid/prepare_bank.py \
  --config robolab/experiments/amp_hybrid/config.local900.json \
  --template /actual/rpo/urdf/rpo.urdf --mesh-dir /actual/rpo/meshes \
  --output-dir /data/amp_hybrid/bank --generate-assets
```

仓库已附带本机短测验证过的HHTools 0.1.0源码、许可证和逐文件校验清单，无需手动上传或另行克隆HHTools。在当前训练环境中执行以下命令，自动创建独立的Python 3.12环境amp_hhtools并安装固定版本依赖；需要服务器能访问Python包源，但不升级训练环境或Isaac：

```bash
python robolab/experiments/amp_hybrid/setup_hhtools.py
```

随后在该独立环境中离线生成参考（以下路径按自己新建的AMP_ROOT填写）：
```bash
conda run --no-capture-output -n amp_hhtools python robolab/experiments/amp_hybrid/prepare_references.py \
  --bank /data/amp_hybrid/bank/bank.json \
  --output /data/amp_hybrid/references \
  --hhtools-root robolab/experiments/amp_hybrid/vendor/hhtools
```

需要现有经过验证的HHTools 0.1.0 / Newton 1.1.0 / Warp 1.12.1环境，脚本会检查版本并强制CPU离线转换。初次可能耗时；未生成参考就不能训练。

训练使用Isaac Python：
```bash
python -u robolab/experiments/amp_hybrid/train.py \
  --config robolab/experiments/amp_hybrid/config.local900.json \
  --bank /data/amp_hybrid/bank/bank.json \
  --reference-library /data/amp_hybrid/references/library.json \
  --output-root /data/amp_hybrid/train --execute
```

默认只做CPU计划，加execute才训练。默认自动选择至少12GiB空闲且利用率≤30%的GPU，这是保护门槛，不是AMP实测显存保证；不要套用Rough的显存数据。支持device（当前可见GPU列表的逻辑编号）、gpu-min-free-gb、gpu-max-utilization、python、kit-args。未擅自放宽共用GPU门槛。

## 恢复与安全

- 新实验从零；不同bank、旧AMP/旧Rough/背包检查点拒绝加载。
- 同一新版实验中断可加resume，仅从最后完整提交组恢复，中断组最多重做U次。全局绝对预算不增加。
- 初始组已经尝试训练但没有完整提交时拒绝隐式从零；先查错，不自动重跑。
- Ctrl+C停止controller及同链worker，创建本实验USER_PAUSED；恢复前由用户确认移除，不操作背包标记。
- bank/物理/代码/参考库身份冻结；代码或参考改变拒绝续训，不绕过校验。
- GPU锁与原Rough共用，不重复占用同一协作锁；同bank进程保护不因无关Python或另一bank专家阻塞。进程退出竞争安全处理，不杀其他任务。
- USD按原生转换哈希复用本实验缓存，缺失正常转换；外部显式asset-cache严格匹配。每组仍重建一次场景，不承诺已消除全部换组耗时。

输出plan.json、attempts/stage_*/train/日志/manifest/geometry_audit/reference_routing_audit/restore_audit、commits/stage_*.json和最终completed.json。模型编号为全局0起始：5000预算对应model_4999.pt。完成只表示预算完成，不表示步态合格；不自动评估、录像、蒸馏或扩预算。

## 验证与隔离

独立复制并修改时空混合调度、原生空间AMP参考路由和HHTools适配；不修改原AMP、Rough代码及任何旧结果。共享依赖是原生AMP任务/RSL-RL、物理快照、几何审计、GPU选择、原生spawner和便携参考序列化。

CPU测试：
```bash
python -m pytest robolab/experiments/amp_hybrid/test_amp_hybrid.py -q
```

已验证真实PPOAMP小规模CPU更新、三组完整恢复、判别器双Adam/归一化、缓冲区清空、参考路由/哈希、调度中断恢复、原生AMPRunner构造和结构后缀隔离；真实Isaac三组短测也已通过。未做全规模HHTools生成或长训，不宣称跑步效果/遗忘或计算收益已验证。详见VALIDATION.md。
