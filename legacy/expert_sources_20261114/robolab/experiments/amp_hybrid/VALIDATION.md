# AMP时空混合改造验证记录

## 2026-10-28：与新版Rough对齐

以下为当前追加记录；后续旧日期段落仅为历史，不代表仍使用4310P或仅通过14项。

- 配置踝池更新为4340P/6216P/RS06，与当前Rough grid及4轮排表逐项一致。
- --rough-bank入口实际生成了本机900设计AMP清单，未转换或复制物理资产。
  独立完整审计通过：所有共享字段、900个ID、URDF/report路径及哈希、物理模型、
  23关节契约和24维context完全相同。绑定到本机Rough bank，不声称服务器原URDF字节一致。
- 本地审计：工作区amp_rough_alignment_20261028/parity_audit.json；
  AMP bank SHA256=32506606b834471c1d07f46e4322c58b82dfcc4e79f810fd71340f08cb0a7fe6。
- 增加同Rough的导入兼容、未用缓存释放、显存记录、失败清理超时和有效退出码。
  共用辅助代码哈希被冻结；没有完整自动续训循环，没有改依赖版本。
- 对合并固定基体，仅映射质量/质心事件选择器到torso_link，原事件参数不变。
- AMP+Rough联合CPU测试96项通过，最终重跑42.86秒；覆盖真实CPU PPOAMP梯度、
  两个Adam、判别器归一化、组切换/恢复、物理对齐拒绝和故障退出契约。
  使用PYTEST_DISABLE_PLUGIN_AUTOLOAD=1排除无关插件。
- AMP清单加载仍逐项校验每个资产哈希及腿长；已验证电机组合的context固定后缀
  按组合复用，避免900次重复读取物理快照，不降低各设计编码核验。
- 没有执行新版Isaac/GPU训练或生成新版900设计HHTools库；不能宣称跑步效果已验证。
  代码未提交/推送。详见ROUGH_ALIGNMENT.md。

日期：2026-10-24。仅代码改造和CPU验证；未启动Isaac/GPU、全规模HHTools参考生成、长训练、评估或录像，未操作服务器。

## 同进程正式入口追加验证（2026-10-25本地时间）

默认入口现为amp_persistent_v1：同一Kit和AMP learner跨组运行，保留策略、PPO Adam、rollout分配、判别器、判别器归一化和判别器Adam。换组重建场景及当前组参考，原地清空rollout/两类判别缓存，重绑形态均衡采样和镜像环境。场景初始化后恢复学习随机流；环境轨迹不连续。

AMP专用CPU测试14项通过（此前11项为逐组版本历史记录）：新增同一真实PPOAMP learner连续三组/六次更新，验证对象及buffer内存身份不变、Actor/判别器梯度有效、Adam持续累计、缓存清空前后学习状态精确一致、inference tensor安全清理及RNG恢复。另用真实CPU PPOAMP与仿真入口替身验证同进程调度、组提交、异常中断后恢复、末组只做1次更新；零退出码但无摘要会拒绝。测试目录amp_persistent_cpu_tests_20261024_v2。

没有进行新版Isaac换场景实测，也没有生成全量HHTools参考；这两项仍是运行验证门槛，不能以CPU通过替代。背包未暂停、未改动，没有启动AMP长训，也未提交或推送Git。

随后重跑AMP、Rough混合及公共GPU/启动CPU测试，合计52项通过（9.30秒），临时目录amp_persistent_combined_tests_20261024；19个AMP Python源文件均通过AST语法检查。Git只显示独立amp_hybrid目录未跟踪，没有改动已跟踪的Rough/RSL-RL文件。

## 实现

- 独立目录experiments/amp_hybrid，新的bank kind和experiment身份；原AMP、Rough和正在训练的背包不改。
- 4096环境、900库、30结构/组、50更新切换的示例；AMP总预算保留原5000，均可配置。
- 当前组内真实形态并行联合更新；换组保留Actor/Critic/归一化/PPO Adam/LR和AMP判别器/归一化/独立Adam、Torch随机状态。
- 环境、动画和当前组参考路由重建；rollout及判别器策略/示范缓冲区清空但保留分配，不混入旧组样本。
- 无每组预检，逐更新实时打印并写日志，记录全局更新边界，最后完整提交组恢复。
- 每设计绑定目标HHTools参考，沿用原八段/权重，统一六关键点；只加载当前组的参考。
- 原生AMP参数和MLP保留，结构24维后缀只进Actor/Critic。只做平面视觉离线替代、形态物理及出生高度/路由适配，没有改任务奖励或学习参数。
- source/bank/URDF/reference/代码与本地平面资源哈希冻结，拒绝跨实验、跨任务或损坏检查点。

## 测试

此前逐组版本联合测试共47项通过（当时AMP自身11项，加既有Rough混合及公共GPU/启动测试36项），在工作区独立临时目录执行：

python -m pytest robolab/experiments/amp_hybrid/test_amp_hybrid.py robolab/experiments/rough_hybrid/test_hybrid.py robolab/tests/test_rough_gpu.py robolab/tests/test_rough_launcher.py robolab/tests/test_rough_standalone.py -q -p no:cacheprovider

覆盖：

1. 900结构/30组精确腿长和电机均衡、4096布局136/137、5000绝对预算和尾阶段。
2. 真实PPOAMP算法在CPU合成观测上连续三组、每组2次更新；Actor/Critic和判别器确实更新，两个Adam状态续接，归一化及学习率恢复，运动缓冲区保持新建。
3. 判别器状态/归一化/Adam缺失或非有限值拒绝，旧任务身份拒绝。
4. 原生AMPRunner实际构造路径及LoggerAMP、判别器维度/历史与CircularBuffer适配。
5. 结构后缀不进入disc/disc_demo，原AMP镜像/学习率/判别器配置未被适配器改动。
6. 目标参考目录按物理环境路由，非法motion_id拒绝；23关节、六关键点、八段权重及bank/URDF/PKL哈希检查。
7. controller用模拟worker加真实CPU PPOAMP检查中断恢复、完整阶段提交、全局预算、暂停及完成后不再训练，不产生预检。
8. 相同bank重复工作者拒绝，不误阻塞不同bank专家。
9. 原Rough及原AMP/RSL-RL跟踪文件与HEAD无差异。

初次controller测试发现AMP阶段结果标签遗漏了一处，已修正并重新全测通过；没有将失败测试当作已完成训练。

另外，8环境/4结构的小配置实际完成URDF/GP质量惯量/中立姿态FK与bank哈希校验。这里只生成结构，0次物理仿真、0次PPO：
工作区amp_hybrid_validation_20261024_v1/bank/bank.json
SHA256：0c013b23f88a4cfba635c08ded26a8434f56329ecf6a3634302852771fa75926

## 尚未验证

- 新版在Isaac中的真实环境/参考路由、接触动力学、跨组PPO；当前几何审计/恢复审计代码尚需真实运行。
- 900设计HHTools参考库的全量生成与耗时；采用现有已使用的适配方法，不冒充本次已全部生成。
- 4096环境AMP显存、换组时间、训练稳定性、跑步质量、泛化及遗忘；不能套用Rough的显存结论。
- 没有自动启动GPU测试：本机背包正在运行，后续真实AMP短测需要用户安排GPU。

## 最新本机验证（2026-10-29整理，覆盖上述历史待验证状态）

AMP真实Isaac短测已完成：18环境、9种中点腿长/3×3电机结构，分3组、每组2次更新，共6次。使用HHTools生成的原8段完整参考，无合成参考替代。同一工作进程PID 22724及学习对象跨组保持；Actor、Critic和判别器参数逐组更新，PPO Adam与判别器Adam步数均为40→80→120。参考路由、结构绑定、几何审计、有限值及完成检查全部通过。验证结果：REAL_ISAAC_AMP_3_GROUP_6_PPO_PASSED。首轮启动暴露AppLauncher参数转发问题，修复后另建run_v2重测通过，失败记录保留。

AMP与Rough联合CPU回归96项通过（最终65.25秒），包含参数转发回归。900种正式结构清单已与所绑定本机Rough库逐项核对。Rough资源修复另已完成200组小规模真实场景重建验证。本次未验证4096环境AMP显存、900设计全量参考生成、长训收敛或步态质量。

本机证据位于工作区amp_real_smoke_20261028/run_v2_verification.json、cpu_junit.xml及rough_rebuild_200_20261028/verification.json；这些运行产物不纳入Git。服务器应绑定服务器实际Rough bank并生成匹配参考。

## 自动参考准备入口（2026-10-29）

train.py现在可省略--reference-library。--execute时缺失参考先调用独立amp_hhtools环境生成，完整校验后发布清单，再创建训练计划和申请GPU。已有清单仅校验复用；续训缺失原清单拒绝生成；预览不创建参考目录；失败产物保留，禁止覆盖已有参考。生成器增加逐几何/动作进度输出。

AMP与Rough联合回归113项通过（21.89秒）：包括此前训练/恢复测试、8项HHTools安装保护，以及9项自动准备测试，覆盖预览无副作用、已有库身份错误拒绝、续训保护、生成后校验与路径保留、非零退出/缺失清单/错误身份/validation-only均不发布清单、显式Python选择。新增准备过程以替身生成器验证控制逻辑；本次没有再次生成全库参考或重跑GPU训练。此前真实Isaac三组短测和8段真实HHTools生成结果仍是物理/生成器验证依据。

## 默认GPU＋CUDA Graph与固定缓存（2026-10-29）

自动入口缺失参考时默认GPU＋Graph，沿用--device和训练显存/利用率门槛选卡，并持有共用GPU协作锁；隔离环境子进程按UUID只看到该卡。参考完成并释放锁后，Isaac训练重新申请GPU。每段动作必须实际使用cuda和Graph且无fallback，否则不生成合格清单、不启动训练。默认固定Warp缓存跨实验和新进程复用。既有合格参考不重生成；显式--reference-device cpu保留CPU选项。

回归120项通过（11.88秒），新增GPU可见性/缓存传参、Graph未确认不发布以及CPU/普通GPU/Graph回退拒绝测试。未升级任何Isaac/HHTools依赖版本或修改任务奖励、物理、动作帧数/权重和IK迭代。

真实RTX3060验证：18环境、9种中点腿长/电机结构，自动GPU生成全部8段完整参考（1套独立几何），8段全部确认Graph启用且无fallback；相对既有CPU参考，最大关节差0.000450 rad、最大逐段关键点位置RMS差约0.0061 mm，帧数和fps一致。验证事先阈值为最大关节差0.01 rad、关键点RMS及根位置最大差各0.001 m，全部通过。

首次自动流程成功生成参考并自动进入Isaac，完成前两组4次更新，但第三组停在原生USD导出left_arm_yaw_link.tmp.usd处。人工结束此次短测，原进程/日志保留，run标记USER_PAUSED；不把停滞当作完整训练成功。随后使用同一GPU参考、已核验的同资产USD缓存另建run_cached复测，三组/6次更新全部通过；同PID和学习对象续接，Actor/Critic/判别器均更新，两套Adam步数40→80→120，几何和参考路由审计、有限值及完成模型检查通过。此处没有修复原生USD冷导出偶发停滞，也没有自动重启机制。

另外在全新HHTools进程中复用固定缓存生成一段完整362帧动作，关键Warp模块均显示cached，Graph仍实际启用；不是只在同一进程内重复求解的缓存测试。本次不代表4096环境、100套几何、完整长训或步态质量已验证。

证据目录：工作区amp_gpu_auto_smoke_20261029（references_verification.json、run_cached_verification.json、cache_stdout.log及全部原始日志/模型）；此前设备对照tmp/amp_gpu_reference_probe_20261029显示第二次求解CPU15.95s、普通GPU26.62s、GPU Graph5.77s。模型、参考和编译缓存不纳入Git。
