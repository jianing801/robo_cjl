# 三任务完整候选池与负重跨形态入口

## 统一配置

Rough、AMP、负重的 `config.fullbody3000.json` 使用完全相同的结构网格：大腿10点×小腿10点×膝5款×踝6款，共3000种结构；4096个环境，每组30种结构、50次更新。100组完整覆盖一遍。

膝：RS04、DM-J10010L-2EC、DM-J10010-2EC、DM-J6248P-2EC、DM-J10422P-2EC。
踝：RS06、DM-J4340P-2EC、DM-J8009P-2EC-V1.1、DM-J4340P-2EC-V1.1-48V、DM-J8006-2EC-V1.1、DM-J6216P-2EC。4310P已排除。

当前腿长范围为大腿0.20–0.40m、小腿0.25–0.45m，按用户明确授权启用GP外推。结构条件29维；三任务的结构ID、条件向量、关节语义顺序和电机参数保持一致。原版4340P与V1.1型号明确分开。

Rough/负重总目标仍12000更新，AMP仍5000更新。本次不增加预算。三任务轮换的前两轮排表逐项一致；一轮中每个电机组合恰好100种结构，每个腿长组合恰好30种结构。

## 基础资产只生成一次

1. 用 `rough_hybrid/prepare_bank.py --config rough_hybrid/config.fullbody3000.json --template <本机原始URDF> --mesh-dir <本机网格目录> --output-dir <新Rough库> --generate-assets` 生成基础库。
2. 用 `backpack_hybrid/prepare_bank.py --config backpack_hybrid/config.fullbody3000.json --rough-bank <Rough库/bank.json> --output-dir <新负重库> --generate-assets` 创建负重库。
3. 用 `amp_hybrid/prepare_bank.py --config amp_hybrid/config.fullbody3000.json --rough-bank <Rough库/bank.json> --output-dir <新AMP库> --generate-assets` 创建AMP库。

上面路径以experiments目录为基准。AMP复用Rough的同一URDF；负重仅新增背包链接，基础机器人原有各link/joint内容不变。负重初始URDF背包质量0.30kg是占位质量，训练开始后会被原负载阶段的采样值覆盖，并不是另加0.30kg负重。

本机实际生成并验证了三套3000条记录，位于项目根目录 `pool_wide_20261113/{rough,amp,backpack}/bank.json`。实际URDF哈希、基础部件内容、30个电机组合、100组腿长配对与条件向量审计见 `pool_wide_20261113/three_bank_audit.json`；轮换排表审计见 `schedule_audit.json`。

## 负重正式入口

`backpack_hybrid/train.py --config backpack_hybrid/config.fullbody3000.json --bank <新负重库/bank.json> --output-root <新训练目录> --device cuda:0 --execute`

该入口调用与Rough相同的持久进程引擎，保留策略、Adam、学习率与rollout分配，只重建当前组场景。原负载计划按累计更新推进：0–999 warmup，1000–2999 expansion，3000起full uniform；换组不把负载阶段重置回warmup。两个raw.reset与属性写入都在inference_mode内，PPO优化在外。

三个新配置都启用全身23关节真实曲线硬约束及动态扭矩惩罚−12。禁止叠加旧六关节惩罚。来源身份含任务、配置、候选曲线、资产与执行代码；旧模型不能直接按新结构库resume。

## 原生缓存准备

新full-body配置首次启动持久训练时，会先自动在独立进程中准备全库原生USD缓存，然后再启动PPO。原生转换器连续导出在本机出现过停滞，因此转换按4个资产一批的新进程进行，并在每个资产后释放转换对象、处理Kit更新，避免保留GUI撤销记录；每批180秒上限，最多3次，所有失败文件保留。缓存完整证明检查结构库哈希、每个URDF、转换配置及主USD哈希；不完整缓存不能直接进入训练。可通过已有 --asset-cache 显式提供匹配缓存。

这一步是资产转换，不会更新任何策略。任务暂停标记在准备期间同样生效。

AMP首次训练还会按本库实际运动学自动生成、校验新参考动作，不能沿用旧900结构库的参考清单。此次没有启动3000结构的完整AMP训练或提前宣称已生成全库动作。

## 上一范围的本机验证（历史记录）

- 三任务各3000种结构已实际生成/绑定并校验，5款膝电机、6款踝电机、30个组合完整覆盖；每个电机组合100种腿长搭配。AMP与Rough的URDF相同，负重原有基础机器人部件逐项相同，仅增加背包链接。
- 11款电机的峰值、速度上限、关节侧等效惯量、曲线CSV哈希与当前候选目录全部一致：motor_profile_audit.json。
- 两轮轮换排表分别比较，三个任务每一组的结构ID列表完全相同；各轮100组×30种，无遗漏或重复。
- 负重真实短测：120种结构（2×2腿长×5×6电机），20环境，6组各1次PPO更新。单一实际进程PID26524；策略、Adam和rollout storage对象始终相同。Adam step连续20→40→60→80→100→120；Actor参数实际变化；全部参数和优化器状态有限。
- 每组23关节完整覆盖，硬裁剪重算误差0Nm；最大腿部几何误差1.3411e-6m。原生缓存主USD哈希在仿真后再次校验，未改变。短测根目录bps4；完整证据为pool_full_20261113/native_smoke.json。
- 120/120缓存转换完成，采用4个一批的独立进程、转换对象释放与Kit更新。保留此前停滞的bps1/bps2/bps3及bpc1原始证据，不宣称原生SDK永不再卡；超时隔离和有限重试提供恢复边界。
- 211项CPU回归测试与73项子测试通过。

这次验收是完整候选池、资产一致性、代码接入和有限连续换组测试；没有启动3000结构完整长训练，也没有把6更新的短测模型当作专家策略。AMP全库参考动作将在首次正式启动时按实际运动学生成并验证。

## 当前扩展范围验证

当前正式库为pool_wide_20261113，不覆盖旧pool_full_20261113。三个任务各3000种结构的范围都是大腿0.20–0.40m、小腿0.25–0.45m，仍各采样10点，膝5款、踝6款；三任务结构、条件输入和轮换顺序再次核对一致。

配置grid.gp_extrapolation=true。新适配器只在隔离的模块副本中扩展输入域，原始GP拟合系数、已冻结源码和注册范围保持不变；授权范围和适配器哈希进入新库身份。超出训练域/原注册预测域的标记逐设计保存；1530种结构涉及原注册预测域之外的输入。原有质量为正、质心有限、惯量正定和主值三角不等式检查全部保留，不做静默裁剪修正。

110次逐电机/腿长点预测全部通过物性检查；新库9000条任务记录及实际URDF已生成并校验。

新范围原生短测位于wp1：16环境、16种端点搭配（大腿0.20/0.40m、小腿0.25/0.45m，包含两款膝和两款踝），完成2次PPO更新、40次Adam step，Actor实际更新，参数和优化器状态均有限；23关节裁剪重算误差0Nm，最大几何误差8.6427e-7m。它是边界接入验证，不是完整训练或步态质量结论。

完整证据：pool_wide_20261113/{gp_prediction_audit,three_bank_audit,schedule_audit,native_smoke}.json。214项回归测试及73项子测试通过。AMP正式启动时会为实际新运动学生成并校验参考动作；没有启动完整长训练。
