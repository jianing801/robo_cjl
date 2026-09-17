# 基于SW拟合参数的URDF更新

## 本次交付与来源

依据仓库 guagua1413028593-sketch/robo_cjl 的提交 `0a99617706f375a25efc88735256bb8bd7031f81`，新增一个运行文件：

`robolab/scripts/tools/generate_urdf_sw.py`

生成器初版采用新增文件方式。当前已将训练、评估和BO入口接入SW模型，原generate_urdf.py及URDF源模板保留。四套拟合系数保存在`robolab/robolab/assets/motor_data/motor_urdf_models.json`，运行不需要CSV或scikit-learn。膝电机型号和踝电机型号作为两个离散设计变量接入外层BO；型号变化会同时修改URDF质量、质心、刚体惯量，以及Isaac Lab执行器的峰值扭矩、最高转速、关节侧`armature`和可用的转矩—转速动态包络。外层BO采用速度跟踪代价与机械CoT双目标优化。

原代码参考：https://github.com/guagua1413028593-sketch/robo_cjl/blob/0a99617706f375a25efc88735256bb8bd7031f81/robolab/scripts/tools/generate_urdf.py

## 已确认的最终模型

| 型号与部位 | 质量 | 质心及六个惯量分量 | 训练长度m | 允许预测m | 归一化输入 |
|---|---|---|---|---|---|
| RS04大腿 | 一次 | 四次 | 0.26～0.40，15点 | 0.25～0.40 | (l−0.325)/0.075 |
| DM-J10010L-2EC大腿 | 一次 | 四次 | 0.26～0.40，15点 | 0.25～0.40 | (l−0.325)/0.075 |
| RS06小腿 | 一次 | 四次 | 0.31～0.45，15点 | 0.30～0.45 | (l−0.375)/0.075 |
| DM-J4340P-2EC小腿 | 一次 | 四次 | 0.31～0.45，15点 | 0.30～0.45 | (l−0.375)/0.075 |

四套模型在各自训练点上拟合；0.25m大腿与0.30m小腿属于允许的一小段端点外推，生成报告会保留`extrapolated`状态。每个型号使用自己的结构模型，通过显式坐标变换/镜像映射至左右link；若左右部件配置不同，不能用镜像替代另一侧CAD参数。

系数取自本次已验证的拟合结果。系数已经恢复为物理输出：质量kg、质心m、惯量kg·m²。按升幂存储，运行时用Horner法求值，不再进行训练输出标准化。JSON记录完整系数、归一化中心及半宽。

## 具体更新哪里

| 内容 | 原方法 | 新文件 |
|---|---|---|
| 膝关节origin.z | −大腿长度 | 保留该方法，左右各一个 |
| 踝pitch关节origin.z | −小腿长度 | 保留该方法，左右各一个 |
| 质量 | 原质量乘长度比例 | 一次拟合结果直接覆盖 |
| 质心 | 只缩放原质心z | 三个坐标均由拟合预测后变换至link |
| 惯量 | 根据旧惯量做z向尺度变换 | 六个质心惯量分量分别由拟合得到 |
| 惯性坐标rpy | 保留旧值 | 张量已变换至link轴，写为0 0 0 |
| 碰撞盒 | origin.z和size.z按长度比例缩放 | 保留；非盒或旋转盒显式报错 |
| 网格路径 | 固定目录软链接 | 根据模板位置或--mesh-dir解析实际路径 |

更新的link为left/right_thigh_pitch_link与left/right_knee_link。其他link和关节参数保留（mesh文件路径统一改为绝对路径）。保留原关节origin的x、y和rpy、轴、限位等。视觉STL不随长度重新建模，与旧生成器的行为一致；新碰撞盒仍是比例缩放近似，并非重新从参数化SW导出的精确几何。

新脚本不会用几何缩放比例再次乘以拟合质量或惯量。输出固定为新目录的urdf/rpo.urdf，附rpo.report.json记录实际写入值、坐标映射和外推状态。拒绝覆盖输入模板或既有输出，避免误改源模型。每次使用新的output目录。

## 必须明确的坐标关系

DM-J10010L-2EC、RS06和DM-J4340P-2EC报告坐标与左侧对应link对齐。RS04报告来自另一装配体坐标：先用固定旋转`Q=[[0,0,-1],[0,1,0],[1,0,0]]`，再使用`t=[0.51755062,-0.08250093,0.52306493-l] m`转到左大腿link坐标。该变换用相同大腿结构的质心轨迹交叉核对，三个轴的最大残差为0.347 mm。右侧继续采用左右结构对称假设，沿y方向镜像。

支持两种方式：

1. 默认方式：无需额外参数，相当于 `--aligned-side left`，左侧直接写入，右侧关于y=0镜像。
2. `--frame-map frames.json`：以后若坐标定义改变，可按四个link覆盖默认关系。

数学定义：c_link=Q*c_SW+t，I_link=Q*I_SW*Q^T。Q须为正交矩阵。正常坐标旋转det(Q)=+1；若明确建模左右镜像，可使用det(Q)=−1。t是SW原点在目标link坐标下的位置（镜像映射时是对应仿射偏移），单位m。不要将SW报告的质心再误当作t。

SW输出Lxy、Lxz、Lyz采用正惯性积约定。构造标准惯性张量时非对角项取负，然后再进行Q变换。张量以质心为参考，不再加质量×质心偏移的平行轴项。质心写入inertial/origin。

默认镜像矩阵为S=diag(1,−1,1)。右腿c=S*c_left，I=S*I_left*S^T：质心y、标准矩阵ixy和iyz变号，ixz不变。

## 单独生成URDF

在项目根目录运行（替换资产路径）：

```bash
python robolab/scripts/tools/generate_urdf_sw.py \
  --thigh 0.30 --calf 0.35 \
  --knee-motor RS04 --ankle-motor RS06 \
  --template /你的路径/rpo.urdf \
  --mesh-dir /你的路径/meshes \
  --output /tmp/rpo_sw_030_035
```

当前已确认对齐，因此无需提供 `--frame-map`。如果以后使用右腿单独导出的SW数据，可显式使用 `--aligned-side right`；若轴向或原点不再一致，应提供完整 `--frame-map`。

若本地原项目默认rpo.urdf及相对STL路径齐全，可省略--template和--mesh-dir。当前公开仓库克隆没有包含默认资产目录，所以本次测试用了用户已上传的roboto_origin(1).urdf及03_URDF中的meshes。生成文件引用本机绝对STL路径；移动到另一台机器时应在那里重新生成。

## 不改原文件启动训练或评估

新文件可以在当前进程替换旧generate_urdf模块入口，然后运行原脚本：

```bash
python robolab/scripts/tools/generate_urdf_sw.py \
  --template /你的路径/rpo.urdf \
  --mesh-dir /你的路径/meshes \
  --run co_design_train -- \
  --thigh 0.30 --calf 0.35 --headless
```

`--`后参数原样传给原训练脚本，可追加原有训练选项。评估时改成`--run co_design_eval`并追加原评估脚本所需参数。应在原Isaac Lab环境及其原有Python启动方式下运行。

当前直接运行co_design_train.py或co_design_eval.py默认使用SW生成器，无需包装入口。上述包装方式仍支持自定义模板、网格与坐标映射。评估旧模型训练的检查点时，可显式传入`--urdf-model legacy`；训练和评估应选择相同模型。

外层co_design.py显式向训练和评估传入`--urdf-model sw`，搜索范围为大腿0.25～0.40m、小腿0.30～0.45m，并离散选择一类膝电机和一类踝电机。左右膝使用同一膝电机；左右腿的踝俯仰和踝横滚共四个关节使用同一踝电机。默认膝候选为`RS04,DM-J10010L-2EC`，默认踝候选为`RS06,DM-J4340P-2EC`。DM-J8006-2EC因为没有经过验证的SW结构模型和关节侧惯量，保留在通用电机目录中，但不进入优化候选。默认数据库为`co_design_track_cot_motor_urdf_v1.db`、study名称为`rpo_flat_track_cot_motor_urdf_v1`，与旧的“只改执行器、不改URDF”的试验隔离；已有study还会核对候选列表、URDF模型与生成器文件哈希、型号映射和曲线哈希，避免混合不同模型。

当前电机变量同时改变URDF结构参数和仿真执行器：

- `effort_limit`与`effort_limit_sim`使用候选电机峰值输出扭矩；
- `velocity_limit`与`velocity_limit_sim`使用候选电机输出端最高转速；
- 有CSV包络的型号在每个仿真步按关节绝对转速插值，并动态裁剪正、反向输出扭矩；
- `armature`随型号变化：RS04为0.04000、DM-J10010L-2EC为0.05556、RS06为0.01200、DM-J4340P-2EC为0.03200 kg·m²；灵足使用厂家低速端等效惯量，达妙使用上位机电机转动惯量乘减速比平方；
- 所选型号与腿长共同选择对应的SW多项式，覆盖大腿或小腿link的质量、三维质心和六个质心惯量分量；
- 暂无可靠惯量和SW模型的DM-J8006-2EC以及兼容保留的DM-J4340-2EC不进入优化；
- PD刚度、阻尼和延迟保持原值；
- 关节位置与碰撞盒仍按腿长更新，视觉网格仍未针对每个电机和长度重新导出。

PPO训练奖励保持不变。结构评价不再使用episode return作为优化目标，而是同时最小化：

```text
tracking_cost = mean(||v_xy_cmd - v_xy||^2 + (wz_cmd - wz)^2)
mechanical_CoT = sum(|tau * joint_velocity| * dt) / (mass * 9.81 * traveled_distance)
```

线速度在机器人yaw坐标系下比较，偏航角速度在世界坐标系下比较，与训练跟踪项的坐标定义一致。评价结果同时输出线速度和偏航角速度的MSE/RMSE。机械距离按逐步水平路径长度累计；仅允许含非零平移的BO评价指令。提前终止、距离不足或非有限指标作为不可行设计，赋予被正常解支配的惩罚值。训练reward和evaluation return仍记录为诊断数据，不参与外层优化。

Optuna study使用两个`minimize`方向，并要求Optuna>=4.4，由`GPSampler`通过多目标logEHVI采集函数选择新腿长；依赖或版本不满足时明确报错，不在同一实验中替换优化算法。优化完成后终端列出可行帕累托前沿，并在数据库同目录写出`*_pareto.csv`。默认固定测试指令为`0.5,0,0`，即沿机器人前向以0.5m/s运动；CoT按总机械能与总路径距离计算。可通过`--eval-commands`显式增加其他平移工况。

参考Du等人2025年论文表1，训练奖励中的平面线速度跟踪权重由1.0提高到2.0，偏航角速度跟踪权重由1.0提高到1.5，指数误差尺度`std=0.5`保持不变。这两个权重同时作用于使用`RPORewardCfg`的Flat及其派生任务；外层评价仍使用原始跟踪误差和机械CoT，不使用加权训练回报。

在原项目Isaac Lab环境、机器人资产齐全的前提下，从仓库根目录运行：

```bash
python robolab/scripts/tools/co_design_train.py --thigh 0.30 --calf 0.35 --knee-motor RS04 --ankle-motor RS06 --headless
python robolab/scripts/tools/co_design_eval.py --checkpoint /路径/model.pt --thigh 0.30 --calf 0.35 --knee-motor RS04 --ankle-motor RS06 --headless
python robolab/scripts/tools/co_design.py --trials 30 --max-iterations 12000 --num-envs 4096
```

本次接入仅作离线检查，没有在当前环境执行Isaac训练；以下初版生成器验证记录不代表新策略性能。

也可以在新的调用程序中导入generate_urdf_sw.generate_urdf，保留旧函数前四个参数即可使用默认左腿映射；需要覆盖时再传frame_map或aligned_side。

## 验证结果与边界

已验证：

- 四套型号模型覆盖各自允许预测范围，质量为正，惯量正定且满足主惯量三角不等式。
- 四种膝/踝型号组合均可生成完整URDF。
- 三维质心、SW非对角符号、旋转、镜像及平移的写入正确。
- 关节z与碰撞盒缩放正确，非目标link/关节参数保持，源模板哈希不变。
- 输入范围和已有输出保护生效。
- 嵌入系数在原验证点上的预测与已记录多项式预测一致（绝对差<1e-14）。
- --run入口通过模拟训练脚本检验，正确转发参数并注入新生成器。

未在当前环境执行Isaac Sim导入、PPO训练或真实运动验证。602点物理一致性检查不能替代CAD预测精度或动力学验证，也不能证明外推精度。

## 符号判定与新旧策略比较

非对角项取负的依据是用户SW报告中的数据关系，而非软件版本名称。小腿0.31m处，SW质心Lxy=0.00002173，m*cx*cy=−0.00000616331，相加得到0.00001556669，与SW报告的原点Ixy=0.00001557一致。因此报告输出的是正惯性积；标准矩阵非对角项应取负。其余两项也通过此检查。若读取的是导出器已经转换过的URDF，则不能再次取负。

新旧模型两个交叉项异号不能证明旧文件错误，也不能排除坐标差异。例如绕x轴旋转180°可使xy、xz变号而yz不变。不能通过匹配旧符号决定新数据的约定。

旧策略在旧URDF正常，不保证在新URDF保持同样表现。新大腿0.25m处主要惯量相较旧模型约降低24%，影响可能比两个小交叉项变号更明显。首次验证宜固定相同腿长、策略、指令和初始状态，只比较惯性参数更新造成的影响；尚无新URDF上的实测训练/评估结论。

视觉网格不随腿长改变，碰撞盒与关节位置随长度更新。动力学采用inertial，接触采用collision；视觉模型可影响渲染和视觉传感器输入。Isaac导入后应实际核对质量/惯量被采用、碰撞几何正确，不能仅凭源配置未出现覆盖选项推断所有导入默认值。
