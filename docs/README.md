# 文档导航

这里仅保留可复用的现行说明。当前运行配置以代码和机器可读数据为准，历史阶段报告不作为候选状态依据。

## 当前训练操作主线

| 阶段 | 文档 |
|---|---|
| 从环境到最终共享策略的完整顺序 | [完整操作流程](WORKFLOW.md) |
| 已验证环境、原始URDF/网格、HHTools | [环境与资产](SETUP.md) |
| 真实教师/学生评估、录像和索引 | [评估与录像](EVALUATION.md) |
| 暂停续训、部分生成、Git与API更新 | [恢复与排错](RECOVERY.md) |
| 实验目录、身份、磁盘与交付 | [数据与交付](ARTIFACTS.md) |
| BO、新结构、共享PPO与遗忘评估边界 | [扩展说明](EXTENSIONS.md) |
| 当前所有入口、旧代码去向 | [入口索引](../ENTRYPOINTS.md) |

下面的CAD、GP和电机资料用于方法与资产交接，具体任务选择以正式配置和实际bank为准。目录中可选电机不等于该任务全部启用的电机。

## 按资产任务阅读

| 你要做什么 | 文档 |
|---|---|
| 接入新电机、对齐 CAD、覆盖质量属性、扫描并交接 SW 数据 | [新电机 SolidWorks 数据获取流程](SW数据获取与新电机交接完整流程.md) |
| 查看当前 BO 电机名单、每款电机对应的扭矩曲线和 SW-GP | [电机候选池与模型映射](电机候选池与模型映射.md) |
| 理解 SW-GP 如何写入 URDF、如何生成并检查结果 | [基于 SW-GP 的 URDF 更新](基于SW拟合参数的URDF更新.md) |
| 理解机器人强化学习训练流程 | [训练流程详解](训练流程详解.md) |

## 可直接复用的样例

- [DM-J6216P 小腿样例](examples/DM-J6216P_calf_SW/README_CN.md)：输出端面中心与 DM4340 对齐，含对齐 STEP 和扫描宏。
- [RS04 大腿样例](examples/RS04_thigh_B10010L/README_CN.md)：按 DM-J10010L 的 B 方案对齐，含对齐 STEP 和扫描宏。
- [SW 质量属性 CSV 模板](templates/motor_sw_mass_properties.csv)。

## 数据真值来源

- 当前电机名单及执行器参数：robolab/robolab/assets/rpo_motor_catalog.py。
- BO 可读映射：robolab/robolab/assets/motor_data/selected/bo_ready_candidates.csv。
- SW-GP 注册与预测范围：robolab/robolab/assets/motor_data/motor_urdf_gp_registry.json。
- 运行中的质量属性模型、逐电机扭矩—转速曲线及拟合源数据：robolab/robolab/assets/motor_data/。

三任务正式选择和范围见 `robolab/experiments/multitask_morphology/configs/`，分组见同目录 `scheduling/formal_schedule.json`。`vendor/physical_1450a51` 是带校验的物理依赖来源之一；某次运行最终实际用了哪些生成器、曲线、URDF及适配器，以bank中的物理记录和plan/checkpoint的源码身份为准，不能只看目录名称或候选CSV。

发生冲突时以运行代码和上述机器可读文件为准，并同步修正文档；不要用带日期的历史总结覆盖当前配置。
