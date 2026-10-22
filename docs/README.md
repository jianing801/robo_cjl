# 文档导航

这里仅保留可复用的现行说明。当前运行配置以代码和机器可读数据为准，历史阶段报告不作为候选状态依据。

## 按任务阅读

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

发生冲突时以运行代码和上述机器可读文件为准，并同步修正文档；不要用带日期的历史总结覆盖当前配置。
