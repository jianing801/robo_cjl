# 电机数据目录

这里保留运行所需数据、当前选型索引，以及能复现模型的源资料。当前状态以
`selected/bo_ready_candidates.csv` 和仓库文档
`docs/电机BO候选池与GP映射_2026-10-17.md` 为准。

新型号 SolidWorks 结构数据采集与交接（坐标对齐、质量属性覆盖、扫描宏和 GP 输入）流程见[仓库文档](../../../../docs/SW数据获取与新电机交接完整流程.md)。

## 根目录运行文件

| 文件 | 用途 |
|---|---|
| `motor_urdf_gp_registry.json` | 唯一的 URDF 结构 GP 注册表：把电机 ID 映射到大腿/小腿模型和预测范围。 |
| `torque_speed_curves/` | 当前 12 款 BO 候选电机的一机一 CSV 运行曲线目录。执行器仍按候选型号逐份读取，不同电机的曲线不能互相替代。 |
| `DM_selection_2026_09_23.csv` | 厂家选型参数原始整理表，保留作来源核对，不是执行器曲线或 GP 模型。 |

每份 CSV 只保存一个电机的曲线。曲线集中放在 `torque_speed_curves/`，方便查找和按型号单独替换。

## 子目录

- `urdf_gp_models/`：训练/评估/BO 直接读取的 12 个 GP 模型 JSON。
- `urdf_gp_sources/`：原始 SW 扫描、可复现拟合脚本、MATLAB模型、图表和误差/物理检查结果；不是运行时查找目录。原始 BO 四款电机（RS04、DM-J10010L、RS06、DM-J4340P）的当前 SW 扫描与 GP 拟合资料均已纳入。
- `selected/`：加入选型的参数资料和精简 BO 映射表。
- `not_selected/`：未加入选型的候选及其参数/来源归档；不参与运行。
- `torque_speed_curves/`：所有当前 BO 候选电机各自独立的扭矩—转速 CSV。

旧 URDF 多项式模型索引和多项式拟合脚本已移除；URDF 结构预测统一使用 GP，不存在多项式回退。达妙 J4340P 当前 65 点峰值包络、464 点约束 GP 曲线及其 MATLAB 输入/输出均归入
`urdf_gp_sources/DM-J4340P-2EC/`，运行时单电机曲线位于 `torque_speed_curves/DM_J4340P_gp_envelope.csv`。
