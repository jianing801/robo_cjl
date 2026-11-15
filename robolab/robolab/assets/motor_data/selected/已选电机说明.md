# 加入选型

当前实际电机 ID、关节分配和 GP 映射以 `bo_ready_candidates.csv` 为准；`电机清单.csv` 是整理前的状态快照，里面的 `SW_PENDING` 不代表当前状态。达妙原厂选型表副本位于 `DM_selection_2026_09_23.csv`，新增参数化电气模型位于 `DM_parameter_models_2026_10_09.csv`。

原有 RS04、DM-J10010L、RS06、旧版 DM-J4340P 保持独立；新增 DM-J10010（非 L）、J6248P、J10422P、J8009P V1.1、J4340P V1.1、J8006 V1.1、J4310P-48V、J6216P 已分别绑定各自关节段的 SW-GP 数据和电机曲线，进入对应 BO 候选池。特别是 DM-J10010 与 DM-J10010L 的模型、几何数据和电机包络互不复用。

SW-GP 的运行模型放在 `../urdf_gp_models/`，统一索引为 `../motor_urdf_gp_registry.json`；DM-J10010 的原始质量属性 CSV、MATLAB 拟合脚本、保存的 GPR 模型、预测表、留一交叉验证指标、物理检查及图均整理在 `../urdf_gp_sources/DM-J10010-2EC/`。原始资料库未移动或删除。

2026-10-17：J6216P 仍按用户约定使用 260 rpm 作为 48 V 端点及 45 N·m 峰值，手册 V1.3 额定相电流采用 22 A。参数定义和非候选资料分组详见 `../../../../../docs/电机选型分组与SW待补清单.md`。
