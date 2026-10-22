# 原始 BO 电机数据源

下列原始 BO 候选均保存当前 SolidWorks 扫描、训练数据、拟合脚本和验证输出。运行时只读取 `../urdf_gp_models/` 与 `../torque_speed_curves/`；本目录用于追溯和复现，不作为运行时入口。

| 电机 | 装配体 SW 数据 | GP 源包 | 当前预测模型 |
|---|---|---|---|
| DM-J10010L-2EC | `DM-J10010L-2EC/SW_GP/data/`，15 点大腿扫描 | `DM-J10010L-2EC/SW_GP/` | `../urdf_gp_models/DM10010L_thigh.json` |
| RS04 | `RS04/SW_GP/data/`，15 点大腿扫描 | `RS04/SW_GP/` | `../urdf_gp_models/RS04_thigh.json` |
| RS06 | `RS06/SW_GP/data/`，15 点小腿扫描 | `RS06/SW_GP/` | `../urdf_gp_models/RS06_calf.json` |
| DM-J4340P-2EC | `DM-J4340P-2EC/SW_GP/data/`，15 点小腿扫描；同目录另存扭矩—转速曲线建模资料 | `DM-J4340P-2EC/` | `../urdf_gp_models/DM4340P_calf.json` |

结构 GP 的运行时 JSON 已与各自当前拟合输出逐字节核对一致。四条当前扭矩—转速曲线集中在 `../torque_speed_curves/`；RS04、RS06、DM-J10010L 与现有最新输出一致，DM-J4340P 已更新为 464 点版本。
