# 三任务全身动态扭矩约束接入与验证

日期：2026-11-13。当前统一权重为 **−12**，阈值 **80%**。这是功能接入和有限短测结果，不是完整训练收敛或权重最优的结论。

## 实际计算

每个控制步：`r = -12 * dt * sum_j(max(abs(tau_applied_j) / tau_available_j - 0.8, 0)^2)`。

23 个关节求和，不取平均。`tau_available` 使用执行器当次物理子步硬裁剪所用的转速曲线上限；奖励读取最后子步实际输出扭矩。原生 RewardManager 乘一次控制步长，函数内部不再乘。保留原有能耗、torque L2 等不同含义的奖励项，拒绝同时启用旧 design_motor_torque_limits 项。

可用扭矩不大于峰值的 1e-6 时，按之前确定的处理方式排除该比例项并记录 unavailable，不除零。这仍是零可用扭矩区域的已知盲点，并未宣称解决。

## 电机与覆盖

| 执行器组 | 关节数 | 曲线来源 |
|---|---:|---|
| legs：髋/大腿与躯干 | 7 | 候选池 DM-J10010L-2EC |
| shoulders：肩/上臂 | 6 | 候选池原版 DM-J4340P-2EC |
| arms：肘/前臂 | 4 | 候选池原版 DM-J4340P-2EC |
| knees | 2 | 当前设计的膝电机 |
| feet：踝 pitch/roll | 4 | 当前设计的踝电机 |

固定组使用候选池自己的曲线、峰值、转速上限和关节侧等效惯量，保留已有 PD 与延迟配置。不把原版4340P与40Nm的V1.1型号混为一谈。候选目录、曲线CSV及全身约束代码加入检查点来源身份校验。

## 接入位置

- Rough 时空混合：hybrid_runtime.configure_spatial；配置 full_body_torque=true。
- AMP 时空混合：amp_mixed_runtime.configure_spatial；同一开关、同一奖励函数。AMP原有任务/风格奖励混合方式保持原样，因此这里的−12指环境原生奖励项权重。
- 负重分布训练入口：run_backpack_distribution_pilot.py 与 train_backpack_distribution_stage.py 新增 --full-body-torque；保存开关及曲线身份，拒绝把旧检查点按新物理约束直接续训。此单结构入口保留；后续已新增 backpack_hybrid/train.py，使用相同的持久换组引擎。
- 三任务跨形态采样/共享PPO：rollout.py、rollout_amp.py、joint_finetune.py 同时接入 --full-body-torque。用于本次真实物理与优化短测。

Rough/AMP另提供 config.fullbody3000.json：腿长10×10、膝5款、踝6款（排除4310P）、共3000结构，4096环境，两个配置的结构网格完全相同。该配置要求新建对应结构库；后续已经生成并逐一校验了三任务3000结构资产，并补齐负重跨形态入口；见 backpack_hybrid/README.md 和 pool_full_20261113/three_bank_audit.json。完整长训练尚未启动。

## 实际短测

最终证据目录：C:\Users\Admin\.codex\.chatgpt-projects\g-p-6a967dbd08f4819188bc02c144f1a591\fbsmoke3。

每任务8环境、每轮24步、共2轮，合计每任务384个环境步，三个任务1152个环境步。单个结构为0.25m/0.30m、膝10010L、踝RS06，读取既有共享策略初始化。不是3000结构覆盖测试，也不是三个专家完整训练。

| 任务 | 关节覆盖 | 动态上限超出 | 平均惩罚/环境步 | 超80%关节样本比例 |
|---|---:|---:|---:|---:|
| Rough | 23/23 | 0 Nm | −0.00004519 | 0.1698% |
| 负重 | 23/23 | 0 Nm | −0.00001167 | 0.0793% |
| AMP | 23/23 | 0 Nm | −0.00170891 | 0.9058% |

全部奖励有限；每个任务只出现一次 full_body_torque_limits，旧六关节项未启用。逐步核对原生RewardManager的实际贡献，验证−12与dt=0.02只乘一次。此短测没有零可用扭矩样本。

共享PPO完成2次同步更新；24次Adam step；Actor与critic参数均实际变化且有限。这里只证明新约束能参与真实仿真和优化，不能说明−12已是最终最优权重。

首次短测发现固定组未导出实时限值，运行被明确拒绝，已修复全部五组导出；失败输出保留在fbsmoke1。修复后fbsmoke2与带逐步审计的fbsmoke3均成功。

机器可读证据：fbsmoke3/verification.json、各任务_000.pt与_001.pt、ppo_002.pt、completed.json、worker_*/full_body_configuration.json。既有专家模型及冻结源码没有修改。

## 回归检查

154 项测试全部通过（使用实际共享检查点运行原先可选的梯度测试，无跳过）；26 个集成/入口文件语法检查通过；git diff --check通过。测试包括Rough/AMP调度、资源清理、恢复、健康日志、动态惩罚及负重配置回归。
