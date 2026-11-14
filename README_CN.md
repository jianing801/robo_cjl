# robo_cjl：结构—执行器—控制协同优化

本仓库包含RPO双足机器人跨形态专家训练、三任务共享策略蒸馏、SW-GP/URDF资产生成和独立Rough BO工作流。当前三任务是Rough楼梯、负重行走和AMP跑步。

**从 [完整操作流程](docs/WORKFLOW.md) 开始。** 已生成结构库和AMP参考动作时，直接进入该文档第4节，不必重做准备。

## 当前正式配置

| 项目 | 设置 |
|---|---|
| 结构库 | 10×10腿长搭配 × 5款膝电机 × 6款踝电机＝3000种 |
| 腿长 | 大腿0.20–0.40m，小腿0.25–0.45m，允许已授权GP外推 |
| 每组 / 每轮 | 60种 / 50组；每组50次PPO更新 |
| 环境 | 4096个仿真副本，共同更新当前任务策略 |
| 预算 | Rough和负重各12000更新；AMP 5000更新 |
| 执行器 | 全身23关节对应曲线硬约束＋动态扭矩利用率软惩罚，统一权重−12 |
| 共享策略 | 三个冻结教师的BC＋DAgger共享Actor，另做真实配对评估 |

这里的“代”指一次PPO更新，不是回合。通过名义步态评估不代表所有扰动和新结构都已验证。

## 文档导航

| 你要做什么 | 文档 |
|---|---|
| 环境、原始机器人资产、两个Python环境 | [环境与资产](docs/SETUP.md) |
| 准备结构库、参考动作，启动三专家和蒸馏 | [完整操作流程](docs/WORKFLOW.md) |
| 教师/学生三任务评估、单机器人画面与录像索引 | [评估与录像](docs/EVALUATION.md) |
| Ctrl-C恢复、缺文件、参考中断、Git/API更新 | [恢复与排错](docs/RECOVERY.md) |
| 输出文件、模型身份、数据备份和资源估算 | [实验文件与交付](docs/ARTIFACTS.md) |
| BO、共享PPO和新结构的当前能力边界 | [扩展说明](docs/EXTENSIONS.md) |
| 全部现行入口与旧代码去向 | [入口索引](ENTRYPOINTS.md) |
| 分组构造、模拟退火、覆盖对比 | [正式分组依据](robolab/experiments/multitask_morphology/scheduling/README.md) |
| SW、电机映射、GP与CAD交接 | [文档总目录](docs/README.md) |

## 代码结构

```text
robolab/experiments/multitask_morphology/  正式三任务训练和共享策略
rough_bo/                                独立的历史Rough教师BO协议
robolab/robolab/                          原生任务、物理、执行器与资产工具
robolab/scripts/                         原生工具及API在线更新器
rsl_rl/                                  随仓库提供的算法源码
vendor/physical_1450a51/                  带哈希校验的物理依赖快照
legacy/                                  旧入口与旧专家源码档案
```

`rough_hybrid`只保留旧力矩兼容模块；`amp_hybrid/vendor`暂保留既有HHTools editable环境的依赖位置。不能为了目录整洁破坏已经生成的参考动作或历史模型身份。详见入口索引。

## 获取代码

```bash
git clone --branch feature/rough-hybrid https://github.com/jianing801/robo_cjl.git
cd robo_cjl
```

分支名沿用历史名称，当前内容覆盖三个任务。robolab与rsl_rl已经作为源码提供，不是原始上游说明中的两个待初始化子模块；原始机器人URDF和显示网格需另按已验证版本准备。

已部署服务器沿用现有环境，不升级Isaac、PyTorch或URDF导入器。Git连接故障、本地补丁和已有提交的合并按恢复文档处理，不使用强制覆盖。源码更新不会自动启动训练。

[English](README.md) · [历史上游工作区说明](docs/reference/upstream_workspace_README_CN.md)
