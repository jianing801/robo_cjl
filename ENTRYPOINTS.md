# 当前正式入口与历史入口

新实验先从本页选择工作流，不要根据文件名相似程度调用旧脚本。完整启动顺序和可复制命令见 [完整操作流程](docs/WORKFLOW.md)，评估见 [评估与录像](docs/EVALUATION.md)，异常处理见 [恢复说明](docs/RECOVERY.md)。

| 工作流 | 入口 | 说明 |
|---|---|---|
| 三任务结构库 | `robolab/experiments/multitask_morphology/prepare.py` | 一次准备Rough、负重、AMP匹配的完整结构库 |
| AMP参考动作 | `robolab/experiments/multitask_morphology/amp_engine/prepare_references.py` | 可独立CPU/GPU生成，不启动PPO |
| 三个跨形态专家 | `robolab/experiments/multitask_morphology/train.py --task rough/backpack/amp` | 统一正式入口；先完成结构库 |
| 三专家共享策略 | `robolab/experiments/multitask_morphology/prepare_shared.py`、`distill.py` | 完成并校验教师后，生成计划并做BC/DAgger |
| Rough候选结构BO | `rough_bo/bo_finetune_controller.py` | 独立工作流，明确传入教师、结构库与源码参数；不是完整三任务预训练 |
| 原生任务开发/单任务 | `robolab/scripts/rsl_rl/train.py`、`play.py`、`play_amp.py` | 原始原生任务工具，继续保留 |

完整三任务参数与命令见 [统一训练说明](robolab/experiments/multitask_morphology/README.md)。BO协议见 [Rough BO说明](rough_bo/README.md)。这些入口不会因为更新代码或阅读文档而自动执行。

## 已归档的旧Flat BO顶层脚本

以下5个脚本原来位于仓库根目录，现在完整保存在 [历史入口目录](legacy/flat_co_design_2026/README.md)：

- `bo_watchdog.sh`
- `resume_co_design.sh`
- `re_eval_clean.sh`
- `re_eval_cot.sh`
- `eval_all_multi.py`

它们绑定旧Flat Optuna研究、旧服务器目录和旧评估批次，不属于当前Rough BO或三任务方案。原始版本与SHA256记录一并保留，可追溯；未删除数据库、实验、模型或视频。旧cron/外部自动化若仍指向这些根目录文件，需要单独核实，不能自动转接成新训练。

## 已归档的四套旧专家与蒸馏源码

完整源码已移至 [历史专家档案](legacy/expert_sources_20261114/README.md)，原始Git字节及SHA256记录保留：`rough_hybrid`、`amp_hybrid`、`backpack_hybrid`、`rough_backpack_distill`。不要在档案目录直接执行旧脚本；需要历史复现时，使用记录的源提交建立独立checkout。

experiments中仅特意保留两处兼容依赖：

- `rough_hybrid/full_body_torque.py`：旧负重分布工具按原路径读取，原字节不变，已没有旧专家训练入口。
- `amp_hybrid/vendor/`：现有HHTools editable环境可能指向它，暂时保留，不影响正在生成的参考动作；已没有旧AMP专家训练入口。

旧Rough BO的检查点源码恢复器会按检查点SHA从档案恢复所需旧Rough文件到独立审计缓存，不改用新引擎。已有冻结代码树、所有模型、结构库、参考动作、录像和实验记录未移动或删除。

历史 `HANDOFF.md` 保留为研究过程记录，不作为当前默认配置或启动规范。原上游通用介绍已移至 `docs/reference/`，不再作为当前克隆、子模块或启动规范。
