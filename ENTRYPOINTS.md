# 当前正式入口与历史入口

新实验先从本页选择工作流，不要根据文件名相似程度调用旧脚本。

| 工作流 | 入口 | 说明 |
|---|---|---|
| 三任务结构库 | `robolab/experiments/multitask_morphology/prepare.py` | 一次准备Rough、负重、AMP匹配的完整结构库 |
| AMP参考动作 | `robolab/experiments/multitask_morphology/amp_engine/prepare_references.py` | 可独立CPU/GPU生成，不启动PPO |
| 三个跨形态专家 | `robolab/experiments/multitask_morphology/train.py --task rough/backpack/amp` | 统一正式入口；先完成结构库 |
| 三专家共享策略 | `robolab/experiments/multitask_morphology/prepare_shared.py`、`distill.py` | 完成并校验教师后，生成计划并做BC/DAgger |
| Rough候选结构BO | `rough_bo/bo_finetune_controller.py` | 独立工作流，明确传入教师、结构库与源码参数；不是完整三任务预训练 |
| 原生任务开发/单任务 | `robolab/scripts/rsl_rl/train.py`、`play.py`、`play_amp.py` | 原始原生任务工具，继续保留 |

完整三任务参数与命令见 [统一训练说明](robolab/experiments/multitask_morphology/README.md)。BO协议见 [Rough BO说明](rough_bo/README.md)。本次只精简入口，不启动训练、参考动作生成、评估或BO。

## 已归档的旧Flat BO顶层脚本

以下5个脚本原来位于仓库根目录，现在完整保存在 [历史入口目录](legacy/flat_co_design_2026/README.md)：

- `bo_watchdog.sh`
- `resume_co_design.sh`
- `re_eval_clean.sh`
- `re_eval_cot.sh`
- `eval_all_multi.py`

它们绑定旧Flat Optuna研究、旧服务器目录和旧评估批次，不属于当前Rough BO或三任务方案。原始版本与SHA256记录一并保留，可追溯；未删除数据库、实验、模型或视频。旧cron/外部自动化若仍指向这些根目录文件，需要单独核实，不能自动转接成新训练。

## 仍保留的历史兼容目录

`rough_hybrid`、`amp_hybrid`、`backpack_hybrid`、`rough_backpack_distill` 暂不搬动、不删除、不重定向。它们包含历史训练与蒸馏代码、源码身份和测试依赖。新实验使用 `multitask_morphology`，历史模型使用它训练时对应的冻结版本，不能仅因入口相似就换成新引擎。

历史 `HANDOFF.md` 保留为研究过程记录，不作为当前默认配置或启动规范。根目录中英文README下方仍含上游通用介绍，应以本入口索引和对应工作流文档为准。
