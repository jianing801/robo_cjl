# Rough 与 AMP 专家参考评估代码

这两个入口复现[已归档的 50 结构专家评估](../../../../docs/Rough与AMP专家评估及评分记录.md)。它们复用仓库的原生策略、BO 指标采样和 GPU 租约，按冻结的结构顺序记录原始轨迹和试次结果。

默认读取仓库中的 `robolab/logs/multitask_full2500_physical_migrated`、结构库、原生 USD 缓存及已完成的专家。部署位置不同，可将环境变量 `ROBO_REFERENCE_PROJECT` 指向该次实验根目录。AMP 还读取自己的参考库和导入记录。Rough 校验租卡原始或已核验的服务器重绑检查点，以及对应的结构库和缓存。选样来自资料目录中的 AMP 清单。

从仓库根目录执行。将 GPU 号改为当前可见列表中的空闲 GPU 序号。先用预检查核对数据和检查点身份。该步骤使用临时目录，无需占用 GPU 或改动既有评估结果。

```bash
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/reference_evaluation/evaluate_rough.py evaluate --root ./rough_reference_eval --gpu 0 --check-only
conda run --no-capture-output -n robo_cjl python robolab/experiments/multitask_morphology/reference_evaluation/evaluate_amp.py evaluate --root ./amp_reference_eval --gpu 0 --check-only
```

去掉 `--check-only` 开始评估。请用新的输出目录执行，保留已经归档的基准结果。历史执行脚本随原始数据包保存。
