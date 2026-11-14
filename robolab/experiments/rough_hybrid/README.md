# 历史力矩兼容模块，不是专家训练入口

这里仅保留旧负重分布工具依赖的 `full_body_torque.py`，原字节不变。

完整旧Rough源码及SHA记录已归档到 [历史专家档案](../../../legacy/expert_sources_20261114/README.md)。历史Rough BO通过检查点SHA恢复独立审计源码树，不调用新的专家引擎冒充旧源码。

新专家训练、结构库准备及共享策略统一使用 [multitask_morphology](../multitask_morphology/README.md)。不要从这里启动旧训练。
