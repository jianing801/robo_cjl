# 正式同进程Rough与旧检查点升级验收

2026-10-24。只涉及独立Rough目录；原空间Rough、AMP、背包冻结代码和所有旧结果不改。背包保持暂停。

## 正式实现

- train.py新建实验默认execution_protocol=persistent_v1，启动一个持有整轮GPU租约的控制器及一个持续worker。不是每组重启解释器或Kit。
- 同一worker内只重建真实场景，复用policy、normalization、Adam、学习率、rollout存储；均衡采样分组、镜像环境引用切换为新组；旧轨迹索引清零。
- 原生环境初始化会重设Torch种子，所以独立保存学习RNG，并在新场景完成后恢复CPU/CUDA学习RNG。物理环境重新初始化，不声称轨迹连续。
- 清理旧RayCaster.meshes和USD stage；新组仍完整检查23关节顺序、默认姿态、实际腿长及上下文关联。
- 每组完整结束才写commits；中断后只从最后提交组恢复，最多重做一组，目标仍为绝对12000。首组没有完整检查点则拒绝隐式从头重跑。
- worker异常、返回0但有failure、缺少summary、提交不足目标均失败；原始退出码与有效状态分开记录。Ctrl+C先停调度并终止同链worker，写USER_PAUSED，不自动复活。
- 原生转换哈希缓存可复用；缓存不是另一个设计的替代模型。没有每组PPO预检。

## 已完成验证

1. 42项CPU回归通过（Rough专用测试及共享GPU/launcher/standalone回归）。包括连续组对象复用、真实Torch优化梯度/Adam更新、失败不提交、从最后完整组续接、最后不足一组的预算、学习RNG跨环境重设恢复、旧源不匹配拒绝、bootstrap篡改拒绝。
2. 4096环境、900全库、30结构/组、两组各2次更新，共4次真实PPO，使用正式train.py而不是probe入口。
   - 根：C:/Users/Admin/.codex/.chatgpt-projects/g-p-6a967dbd08f4819188bc02c144f1a591/rhf26/cached
   - 两组PID均29492，policy/optimizer/storage三个对象ID全程相同。
   - 最终model_1.pt、model_3.pt完整加载，内部iter=1、3；Adam step=40、80；Actor/Critic在换组后继续改变。
   - 两份geometry_audit、manifest、summary与commits、最终completed_updates=4及退出0一致。会话wall_seconds=395.046。
   - 本机12GB显存几乎满载；不能用这一速度外推服务器24GB表现。PyTorch缓存不等于独占GPU资源。
   - 此4096验证先于最后的显式学习RNG保护/同bank进程扫描补丁；这些补丁由CPU回归及后续小规模正式入口复测覆盖，不冒称完全相同版本已重跑4096。
3. 旧版升级CPU回归：模拟已审核b313646完整组，从2更新升级后继续到绝对5更新，Adam最终step=5。原旧目录逐文件哈希完全不变，模型/归一化张量、优化器、LR、迭代与RNG逐项相等，只重新绑定身份元数据。
4. 最终RNG/进程保护补丁后，正式train.py再次完成8环境、3组各2更新的真实回归：rhf26/final8，PID三组均13288，对象身份相同，检查点iter=1、3、5，总计6更新。再次以resume/execute调用，只返回TRAINING_ALREADY_COMPLETE_NOT_GAIT_VALIDATED，没有启动worker或增加更新。

## 显式升级边界

upgrade_run.py只接受随仓库保存的b313646源文件哈希清单（LF/CRLF）；native_sources、bank SHA、配置、物理和输入合同必须完全一致。核验所有历史提交路径/摘要/hash，只选择最后提交检查点并进行30秒稳定检查。新目录保留old_plan SHA、旧模型SHA、完整旧身份与新bootstrap SHA。

升级并非“忽略source mismatch”，而是有白名单、审计和逐状态相等检查的版本变更。未知源版本、资产变化、预算/上下文变化、活动同bank进程、USER_PAUSED、坏模型均拒绝，不自动修改旧plan或原模型。

服务器真实600代检查点尚未在本机读取；服务器执行升级工具后才能确认该链满足全部条件。README给出命令。没有远程操作服务器。

## 尚未声称的结果

没有12000长训、长期内存泄漏验证、服务器Linux实际GPU验证或步态验收。短测不证明收敛或降低遗忘。同进程减少释放训练分配的空档，但不提供GPU显存预约/独占，场景显存仍波动，别人任务仍可能造成OOM。

本机首次直接批量导入停在第19个USD，相关rhf26/run输出保留。成功4096测试使用hy24已有精确转换哈希缓存，新旧900份URDF逐份SHA相等；没有更改缓存哈希绕过转换或物理检查。
