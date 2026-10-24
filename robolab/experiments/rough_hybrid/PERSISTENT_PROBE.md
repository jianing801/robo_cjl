# Rough异常退出修复与同进程换组试验

日期：2026-10-24。本地完成；未提交或推送Git，未操作服务器。背包保持USER_PAUSED。

## 异常退出

原问题：Kit关闭可能掩盖Python异常，子进程返回0，但failure.json存在且summary.json缺失，控制器误读摘要后才报FileNotFoundError。

现在控制器同时检查原始退出码、failure.json和完成摘要。exit.json分别记录raw_returncode与有效returncode，异常返回0也记为有效失败1；失败记录优先于任何摘要，拒绝提交/启动下一组。worker异常时保留原始traceback，并注册失败退出处理；即便底层绕过退出处理，控制器仍独立拒绝该结果。没有把OOM变成成功，也没有自动改环境数或重试。

38项CPU回归通过，包括正常逐迭代日志、返回0却有failure、返回0缺summary以及TensorDict键接口。

## 实验入口（非正式训练入口）

probe_persistent.py：仅接受最多64环境、最多12次更新的独立新bank/配置和新输出目录。生产train.py默认仍是逐组worker，不会悄悄切换。

同一Kit应用跨组运行，保留runner/policy/Adam/normalization/rollout storage；每组销毁旧环境、清理旧RayCaster地形网格缓存、建立新USD stage，再创建新真实形态组。重绑环境、镜像增强环境引用及均衡采样分组。rollout分配保留但索引清零，不复用旧组轨迹；统计中的未完成回合清零。原生learn最后停在已完成iter，下一组显式推进到下一个全局编号，不重复更新。

首次试验发现RayCaster.meshes类缓存含Warp runtime循环引用，使第二组cfg.validate递归溢出。已定位并清理，不关闭扫描、不改物理。配置从未初始化的独立模板深拷贝。

## 真实短测证据

根：C:/Users/Admin/.codex/.chatgpt-projects/g-p-6a967dbd08f4819188bc02c144f1a591/rhp26/probe6

- 配置：8环境、4种真实URDF完整库、2结构/组、2更新/组、3组共6次PPO；组0和组2同一批结构，经历两次换组。
- 模型：group_0/models/model_1.pt、model_3.pt、model_5.pt。日志目录保持同一处，不是每组新建策略。
- 三组PID均3528；policy/optimizer/storage对象身份全程相同。
- 三组Adam step为40、80、120；内部iter为1、3、5。
- 两次跨组后全部Actor/Critic层参数均变化；模型张量有限，严格检查点身份验证通过。
- 每组geometry_audit、manifest与summary均存在；probe_complete.json及group_0/1/2_audit.json记录原始证据。
- PyTorch已分配显存约59.94/60.51/60.51 MB，保留缓存约98.57/100.66/100.66 MB。这不包含PhysX、Kit及其他非PyTorch显存，不是整项任务显存。
- 首组32.16秒，后两组14.09/14.42秒；仅小规模含加载/首次缓存差异的数据，不能据此宣称4096环境提速比例。

原失败试验probe1至probe5及日志保留，不作为成功结果。没有新增长训或录像。

## 尚未完成的生产验收

尚未做4096环境、30结构/组的同进程验证、长时间泄漏测试、Linux服务器测试、生产调度/中断恢复集成。当前试验入口不支持生产续训。不得直接替换原服务器训练或绕过冻结源身份检查。生产版本源变动后需明确的新身份/迁移流程，不能修改旧plan冒充原实验。

同进程可减少释放全部训练分配的空档，但不能预留/独占GPU显存，场景内存仍有波动；其他任务仍可能造成OOM。真正资源保证需服务器调度预约/独占。
