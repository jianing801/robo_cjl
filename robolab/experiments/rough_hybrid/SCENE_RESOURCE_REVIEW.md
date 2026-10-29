# 换组显存与故障清理处理

服务器 stderr 最早记录 Warp alloc_device_async out of memory，随后出现
invalid device ordinal、PhysX illegal memory access，以及初始化失败后的关闭异常。
这证明显存分配失败在日志中先出现，但不能区分外部竞争、未用缓存或资源累计。

## 本次变更

- 每组 scene_memory.jsonl 记录创建前缓存释放前后、初始化后、关闭前后：
  PID、时间、当前逻辑设备、CUDA free/total、PyTorch allocated/reserved，
  以及 nvidia-smi 全卡 UUID/free/used。查询超时或不可用仅记录诊断错误。
- 下一组初始化前，在上一组函数已返回后运行 gc.collect、CUDA synchronize、
  empty_cache。仅释放未使用的 PyTorch 缓存，不删除仍存活的模型、Adam或rollout。
  不清除 Warp 全局运行时，不更改物理、奖励、设计分组或软件版本。
- 权威 failure.json 写入、traceback 刷新后启动独立守护线程：故障清理若120秒
  未完成，os._exit(1) 退出本 worker。不是健康训练超时，也不会自动重试。
  Python 守护线程不能保证解决驱动级不可中断阻塞，必要时仍需外部终止。
- 控制器另外监视本次会话的 failure.json/cleanup_failure.json；只有明确失败
  持续120秒仍未退出才停止本次子进程树，5秒后仍存活则强制终止。
  不按日志安静时间判定故障，不扫描旧组错误，不自动重启训练。
- 显存查询异常仅记诊断，不覆盖原始故障，也不让诊断文件写入失败中断训练。

## 验证边界与部署

CPU 回归检查缓存清理顺序、诊断字段、故障超时、原持久化训练/恢复契约。
尚未在服务器4096环境做长期重建验证，不能声称泄漏已解决。
本次不启动训练、不更新依赖、不推送Git、不修改任何旧计划或检查点。
源码属于严格身份的一部分：旧训练不能直接覆盖代码后 --resume。
部署前需确认服务器实际版本，并单独审核仅运行时变更的接续身份；不能绕过校验。
