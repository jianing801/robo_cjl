# GitHub Git连接超时、API可达时的在线代码更新

正常网络下优先使用Git fetch/merge。本工具用于已确认 `api.github.com` 可达但 `github.com:443` Git连接失败的服务器，不需要人工上传离线包，不安装依赖，不启动或停止训练。

```bash
python /tmp/robo_api_update.py --repo /data/chenjiale/robo_cjl --apply
```

脚本从本仓库API获取目标分支及其源码快照。压缩源码由GitHub官方codeload通道传输，因此该域名也必须能访问；网络超时会停止，不修改工作区。已有 `GH_TOKEN`/`GITHUB_TOKEN` 可选，公开仓库不要求令牌。不会打印令牌，重定向到其他官方域名时不转发认证头。

具体行为：

1. 要求分支正确、工作区干净、没有未完成的am/rebase/merge，也没有正在使用本仓库相关源码的训练、参考动作生成、资产准备或控制器。
2. 读取GitHub提交的tree SHA；在内存/临时文件中处理快照，不向工作区解压。不执行下载内容。
3. 按原始文件字节、可执行位和符号链接重新写入Git对象并构建tree；必须与GitHub的tree SHA完全相同。对不完整快照、Git子模块缺失、导出过滤导致的不一致均拒绝合并。
4. 以本地已有的origin跟踪分支与当前HEAD的共同祖先作为导入基线，创建明确标注 `Import verified GitHub API snapshot ...` 的导入提交及 `api-snapshot/...` 分支。
5. 保存当前HEAD到唯一的 `backup/server-before-api-update-...` 分支，再执行普通三方合并，保留服务器本地提交。不会自动选择ours/theirs，不使用reset或强制覆盖；冲突时保留合并现场，请运行 `git status` 后处理。

这是**带来源校验的快照导入合并**，不是完整远端Git历史fetch。不会伪造远端提交SHA，也不会把本地origin跟踪分支改成快照提交。因此保留本地改动后，最终tree可能与纯上游tree不同，这是三方合并的正常结果。脚本会校验原本本地HEAD仍是最终HEAD的祖先。

后续普通Git网络恢复时，应先获取并检查远端历史，再正常合并，不能期待存在本地提交的分支始终支持 `pull --ff-only`。不要把API导入分支当作远端原始提交。历史实验、模型和视频不由脚本清理；Git本身在合并时修改的是受版本管理的源码。

本地验证包含：精确tree重建、错误SHA拒绝、非冲突更新保留本地提交、冲突时保留HEAD/备份/合并现场、origin跟踪分支不变。没有声称已经代替用户在服务器执行。

旧专家源码归档期间，若正在生成AMP参考动作，应先让它完成。更新器现在也会检查参考动作/资产准备进程，遇到活动任务会拒绝更新，不会停止任务。
