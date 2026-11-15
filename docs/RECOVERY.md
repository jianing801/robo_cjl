# 暂停、续训、排错与Git更新

以下处理保留已有实验、模型、录像和参考动作。不要用删除ROOT、强制reset、改bank或覆盖检查点来“清掉报错”。路径变量见 [完整流程](WORKFLOW.md)。

## 先区分当前阶段

| 报错/现象 | 含义与处理 |
|---|---|
| `can't open file '/train.py'` | 当前终端ENTRY为空；重设路径变量，不是训练失败 |
| 缺`banks/.../bank.json` | 首次结构库尚未生成或ROOT指错；先检查文件，再prepare |
| 首次prepare提示ROOT存在 | 输出保护生效；区分完整库与部分生成，不要删除旧目录 |
| `references`目录存在但没有完整library | 可能是中断后的部分结果；不能当完整参考库使用 |
| 训练输出目录已存在 | 可能是预览建了plan，也可能已有训练；先检查plan和commits |
| `USER_PAUSED` | 人工暂停标记，程序不会自动清除 |
| worker返回0但有failure | 控制器会拒绝继续；看原始failure和stderr，不以退出码0当成功 |
| CUDA illegal memory access / OOM | 读最早的原生错误，确认进程和显存；不能在损坏进程里直接重试 |
| URDF importer版本/依赖不匹配 | 沿用已验证版本排查本地扩展和进程参数，不自动升级Isaac |
| 校验提示源码/bank/参考/排表身份变化 | 停止；使用原冻结输入或创建有新身份的实验，不手改哈希 |

## 1. 先查看进程和证据

```bash
pgrep -u "$USER" -af 'multitask_morphology/.*(train|worker)'
nvidia-smi
find "$ROOT/experts" -name failure.json -o -name cleanup_failure.json
```

有driver/worker或同任务GPU残留时，不再启动第二份。安静日志不能证明退出。先读相应attempt下的`stdout.log`、`stderr.log`、`launch.json`、`exit.json`及result失败记录；CUDA异步报错最后出现的位置不一定是最早失败的位置。

## 2. 人工暂停与恢复

运行中的前台训练按Ctrl-C会请求暂停；复制终端文本通常使用Ctrl+Shift+C，具体以终端快捷键为准。暂停后确认匹配的driver先停止、worker随后停止；不要用广泛的`pkill python`影响其他用户或任务。

明确恢复时：确认进程已退出、输入身份与原plan一致，再仅清除对应任务输出下的USER_PAUSED，并在原启动命令中增加`--resume`。例如Rough：

```bash
# 仅在你已明确决定恢复且相关进程全部停止后执行。
rm -f "$ROOT/experts/rough/USER_PAUSED"
python -u "$ENTRY/train.py" --task rough \
  --bank "$ROOT/banks/rough/bank.json" --output-root "$ROOT/experts/rough" \
  --device "$GPU" --gpu-min-free-gb 12.5 --gpu-max-utilization 100 --resume --execute
```

负重/AMP用各自原命令和各自暂停标记。AMP保留原reference-library。只续已提交完整阶段的同链状态，恢复Adam、学习率与绝对更新编号；未提交阶段可能重做，仿真环境重置，不保证物理轨迹逐位连续。

若先前只是去掉`--execute`预览而创建了plan，可在确认没有实际训练失败、没有worker后，用相同命令加`--resume --execute`执行已有计划。初始阶段实际失败且没有已提交检查点时，控制器会拒绝隐式从头重启；应保留失败目录另行分析，不绕过检查。

## 3. 部分生成如何处理

完整结构库已验证：跳过prepare。部分结构库：保留现场，先确定哪一个bank完整。不要把空目录等同于有效bank，也不要手写JSON补“完成”。

若只有Rough bank完整，已确认它通过当前原生身份检查，可以用对应内部准备入口，只创建缺失任务的**新输出目录**：

```bash
python "$ENTRY/engine/prepare_bank.py" --task-key backpack \
  --config "$ENTRY/configs/backpack.json" --rough-bank "$ROOT/banks/rough/bank.json" \
  --output-dir "$ROOT/banks/backpack_recovered" --generate-assets
python "$ENTRY/amp_engine/prepare_bank.py" \
  --config "$ENTRY/configs/amp.json" --rough-bank "$ROOT/banks/rough/bank.json" \
  --output-dir "$ROOT/banks/amp_recovered" --generate-assets
```

这些是另名的任务bank，不自动替换原路径；后续审计和启动必须显式使用实际恢复后的路径。已有AMP reference library绑定具体bank SHA，不能把它无提示地换绑到新bank，即使腿长看起来相同。

部分AMP参考动作：目前没有自动续生成接口。保留原目录，检查生成日志和完成的运动学片段；如需重新生成，使用一个新的输出目录并在训练命令里显式指定最终library。不要移动生成目录，因为library内保存绝对路径。完整库可直接校验复用。

## 4. 普通Git更新

更新前让本仓库相关训练、参考动作生成和资产转换结束；不要为了更新停止别人任务。源码身份会进入训练检查点，不能在运行中升级代码或换分组后冒充原链。

```bash
cd /data/chenjiale/robo_cjl
git status
git branch --show-current
git -c http.version=HTTP/1.1 fetch https://github.com/jianing801/robo_cjl.git feature/rough-hybrid
git log --oneline --left-right HEAD...FETCH_HEAD
```

工作区干净且确认没有本地独有提交时，才用`git merge --ff-only FETCH_HEAD`。有服务器本地提交时，应先备份、检查差异后正常合并，不能强制覆盖。之前API快照合并会留下本地提交，因此“领先origin若干提交”不自动表示错误，也不是要求执行push。

若仍在`git am`会话，即使工作区干净也不能切分支。先保存当前补丁/提交并确认是否要结束旧导入；`git am --quit`保留当前HEAD和文件，但结束剩余会话，不等于`--abort`。不要未核对就丢弃尚需应用的补丁。

## 5. Git连接超时但API可达

```bash
curl -I --connect-timeout 5 --max-time 12 https://api.github.com
```

API可达后可使用仓库的在线快照合并工具：

```bash
curl -fL --connect-timeout 5 --max-time 60 \
  -H 'Accept: application/vnd.github.raw+json' \
  'https://api.github.com/repos/jianing801/robo_cjl/contents/robolab/scripts/tools/update_github_api.py?ref=feature%2Frough-hybrid' \
  -o /tmp/robo_api_update.py \
&& python /tmp/robo_api_update.py --repo /data/chenjiale/robo_cjl --apply
```

工具校验GitHub原始tree、备份当前HEAD、普通三方合并，保留本地提交；不伪造完整远端历史，也不改origin跟踪。API的官方codeload下载地址也要可达。成功标志为`API_UPDATE_COMPLETE`，有冲突则停在现场。详细机制见 [API更新说明](../robolab/scripts/tools/GITHUB_API_UPDATE.md)。

有冲突时先执行`git status`和`git diff --name-only --diff-filter=U`。`.gitattributes`中的冲突标记可能导致“不是有效属性名”警告，应先核对两边规则。只有确认对方完整包含本地规则时才能选择对方文件；不要把过去一次已核对的解决命令套到所有未来冲突。解决后`git add`相应文件，再`git commit --no-edit`。

API合并后的本地HEAD通常不同于上游SHA，这是保留本地历史的结果。用`git log HEAD --oneline --grep='Import verified GitHub API snapshot' -1`查看实际合入的来源版本，不要仅凭两个提交号不同就判断更新失败。

## 原生资产持续转换与恢复

首次准备默认一个Isaac Sim子进程持续转换全部待完成设计，不按固定4或32种重启。每完成一个设计，即保存USD、URDF/网格/几何层哈希及原子发布的 `batch_*/assets/DESIGN.json`。控制器只有校验通过后才计入进度，临时 `.tmp` 不算完成。

首次启动最多允许300秒未完成首个资产；开始产生完整资产后，连续30秒无新完成资产才判定停滞。持续有进度时没有整场转换总时限。意外退出或停滞后，保留文件和错误日志，仅向新进程传递剩余设计。连续3个进程均无进度则停止；明确Python转换错误或身份不匹配直接停止，用户暂停不恢复。全部资产完成但关闭进程停滞时，终止对应进程并使用已校验的逐资产证据汇总。

旧的完整小批次summary及新的逐资产证据均可在原训练目录恢复时复用，不能仅按USD文件存在就认定完成。默认不跨任务共享缓存。不删除失败目录。原生导出器、物理配置和最终缓存schema保持原样。

独立准备入口可用 `--batch-size N` 显式指定定期重启间隔；默认0为持续模式。`--startup-timeout`、`--stall-timeout`只用于控制器看门狗。3000个ID通过JSON清单传给子进程，避免Windows命令行长度限制。

正在运行的旧转换进程不会热切换；更新前先停止匹配的控制器和转换子进程并确认退出，再在原输出目录使用 `--resume --execute`。已有成功批次会复用。如果已进入PPO，不为缓存提速中途更新源码。

更新转换代码会改变旧计划记录的源码身份。如果原目录仅转换缓存、尚未开始PPO，可以在停止相关进程后执行：

```bash
python robolab/experiments/multitask_morphology/engine/refresh_cache_plan.py --root "$OUT"
```

该入口只接受已记录的两个旧转换器版本，检查训练/物理源码、bank和配置不变，并拒绝任何PPO启动记录、提交或检查点；保留原计划和审计后更新转换器身份。之后用原目录 `--resume --execute`，已有成功缓存可复用。已开始PPO则不允许这项迁移。
