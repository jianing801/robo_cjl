# USD资产准备启动说明

本页只用于跨形态多任务流程的仿真资产准备：把结构库里的URDF转换成Isaac可加载的USD。**不启动PPO训练，也不生成AMP参考动作。** 已有资产会校验复用，不删除旧结果。

## 1. 先更新代码

在相关启动器退出后执行；不要在转换或训练进程仍使用代码时更新。

```bash
git -C /data/chenjiale/robo_cjl pull --ff-only origin main
```

下面命令已经包含环境和完整路径，不需要设置ENTRY、ROOT，也不需要先切换目录。前提是三个任务结构库已生成；你已完成这一步。

## 2. 选一种启动方式

### 一张卡依次完成三个任务

使用5号卡，依次处理Rough、负重、AMP；任一任务失败会停止。

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --gpu 5
```

### 只处理某一个任务

**只准备Rough，5号卡：**

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 rough --gpu 5
```

**只准备负重，5号卡：**

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 负重 --gpu 5
```

**只准备AMP，1号卡：**

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 amp --gpu 1
```

不同任务可以在不同终端、不同GPU上并行。不要同时运行“全部任务”和其中一个单独任务，也不要重复启动同一任务。编号与nvidia-smi一致，0表示第一张卡。每张卡转换启动门槛为空闲4GiB，不是峰值显存保证；并行仍可能争用/data存储。

## 3. 看懂进度和完成标志

| 输出 | 正在做什么 |
|---|---|
| USD_EXISTING_CACHE_CHECK_BEFORE_GPU | 发现完整缓存，先用CPU校验，不申请GPU |
| CACHE_VALIDATE_SOURCE_MESHES | 本次要检查的结构数量 |
| CACHE_URDF_PROGRESS / CACHE_MESH_PROGRESS | 结构源文件与网格依赖检查进度，如60 / 60 |
| CACHE_VALIDATE_PROGRESS 12 / 3000 | 本次完整缓存检查已有12个结构通过 |
| CACHE_VALIDATE_WAIT | 每10秒提示校验尚未结束；阶段及已完成数见最近进度 |
| CACHE_REUSE_PROGRESS 2026 / 3000 | 已从历史尝试校验并复用2026个结构，不是重新转换 |
| CACHE_SESSION_START | 启动原生转换子进程，处理仍缺失的结构 |
| NATIVE_CACHE_PROGRESS | 已接受的总资产数，可能包含复用项 |
| CACHE_HEARTBEAT | 每10秒报告运行状态及最近阶段/进度，不再输出打开文件路径 |
| CACHE_FINALIZED | 本次协调器完成报告核对成功，无需外层重读全部USD |
| USD_ASSETS_READY rough / backpack / amp | 对应任务已完成缓存登记 |
| USD_ASSETS_ALL_SELECTED_READY | 此次选定的任务全部完成 |

**等到USD_ASSETS_READY及正常结束才算该任务完成。** 单独的CACHE_CHILD_EXIT、returncode=0或3000/3000不是外层登记完成标志。全库检查仍可能受存储影响，但本次转换完成后不再在外层重复全量读USD。完成标志不代表步态或训练验证成功。

## 4. 暂停与继续

暂停：在对应终端按一次Ctrl+C，等待退出。已完成资产保留。

如果提示用户暂停标记，确认自己要继续后，使用对应任务的恢复命令：

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 rough --gpu 5 --resume-paused
```

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 负重 --gpu 5 --resume-paused
```

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py 资产 --任务 amp --gpu 1 --resume-paused
```

恢复选项只恢复尚无PPO记录的资产准备暂停，不会解除已有训练暂停。不要直接删除暂停标记或GPU锁。恢复时历史缓存仍需检查；进度重新从小数字开始不等于重新转换。

## 5. 准备好后启动训练

例如3号卡空闲时启动Rough：

```bash
conda run --no-capture-output -n robo_cjl python /data/chenjiale/robo_cjl/start.py rough --gpu 3
```

入口自动寻找已登记缓存，无需复制USD路径。训练需要空闲12.5GiB；已有完整缓存时，首组60个结构先校验，通过后再申请GPU。AMP训练仍需其参考动作库，USD准备不代替参考动作生成。

完整流程见[跨形态多任务共享策略训练流程](跨形态多任务共享策略训练流程.md)。


### 启动训练停在计划检查时

`CACHE_PLAN_CHECK_PPO_EVIDENCE`是在确认尚未开始PPO，随后才能更新转换相关代码身份。检查器排除顶层`native_cache_*`和有明确转换启动记录的`attempts/cache_*`，不为查找模型遍历USD目录。未知目录和未证明为转换任务的尝试仍检查；已存在训练启动或提交记录时继续拒绝修改身份。

检查阶段不逐目录输出路径；明确的纯转换尝试自动跳过；`CACHE_PLAN_NO_CHECKPOINTS`表示未发现检查点，随后进入代码身份检查。此步骤不申请GPU，不转换USD。


### 简洁进度输出

正常情况下只显示阶段、已完成/总数，以及必要的启动、结束信息。源文件、网格、USD检查和历史资产复用的计数通常按约5%刷新，处理较慢时也会更新已完成数；等待提示每10秒保留。不会逐个打印URDF、USD、网格、证明文件或结构ID。失败时仍保留具体文件路径及完整错误，方便定位。

典型输出：

```text
CACHE_PHASE CHECKING_PPO_EVIDENCE
CACHE_URDF_PROGRESS 0 / 60
CACHE_URDF_PROGRESS 30 / 60
CACHE_URDF_PROGRESS 60 / 60
CACHE_MESH_PROGRESS 24 / 24
CACHE_VALIDATE_PROGRESS 30 / 60
CACHE_VALIDATE_PROGRESS 60 / 60
```


### 中断后继续与校验记录

启动命令不变。每个已验证资产成功深检后立即保存恢复记录；下次启动比较该资产的文件大小、修改时间、设备和文件标识。未变化时跳过大USD内容读取，变化时重新完整哈希验证。旧资产没有恢复记录时，需要一次深检；因此首次升级仍可能慢，已完成的逐资产检查不会因再次中断而全部白做。

此检查仍遍历资产元数据和源URDF/网格，不承诺完全没有磁盘等待。它依赖文件元数据发现变化，不等于每次重新证明所有内容；完整审计接口 `validate_cache` 的默认行为仍为深检。恢复记录写在资产目录的上一层，不进入USD文件清单。

原生worker启动前不再逐一解析全部剩余结构的URDF；协调器先检查共享源，worker随后逐个验证并转换。`NATIVE_CACHE_PROGRESS 已完成 / 总数`表示已接受资产，`CACHE_REUSE_PROGRESS`表示历史资产检查复用。连续三次没有新增资产仍停止并保留日志；原生进程崩溃不视为成功，也不无限重试。
