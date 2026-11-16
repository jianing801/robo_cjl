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
| CACHE_VALIDATE_URDF_BEGIN / MESH_BEGIN / FILE_BEGIN | 正在读取的具体文件 |
| CACHE_VALIDATE_PROGRESS 12 / 3000 | 本次完整缓存检查已有12个结构通过 |
| CACHE_VALIDATE_WAIT | 每10秒提示校验尚未结束；结合最近BEGIN定位文件 |
| CACHE_REUSE_PROGRESS 2026 / 3000 | 已从历史尝试校验并复用2026个结构，不是重新转换 |
| CACHE_SESSION_START | 启动原生转换子进程，处理仍缺失的结构 |
| NATIVE_CACHE_PROGRESS | 已接受的总资产数，可能包含复用项 |
| CACHE_HEARTBEAT | 转换协调器仍在运行；disk-sleep表示当时等磁盘 |
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
