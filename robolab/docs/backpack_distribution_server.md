# 背包平地行走：固定进度课程，服务器训练

独立单任务 MLP + PPO，从头训练；不是共享策略，不加载旧背包或 AMP。
首轮只用大腿 0.25 m、小腿 0.30 m，膝 DM-J10010L-2EC、踝 DM-J4340P-2EC。
使用冻结的 SW 质量/惯量与动态扭矩曲线。暂不扩展多形态。

## 冻结预算

| 完成更新数（不是零基 checkpoint 编号） | 更新量 | 每环境质量均匀分布 |
|---|---:|---|
| 0–1000 | 1000 | 0.1–0.5 kg |
| 1000–1500 | 500 | 0.1–4.125 kg |
| 1500–2000 | 500 | 0.1–7.75 kg |
| 2000–2500 | 500 | 0.1–11.375 kg |
| 2500–3000 | 500 | 0.1–15 kg |
| 3000–12000 | 9000 | 0.1–15 kg |

4096 环境，24 steps/update，每 100 更新保存完整策略/critic/归一化/Adam。
无达标升重门槛，不会自动增加预算。各阶段独立建环境，按固定种子采样质量、
惯量和等效重心偏移，同阶段每个环境的背包属性保持固定，不在行走中改质量。
因此这是“阶段内固定样本”，不是每次 reset 重采样。轻负载在后续阶段仍被覆盖。
阶段切换恢复 PPO 状态和自适应学习率；物理环境和 rollout 重新初始化，
不是逐位连续的仿真轨迹。执行策略只使用原本体观测历史，不提供真实负载参数。
保留原 compliant 背包俯仰启发式、摔倒终止、跟踪 std=0.25、跟踪权重各 1；
沿用此前背包基线的 DR：躯干附加质量 ±1kg、armature 比例 0.8–1.2、
关节 reset offset ±0.15rad。这些设置仅在本任务生效，不修改 AMP/Rough。

## 服务器准备

使用已能运行该项目的 Isaac Lab / Isaac Sim + Python 环境（同本机兼容版本，
Python 3.11、RSL-RL 使用本仓库版本），安装本项目及 `rsl_rl`：

```bash
# 在仓库顶层（有 robolab/、rsl_rl/、vendor/）执行
python -m pip install -e ./rsl_rl -e ./robolab
python -m pip install psutil pytest
cd robolab
```

URDF/STL 是项目原有的大体积资源，本提交不上传机器人网格、模型、日志。
`--template` 必须指向服务器上**未经过腿长生成器处理**的原 RPO 模板，
`--mesh-dir` 指向原 STL 文件目录。不要把 Windows 上生成的 plan.json 拷过来；
服务器自己生成 plan，记录本机绝对路径、源文件和资源哈希。

```bash
# 按服务器实际目录修改这两个参数。先只生成计划，不占 GPU。
python scripts/tools/run_backpack_distribution_pilot.py \
  --root logs/backpack_distribution_server_v1 \
  --template /absolute/path/to/rpo/urdf/rpo.urdf \
  --mesh-dir /absolute/path/to/rpo/meshes
```

建议先做两步真实物理 smoke（仍为4096环境、零 PPO 更新）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/tools/run_backpack_distribution_pilot.py \
  --root logs/backpack_distribution_server_v1 --execute --smoke-only
```

检查 smoke 的 `manifest.json` 中背包质量、动态执行器审计及有限数值。
smoke 不算训练，不产生可恢复的策略，也不修改正式进度。
正式启动（示例用 GPU0；需要 GPU2 时改 CUDA_VISIBLE_DEVICES=2，内部仍用 cuda:0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/tools/run_backpack_distribution_pilot.py \
  --root logs/backpack_distribution_server_v1 --execute \
  > logs/backpack_distribution_server_v1/controller.stdout.log 2>&1 &
```

首次启动前确认日志文件尚不存在，避免 shell 重定向覆盖旧日志。
恢复时使用新的控制器日志名：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/tools/run_backpack_distribution_pilot.py \
  --root logs/backpack_distribution_server_v1 --execute --resume \
  > logs/backpack_distribution_server_v1/controller.resume1.stdout.log 2>&1 &
```

恢复会校验实验身份、完整 checkpoint 和 Adam，保留坏文件，尝试当前阶段
最新有效 `model_N.pt`，从 N+1 接续，不会把12000重新当作额外训练量。
若第一阶段尚未保存任何有效 checkpoint，恢复报错，不擅自从头重跑。
进程冲突时拒绝启动；该检查是保守的，发现其他训练/评估 Python 会停手，
请在空闲的服务器训练会话中执行，不要同时启动两个本实验控制器。
阶段输出和每次启动日志使用唯一目录，只有 `state.json` 是可更新的进度指针。
执行期间不要修改已冻结代码、计划、URDF或网格，也不要 git pull 到该运行目录；
后续代码更新用另外的 checkout。

## 结果解释

最终 `model_11999.pt` / state `complete_requires_gait_review` 只表示训练预算完成。
必须另行做固定初始条件、无训练随机化/无推扰的轻中重负载评估和真实仿真录像。
平均速度只报告，不作为单独的硬通过条件；正常迈步、持续前进、不原地踏步，
以及摔倒、姿态和电机饱和情况一起判断。尚未证明本暖身方案能学会15kg。
