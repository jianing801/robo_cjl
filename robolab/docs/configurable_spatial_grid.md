# 可配置形态网格（Rough空间联合训练）

编辑 `configs/spatial_rough_grid.json`。所有修改只对新生成的实验生效，不修改旧bank、检查点或正在运行的仿真。

| 字段 | 含义 |
|---|---|
| num_envs | 总环境数，正整数，不再固定2048 |
| length_count | 大腿、小腿各取多少个长度值 |
| thigh_range_m / calf_range_m | 各自长度范围，单位米 |
| knee_motor_count / ankle_motor_count | 各抽多少个不同型号 |
| motor_seed | 固定抽样种子；同配置重复生成结果一致 |
| knee_pool / ankle_pool（可选） | 自定义候选型号列表；不填使用全部已注册且资料完整的候选池 |

长度是包含两端点的等间距网格，不是随机长度。数量为1时取范围中点。所有大腿值、小腿值、抽中的膝电机、踝电机做笛卡尔积；没有暗中增加中心点或留出结构。总数=N²×K×A。例如示例配置3²×2×2=36种。示例数值不是已启动的训练。

电机从池中不放回抽取，按膝/踝独立种子流选择，在bank里冻结；不是每次重置重新抽。型号数量不能超过池容量。腿长不得超出所选型号SW GP的有效范围，生成时检查。现有资料支持的完整范围是大腿0.25–0.40m、小腿0.30–0.45m，具体以所选模型校验为准。

每个环境固定一个真实URDF，同一批环境覆盖全部结构，按环境编号轮流分配。不能让结构数超过环境数；环境数不能整除结构数时，每种结构分到的环境数最多差1。原on-policy样本不丢弃，按形态分散到小批次，不额外改奖励或损失权重。结构数变多、环境数不变时，每种结构获得的样本会减少。

结构输入维数=18+K+A（长度2、型号one-hot K+A、两组扭矩/速度/armature共6、曲线采样10）。改变型号数量或bank后不能直接恢复旧模型，必须使用匹配bank/输入定义的模型；本接口拒绝错配。修改环境数后也必须重新做匹配的预检。没有自动启动、自动扩预算。

## 使用

在robolab工作目录、项目Python环境下：

```powershell
python scripts/tools/prepare_spatial_grid.py --config configs/spatial_rough_grid.json
```

只预览取点、完整候选池、抽取型号、结构数，不启动仿真。

```powershell
python scripts/tools/prepare_spatial_grid.py --config configs/spatial_rough_grid.json --template /path/to/rpo/urdf/rpo.urdf --mesh-dir /path/to/rpo/meshes --output-dir logs/my_new_grid/bank --generate-assets
python scripts/tools/train_spatial_rough.py --bank logs/my_new_grid/bank/bank.json --output-dir logs/my_new_grid/plan
```

第二步只生成CPU训练计划。输出目录必须全新。环境数默认读取bank，也可用`--num-envs`指定新计划的环境数。物理预检使用`--preflight-only`，正式训练仍需`--execute --updates N --preflight-report <匹配预检summary.json>`，保留原来的源配置/物理校验和GPU冲突检查。

当前接通的是Rough空间联合训练，不代表AMP空间联合训练已实现。AMP仍需为同批不同形态绑定各自重定向参考，不能直接套用一个结构的参考。

## 独立发布

默认配置为4096环境、10²×2×2=400种结构；不是已启动的训练。

新入口仅依赖原RPO-Rough/本仓库RSL-RL、既有物理快照，以及本次Rough辅助模块，不需要旧AMP/MxT/共享策略模块、旧模型或旧日志bank。原始URDF和mesh需要在训练机器上存在；上面的路径必须替换为该机器的实际路径。默认关节顺序/初始姿态在`configs/rough_robot_contract.json`，真实仿真启动时再次核对；不匹配会停止，不能忽略。

v2 bank需要在训练机器重新生成，不直接使用之前v1 bank或旧检查点。修改不改变旧文件。模型/输出保存到logs，未纳入Git。

Linux服务器准备好Isaac Lab和本仓库依赖后，在robolab目录先运行配置预览，再生成bank。随后可以明确运行两次更新的预检（非长训练）：

```bash
python scripts/tools/train_spatial_rough.py --bank logs/my_new_grid/bank/bank.json --output-dir logs/my_new_grid/preflight --preflight-only --headless
```

正式更新预算仍需用户明确指定，不能从“400种结构”推导训练代数。400真实URDF、4096环境的完整显存/吞吐尚未验证；不要把CPU测试通过视为该规模训练已通过。
