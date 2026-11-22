# AMP 换组随机回合对照实验

轮换训练仍是 4096 个环境、每组 50 种结构、每 50 次 PPO 更新换组，共 5000 次更新。首组和每次重建场景后，训练器先为每个环境独立随机设置 0–999 步的首回合起始计数，再开始正式采样。一个环境完成首次重置后，后续回合从 0 开始。该入口采用普通 AMP 的首回合计时和回报记录规则，预热步数为 0。

在仓库根目录使用已迁移的 2500 结构库启动，例如用 3 号卡：

```bash
conda run --no-capture-output -n robo_cjl python start_amp_random_episode.py --migrated --gpu 3
```

用默认实验根时去掉 `--migrated`。末尾加 `--dry-run` 可先查看命令。入口复用 AMP 结构库、已登记的 USD 缓存和参考动作库。

输出在实验根的 `experts_random_episode/amp/`。同一命令会从该实验最后一个已提交的组续训。有用户暂停标记且已有已提交断点时，加 `--resume-paused`。如果想从头另跑一遍，使用 `--run-name amp_repeat`，保留原实验目录。

用 TensorBoard 的 `Train/mean_reward` 和 `Train/mean_episode_length` 对照原专家。训练日志位于 `attempts/session_*/train/stdout.log`。每组的 `Mean reward` 仍用最近 100 个已结束回合的累计回报求均值，随机起始计时可能缩短首回合，首回合仍正常计入统计。
