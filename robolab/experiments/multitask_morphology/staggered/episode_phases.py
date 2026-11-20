"""错开回合起点，用原生记录器统计完整回合。"""

from functools import wraps
import json
import time

import torch


def warm_up_episode_phases(runner, pause_marker=None):
    if runner.alg_cfg.get("rnd_cfg") or runner.alg.policy.is_recurrent or getattr(runner, "is_distributed", False):
        raise ValueError("错开回合入口支持单卡、前馈策略和无 RND 的原生专家配置。")

    env, logger, alg = runner.env, runner.logger, runner.alg
    age = env.episode_length_buf.reshape(-1)
    num_envs = age.numel()
    horizon = int(env.max_episode_length)
    steps = horizon + 4 * runner.cfg["num_steps_per_env"]
    groups = alg.storage.spatial_groups.to(age.device)
    if horizon <= 0 or groups.shape != age.shape:
        raise ValueError("回合长度或结构与环境的对应关系无效。")

    disc = getattr(alg, "amp_discriminator", None)
    counters = [logger.cur_episode_length, logger.cur_reward_sum]
    windows = [logger.rewbuffer, logger.lenbuffer]
    if disc is not None:
        counters.extend((logger.cur_total_reward_sum, logger.cur_style_reward_sum))
        windows.extend((logger.total_rewbuffer, logger.style_rewbuffer))

    runner.train_mode()
    started = time.monotonic()
    print(f"EPISODE_PHASE_WARMUP_BEGIN steps={steps} environments={num_envs}", flush=True)
    with torch.inference_mode():
        # 负重阶段可能因课程调整再次物理重置，此时也重新建立统计窗口。
        for window in windows:
            window.clear()
        for counter in counters:
            counter.zero_()
        logger.ep_extras.clear()

        # 每个结构内部均匀分配起始回合步数，再加独立的随机相位。
        _, membership, sizes = torch.unique(groups, return_inverse=True, return_counts=True)
        order = torch.argsort(membership)
        sorted_groups = membership[order]
        starts = sizes.cumsum(0) - sizes
        ranks = torch.arange(num_envs, device=age.device) - starts[sorted_groups]
        shifts = torch.randint(horizon, sizes.shape, device=age.device)
        age[order] = (ranks * horizon // sizes[sorted_groups] + shifts[sorted_groups]) % horizon

        reset_once = torch.zeros(num_envs, dtype=torch.bool, device=runner.device)
        valid_episodes = torch.zeros((), dtype=torch.long, device=runner.device)
        obs = env.get_observations().to(runner.device)
        was_training = disc.training if disc is not None else None
        if disc is not None:
            disc.eval()
        try:
            for step in range(1, steps + 1):
                if step == 1 or step % 250 == 0:
                    if pause_marker is not None and pause_marker.exists():
                        raise RuntimeError("USER_PAUSED")
                # 直接使用策略和仿真。采样缓冲区与优化器由随后的原生 learn 调用处理。
                action = alg.policy.act(obs).to(env.device)
                obs, reward, done, _ = env.step(action)
                obs, reward, done = obs.to(runner.device), reward.to(runner.device), done.to(runner.device)
                finished = done.reshape(-1) > 0
                valid = finished & reset_once
                provisional = finished & ~reset_once
                valid_done = valid.reshape_as(done)
                if disc is None:
                    logger.process_env_step(reward, valid_done, {})
                else:
                    disc_obs = disc.get_disc_obs(obs, flatten_history_dim=False)
                    style, _ = disc.predict_style_reward(disc_obs, dt=alg.amp_cfg["step_dt"])
                    total = disc.lerp_reward(task_reward=reward, style_reward=style)
                    # AMP 记录器会把风格奖励写进当步的字典。
                    logger.process_env_step(reward, valid_done, {"log": {}}, None, style, total)
                    logger.ep_extras.clear()
                for counter in counters:
                    counter.masked_fill_(provisional, 0)
                reset_once.logical_or_(finished)
                valid_episodes.add_(valid.count_nonzero())
                if step % 250 == 0:
                    print(f"EPISODE_PHASE_WARMUP_PROGRESS {step}/{steps}", flush=True)
        finally:
            if disc is not None:
                disc.train(was_training)

        reset_count = int(reset_once.count_nonzero().item())
        if reset_count != num_envs or not logger.rewbuffer:
            raise RuntimeError("预热结束时尚未得到可统计的完整回合。")

    info = dict(
        stage=runner.hybrid_stage["index"],
        preceding_updates=runner.current_learning_iteration,
        warmup_control_steps=steps,
        warmup_transitions=steps * num_envs,
        provisional_episodes_excluded=reset_count,
        genuine_warmup_episodes=int(valid_episodes.item()),
        window_episodes=len(logger.rewbuffer),
        warmup_seconds=time.monotonic() - started,
        ppo_collection_during_warmup=False,
    )
    print("EPISODE_PHASE_WARMUP_COMPLETE " + json.dumps(info), flush=True)
    return info


def attach_phase_warmup(runner, pause_marker):
    original_learn = runner.learn

    @wraps(original_learn)
    def learn(num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("错开回合入口使用结构内均匀的初始步数。")
        if pause_marker.exists():
            raise RuntimeError("USER_PAUSED")
        warm_up_episode_phases(runner, pause_marker)
        return original_learn(num_learning_iterations)

    runner.learn = learn
