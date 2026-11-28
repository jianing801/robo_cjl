"""Independent-timer warm-up for a native Rough singleton checkpoint.

Run the original, identity-checked training script unchanged; reuse the
repository's original staggered episode helper before the first PPO update.
"""
from pathlib import Path
import runpy
import sys
from statistics import fmean


def snapshot(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    return value


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def install_warmup(runner, output, pause_root, stage, native_ledger, phase_helper, publish):
    native_ledger(runner, output, pause_root, stage)
    beginning = int(runner.current_learning_iteration)
    if not 0 < beginning < int(stage["end_update"]):
        raise ValueError(f"Expected a partial singleton checkpoint, got {beginning}")
    if int(runner.env.num_envs) != 4096 or len(stage["design_ids"]) != 1:
        raise ValueError("Expected the native 4096-env singleton experiment")
    policy = snapshot(runner.alg.policy.state_dict())
    adam = snapshot(runner.alg.optimizer.state_dict())
    lr = runner.alg.learning_rate
    storage_step = runner.alg.storage.step
    info = phase_helper(runner, Path(pause_root)/'USER_PAUSED', independent=True)
    info.update(
        policy_and_normalization_unchanged=equal(policy, runner.alg.policy.state_dict()),
        adam_unchanged=equal(adam, runner.alg.optimizer.state_dict()),
        learning_rate_unchanged=lr == runner.alg.learning_rate,
        update_count_unchanged=beginning == runner.current_learning_iteration,
        ppo_storage_unchanged=storage_step == runner.alg.storage.step,
        actual_warmup_mean_reward=fmean(runner.logger.rewbuffer),
        actual_warmup_mean_episode_length=fmean(runner.logger.lenbuffer),
        actual_window_rewards=list(runner.logger.rewbuffer),
        actual_window_lengths=list(runner.logger.lenbuffer),
    )
    import torch
    age = runner.env.episode_length_buf.detach().float().cpu()
    info['post_warmup_episode_age_histogram'] = torch.histc(age, bins=20, min=0, max=float(runner.env.max_episode_length)).int().tolist()
    if not all(info[k] for k in ('policy_and_normalization_unchanged','adam_unchanged','learning_rate_unchanged','update_count_unchanged','ppo_storage_unchanged')):
        raise RuntimeError(f'Warm-up changed learning state: {info}')
    if info['window_episodes'] != 100:
        raise RuntimeError(f'Warm-up did not fill the original 100-episode window: {info}')
    publish(Path(output)/'warmup_receipt.json', info)
    print('STAGGERED_RESUME_WARMUP_VERIFIED', {k:v for k,v in info.items() if not k.startswith('actual_window_')}, flush=True)


def main():
    native_script = Path(sys.argv[1]).resolve()
    engine = native_script.parent
    sys.path[:0] = [str(engine), str(engine.parent)]
    import singleton_native
    native_ledger = singleton_native.install_ledger

    def ledger(runner, output, pause_root, stage):
        # Import torch/Isaac-facing helper only after the simulator is running.
        from episode_phases import warm_up_episode_phases
        install_warmup(runner, output, pause_root, stage, native_ledger,
                       warm_up_episode_phases, singleton_native.publish)

    singleton_native.install_ledger = ledger
    sys.argv = sys.argv[1:]
    runpy.run_path(str(native_script), run_name='__main__')


if __name__ == '__main__':
    main()
