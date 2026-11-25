"""Exercise the mixed observation/symmetry bridge without simulator imports."""
import ast
from functools import lru_cache
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from native_observations import NativeObservationAdapter, amp_native_observations, native_amp_setup


TASKS = ("stairs", "backpack", "run")
CORES = (3260, 1390, 243)
WIDTHS = tuple(x + 28 for x in CORES)
SOURCE = Path(__file__).resolve().parents[3] / "robolab" / "tasks"


def _native_mirrors():
    # Execute only the existing pure-Torch native mirror functions. The
    # configuration module itself depends on simulator/Kit imports.
    path = SOURCE / "direct/base/agents/rpo_agent_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    first = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_height_scan_mirror")
    last = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "data_augmentation_func")
    space = {"torch": torch, "lru_cache": lru_cache}
    exec(compile(ast.Module(body=[n for n in tree.body if first <= n.lineno < last], type_ignores=[]),
                 str(path), "exec"), space)
    path = SOURCE / "manager_based/amp/mdp/symmetry/rpo.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = {"_history_length", "_transform_policy_obs_left_right", "_transform_critic_obs_left_right",
             "_apply_xyz_sign", "_switch_joints_left_right_flat", "_switch_joints_left_right",
             "_transform_actions_left_right"}
    amp = {"torch": torch, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), str(path), "exec"), amp)
    return space, amp


def _rows():
    gen = torch.Generator().manual_seed(53)
    adapter = NativeObservationAdapter(WIDTHS)
    sizes = (4, 3, 2)
    inputs = []
    native = []
    for i, (task, width, n) in enumerate(zip(TASKS, WIDTHS, sizes, strict=True)):
        base = torch.randn(n, 1161, generator=gen)
        base[:, 780:783] = 0
        base[:, 780 + i] = 1
        privileged = torch.randn(n, width, generator=gen)
        inputs.append(adapter.pack(task, base, privileged))
        native.append(privileged)
    return adapter, inputs, native


def test_exact_raw_native_critics_and_explicit_task_ids_without_mixed_padding():
    adapter, inputs, native = _rows()
    for i, (value, privileged, width) in enumerate(zip(inputs, native, WIDTHS, strict=True)):
        assert value["task_id"].dtype == torch.long
        assert value["task_id"].shape == (len(privileged), 1)
        assert value["task_id"].eq(i).all()
        torch.testing.assert_close(value["critic"][:, :width], privileged, rtol=0, atol=0)
        assert value["critic"].shape[-1] == WIDTHS[0]
        assert value["critic"][:, width:].eq(0).all()
        assert set(value.keys()) == {"policy", "critic", "task_id"}
    combined = adapter.concat(inputs)
    assert combined.batch_size == torch.Size([9])
    assert combined["task_id"].squeeze(1).tolist() == [0] * 4 + [1] * 3 + [2] * 2
    with pytest.raises(ValueError, match="critic"):
        adapter.pack("run", inputs[2]["policy"], native[2][:, :-1])
    with pytest.raises(ValueError, match="critic"):
        NativeObservationAdapter((WIDTHS[0] - 1, WIDTHS[1], WIDTHS[2]))


def test_mirror_exact_native_rough_and_amp_layouts_and_backpack_identity():
    adapter, inputs, native = _rows()
    obs = adapter.concat(inputs)
    before = obs.clone()
    actions = torch.randn(9, 23)
    actions_before = actions.clone()
    direct, amp = _native_mirrors()
    env = SimpleNamespace(cfg=SimpleNamespace(observations=SimpleNamespace(critic=SimpleNamespace(history_length=3))))
    aug, act = adapter.augment(obs=obs, actions=actions, env=None)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(aug[key][:len(obs)], before[key], rtol=0, atol=0)
    torch.testing.assert_close(actions, actions_before, rtol=0, atol=0)
    torch.testing.assert_close(act[:9], actions_before, rtol=0, atol=0)
    enabled = torch.tensor([True] * 4 + [False] * 3 + [True] * 2)
    torch.testing.assert_close(act[9:][enabled], direct["mirror_actions"](actions[enabled]), rtol=0, atol=0)
    torch.testing.assert_close(act[9:][enabled], amp["_transform_actions_left_right"](actions[enabled]), rtol=0, atol=0)
    torch.testing.assert_close(act[9:][~enabled], actions[~enabled], rtol=0, atol=0)
    mirrored = aug[9:]
    # Canonical actor is ten frame-major frames for ALL three tasks, even when
    # native AMP actor history is three frames and term-major.
    torch.testing.assert_close(mirrored["policy"][enabled, :780], direct["mirror_policy_observation"](obs["policy"][enabled, :780]), rtol=0, atol=0)
    torch.testing.assert_close(mirrored["policy"][~enabled], obs["policy"][~enabled], rtol=0, atol=0)
    torch.testing.assert_close(mirrored["policy"][:, 780:], obs["policy"][:, 780:], rtol=0, atol=0)
    torch.testing.assert_close(mirrored["task_id"], obs["task_id"], rtol=0, atol=0)
    torch.testing.assert_close(mirrored["critic"][:4, :CORES[0]], direct["mirror_critic_observation"](native[0][:, :CORES[0]]), rtol=0, atol=0)
    # Backpack's actual source agent is RPOFlatAgentCfg, symmetry_cfg=None.
    # Repeat unchanged rows for native fixed-size 2x augmentation contract.
    torch.testing.assert_close(mirrored["critic"][4:7], obs["critic"][4:7], rtol=0, atol=0)
    torch.testing.assert_close(mirrored["critic"][7:, :CORES[2]], amp["_transform_critic_obs_left_right"](env, native[2][:, :CORES[2]]), rtol=0, atol=0)
    for start, end, core in zip((0, 4, 7), (4, 7, 9), CORES, strict=True):
        # The left-leg physical context describes the same unmodified design;
        # padded columns are not dynamically created or transformed.
        torch.testing.assert_close(mirrored["critic"][start:end, core:], obs["critic"][start:end, core:], rtol=0, atol=0)
    repeated, repeated_actions = adapter.augment(obs=mirrored, actions=act[9:])
    for key in obs.keys():
        torch.testing.assert_close(repeated[key][9:], obs[key], rtol=0, atol=0)
    torch.testing.assert_close(repeated_actions[9:], actions, rtol=0, atol=0)
    none, target = adapter.augment(obs=None, actions=actions)
    assert none is None
    torch.testing.assert_close(target, act, rtol=0, atol=0)
    with pytest.raises(ValueError, match="minibatch"):
        adapter.augment(obs=None, actions=actions)  # pending route is consumed
    assert adapter.augment(obs=obs, actions=None)[1] is None


def test_action_only_mirror_uses_exact_immediately_preceding_minibatch_route():
    adapter, inputs, _ = _rows()
    shuffled = adapter.concat(inputs)[torch.tensor([4, 8, 6, 0, 2, 1, 3, 7, 5])]
    actions = torch.randn(len(shuffled), 23)
    _, expected = adapter.augment(obs=shuffled, actions=actions)
    _, mirrored = adapter.augment(actions=actions)
    torch.testing.assert_close(mirrored, expected, rtol=0, atol=0)
    adapter.augment(obs=shuffled, actions=None)
    with pytest.raises(ValueError, match="minibatch"):
        adapter.augment(actions=actions[:-1])
    with pytest.raises(ValueError, match="minibatch"):
        adapter.augment(actions=actions)


def test_amp_observations_and_native_teacher_state_are_separate_from_policy_storage():
    disc = torch.randn(7, 3, 67)
    demo = torch.randn_like(disc)
    obs = {"policy": torch.randn(7, 1161), "critic": torch.randn(7, WIDTHS[2]),
           "disc": disc, "disc_demo": demo}
    native = amp_native_observations(obs)
    assert set(native.keys()) == {"disc", "disc_demo"}
    assert native.batch_size == torch.Size([7])
    torch.testing.assert_close(native["disc"], disc, rtol=0, atol=0)
    torch.testing.assert_close(native["disc_demo"], demo, rtol=0, atol=0)
    agent = {"algorithm": {"amp_cfg": {"loss_type": "LSGAN", "amp_discriminator": {"task_style_lerp": .6}}},
             "obs_groups": {"discriminator": ["disc"], "discriminator_demonstration": ["disc_demo"]}}
    teacher = {"amp_discriminator_state_dict": {"weight": torch.ones(3)},
               "amp_discriminator_normalizer_state_dict": {"mean": torch.randn(67)}}
    with torch.inference_mode():
        setup = native_amp_setup(agent, teacher, obs, step_dt=.02)
    assert setup["amp_cfg"] == {"loss_type": "LSGAN", "amp_discriminator": {"task_style_lerp": .6},
                                "disc_obs_dim": 67, "disc_obs_steps": 3, "step_dt": .02}
    setup["amp_state"]["amp_discriminator_state_dict"]["weight"].zero_()
    assert teacher["amp_discriminator_state_dict"]["weight"].eq(1).all()
    setup["amp_cfg"]["amp_discriminator"]["task_style_lerp"] = 0.
    assert agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"] == .6
    assert "disc_obs_dim" not in agent["algorithm"]["amp_cfg"]
    with pytest.raises(ValueError, match="AMP"):
        amp_native_observations({**obs, "disc_demo": demo[:, :, :-1]})


def test_symmetry_cache_safe_for_diagnostics_before_gradient_update():
    adapter, inputs, _ = _rows()
    # A diagnostics pass must not poison the cached sign/index tensors used
    # later by the real PPO mirror-action loss.
    with torch.inference_mode():
        adapter.augment(obs=inputs[0][:2], actions=torch.ones(2, 23, dtype=torch.float64))
    action = torch.randn(2, 23, dtype=torch.float64, requires_grad=True)
    _, mirrored = adapter.augment(actions=action)
    mirrored[2:].square().sum().backward()
    torch.testing.assert_close(action.grad, 2 * action.detach())


def test_real_native_ppo_and_ppoamp_callback_sequence_routes_backpack_unchanged():
    # Both the critic-only native PPO and the regular native PPOAMP path use
    # an observations/actions call followed by an actions-only target call.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[3] / "rsl_rl"))
    from native_ppo import build_native_learner
    from rsl_rl.modules.amp import AMPDiscriminator

    class Actor(torch.nn.Module):
        width = 1161
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(self.width, 23)
        def forward(self, obs):
            return self.net(obs)

    cfg = dict(disc_obs_dim=67, disc_obs_steps=3, step_dt=.02,
               disc_obs_buffer_size=8, loss_type="LSGAN", grad_penalty_scale=10.,
               disc_trunk_weight_decay=.001, disc_linear_weight_decay=.1,
               disc_learning_rate=.0001, disc_max_grad_norm=1.,
               amp_discriminator=dict(hidden_dims=[8], activation="elu",
                                      style_reward_scale=1.5, task_style_lerp=.6))
    groups = {"discriminator": ["disc"], "discriminator_demonstration": ["disc_demo"]}
    teacher = AMPDiscriminator(67, 3, groups, **cfg["amp_discriminator"])
    state = dict(amp_discriminator_state_dict=teacher.state_dict(),
                 amp_discriminator_normalizer_state_dict=teacher.disc_obs_normalizer.state_dict())
    adapter, inputs, _ = _rows()
    counts = tuple(len(x) for x in inputs)
    observed = []

    def symmetry(obs=None, actions=None, env=None):
        if obs is not None:
            observed.append(obs["task_id"].squeeze(-1).clone())
        aug, act = adapter.augment(obs=obs, actions=actions, env=env)
        if obs is None:
            backpack = observed[-1] == 1
            torch.testing.assert_close(act[len(actions):][backpack], actions[backpack], atol=0, rtol=0)
            observed.append("target")
        return aug, act

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        learner = build_native_learner(Actor(), counts, WIDTHS, cfg, state, symmetry,
                                       "cpu", steps=2, num_epochs=1, num_minibatches=2)
        for critic_only in (True, False):
            learner.configure_critic_only(critic_only)
            obs = adapter.concat(inputs)
            with torch.inference_mode():
                for _ in range(2):
                    learner.act(obs)
                    n = sum(counts)
                    disc = torch.randn(counts[-1], 3, 67)
                    amp = amp_native_observations({"disc": disc, "disc_demo": disc + .5})
                    learner.process_mixed_env_step(obs, torch.randn(n), torch.zeros(n, dtype=torch.bool),
                                                   torch.zeros(n, dtype=torch.bool), amp)
            metrics = learner.compute_returns_and_update(obs, critic_only=critic_only)
            assert metrics["critic_only"] is critic_only
            assert adapter._pending_action_mask is None
    finally:
        torch.set_num_threads(previous_threads)
    assert len(observed) == 8
    assert all(observed[i] == "target" for i in range(1, len(observed), 2))
