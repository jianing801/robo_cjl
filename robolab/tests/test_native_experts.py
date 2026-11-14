"""CPU contract tests only; not evidence of trained gait or Isaac integration."""
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from robolab.native_expert_contract import RECIPE, TASKS, extend_agent, validate_native_bank, source_identity
from robolab.native_expert_policy import NativeMorphActorCritic, PrefixNormalization, append_context, native_design_symmetry
from rsl_rl.networks import EmpiricalNormalization

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ("run", "stairs"))
def test_agent_keeps_all_source_hyperparameters(task):
    source = {"policy": {"class_name": "ActorCritic", "init_noise_std": 1.,
               "actor_obs_normalization": True, "critic_obs_normalization": True},
        "algorithm": {"class_name": "PPOAMP" if task == "run" else "PPO",
            "learning_rate": 1e-4, "num_learning_epochs": 5, "num_mini_batches": 4,
            "symmetry_cfg": {"use_mirror_loss": True, "mirror_loss_coeff": .2,
                             "data_augmentation_func": "original:mirror"},
            "amp_cfg": {"disc_learning_rate": 1e-4, "task_style_lerp": .6}},
        "num_steps_per_env": 24, "save_interval": 100, "obs_groups": {"policy": ["policy"]}}
    saved = deepcopy(source)
    result = extend_agent(source, task=task, bank_sha="bank")
    result["policy"].pop("native_config")
    result["policy"]["class_name"] = source["policy"]["class_name"]
    result["algorithm"]["symmetry_cfg"]["data_augmentation_func"] = "original:mirror"
    result["algorithm"]["class_name"] = source["algorithm"]["class_name"]
    assert result == saved and source == saved


def test_newer_config_compatibility_does_not_silently_change_optimizer():
    source = {"policy": {}, "algorithm": {"optimizer": "adam", "share_cnn_encoders": False}}
    assert "optimizer" not in extend_agent(source, task="stairs", bank_sha="b")["algorithm"]
    source["algorithm"]["optimizer"] = "adamw"
    with pytest.raises(ValueError, match="refusing"):
        extend_agent(source, task="stairs", bank_sha="b")


def test_native_amp_does_not_override_original_transition_or_update():
    from robolab.morph_expert_runner import NativePPOAMP
    from rsl_rl.algorithms import PPOAMP
    assert NativePPOAMP.process_env_step is PPOAMP.process_env_step
    assert NativePPOAMP.update is PPOAMP.update


def test_full_native_amp_algorithm_factory_accepts_suffix_and_original_symmetry():
    from robolab.morph_expert_runner import NativeMorphRunRunner, NativePPOAMP
    from rsl_rl.algorithms import PPOAMP
    callback = load_amp_mirror()
    cfg = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(history_length=3),
                                                       critic=SimpleNamespace(history_length=3)))
    env = SimpleNamespace(num_envs=4, num_actions=23, native_source_symmetry=callback,
                          cfg=cfg, env=SimpleNamespace(unwrapped=SimpleNamespace(step_dt=.02)))
    env.unwrapped = SimpleNamespace(cfg=cfg)
    obs = append_context(TensorDict({"policy": torch.randn(4, 234), "critic": torch.randn(4, 243),
                                    "disc": torch.randn(4, 3, 67), "disc_demo": torch.randn(4, 3, 67)}, [4]), torch.ones(22))
    source = {"policy": {"class_name": "ActorCritic", "init_noise_std": 1., "activation": "elu",
                         "actor_hidden_dims": [32], "critic_hidden_dims": [32],
                         "actor_obs_normalization": True, "critic_obs_normalization": True},
        "algorithm": {"class_name": "PPOAMP", "rnd_cfg": None,
            "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-4,
            "symmetry_cfg": {"use_data_augmentation": True, "use_mirror_loss": True,
                             "mirror_loss_coeff": .2, "data_augmentation_func": callback},
            "amp_cfg": {"disc_obs_buffer_size": 100, "loss_type": "LSGAN",
                        "disc_trunk_weight_decay": .001, "disc_linear_weight_decay": .1,
                        "disc_learning_rate": .0001, "disc_max_grad_norm": 1.,
                        "grad_penalty_scale": 10., "amp_discriminator": {"hidden_dims": [32], "activation": "elu",
                            "style_reward_scale": 1.5, "task_style_lerp": .6}}},
        "num_steps_per_env": 4,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"],
                       "discriminator": ["disc"], "discriminator_demonstration": ["disc_demo"]}}
    agent = extend_agent(source, task="run", bank_sha="bank")
    runner = NativeMorphRunRunner.__new__(NativeMorphRunRunner)
    runner.cfg = agent; runner.policy_cfg = agent["policy"]; runner.alg_cfg = agent["algorithm"]
    runner.env = env; runner.device = "cpu"; runner.multi_gpu_cfg = None
    alg = runner._construct_algorithm(obs)
    assert type(alg) is NativePPOAMP and alg.process_env_step.__func__ is PPOAMP.process_env_step
    assert alg.num_learning_epochs == 5 and alg.num_mini_batches == 4
    assert alg.symmetry["use_mirror_loss"] and alg.symmetry["mirror_loss_coeff"] == .2
    assert alg.policy.act_inference(obs).shape == (4, 23)


def test_old_mixed_bank_and_changed_source_are_rejected():
    with pytest.raises(ValueError, match="retired"):
        validate_native_bank({"training": {}})
    bank = {"training": {"recipe": RECIPE, "source_hashes": source_identity()}, "num_envs": 2048}
    validate_native_bank(bank)
    bank["training"]["source_hashes"][next(iter(bank["training"]["source_hashes"]))] = "changed"
    with pytest.raises(ValueError, match="implementation changed"):
        validate_native_bank(bank)


def test_normalizer_guard_rejects_uncovered_designs_without_changing_source_cap():
    from robolab.native_expert_contract import require_normalizer_coverage
    bank = {"num_envs": 2048, "designs": [{"id": str(i), "split": "train"} for i in range(20)]}
    source = {"num_steps_per_env": 24, "algorithm": {"num_learning_epochs": 5}}
    unsafe = [{"design_id": str(i), "updates": 50} for i in range(20)]
    with pytest.raises(ValueError, match="3/20"):
        require_normalizer_coverage(bank, unsafe, source, 3)
    safe = [{"design_id": str(i), "updates": 4} for i in range(20)]
    assert require_normalizer_coverage(bank, safe, source, 3)["source_sample_cap"] == 1e8


def test_coverage_sweep_then_normal_blocks_preserves_total_budget_and_balance():
    from robolab.morph_experiments import morphology_rows, block_schedule
    from robolab.native_expert_contract import require_normalizer_coverage
    rows = morphology_rows()
    schedule = block_schedule(rows, updates=5000, block_updates=50, seed=42, initial_sweep_updates=4)
    assert len(schedule) == 119
    assert [b["updates"] for b in schedule[:20]] == [4] * 20
    assert len({b["design_id"] for b in schedule[:20]}) == 20
    assert sum(b["updates"] for b in schedule) == 5000
    assert schedule[20]["first_update"] == 80 and schedule[-1]["updates"] == 20
    require_normalizer_coverage({"num_envs": 2048, "designs": rows}, schedule,
        {"num_steps_per_env": 24, "algorithm": {"num_learning_epochs": 5}}, 3)


def test_suffix_does_not_change_or_extend_amp_discriminator():
    obs = TensorDict({"policy": torch.randn(3, 234), "critic": torch.randn(3, 243),
                      "disc": torch.randn(3, 3, 67), "disc_demo": torch.randn(3, 3, 67)}, [3])
    tail = torch.linspace(-1, 1, 22)
    result = append_context(obs, tail)
    assert result["policy"].shape == (3, 256)
    assert result["critic"].shape == (3, 265)
    for key in ("policy", "critic"):
        assert torch.equal(result[key][:, :-22], obs[key])
        assert torch.equal(result[key][:, -22:], tail.expand(3, -1))
    for key in ("disc", "disc_demo"):
        assert torch.equal(result[key], obs[key])
    assert obs["policy"].shape[-1] == 234


def test_native_normalization_matches_original_for_source_features():
    original, adapter = EmpiricalNormalization(7), PrefixNormalization(7)
    x = torch.randn(8, 7) * 3
    context = torch.randn(8, 22)
    full = torch.cat((x, context), -1)
    original.update(x); adapter.update(full)
    assert torch.equal(original(x), adapter(full)[:, :7])
    assert torch.equal(context, adapter(full)[:, 7:])


@pytest.mark.parametrize("normalized", (True, False))
def test_native_model_normalized_inference_and_strict_roundtrip(normalized):
    obs = append_context(TensorDict({"policy": torch.randn(4, 234), "critic": torch.randn(4, 243)}, [4]), torch.ones(22))
    kwargs = dict(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]}, num_actions=23,
        actor_hidden_dims=[32], critic_hidden_dims=[32], init_noise_std=1.,
        actor_obs_normalization=normalized, critic_obs_normalization=normalized,
        native_config={"recipe": RECIPE, "role": "run", "bank_sha256": "test"})
    model = NativeMorphActorCritic(**kwargs)
    model.update_normalization(obs)
    model.act_inference(obs).square().mean().backward()
    assert any(p.grad is not None for p in model.actor.parameters())
    clone = NativeMorphActorCritic(**kwargs)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(clone.act_inference(obs), model.act_inference(obs))
    assert torch.equal(model.std, torch.ones(23))
    bad = deepcopy(model.state_dict()); bad["_extra_state"]["config"]["role"] = "stairs"
    with pytest.raises(ValueError, match="role"):
        clone.load_state_dict(bad)


def load_amp_mirror():
    path = ROOT / "robolab/tasks/manager_based/amp/mdp/symmetry/rpo.py"
    spec = importlib.util.spec_from_file_location("native_test_amp_mirror", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.compute_symmetric_states


def test_original_amp_symmetry_is_not_disabled_and_design_is_unchanged():
    callback = load_amp_mirror()
    env = SimpleNamespace(cfg=SimpleNamespace(observations=SimpleNamespace(
        policy=SimpleNamespace(history_length=3), critic=SimpleNamespace(history_length=3))),
        native_source_symmetry=callback)
    source = TensorDict({"policy": torch.randn(3, 234), "critic": torch.randn(3, 243)}, [3])
    context = torch.randn(22); obs = append_context(source, context); actions = torch.randn(3, 23)
    expected, expected_actions = callback(env, source, actions)
    actual, actual_actions = native_design_symmetry(env, obs, actions)
    assert torch.equal(actual_actions, expected_actions)
    for key in ("policy", "critic"):
        assert torch.equal(actual[key][:, :-22], expected[key])
        assert torch.equal(actual[key][:, -22:], context.expand(6, -1))
    assert native_design_symmetry(env, None, actions)[0] is None


def test_worker_has_no_shared_environment_or_training_parameter_overrides():
    source = (ROOT / "scripts/tools/train_multimorph_expert_block.py").read_text()
    for forbidden in ("from multimorph_runtime", "RPO-MxT-Task-Routed-AMP", "bank[\"evaluation\"]",
                      "init_noise_std=.3", "num_learning_epochs=2", "symmetry_cfg=None"):
        assert forbidden not in source
    tree = ast.parse((ROOT / "scripts/tools/native_multimorph_runtime.py").read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    assert not attrs & {"heading_command", "rel_standing_envs", "position_range", "velocity_range",
                       "episode_length_s", "run_speed_range", "num_learning_epochs", "learning_rate"}
    assert TASKS == {"run": "RPO-AMP", "stairs": "RPO-Rough"}
