from copy import deepcopy
import pytest
import torch
from robolab.configurable_morphology import select_grid, evenly_spaced
from robolab.spatial_morphology import make_layout, balanced_indices


def config():
    return dict(num_envs=101, length_count=3, thigh_range_m=[.25, .4], calf_range_m=[.3, .45],
                knee_motor_count=2, ankle_motor_count=3, motor_seed=42)


def test_full_cartesian_and_uniform_points():
    g = select_grid(config(), ["k1", "k2", "k3"], ["a1", "a2", "a3", "a4"])
    assert g["design_count"] == 54
    assert len({d["id"] for d in g["designs"]}) == 54
    assert g["thigh_values"] == pytest.approx([.25, .325, .4])
    assert g["calf_values"] == pytest.approx([.3, .375, .45])
    assert g["context_width"] == 23
    assert all(d["split"] == "train" for d in g["designs"])


def test_motor_selection_reproducible_without_replacement_and_independent():
    c = config()
    a = select_grid(c, ["k1", "k2", "k3"], ["a1", "a2", "a3", "a4"])
    b = select_grid(c, ["k3", "k2", "k1"], ["a4", "a3", "a2", "a1"])
    assert a == b
    assert len(set(a["knees"])) == 2 and len(set(a["ankles"])) == 3
    c["knee_motor_count"] = 1
    assert select_grid(c, ["k1", "k2", "k3"], ["a1", "a2", "a3", "a4"])["ankles"] == a["ankles"]


@pytest.mark.parametrize("key,value", [("length_count", 0), ("num_envs", 5), ("motor_seed", 1.2),
    ("knee_motor_count", 4), ("ankle_motor_count", True), ("knee_pool", ["bad"]),
    ("ankle_pool", ["a1", "a1"]), ("thigh_range_m", [.4, .25]), ("typo", 1)])
def test_invalid_rejected(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        select_grid(c, ["k1", "k2", "k3"], ["a1", "a2", "a3", "a4"])


def test_single_length_midpoint_and_flexible_env_context():
    assert evenly_spaced([.25, .4], 1) == pytest.approx([.325])
    c = config(); c.update(length_count=1, knee_motor_count=1, num_envs=17)
    g = select_grid(c, ["k1"], ["a1", "a2", "a3"])
    designs = [dict(d, context=[0.]*g["context_width"], graph={"joint_names": ["j"]}) for d in g["designs"]]
    layout = make_layout(dict(designs=designs, context_width=g["context_width"]), 17)
    assert sorted(layout["counts"].values()) == [5, 6, 6]
    batches = balanced_indices(layout["env_design_index"], 24, 4)
    assert torch.equal(torch.cat(batches).sort().values, torch.arange(17*24))


def test_variable_width_symmetry_preserves_context():
    from types import SimpleNamespace
    from tensordict import TensorDict
    from robolab.spatial_policy import spatial_design_symmetry
    width = 23
    env = SimpleNamespace(unwrapped=SimpleNamespace(spatial_context=torch.zeros(2, width)))
    def mirror(env, obs, actions):
        assert obs["policy"].shape[-1] == 5
        return TensorDict({"policy": torch.cat((obs["policy"], -obs["policy"]))}, batch_size=[4]), None
    env.native_source_symmetry = mirror
    x = torch.arange(56.).reshape(2, 28)
    obs, _ = spatial_design_symmetry(env, TensorDict({"policy": x}, batch_size=[2]))
    assert torch.equal(obs["policy"][:, -width:], x[:, -width:].repeat(2, 1))


@pytest.mark.parametrize("width", [20, 21, 23, 30])
@pytest.mark.parametrize("normalized", [True, False])
def test_variable_policy_width_normalization_roundtrip(width, normalized):
    from tensordict import TensorDict
    from robolab.spatial_policy import SpatialActorCritic
    obs = TensorDict({"policy": torch.randn(4, 10+width), "critic": torch.randn(4, 14+width)}, [4])
    kwargs = dict(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]}, num_actions=23,
        actor_hidden_dims=[16], critic_hidden_dims=[16], actor_obs_normalization=normalized,
        critic_obs_normalization=normalized, native_config=dict(role="stairs", context_width=width, bank_sha256="test"))
    model = SpatialActorCritic(**kwargs)
    model.update_normalization(obs)
    if normalized:
        assert torch.equal(model.actor_obs_normalizer(obs["policy"])[:, -width:], obs["policy"][:, -width:])
    model.act_inference(obs).square().mean().backward()
    clone = SpatialActorCritic(**kwargs)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.act_inference(obs), model.act_inference(obs))
    bad = deepcopy(model.state_dict()); bad["_extra_state"]["config"]["bank_sha256"] = "other"
    with pytest.raises(ValueError, match="incompatible"):
        clone.load_state_dict(bad)
