"""Standalone Rough must work without uploading historical AMP/MxT helpers."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from robolab.rough_contract import digest, interpolate, extend_agent, write_new

ROOT = Path(__file__).resolve().parents[1]
FILES = ["robolab/rough_contract.py", "robolab/rough_geometry.py", "robolab/configurable_morphology.py",
         "robolab/spatial_morphology.py", "robolab/spatial_policy.py", "scripts/tools/prepare_spatial_grid.py",
         "scripts/tools/train_spatial_rough.py", "scripts/tools/spatial_rough_runtime.py"]
ALLOWED = {"robolab.rough_contract", "robolab.rough_geometry", "robolab.configurable_morphology",
           "robolab.spatial_morphology", "robolab.spatial_policy", "robolab.physical_snapshot", "robolab.rough_gpu",
           "robolab.tasks", "robolab.tasks.direct.base.base_env", "robolab.actuators.torque_speed_envelope"}


def test_no_historical_local_imports():
    for filename in FILES:
        for node in ast.walk(ast.parse((ROOT/filename).read_text(encoding="utf8"))):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.startswith("robolab"):
                    assert name in ALLOWED, (filename, name)


def test_adapter_changes_only_policy_identity_and_mirror_callback():
    source = dict(policy=dict(class_name="ActorCritic", actor_hidden_dims=[512, 256], init_noise_std=1.),
        algorithm=dict(class_name="PPO", learning_rate=1e-3, optimizer="adam", share_cnn_encoders=False,
            symmetry_cfg=dict(use_mirror_loss=True, mirror_loss_coeff=.1, data_augmentation_func="original")),
        num_steps_per_env=24, seed=42)
    before = deepcopy(source)
    result = extend_agent(source, bank_sha="test", context_width=23)
    assert source == before
    assert result["algorithm"]["class_name"] == "PPO"
    assert result["algorithm"]["learning_rate"] == source["algorithm"]["learning_rate"]
    assert result["algorithm"]["symmetry_cfg"]["use_mirror_loss"]
    assert result["policy"]["native_config"]["context_width"] == 23
    assert result["policy"]["actor_hidden_dims"] == source["policy"]["actor_hidden_dims"]


def test_immutable_json_and_pure_helpers(tmp_path):
    p = tmp_path/"result.json"
    write_new(p, dict(a=1))
    with pytest.raises(FileExistsError):
        write_new(p, dict(a=2))
    assert digest(dict(a=1, b=2)) == digest(dict(b=2, a=1))
    assert interpolate([[0, 10], [5, 0]], 2.5) == 5
    assert interpolate([[0, 10], [5, 0]], 6) == 0
