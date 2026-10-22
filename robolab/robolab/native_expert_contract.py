"""CPU-only contracts for independent source-native morphology experts.

No simulator imports, launches, reward changes or implicit legacy migration.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

RECIPE = "rpo_native_independent_experts_v2"
TASKS = {"run": "RPO-AMP", "stairs": "RPO-Rough"}
ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    paths = [
        "scripts/tools/train_multimorph_expert_block.py",
        "scripts/tools/native_multimorph_runtime.py",
        "robolab/native_expert_contract.py", "robolab/native_expert_policy.py",
        "robolab/native_ground.py", "data/native_probe_ground.usda",
        "robolab/amp_reference_retarget.py", "scripts/tools/hhtools_amp_backend.py",
        "robolab/amp_reference_serialization.py",
        "robolab/amp_reference_source_audit.py",
        "scripts/tools/prepare_amp_retarget_control.py",
        "robolab/morph_expert_runner.py", "robolab/native_checkpoint_contract.py", "robolab/physical_snapshot.py",
        "robolab/morph_experiments.py", "scripts/tools/run_multimorph_experts.py",
        "robolab/tasks/manager_based/amp/rpo_amp_env_cfg.py",
        "robolab/tasks/manager_based/amp/amp_env_cfg.py",
        "robolab/tasks/manager_based/amp/amp_env.py",
        "robolab/tasks/manager_based/amp/animation_env.py",
        "robolab/tasks/manager_based/amp/agents/rpo_amp_agent_cfg.py",
        "robolab/tasks/manager_based/amp/mdp/symmetry/rpo.py",
        "robolab/tasks/direct/base/base_env.py",
        "robolab/tasks/direct/base/base_config.py",
        "robolab/tasks/direct/base/rpo_env_cfg.py",
        "robolab/tasks/direct/base/agents/rpo_agent_cfg.py",
        "../rsl_rl/rsl_rl/algorithms/ppo.py", "../rsl_rl/rsl_rl/algorithms/ppo_amp.py",
        "../rsl_rl/rsl_rl/modules/actor_critic.py", "../rsl_rl/rsl_rl/modules/amp.py",
    ]
    # Include reward/data-loader dependencies, not just the launcher's SHA.
    for folder in ("robolab/tasks/manager_based/amp", "robolab/tasks/direct/base", "../rsl_rl/rsl_rl"):
        paths += [folder + "/" + p.relative_to(ROOT / folder).as_posix()
                  for p in sorted((ROOT / folder).rglob("*.py"))]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(set(paths))}


def validate_native_bank(bank):
    training = bank.get("training", {})
    if training.get("recipe") != RECIPE:
        raise ValueError("legacy mixed-runtime bank is retired; prepare a NEW native-v2 bank, never resume the deleted experiment")
    if bank.get("num_envs") != 2048:
        raise ValueError("formal native experts require the user-selected 2048 environments")
    if training.get("reference_retargeting") and not training.get("retargeted_references"):
        raise ValueError("retargeting requested without a reviewed reference library")
    if training.get("source_hashes") != source_identity():
        raise ValueError("native training implementation changed since bank preparation; do not bypass identity checks")


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if callable(value):
        return value.__module__ + ":" + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_digest(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_normalizer_coverage(bank, schedule, source_agent, history_length):
    """Fail closed rather than silently changing the original AMP sample cap.

    Current source AMP uses until=1e8 and updates once per discriminator mini-
    batch with all history frames. Cover every design before that cap is hit.
    This check changes no normalizer, optimizer or curriculum setting.
    """
    import ast
    source = ast.parse((ROOT.parent / "rsl_rl/rsl_rl/modules/amp.py").read_text(encoding="utf8"))
    assignments = [node for node in ast.walk(source) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "disc_obs_normalizer" for t in node.targets)]
    caps = [ast.literal_eval(kw.value) for assignment in assignments for n in ast.walk(assignment.value)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "EmpiricalNormalization"
            for kw in n.keywords if kw.arg == "until"]
    if len(caps) != 1:
        raise ValueError("cannot establish original AMP normalizer coverage; inspect source before training")
    cap = caps[0]
    per_update = bank["num_envs"] * source_agent["num_steps_per_env"] * history_length * source_agent["algorithm"]["num_learning_epochs"]
    expected = {r["id"] for r in bank["designs"] if r["split"] == "train"}
    seen, count = set(), 0
    for block in schedule:
        if count >= cap:
            break
        seen.add(block["design_id"])
        count += per_update * block["updates"]
    if seen != expected:
        raise ValueError(f"unsafe morphology schedule: original AMP normalizer freezes after seeing {len(seen)}/{len(expected)} designs; "
                         "review the design schedule, do NOT silently raise the source sample cap")
    return {"source_sample_cap": cap, "all_training_designs_seen_before_freeze": True}


def extend_agent(source, *, task, bank_sha):
    """Only class/identity and the shape-preserving symmetry adapter may change.

    In particular do not override std, normalization, lr, epochs, minibatches,
    rollout length, symmetry enable/coefficient or AMP parameters.
    """
    if task not in TASKS:
        raise ValueError("unknown native expert")
    agent = deepcopy(source)
    policy = agent["policy"]
    policy["class_name"] = "robolab.native_expert_policy:NativeMorphActorCritic"
    policy["native_config"] = {"recipe": RECIPE, "role": task, "bank_sha256": bank_sha}
    algorithm = agent["algorithm"]
    # Installed Isaac config exposes newer-version fields absent from this
    # checkout's constructors. Only their equivalent no-op defaults are safe.
    for key, expected in (("optimizer", "adam"), ("share_cnn_encoders", False)):
        if key in algorithm:
            if algorithm[key] != expected:
                raise ValueError(f"unsupported source {key}; refusing to silently change its meaning")
            algorithm.pop(key)
    if task == "run":
        algorithm["class_name"] = "robolab.morph_expert_runner:NativePPOAMP"
    symmetry = agent["algorithm"].get("symmetry_cfg")
    if symmetry is not None:
        symmetry["data_augmentation_func"] = "robolab.native_expert_policy:native_design_symmetry"
    return agent
