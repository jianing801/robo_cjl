"""CPU contracts for paper-inspired spatial, not serial, morphology training.

No simulator imports. Old banks and checkpoints are reference inputs only.
"""
from collections import Counter
import hashlib
import json

KIND = "rpo_spatial_morphology_v1"


def make_layout(bank, num_envs=2048, design_ids=None):
    designs = [d for d in bank["designs"] if d["split"] == "train"]
    if design_ids is not None:
        if len(set(design_ids)) != len(design_ids):
            raise ValueError("duplicate design")
        lookup = {d["id"]: d for d in designs}
        designs = [lookup[key] for key in design_ids]
    if not designs or num_envs < len(designs):
        raise ValueError("every training morphology requires at least one environment")
    widths = {len(d["context"]) for d in designs}
    orders = {tuple(d["graph"]["joint_names"]) for d in designs}
    width = bank.get("context_width", 22)
    if type(num_envs) is not int or type(width) is not int or width < 1:
        raise ValueError("integer environment/context dimensions required")
    if widths != {width} or len(orders) != 1:
        raise ValueError("incompatible semantic/action/context layout")
    assignment = [i % len(designs) for i in range(num_envs)]
    return dict(kind=KIND, num_envs=num_envs, design_ids=[d["id"] for d in designs],
        env_design_index=assignment, counts=dict(Counter(designs[i]["id"] for i in assignment)),
        replicate_physics=False, random_asset_choice=False, morphology_fixed_per_env=True,
        no_holdout_training=True, context_width=width)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def balanced_indices(env_groups, steps, batches, *, device="cpu"):
    """Use every on-policy sample once; distribute each morphology across batches.

    No replay, importance weights, per-design advantage scaling or reward edits.
    """
    import torch
    groups = torch.as_tensor(env_groups, device=device, dtype=torch.long)
    n = len(groups)
    if steps < 1 or batches < 1 or steps * n % batches:
        raise ValueError("rollout must divide exactly into source mini-batches")
    bins = [[] for _ in range(batches)]
    cursor = 0
    for group in groups.unique(sorted=True):
        envs = (groups == group).nonzero().flatten()
        indices = (torch.arange(steps, device=device)[:, None] * n + envs[None, :]).flatten()
        indices = indices[torch.randperm(len(indices), device=device)]
        for b in range(batches):
            offset = (b - cursor) % batches
            bins[b].append(indices[offset::batches])
        cursor = (cursor + len(indices)) % batches
    result = [torch.cat(parts) for parts in bins]
    return [x[torch.randperm(len(x), device=device)] for x in result]


def envelope_limit(speed, table, peak):
    """Same symmetric piecewise-linear joint-side envelope as native actuator."""
    import torch
    table = torch.as_tensor(table, device=speed.device, dtype=speed.dtype)
    if (table.ndim != 2 or table.shape[1] != 2 or len(table) < 2
            or not torch.isfinite(table).all() or (table < 0).any()
            or not (table[1:, 0] > table[:-1, 0]).all()):
        raise ValueError("invalid motor curve")
    return interpolate_validated_envelope(speed, table, peak)


def interpolate_validated_envelope(speed, table, peak):
    """GPU hot path: table validated/cached once; no per-step host sync."""
    import torch
    x = speed.abs().clamp(max=table[-1, 0])
    hi = torch.searchsorted(table[:, 0].contiguous(), x.contiguous()).clamp(1, len(table)-1)
    lo = hi - 1
    limit = table[lo, 1] + (x-table[lo, 0])*(table[hi, 1]-table[lo, 1])/(table[hi, 0]-table[lo, 0])
    return torch.minimum(limit, torch.as_tensor(peak, device=x.device, dtype=x.dtype))
