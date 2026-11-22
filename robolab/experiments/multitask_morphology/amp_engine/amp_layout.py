"""CPU contracts for spatial mixing inside each rotating AMP group."""
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


def vectorized_balanced_indices(env_groups, steps, batches, *, device="cpu"):
    """Vectorized equivalent of ``balanced_indices`` for large morphology banks.

    Each rollout transition appears exactly once per epoch. Samples are shuffled
    within their morphology and split across minibatches using the same rotating
    offsets as the grouped reference implementation; only the per-morphology
    Python loop is removed.
    """
    import torch
    groups = torch.as_tensor(env_groups, device=device, dtype=torch.long)
    if groups.ndim != 1 or groups.numel() == 0:
        raise ValueError("env_groups must be a nonempty one-dimensional tensor")
    n = groups.numel()
    total = steps * n
    if steps < 1 or batches < 1 or total % batches:
        raise ValueError("rollout must divide exactly into source mini-batches")

    unique_groups, group_index = torch.unique(groups, sorted=True, return_inverse=True)
    group_count = unique_groups.numel()
    env_counts = torch.bincount(group_index, minlength=group_count)
    env_order = torch.argsort(group_index)
    env_starts = torch.cumsum(env_counts, 0) - env_counts
    env_rank = torch.empty_like(group_index)
    env_rank[env_order] = torch.arange(n, device=device) - torch.repeat_interleave(env_starts, env_counts)

    # make_layout assigns environments round-robin, so each morphology has
    # floor(n/G) or ceil(n/G) environments. A padded [G, env, step] tensor
    # lets us shuffle each morphology in parallel instead of looping over G.
    max_envs = (n + group_count - 1) // group_count
    env_slots = torch.full((group_count, max_envs), -1, device=device, dtype=torch.long)
    env_slots[group_index, env_rank] = torch.arange(n, device=device)
    valid_envs = env_slots >= 0
    time = torch.arange(steps, device=device)
    sample_matrix = (time[None, None, :] * n + env_slots[:, :, None]).reshape(group_count, -1)
    valid_matrix = valid_envs[:, :, None].expand(-1, -1, steps).reshape(group_count, -1)

    random_scores = torch.rand(sample_matrix.shape, device=device)
    random_scores.masked_fill_(~valid_matrix, 2.)
    order = torch.argsort(random_scores, dim=1)
    ordered_samples = sample_matrix.gather(1, order)
    valid_samples = valid_matrix.gather(1, order)

    sample_counts = env_counts * steps
    group_offsets = (torch.cumsum(sample_counts, 0) - sample_counts).remainder(batches)
    positions = torch.arange(sample_matrix.shape[1], device=device)[None, :]
    batch_ids = (positions + group_offsets[:, None]).remainder(batches)
    flat_valid = valid_samples.reshape(-1)
    flat_samples = ordered_samples.reshape(-1)
    flat_batches = batch_ids.expand(group_count, -1).reshape(-1)
    bins = [flat_samples[flat_valid & (flat_batches == batch)] for batch in range(batches)]
    return [values[torch.randperm(values.numel(), device=device)] for values in bins]


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
