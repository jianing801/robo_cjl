"""Pure tensor helpers shared by the RPO run/stairs teacher-distillation flow."""

from __future__ import annotations

from collections.abc import Sequence

import torch


FRAME_WIDTH = 78
CANONICAL_HISTORY = 10
CANONICAL_POLICY_WIDTH = FRAME_WIDTH * CANONICAL_HISTORY
TASK_COUNT = 2
TASK_CONTEXT_WIDTH = TASK_COUNT
STUDENT_POLICY_WIDTH = CANONICAL_POLICY_WIDTH + TASK_CONTEXT_WIDTH
ACTION_WIDTH = 23

TASK_TO_ID = {"run": 0, "stairs": 1}
POLICY_TERM_WIDTHS = (3, 3, 3, 23, 23, 23)
POLICY_TERM_NAMES = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")


def source_teacher_inputs(canonical, task):
    """Ten real frame-major frames -> unchanged original teacher layouts."""
    if canonical.ndim != 2 or canonical.shape[1] != CANONICAL_POLICY_WIDTH:
        raise ValueError("teacher adapter requires ten real canonical frames")
    if task == "stairs":
        return canonical
    if task == "run":
        frames = canonical.reshape(-1, CANONICAL_HISTORY, FRAME_WIDTH)[:, -3:]
        return torch.cat([term.flatten(1) for term in frames.split(POLICY_TERM_WIDTHS, dim=-1)], dim=-1)
    raise ValueError("unknown teacher task")


def fixed_teacher_execution_mask(task_ids, fraction):
    """Fixed per-environment teacher/student roles, not action interpolation."""
    if not 0 <= fraction <= 1 or task_ids.ndim != 1 or not torch.all((task_ids == 0) | (task_ids == 1)):
        raise ValueError("invalid role fraction or task IDs")
    mask = torch.zeros_like(task_ids, dtype=torch.bool)
    for task_id in (0, 1):
        indices = (task_ids == task_id).nonzero().flatten()
        count = int(indices.numel() * fraction)
        mask[indices[:count]] = True
    return mask


def validate_fixed_task_columns(*, curriculum, num_cols, terrain_names, proportions):
    """Column indices are task IDs only with deterministic terrain generation.

    Isaac's non-curriculum generator RANDOMLY chooses the geometry for each
    column. Disabling level progression must not disable column assignment.
    """
    if not curriculum:
        raise ValueError("fixed RUN/STAIRS task IDs require deterministic curriculum terrain columns")
    if num_cols != 2 or list(terrain_names) != ["flat_run", "up_stairs"]:
        raise ValueError("expected exactly column 0=flat_run, column 1=up_stairs")
    if len(proportions) != 2 or min(proportions) <= 0 or abs(proportions[0] - proportions[1]) > 1e-8:
        raise ValueError("two task columns require equal positive terrain proportions")


def decode_policy_frames(policy_obs: torch.Tensor, *, history: int, layout: str) -> torch.Tensor:
    """Decode source history without changing the teacher's own input.

    Isaac ObservationManager flattens EACH term's history before concatenation;
    Direct BaseEnv flattens the whole [time, frame] buffer. Width alone cannot
    distinguish these contracts. Both buffers order time oldest to newest.
    """
    if policy_obs.ndim != 2 or policy_obs.shape[1] != history * FRAME_WIDTH or history < 1:
        raise ValueError("source observation width/history mismatch")
    if layout == "frame_major":
        return policy_obs.reshape(-1, history, FRAME_WIDTH)
    if layout == "term_major":
        terms = policy_obs.split([history * width for width in POLICY_TERM_WIDTHS], dim=1)
        return torch.cat([term.reshape(-1, history, width) for term, width in zip(terms, POLICY_TERM_WIDTHS)], dim=-1)
    raise ValueError(f"unknown source history layout: {layout}")


class PolicyHistory:
    """Collect ten REAL consecutive frames per environment, reset at episode boundaries.

    Repeating the first frame is permitted only while a new episode's buffer
    fills, matching Isaac CircularBuffer startup semantics. Short teacher inputs
    must never be padded independently on every simulation step.
    """

    def __init__(self, num_envs: int, *, device: torch.device | str):
        self.frames = torch.zeros(num_envs, CANONICAL_HISTORY, FRAME_WIDTH, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def append(self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        if frame.shape != (self.frames.shape[0], FRAME_WIDTH):
            raise ValueError("one current frame per environment is required")
        reset = ~self.initialized
        if reset_mask is not None:
            if reset_mask.shape != reset.shape:
                raise ValueError("reset mask batch mismatch")
            reset = reset | reset_mask.to(device=reset.device, dtype=torch.bool)
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = frame
        self.frames[reset] = frame[reset, None, :]
        self.initialized[:] = True
        return self.frames.flatten(1).clone()


def split_by_environment(environment_ids: torch.Tensor, fraction: float, generator: torch.Generator):
    """Hold out complete environment streams so overlapping histories cannot leak."""
    if environment_ids.ndim != 1 or not 0 < fraction < 0.5:
        raise ValueError("invalid environment IDs or validation fraction")
    ids = torch.unique(environment_ids.cpu())
    if ids.numel() < 2:
        raise ValueError("collect at least two environments for held-out-stream validation")
    shuffled = ids[torch.randperm(ids.numel(), generator=generator)]
    count = min(ids.numel() - 1, max(1, round(ids.numel() * fraction)))
    validation = torch.isin(environment_ids.cpu(), shuffled[:count])
    return (~validation).nonzero().flatten(), validation.nonzero().flatten()


def require_matching_contracts(run_contract: dict, stairs_contract: dict) -> None:
    """Prevent distillation of identically sized, semantically different actions."""
    required = ("joint_names", "default_joint_pos", "action_scale", "step_dt", "urdf_sha256", "observation_scales")
    optional = ("default_joint_vel", "coordinates", "action_semantics", "mtg")
    required += tuple(field for field in optional if field in run_contract or field in stairs_contract)
    for field in required:
        if field not in run_contract or field not in stairs_contract:
            raise ValueError(f"teacher contract is missing {field}")
        if run_contract[field] != stairs_contract[field]:
            raise ValueError(f"teachers disagree on {field}; reconcile the physical/I-O contract first")


def task_ids_to_one_hot(task_ids: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Map run/stairs IDs to a stable two-column one-hot task context."""
    if task_ids.ndim != 1:
        raise ValueError(f"task_ids must be one-dimensional, got {tuple(task_ids.shape)}")
    if task_ids.numel() and (torch.any(task_ids < 0) or torch.any(task_ids >= TASK_COUNT)):
        raise ValueError("task_ids must contain only 0 (run) or 1 (stairs)")
    result = torch.nn.functional.one_hot(task_ids.to(torch.long), num_classes=TASK_COUNT)
    return result.to(dtype=dtype) if dtype is not None else result


def canonicalize_policy_history(policy_obs: torch.Tensor) -> torch.Tensor:
    """Validate an ALREADY canonical frame-major ten-frame history.

    Raw manager-based AMP observations require decode_policy_frames plus a
    streaming PolicyHistory. Old shape-only, three-frame padding is rejected.
    """
    if policy_obs.ndim != 2:
        raise ValueError(f"policy_obs must have shape [batch, features], got {tuple(policy_obs.shape)}")
    width = policy_obs.shape[-1]
    if width % FRAME_WIDTH:
        raise ValueError(f"policy observation width must be a multiple of {FRAME_WIDTH}, got {width}")
    history = width // FRAME_WIDTH
    if history != CANONICAL_HISTORY:
        raise ValueError("expected ten actual canonical frames; raw teacher histories need the streaming adapter")
    return policy_obs


def make_student_policy_input(policy_obs: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    """Build the shared actor input: 10x78 proprioception plus explicit task ID."""
    canonical = canonicalize_policy_history(policy_obs)
    if task_ids.shape != (canonical.shape[0],):
        raise ValueError("task_ids batch dimension must match policy_obs")
    task_context = task_ids_to_one_hot(task_ids, dtype=canonical.dtype).to(canonical.device)
    return torch.cat((canonical, task_context), dim=-1)


def task_speed_bounds(
    task_ids: torch.Tensor,
    run_range: Sequence[float],
    stairs_range: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment lower/upper forward-command bounds."""
    if len(run_range) != 2 or len(stairs_range) != 2:
        raise ValueError("speed ranges must each contain a (low, high) pair")
    ids = task_ids.to(torch.long)
    if ids.ndim != 1 or (ids.numel() and (torch.any(ids < 0) or torch.any(ids >= TASK_COUNT))):
        raise ValueError("task_ids must be a 1-D tensor of 0 (run) or 1 (stairs)")
    run = ids == TASK_TO_ID["run"]
    low = torch.where(
        run,
        torch.as_tensor(run_range[0], device=ids.device, dtype=torch.float32),
        torch.as_tensor(stairs_range[0], device=ids.device, dtype=torch.float32),
    )
    high = torch.where(
        run,
        torch.as_tensor(run_range[1], device=ids.device, dtype=torch.float32),
        torch.as_tensor(stairs_range[1], device=ids.device, dtype=torch.float32),
    )
    return low, high
