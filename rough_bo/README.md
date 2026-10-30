# Rough BO: 12k shared policy + per-design adaptation

This folder contains the BO workflow that starts from the completed 12,000-update
morphology-conditioned Rough policy, fine-tunes a separate policy for each
candidate design, then scores it by forward-speed tracking error. It is isolated
from the earlier frozen-policy pilot. Nothing in this folder starts training
unless `bo_finetune_controller.py` is explicitly launched.

## Run definition

| Stage | Configuration |
|---|---|
| Initial designs | 50 mixed continuous/categorical OCD rows in `ocd_mixed_design_50_full_cooling/ocd_design_50.csv` |
| Adaptive search | 50 sequential Optuna `GPSampler` proposals by default |
| Variables | Thigh 0.25–0.40 m; calf 0.30–0.45 m; 3 knee motors × 3 ankle motors |
| Per-design adaptation | 4096 copies of that one robot; exactly 200 PPO updates |
| Initialization | Same base actor, critic and normalization weights for every candidate; fresh Adam state, initialized at the checkpoint's saved LR (`5.0625e-5`) |
| Fine-tuning task | Fixed 10 cm stairs; forward command 0.5 m/s; common seed 42; source Rough noise/reset events retained |
| BO score | After fine-tuning, deterministic 20 s test; minimize fixed-horizon mean absolute forward-speed error (m/s) |

There are 100 candidate evaluations by default, or 20,000 PPO update rounds
across candidates. These are sequential runs. A fall does not shorten the score:
the unobserved part of the 20 s horizon is scored as if forward speed were zero;
fall and feasibility are also saved separately.

Each candidate starts from the exact same base policy weights. Its Adam moments
are reset so one candidate's optimization history cannot leak into another.
The old six observations came from a zero-update frozen-policy objective, so
they are archived for context but are not imported into this study.

## Required external files

Large, source-identified assets stay in their original locations and are
validated by SHA256 at launch:

- Base checkpoint `model_11999.pt`: SHA256
  `a0cd01d59c47588eb23a52b440beac41d626f7b4a85473fb7135743930face26`.
- Replay bank `bank.json`: SHA256
  `25c00be6b64257ef5a049f34d2046895982b253c77495b0dfb0e4652116af2ac`.
- Checkpoint-matched Rough source tree, motor snapshots, URDF template and
  sibling `rsl_rl` source tree. The controller checks source identities before
  launching each candidate.

Their current Windows locations are configured as defaults in
`bo_finetune_controller.py`. If they move, pass `--checkpoint`, `--bank`, and
`--code-root`. The local replay bank reproduces the server design semantics and
source identities, but its regenerated URDF/report bytes are not byte-identical
to the server files; results must be described as local source-matched replay.

## Launch

Open PowerShell in this folder and run:

```powershell
$run = Join-Path (Get-Location) ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
& "C:\Roboparty\env_isaaclab\Scripts\python.exe" .\bo_finetune_controller.py `
  --output $run `
  --bo-trials 50
```

The default is 50 OCD observations followed by 50 GP proposals. Use
`--bo-trials 0` for only the initial 50. To resume, rerun with the same
`--output` path and add `--resume`; the immutable run contract must still match.
Completed candidate results and incomplete attempt logs are retained in the run
directory.

### Linux server: one foreground launch command

Activate the checkpoint-matched Isaac Lab environment in the repository root:

```bash
cd /data/chenjiale/robo_cjl
conda activate robo_cjl
python rough_bo/start.py
```

Omit `--gpu` to automatically select a GPU with at least 12.5 GiB free and
utilization at most 20%. Among eligible GPUs, select the one with the most free
memory, breaking ties by lower utilization and then lower GPU index.
If none qualifies, print the reason and return to the terminal without training.

To specify a physical GPU index, use:

```bash
python rough_bo/start.py --gpu 5
```

An explicitly chosen GPU must still have at least 12.5 GiB free; its utilization
is displayed but is not subject to the automatic-selection utilization limit.
These are launch-time checks, not a GPU memory reservation.

The launcher uses the current environment's Python, the original server
`model_11999.pt` and bank paths, and a new timestamped output directory.
It sets the selected index consistently for Torch, Kit rendering and PhysX;
the child environment clears `CUDA_VISIBLE_DEVICES` to keep those physical
indices consistent. The source-matched asset hash override still requires
checkpoint/source/configuration audits to pass. Existing BO processes belonging
to this user and simultaneous launchers are rejected.

The run stays in the foreground. The terminal shows `BO_PROGRESS 1/100`
before the first candidate and its ordinary PPO iteration output, followed by
evaluation and the next candidate. Each candidate trains for 200 updates;
the default plan is 50 initial designs plus 50 GP candidates.
Per-attempt stdout/stderr and results are also saved. Ctrl+C interrupts training
and reaps the active worker; keep the terminal open while it runs.

Check inputs and GPU selection without starting training:

```bash
python rough_bo/start.py --check-only
```

Resume a previously initialized run:

```bash
python rough_bo/start.py --resume /absolute/path/to/bo_run --gpu 5
```

Omit `--gpu` when resuming to select a GPU automatically. The controller still
checks the immutable run contract; use the original device if that contract
rejects a device change. Override `--checkpoint`, `--bank` or `--code-root`
only when their audited files moved.

## Output and audit

The run directory contains an Optuna SQLite study, `trials.csv`, `summary.json`,
an immutable `run_contract.json`, a request per candidate, and separate logs,
fine-tuned checkpoints and evaluation summaries for each attempt. The summary
also states that this BO run is not a final design certification; finalists
should receive repeat-seed/gait confirmation.

The OCD generator and its design-score audit are included beside the 50-row
CSV. `references/historical_frozen_policy_pilot/` contains the earlier
zero-update pilot and source-replay audit strictly as historical provenance;
those scores are not part of the new objective.

## Contents

- `bo_finetune_controller.py`: sequential 50-point initialization + GP BO.
- `bo_finetune_candidate.py`: one-design, 200-update fine-tuning worker.
- `bo_finetune_evaluate.py`: fixed-condition speed-tracking evaluation.
- `bo_finetune_common.py`: checkpoint/bank/source checks and design creation.
- `ocd_mixed_design_50.py`: reproducible mixed-factor OCD generator.
- `ocd_mixed_design_50_full_cooling/`: initial design and score report.
- `references/`: clearly labeled historical pilot and provenance.
