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

### Linux server: foreground output on GPU 5

Run from the repository root using the checkpoint-matched Isaac Lab environment.
The controller stays attached to this terminal. It prints a global BO counter
(`1/100`, `2/100`, ...) before every candidate, then streams that candidate's
ordinary PPO iteration output live. Per-attempt stdout/stderr are also saved.
Do not launch a second copy. Pressing Ctrl+C interrupts the active BO process;
it is not just a log viewer.

The server bank can have different file bytes from the Windows replay bank. The
explicit hash-override below permits that only after checkpoint iteration,
configuration, motor/morphology scope, native source-hash, and checkpoint-matched
source-tree audits pass. Do not use it for an unrelated checkpoint or bank.

```bash
cd /data/chenjiale/robo_cjl
conda activate robo_cjl
set -o pipefail

# Keep CUDA_VISIBLE_DEVICES unset so cuda:5 and Kit device 5 refer to the same GPU.
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT=/data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/persistent_20261026_194622/attempts/group_000141_20261027_235200_111604/result/models/model_11999.pt
BANK=/data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/dynamic_20261026_174456/bank/bank.json
CODE_ROOT=/data/chenjiale/robo_cjl/robolab
RUN_DIR=/data/chenjiale/robo_cjl/robolab/logs/rough_bo/bo_$(date +%Y%m%d_%H%M%S)

mkdir -p "$(dirname "$RUN_DIR")"
if [ ! -s "$CHECKPOINT" ] || [ ! -s "$BANK" ]; then
  echo "Checkpoint or bank file is missing; nothing was launched."
elif [ -e "$RUN_DIR" ] || [ -e "$RUN_DIR.console.log" ]; then
  echo "Output path already exists; choose a new timestamp; nothing was launched."
else
  FREE_MIB=$(nvidia-smi --id=5 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && [ "$FREE_MIB" -ge 12800 ]; then
    echo "Starting BO in the foreground on GPU 5; free memory: ${FREE_MIB} MiB"
    python -u rough_bo/bo_finetune_controller.py \
      --python "$(which python)" \
      --checkpoint "$CHECKPOINT" \
      --bank "$BANK" \
      --code-root "$CODE_ROOT" \
      --output "$RUN_DIR" \
      --device cuda:5 \
      --kit-args "--/renderer/activeGpu=5 --/physics/cudaDevice=5 --/renderer/multiGpu/enabled=false" \
      --allow-source-matched-asset-hash-override \
      --bo-trials 50 2>&1 | tee "$RUN_DIR.console.log"
  else
    echo "GPU 5 has only ${FREE_MIB:-unknown} MiB free; need at least 12800 MiB. Nothing was launched."
  fi
fi
```

This is a foreground run, not a detached job. Keep this terminal open to see
candidate progress and PPO output. The run directory retains all results and
attempt logs; a failed run can be resumed with the same `--output` and
`--resume` after its cause is fixed.
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
