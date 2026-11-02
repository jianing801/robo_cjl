# Rough BO v2: complete-rollout MAE and learned feasibility

Based on `feature/rough-hybrid` commit `724b244`. Keeps the latest Kit file-watcher
fix, foreground progress/output, GPU selection, source recovery and per-candidate
training unchanged. Does not alter Isaac Sim/Isaac Lab versions, frozen Rough
sources, rewards, physics, structure/motor pool, pretrained checkpoint or old runs.

## Objective and feasibility

Every candidate still starts from the same 12000-update model, uses fresh Adam,
and is fine-tuned for exactly 200 PPO updates with 4096 environments. Evaluation
uses one environment, seed 42, forward command 0.5 m/s, fixed 10 cm stairs and a
20-second horizon, with the existing deterministic evaluation settings.

The measured velocity remains the native task's body-frame forward velocity
`root_lin_vel_b[0, 0]`, compared with its body-frame forward command. Do not mix
it with world-frame displacement/time or change coordinate conventions in an
existing run.

For a complete trajectory with `N=1000` control steps and `dt=0.02 s`:

```
MAE = sum(abs(vx[k] - command[k])) / N
    = sum(abs(vx[k] - command[k]) * dt) / 20
```

The old code omitted `dt` from the observed-error sum while including it in the
post-fall tail, inflating full trajectories by 50 and biasing the ranking toward
some early failures. That implementation is not retained.

A physical fall, or termination before all 1000 steps, sets `feasible=false`.
The observed partial-trajectory MAE is saved for diagnostics, but
`objective_value` and `tracking_mae_fixed_horizon_mps` are **null**. These points
remain valid feasibility observations. A runtime exception, OOM, failed source
audit or missing summary is an infrastructure error: stop, preserve the attempt,
and do not tell either GP that the robot structure is infeasible.

Normal horizon timeout at the final step is successful completion. Falling on
the final step is still infeasible. `metrics_schema=rough_complete_rollout_mae_v2`
is required throughout the new controller.

## Two models and mixed variables

`constrained_gp.py` uses CPU scikit-learn models:

* Performance: GaussianProcessRegressor, trained **only on successful complete
  rollouts** and their MAE. Output is normalized inside the model; reported MAE
  stays in m/s. A WhiteKernel accounts for small observation/modeling noise.
* Feasibility: GaussianProcessClassifier with Bernoulli labels and a logistic
  Laplace approximation, trained on **all** evaluated success/fall outcomes.
* Both use a Mat茅rn-5/2 kernel on two normalized leg lengths, multiplied by an
  exponential Hamming kernel on knee/ankle motor identities. Motor identifiers
  are compared for equality, never treated as ordered numeric distances.
* Kernel hyperparameters are fitted by marginal likelihood. Bounds and any
  optimization warnings are recorded in each proposal audit.

Selection maximizes `EI(MAE) * P(feasible)`. The incumbent is the best **actually
observed feasible** candidate. All nine motor pairs are enumerated. Scrambled
Sobol points plus boundaries seed the continuous search; the best point for each
motor pair is refined locally within the original leg-length bounds. Already
observed parameter combinations are excluded. The acquisition optimizer is
approximate and does not guarantee the global maximum.

If all labels are identical, the binary classifier cannot be fitted. An explicit
Beta(1,1) posterior probability is used until both outcomes are observed. If
there is no feasible point, the algorithm explores feasibility and unvisited
regions. An explicitly logged uncertainty fallback handles vanishing EI.

The probability is a model estimate conditional on the fixed 200-update protocol
and evaluation conditions, not a certified physical safety guarantee. A single
fall does not blacklist an entire motor class or a fixed-radius region.

Optuna provides the durable trial database only; its RandomSampler is never
used to select an adaptive design: every proposal is explicitly enqueued from
the two-model acquisition. Completed infeasible trials have a finite `1e6`
**ledger sentinel**, needed by the scalar storage API. This is not a measured
MAE, is never used to fit the performance GP, and cannot win final selection.
Final `summary.json` and CSV select/report feasible MAE explicitly rather than
using Optuna's generic `best_trial`. If every point fails, `best_trial=null`.

Backend package versions are frozen in the run contract. Same-version resume
checks that contract. Pending queued/running candidates retain their parameters.
Worker outputs are still printed live, and all failed attempts are preserved.

## Dependencies

In the existing Isaac environment, install missing CPU BO dependencies only:

```bash
python -m pip install -r rough_bo/requirements-bo.txt
```

No `--upgrade` and no Isaac/PPO package changes are required. The additional
modeling dependency is scikit-learn. Existing satisfactory versions are retained.

## Reuse only the original 50 initial evaluations

Never use `--resume` on an old v1 run. Its GP proposals were selected using the
wrong objective, and the old database must remain an immutable experimental
record. Conversion creates a **new** JSON manifest outside the old run:

```bash
python rough_bo/repair_observations.py \
  --run /data/chenjiale/robo_cjl/robolab/logs/rough_bo/bo_20261030_185932_322114 \
  --output /data/chenjiale/rough_bo_observations_v2.json
```

The converter checks the original contract, source-audit outcomes, checkpoint
and bank identities, 200-update budget, seeds, evaluation settings and the known
v1 formula. It uses observed MAE/sample timing to recover correct successful
MAE, retains failed observations without an objective, and records source hashes
and old scores. No simulator, model loading or database rewrite occurs. It stops
on missing/ambiguous/mismatched evaluations rather than inventing data.

Only original trials 0–49 are imported: 46 successful objective observations and
50 feasibility labels (including 4 falls). Original adaptive trials 50–99 are
excluded from both models and final ranking. Even an older 100-point manifest
is filtered at loading. Original experiment files remain available for audit.

When you explicitly want new training evaluations, launch a **new** run:

```bash
python rough_bo/start.py \
  --observations /data/chenjiale/rough_bo_observations_v2.json \
  --bo-trials 50
```

Omit `--gpu` for automatic selection, or append `--gpu 5` to explicitly choose
physical GPU 5. The original training checkpoint/bank defaults still apply.
The controller checks imported checkpoint/bank/seeds/LR against the actual
current artifacts and validates the initial design table. Imported candidates
are not retrained. Here `--bo-trials 50` means **50 NEW** constrained-BO proposals
on top of the initial 50, with total progress `51/100` through `100/100`.
All 50 adaptive proposals are generated anew using the corrected criterion.
The old adaptive search path is discarded from this run.

For a CPU-only proposal preview (no training, no Isaac, no GPU requirement):

```bash
python rough_bo/preview_proposal.py \
  --observations /data/chenjiale/rough_bo_observations_v2.json \
  --bank /data/chenjiale/robo_cjl/robolab/logs/rough_hybrid/dynamic_20261026_174456/bank/bank.json \
  --output /data/chenjiale/rough_bo_next_proposal.json
```

The preview's feasibility and MAE are **predictions**, not a newly evaluated
result. It checks the manifest/bank; only the training controller validates the
live base checkpoint/source identity before executing any worker.

## Validation

```bash
python -m pytest rough_bo/tests -q
```

Tests cover the factor-50 regression, early/final-step falls, incomplete/nonfinite
data, strict legacy rejection, category-permutation invariance and analytic
kernel gradients, a real GPC's local feasibility discrimination, failure-only
updates, unique reproducible proposals, one-class fallbacks, final exclusion of
failures, idempotent partial warm-start import, CPU controller OCD/adaptive/resume
wiring, and the existing source-recovery/worker-argument/foreground tests.

## Method references

* Gardner et al., ICML 2014, [Bayesian Optimization with Inequality Constraints](https://proceedings.mlr.press/v32/gardner14.html): EI weighted by probability of feasibility under separate objective/constraint models.
* Gelbart et al., UAI 2014, [Bayesian Optimization with Unknown Constraints](https://arxiv.org/abs/1403.5607): probabilistic treatment of unknown constraints.
* [scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html): regression and logistic Laplace classification backend.

This implementation uses binary feasibility classification, not a fabricated
continuous safety-margin measurement or a claim to reproduce every detail of
the cited papers. No long training is part of the code-fix validation.
