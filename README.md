# robo_cjl: morphology–actuator–control co-design

This repository contains RPO biped morphology-conditioned expert training, three-task shared-actor distillation, SW-GP/URDF generation, and a separate Rough Bayesian optimization workflow.

**Start with the [complete workflow](docs/WORKFLOW.md) (Chinese).** It covers setup, assets, banks, AMP references, expert training, validation, BC/DAgger, evaluation, recordings, recovery and Git updates. If banks and references are already ready, proceed to its expert-training section.

## Current formal configuration

- 3000 designs: 10 thigh lengths × 10 calf lengths × 5 knee motors × 6 ankle motors.
- Thigh 0.20–0.40m; calf 0.25–0.45m; explicitly authorized GP extrapolation.
- 4096 simulation replicas; 60 designs per group, 50 groups per sweep, 50 PPO updates per group.
- Rough and backpack: 12000 updates each. AMP: 5000 updates.
- All 23 joints use their motor curves and a dynamic torque-utilization penalty with weight −12.
- Three independent experts feed one task/morphology-conditioned shared Actor through BC/DAgger. Shared PPO is not implicitly enabled.

An update is not an episode. A completed training budget, low imitation MSE, and validated locomotion are distinct outcomes.

## Entry points and documents

| Purpose | Reference |
|---|---|
| Current versus archived launchers | [Entry index](ENTRYPOINTS.md) |
| Verified environments and original robot assets | [Setup](docs/SETUP.md) |
| End-to-end operating commands | [Workflow](docs/WORKFLOW.md) |
| Paired evaluation and recording | [Evaluation](docs/EVALUATION.md) |
| Pause/resume, partial generation, Git/API update | [Recovery](docs/RECOVERY.md) |
| Data identities and deliverables | [Artifacts](docs/ARTIFACTS.md) |
| BO, new structures and experimental shared PPO limits | [Extensions](docs/EXTENSIONS.md) |
| CAD, motors and SW-GP | [Documentation index](docs/README.md) |

The formal code is under `robolab/experiments/multitask_morphology/`. Historical expert sources are archived under `legacy/`. Two original compatibility locations remain for the legacy torque helper and existing editable HHTools installations. Models, references and experiment results are not deleted or migrated by this organization.

## Get the code

```bash
git clone --branch feature/rough-hybrid https://github.com/jianing801/robo_cjl.git
cd robo_cjl
```

The branch name is historical; the current workflow covers all three tasks. `robolab` and `rsl_rl` are included source trees, not the two uninitialized submodules described by the old upstream workspace README. Original robot URDF/mesh assets require a separately verified deployment.

Keep the validated Isaac Sim/Lab and Python environment. Do not upgrade packages or change importer versions just to follow a newer generic installation guide. Existing server commits must be preserved during updates; see the recovery document.

[中文完整首页](README_CN.md) · [Archived upstream workspace README](docs/reference/upstream_workspace_README.md)
