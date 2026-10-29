# 12k Rough checkpoint: source correction and mixed-variable BO pilot

## What was fixed

The earlier local BO pilot compared source-hash fields embedded in the bank with the checkpoint, but did not hash the files that the evaluator actually ran. It also used a different local code tree. Those pilot results are retained, but this clean study imports none of them.

I recovered the checkpoint's recorded training source tree and verified 58/58 native sources, 16/16 Rough hybrid sources, and 32/32 motor data/model files. The original URDF template hash also matches. The BO controller and each candidate evaluator now hash every checkpoint-identified source file before evaluation; the evaluator additionally verifies that its key imported modules resolve to those same files. A mismatch stops the candidate by default.

Isaac Sim also stalled when launched inside the Codex sandbox. A minimal headless Kit launch completed in normal local process context, so the BO run was restarted there. This was a simulator-launch issue, not a sampler or candidate failure.

Two sandbox-started attempts were stopped after Kit initialization made no progress; their directories and logs remain marked as infrastructure-aborted. They produced no objective values and are not included in the study.

## Bank portability boundary

The original server bank is preserved unchanged. Its SHA256 is `7590855bd44aa73af89fbb1b274f811ce8e35cfaaa959fb2eb9a1ced3d7ae2b2`. For Windows replay, I regenerated all 900 URDFs from the recovered training code, motor snapshots, and original URDF template. All 900 design IDs and the split, thigh/calf lengths, motor identities, pair IDs, joint graph, 24-dimensional context, and root-height compensation match. The maximum difference in reported total mass is `3.13e-13 kg`.

The locally regenerated bank SHA256 is `25c00be6b64257ef5a049f34d2046895982b253c77495b0dfb0e4652116af2ac`. None of its 900 URDF/report byte hashes equal the server-generated files. The server bank contains Linux absolute paths; the local URDFs were generated on Windows. Thus the source/model identity is exact and the design metadata comparison passes, but this local replay is not byte-identical server-asset evidence and does not promise bitwise trajectory reproduction.

## Clean pilot protocol

- Final checkpoint: `model_11999.pt`, iteration 11999, 12,000 completed updates; SHA256 `a0cd01d59c47588eb23a52b440beac41d626f7b4a85473fb7135743930face26`.
- One task: fixed 10 cm stairs, forward command 0.5 m/s, 20 s, seed 42, one environment, no domain randomization, noise, initial perturbation, or motor delay.
- Policy stayed frozen; PPO updates were zero. The objective was 20-second forward-speed tracking MAE in m/s; early falls or incomplete rollouts receive a penalty.
- Search variables: thigh 0.25–0.40 m and calf 0.30–0.45 m (continuous); knee from RS04, DM-J10010-2EC, DM-J10010L-2EC and ankle from DM-J6216P-2EC, DM-J4340P-2EC, RS06 (categorical).
- Optuna 4.9 `GPSampler`: five startup trials, then one GP-guided mixed-variable proposal. This is a pipeline pilot, not a converged optimum.
- The study began with zero imported observations. No prior 20-design gait check, previous mismatched pilot, or server result was used.

## Six candidate results

| Trial | Thigh (mm) | Calf (mm) | Knee motor | Ankle motor | Tracking MAE (m/s) | Fell |
|---:|---:|---:|---|---|---:|:---:|
| 0 | 306.2 | 442.6 | RS04 | RS06 | 0.0707 | No |
| 1 | 340.2 | 406.2 | DM-J10010-2EC | DM-J6216P-2EC | 0.0829 | No |
| 2 | 295.6 | 378.7 | DM-J10010L-2EC | RS06 | 0.0906 | No |
| 3 | 318.4 | 417.8 | DM-J10010L-2EC | DM-J4340P-2EC | 0.1082 | No |
| 4 | 259.8 | 442.3 | RS04 | DM-J4340P-2EC | 0.3061 | No |
| 5 | 353.0 | 423.5 | DM-J10010-2EC | RS06 | 0.0777 | No |

All six completed 20 seconds. Trial 0 is best within this small pilot: thigh 306.2 mm, calf 442.6 mm, RS04 knee, RS06 ankle, MAE 0.0707 m/s. It is a candidate for further confirmation, not a declared global optimum.

## Audit files

- Full per-source, bank-replay, and per-trial audit: [`BO_source_and_replay_audit.json`](BO_source_and_replay_audit.json)
- Full study output, SQLite database, plans, requests, summaries, and logs: [`bo_exact_source_pilot_admin_20261029`](bo_exact_source_pilot_admin_20261029)
- Recovered checkpoint training sources and source audit: [`source_snapshot_audit.json`](C:/Users/Admin/.codex/.chatgpt-projects/g-p-6a967dbd08f4819188bc02c144f1a591/rough_training_source_recovery_20261029/exact_source_snapshot_raw_20261029/source_snapshot_audit.json)
- Original bank and locally regenerated replay bank are both preserved under the source recovery directory.
