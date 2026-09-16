# Motor torque-speed envelopes

This directory contains joint-output torque-speed curves consumed by the
RoboLab co-design actuator model. Curves are loaded once at environment
creation and linearly interpolated on the GPU at every simulation step.

## CSV contract

Every curve must contain these columns:

- `output_speed_rad_s`: non-negative, strictly increasing absolute joint speed;
- `peak_envelope_nm`: non-negative maximum absolute output torque.

Additional provenance or uncertainty columns are allowed and ignored by the
runtime loader. The first point must be at zero speed. The last point should
normally be the no-load speed with zero torque.

## Adding another motor

1. Export one CSV using the contract above.
2. Add a `MotorSpec` entry in `../rpo_motor_catalog.py` and set
   `torque_speed_curve_csv` to `motor_data/<file>.csv`.
3. Add the motor ID to the appropriate knee or ankle candidate tuple.
4. Run the actuator smoke test and use a new Optuna study name/database.

The Optuna model protocol stores a SHA-256 signature of every active curve,
so a study cannot silently mix trials produced by different envelope files.

## DM-J4340P-2EC

- `DM_J4340P_peak_envelope.csv`: original 66-point MATLAB Motor Control
  Blockset envelope;
- `DM_J4340P_gp_envelope.csv`: 457-point constrained Gaussian-process curve
  used by default;
- `fit_dm_j4340p_gp_envelope.m`: reproducible MATLAB R2024a fitting script.

The GP uses a Matérn 3/2 kernel only on the descending branch. The 27 N·m
plateau is retained as a hard constraint, and the exported runtime curve is
projected to `[0, 27]` N·m, made non-increasing, and anchored to zero torque at
100 rpm.
