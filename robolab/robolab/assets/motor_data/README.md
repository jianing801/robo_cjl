# Motor torque-speed envelopes

This directory contains both joint-output torque-speed curves and the
motor-dependent SolidWorks polynomial models consumed by the RoboLab
co-design workflow. Curves are loaded once at environment creation and
linearly interpolated on the GPU at every simulation step.

## Motor-dependent URDF model

`motor_urdf_models.json` contains the four validated length-dependent models:

- knee/thigh: `RS04_thigh`, `DM10010L_thigh`;
- common ankle/calf: `RS06_calf`, `DM4340P_calf`.

Mass uses a first-order polynomial. Center of mass and all six SolidWorks
center-of-mass inertia components use fourth-order polynomials. The runtime
converts SolidWorks products of inertia to the standard tensor convention.
RS04 additionally uses a documented source-frame correction because its CAD
report was exported from a rotated assembly frame whose origin moves with
thigh length.

`DM-J8006-2EC` remains in the general catalog for legacy/manual actuator
experiments, but it is not an optimization candidate because neither a
validated URDF model nor a validated armature value is available.

## CSV contract

Every curve must contain these columns:

- `output_speed_rad_s`: non-negative, strictly increasing absolute joint speed;
- `peak_envelope_nm`: non-negative maximum absolute output torque.

Additional provenance or uncertainty columns are allowed and ignored by the
runtime loader. The first point must be at zero speed. The last point should
normally be the no-load speed with zero torque.

## Adding another motor

1. Export one CSV using the contract above.
2. Add a `MotorSpec` entry in `../rpo_motor_catalog.py`, set the joint-side
   `armature_kgm2`, document its `armature_source`, and set
   `torque_speed_curve_csv` to `motor_data/<file>.csv`.
3. Add the matching polynomial entry to `motor_urdf_models.json` and set the
   catalog entry's `urdf_model_id`.
4. Add the motor ID to the appropriate knee or ankle candidate tuple only
   after both actuator and URDF parameters are validated.
5. Run the actuator/URDF smoke tests and use a new Optuna study name/database.

The Optuna model protocol stores every active armature value and a SHA-256
signature of every active curve, motor-to-URDF mapping, and SHA-256 values for
both the URDF model file and generator, so a study cannot silently mix trials
produced by different actuator or structural models.

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
