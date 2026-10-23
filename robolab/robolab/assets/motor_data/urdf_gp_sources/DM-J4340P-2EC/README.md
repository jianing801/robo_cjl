# DM-J4340P-2EC source data

This package keeps the current SW-GP structural fit and the 48 V actuator-curve
model for the original DM-J4340P candidate. The non-P V1.1 and the newer P V1.1
are separate motors and must not use this model.

## SW-GP structural model

`SW_GP/data/` contains the current 15-point SolidWorks scan and normalized
training data. `SW_GP/fit_4340p_calf_gp.py` reproduces the independent
Matern-5/2 GP fit; `SW_GP/output/` contains the fitted model, validation
metrics, physical checks, and plots. The runtime model is
`../../urdf_gp_models/DM4340P_calf.json`.

## 48 V torque-speed envelope

The current MCB model was generated 2026-09-16 19:28 and the constrained GP
envelope 2026-09-16 19:53. It uses 65 peak-envelope samples and exports 464
lookup points. `selected/motor_parameters.csv` is restricted to this model;
`run_dm_j4340p_peak_envelope.m` and `fit_dm_j4340p_gp_envelope.m` are the
MATLAB R2024a generation scripts. Their current outputs, source peak curve,
MATLAB files, plots, and summaries are under `output/DM-J4340P-2EC/`.

The runtime CSV used by the catalog is
`../../torque_speed_curves/DM_J4340P_gp_envelope.csv`. The older 457-point
curve has been replaced by the current 464-point result.
