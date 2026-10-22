# DM-J10010-2EC thigh SW-GP source package

This package is the non-L DM-J10010 model and is kept separate from
`DM-J10010L-2EC`.

- `DM10010_大腿装配体质量属性完整数据.csv`: 0.25 m baseline plus 15 SolidWorks scans from 0.26 to 0.40 m.
- `GP拟合/fit_dm10010_thigh_gp.m`: MATLAB reconstruction script (Matern 5/2 GP).
- `GP拟合/output/DM10010_thigh_gpr_models.mat`: saved MATLAB regression models.
- `GP拟合/output/gp_loo_cv_metrics.csv`: nested leave-one-out validation metrics.
- `GP拟合/output/gp_physical_check.csv`: dense-interval inertia validity check.
- `GP拟合/output/gp_dense_predictions_301.csv`: 301-point means and uncertainty bands.
- `../../urdf_gp_models/DM10010_thigh.json`: runtime parameterization used by the BO/URDF predictor.

The source fit uses SolidWorks coordinate system 1, treated as aligned with the
thigh link frame. SW positive-product inertia fields are converted to the
standard URDF tensor convention by negating off-diagonal terms. The original
source package under the user's motor-selection library is left unchanged.
