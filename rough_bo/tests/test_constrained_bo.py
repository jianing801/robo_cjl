"""CPU regression checks: real GPs, scoring, ledger recovery and legacy conversion."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import create_trial
from sklearn.base import clone

BO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BO))
from bo_metrics import score_rollout, validate_metrics, ledger_value, METRICS_SCHEMA
from constrained_gp import MixedKernel, ConstrainedSurrogate, key
from repair_observations import convert_summary, load_observations
import bo_finetune_controller as controller

RANGES = {"thigh": [.25, .4], "calf": [.3, .45]}
KNEES, ANKLES = ["K0", "K1"], ["A0", "A1"]
PARAMS = dict(thigh_m=.3, calf_m=.4, knee_motor="K0", ankle_motor="A0")


def metrics(*, trial=0, fell=False, steps=1000, error=.1, params=None):
    return {
        "params": params or PARAMS, "trial_number": trial,
        "result": "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED",
        "base_checkpoint_sha256": "base", "bank_sha256": "bank",
        "fine_tuned_checkpoint": f"model_{trial}.pt", "fine_tuned_checkpoint_sha256": "f"*64,
        "fine_tune_updates": 200, "elapsed_s": steps*.02,
        "horizon_s": 20., "fell": fell, "timed_out": steps == 1000,
        **score_rollout(error*steps, steps, .02, 20., fell, steps == 1000),
    }


def test_complete_rollout_uses_samples_not_seconds():
    m = metrics(error=.10347758378041907)
    assert m["objective_value"] == pytest.approx(.10347758378041907)
    assert m["objective_value"] != pytest.approx(5.173879189020954)
    validate_metrics(m)


@pytest.mark.parametrize("fell,steps", [(True, 47), (False, 500), (True, 1000)])
def test_falls_and_incomplete_rollouts_have_no_full_horizon_mae(fell, steps):
    m = metrics(fell=fell, steps=steps)
    assert m["objective_value"] is None and not m["feasible"]
    assert m["tracking_mae_observed_mps"] == pytest.approx(.1)
    assert ledger_value(m) == 1e6
    m["objective_value"] = 0.
    with pytest.raises(ValueError):
        validate_metrics(m)


def test_bad_timing_nonfinite_data_and_legacy_reuse_rejected():
    for args in [(100, 0, .02, 20, False), (float("nan"), 1000, .02, 20, False),
                 (100, 1001, .02, 20, False), (100, 1000, .021, 20, False)]:
        with pytest.raises(ValueError):
            score_rollout(*args)
    m = metrics(); m.pop("metrics_schema")
    with pytest.raises(ValueError, match="legacy"):
        validate_metrics(m)
    m = metrics(); m["elapsed_s"] = float("nan")
    with pytest.raises(ValueError):
        validate_metrics(m)


def test_mixed_kernel_category_permutation_invariance_and_gradient():
    X = np.array([[.1, .2, 0, 0], [.4, .7, 1, 0], [.3, .5, 2, 1]])
    k = MixedKernel()
    K, gradient = k(X, eval_gradient=True)
    permuted = X.copy(); permuted[:, 2] = [2, 0, 1]
    assert np.allclose(K, k(permuted))
    assert np.allclose(clone(k)(X), K)
    for j in range(len(k.theta)):
        plus, minus = k.theta.copy(), k.theta.copy()
        plus[j] += 1e-5; minus[j] -= 1e-5
        numeric = (k.clone_with_theta(plus)(X) - k.clone_with_theta(minus)(X)) / 2e-5
        assert np.allclose(numeric, gradient[:, :, j], atol=1e-6)
    assert np.linalg.eigvalsh(K).min() > 0


def observations():
    rows = []
    for x in np.linspace(0, 1, 12):
        for knee in KNEES:
            p = dict(thigh_m=.25 + .15*x, calf_m=.375, knee_motor=knee, ankle_motor="A0")
            feasible = bool(x > .5)
            rows.append(dict(params=p, feasible=feasible,
                             mae=.07 + (x-.75)**2 if feasible else None))
    return rows


def test_failed_points_train_classifier_only_and_reduce_local_feasibility():
    rows = observations()
    model = ConstrainedSurrogate(RANGES, KNEES, ANKLES, optimize=False).fit(rows)
    assert model.feasibility_count == 24 and model.feasible_count == 12
    assert len(model.regressor.X_train_) == 12
    test = model.encode([dict(PARAMS, thigh_m=.265, calf_m=.375),
                         dict(PARAMS, thigh_m=.385, calf_m=.375)])
    mean, std, probability = model.predict(test)
    assert 0 < probability[0] < probability[1] < 1
    assert probability[1] - probability[0] > .2
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    # Even enormous failed diagnostics must not influence the performance GP.
    modified = copy.deepcopy(rows)
    for row in modified:
        if not row["feasible"]:
            row["mae"] = 1e100
    other = ConstrainedSurrogate(RANGES, KNEES, ANKLES, optimize=False).fit(modified)
    assert np.allclose(model.regressor.y_train_, other.regressor.y_train_)


def test_real_gp_proposal_is_unique_bounded_and_reproducible():
    model = ConstrainedSurrogate(RANGES, KNEES, ANKLES, optimize=False).fit(observations())
    p, audit = model.propose(pool_power=4, refine=False)
    q, again = model.propose(pool_power=4, refine=False)
    assert p == q and audit == again
    assert key(p) not in {key(o["params"]) for o in observations()}
    assert .25 <= p["thigh_m"] <= .4 and .3 <= p["calf_m"] <= .45
    assert audit["performance_training_count"] == 12
    assert audit["feasibility_training_count"] == 24
    assert audit["method"] == "EI_times_probability_feasible"


@pytest.mark.parametrize("success", [False, True])
def test_all_fail_and_all_success_histories_are_handled(success):
    rows = observations()[:4]
    for row in rows:
        row.update(feasible=success, mae=.1 if success else None)
    model = ConstrainedSurrogate(RANGES, KNEES, ANKLES, optimize=False).fit(rows)
    p, audit = model.propose(pool_power=3, refine=False)
    assert audit["classifier"] == "one_class_Beta_prior"
    assert audit["predicted_feasibility"] == pytest.approx(5/6 if success else 1/6)
    assert (model.regressor is None) == (not success)


def test_final_selection_never_recommends_failure_and_all_fail_is_explicit():
    study = optuna.create_study()
    distributions = {"thigh_m": FloatDistribution(.25, .4), "calf_m": FloatDistribution(.3, .45),
                     "knee_motor": CategoricalDistribution(KNEES), "ankle_motor": CategoricalDistribution(ANKLES)}
    bad = metrics(fell=True, steps=47)
    attrs = controller.result_attrs(bad, phase="ocd50", design_id="failed")
    # Adversarially small ledger value still must not win final selection.
    study.add_trial(create_trial(params=bad["params"], distributions=distributions, value=0., user_attrs=attrs))
    provenance = dict(checkpoint_sha256="base", bank_sha256="bank")
    protocol = dict(objective="test", fine_tuning={"environment_count":4096})
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        controller.write_results(out, study, provenance, protocol)
        assert json.loads((out/"summary.json").read_text())["best_trial"] is None
        good = metrics(trial=1, params=dict(PARAMS, thigh_m=.4))
        study.add_trial(create_trial(params=good["params"], distributions=distributions, value=.1,
                        user_attrs=controller.result_attrs(good, phase="gp", design_id="good")))
        controller.write_results(out, study, provenance, protocol)
        assert json.loads((out/"summary.json").read_text())["best_trial"]["number"] == 1
        obs = controller.model_observations(study)
        assert obs[0]["mae"] is None and not obs[0]["feasible"]


def test_warm_start_idempotence_partial_import_and_budget():
    study = optuna.create_study()
    distributions = {"thigh_m": FloatDistribution(.25, .4), "calf_m": FloatDistribution(.3, .45),
                     "knee_motor": CategoricalDistribution(KNEES), "ankle_motor": CategoricalDistribution(ANKLES)}
    rows = []
    for i, phase in enumerate(["ocd50", "historical_gp"]):
        m = metrics(trial=i, params=dict(PARAMS, thigh_m=.3 + i*.05), fell=i==1, steps=47 if i else 1000)
        m["original_objective_value"] = 3.
        rows.append(dict(original_trial_number=i, phase=phase, metrics=m,
                         source_summary=f"old/{i}/summary.json", source_summary_sha256="a"*64))
    controller.import_warm_start(study, {"observations":rows[:1]}, distributions)
    controller.import_warm_start(study, {"observations":rows}, distributions)
    controller.import_warm_start(study, {"observations":rows}, distributions)
    assert len(study.trials) == 2 and controller.target_count(study, 50, 50) == 101


def legacy(fell=False):
    m = metrics(fell=fell, steps=47 if fell else 1000)
    for k in ("metrics_schema", "control_dt_s", "sample_count", "expected_sample_count", "infeasible_reason"):
        m.pop(k)
    steps = round(m["elapsed_s"]/.02)
    m["objective_value"] = (.1*steps + (1000-steps)*.5*.02)/20
    m.update(base_checkpoint_iter=11999, fine_tune_optimizer_mode="fresh_adam", command_mps=.5,
             num_envs=1, domain_randomization=False, observation_noise=False,
             initial_perturbation=False, motor_delay=0, eval_seed=42,
             runtime_source_audit={"mismatch_count":0}, loaded_module_audit={"mismatch_count":0})
    return m


@pytest.mark.parametrize("fell", [False, True])
def test_legacy_scoring_recovered_without_simulation(fell):
    old = legacy(fell)
    before = copy.deepcopy(old)
    fixed = convert_summary(old, dict(base_checkpoint_sha256="base", bank_sha256="bank", evaluation={"seed":42}))
    validate_metrics(fixed)
    assert old == before
    assert fixed["objective_value"] == (None if fell else pytest.approx(.1))
    assert fixed["original_objective_value"] == old["objective_value"]


def test_unknown_legacy_formula_and_identity_not_silently_imported():
    old = legacy(); old["objective_value"] = .123
    contract = dict(base_checkpoint_sha256="base", bank_sha256="bank", evaluation={"seed":42})
    with pytest.raises(ValueError, match="recognized"):
        convert_summary(old, contract)
    old = legacy(); old["bank_sha256"] = "different"
    with pytest.raises(ValueError, match="identity"):
        convert_summary(old, contract)


def test_controller_full_ocd_gp_loop_and_resume_without_isaac():
    import csv
    designs = list(csv.DictReader((BO / "ocd_mixed_design_50_full_cooling/ocd_design_50.csv")
                                 .open(encoding="utf-8-sig")))
    bank = dict(ranges_m=RANGES, knees=sorted({d["knee_motor"] for d in designs}),
                ankles=sorted({d["ankle_motor"] for d in designs}))
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = SimpleNamespace(
            num_envs=4096, updates=200, bo_trials=2, fine_tune_timeout=10, eval_timeout=10,
            learning_rate=None, python=sys.executable, code_root=BO.parent/"robolab",
            checkpoint=root/"base.pt", bank=root/"bank.json", seed=42, eval_seed=42,
            design_csv=BO/"ocd_mixed_design_50_full_cooling/ocd_design_50.csv",
            output=root/"run", resume=False, observations=None, device="cuda:0", kit_args="",
            allow_source_matched_asset_hash_override=True)
        provenance = dict(checkpoint=str(args.checkpoint), bank=str(args.bank),
            checkpoint_sha256="base", bank_sha256="bank", checkpoint_iter=11999,
            completed_updates=12000, code_root=str(args.code_root), source_mismatch_count=0,
            checkpoint_learning_rate=.001,
            runtime_source_audit=dict(matched_source_count=74, native_source_count=58,
                                      hybrid_source_count=16, mismatch_count=0))
        def candidate(_args, output, _provenance, params, number, phase):
            p = {k:params[k] for k in PARAMS}
            calls.append((number, phase))
            return metrics(trial=number, params=p, fell=number==0, steps=47 if number==0 else 1000,
                           error=.05+.002*(number%5))
        real_model = ConstrainedSurrogate
        def fast_model(*a, **kw):
            return real_model(*a, **kw, optimize=False)
        with patch.object(controller, "parse_args", return_value=args), \
             patch.object(controller, "add_source_import_paths"), \
             patch.object(controller, "load_and_validate_assets", return_value=({}, bank, provenance)), \
             patch.object(controller, "ConstrainedSurrogate", side_effect=fast_model), \
             patch.object(controller, "run_candidate", side_effect=candidate):
            assert controller.main() == 0
            assert len(calls) == 52
            result = json.loads((args.output/"summary.json").read_text())
            assert result["infeasible_trials"] == 1 and result["feasible_trials"] == 51
            assert result["best_trial"]["number"] != 0
            proposals = list((args.output/"proposals").glob("*.json"))
            assert len(proposals) == 2
            for p in proposals:
                audit = json.loads(p.read_text())
                assert audit["performance_training_count"] == audit["feasibility_training_count"]-1
            args.resume = True
            assert controller.main() == 0
            assert len(calls) == 52  # Completed observations are never retrained.


def test_new_start_rejects_old_run_before_gpu_lookup():
    import start
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("base.pt", "bank.json", "code/scripts/tools/pretrained_rough_source_audit.py"):
            p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("fixture")
        old = root/"old"; old.mkdir()
        (old/"run_contract.json").write_text(json.dumps({"result_kind":"v1"}))
        with patch.object(start, "query_gpus") as gpu, patch.object(start.subprocess, "Popen") as launch:
            with pytest.raises(ValueError, match="Old BO"):
                start.main(["--checkpoint",str(root/"base.pt"),"--bank",str(root/"bank.json"),
                            "--code-root",str(root/"code"),"--resume",str(old)])
            gpu.assert_not_called(); launch.assert_not_called()


def test_only_initial_50_retained_from_old_100():
    from repair_observations import initial_observations
    rows = [dict(original_trial_number=i, phase="ocd50" if i < 50 else "historical_gp") for i in range(100)]
    before = copy.deepcopy(rows)
    assert initial_observations(rows) == rows[:50]
    assert rows == before
    with pytest.raises(ValueError, match="all original 50"):
        initial_observations(rows[1:])
    rows[2]["phase"] = "historical_gp"
    with pytest.raises(ValueError, match="original OCD"):
        initial_observations(rows)
