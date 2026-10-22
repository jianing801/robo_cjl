"""Lightweight tests for catalog readiness; Isaac Sim is not required."""

import unittest
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from robolab.assets.rpo_motor_catalog import (
    ANKLE_MOTOR_BO_CANDIDATES,
    ANKLE_MOTOR_CANDIDATES,
    ANKLE_MOTOR_DESIGN_CANDIDATES,
    DM_STAGED_ANKLE_IDS,
    DM_STAGED_KNEE_IDS,
    KNEE_MOTOR_BO_CANDIDATES,
    KNEE_MOTOR_CANDIDATES,
    KNEE_MOTOR_DESIGN_CANDIDATES,
    MOTOR_CATALOG,
    get_motor_spec,
    load_torque_speed_envelope,
    require_motor_actuator_ready,
    require_motor_ready,
)


class MotorCandidateCatalogTest(unittest.TestCase):
    def test_original_roboparty_rectangular_defaults_are_preserved(self):
        from robolab.assets.rpo_motor_catalog import (
            DEFAULT_ANKLE_MOTOR, DEFAULT_KNEE_MOTOR,
            KNEE_MOTOR_CHOICES, ANKLE_MOTOR_CHOICES,
        )
        self.assertEqual(DEFAULT_KNEE_MOTOR, "legacy-knee")
        self.assertEqual(DEFAULT_ANKLE_MOTOR, "legacy-ankle")
        for motor_id, torque, speed, choices in (
            (DEFAULT_KNEE_MOTOR, 120.0, 25.0, KNEE_MOTOR_CHOICES),
            (DEFAULT_ANKLE_MOTOR, 27.0, 8.0, ANKLE_MOTOR_CHOICES),
        ):
            motor = get_motor_spec(motor_id)
            self.assertIn(motor_id, choices)
            self.assertEqual(motor.peak_torque_nm, torque)
            self.assertAlmostEqual(motor.max_speed_rad_s, speed)
            self.assertEqual(motor.armature_kgm2, 0.01)
            self.assertIsNone(load_torque_speed_envelope(motor))

    def test_only_incomplete_added_rectangle_entries_are_removed(self):
        from robolab.assets.rpo_motor_catalog import ANKLE_MOTOR_CHOICES
        for motor_id in ("DM-J8006-2EC", "DM-J4340-2EC"):
            self.assertNotIn(motor_id, MOTOR_CATALOG)
            self.assertNotIn(motor_id, ANKLE_MOTOR_CHOICES)
            with self.assertRaisesRegex(ValueError, "Unknown motor"):
                get_motor_spec(motor_id)
        require_motor_actuator_ready(get_motor_spec("DM-J8006-2EC-V1.1"))
        require_motor_actuator_ready(get_motor_spec("DM-J4340P-2EC-V1.1-48V"))

    def test_versions_have_separate_identity_curves_and_labels(self):
        old = get_motor_spec("DM-J4340P-2EC")
        new = get_motor_spec("DM-J4340P-2EC-V1.1-48V")
        self.assertIn("原款", old.display_name)
        self.assertIn("V1.1", new.display_name)
        self.assertNotEqual(old.motor_id, new.motor_id)
        self.assertNotEqual(old.torque_speed_curve_csv, new.torque_speed_curve_csv)
        self.assertEqual(old.as_dict()["display_name"], old.display_name)
        self.assertEqual((old.peak_torque_nm, new.peak_torque_nm), (27.0, 40.0))

    def test_configuration_keeps_legacy_path_and_explicit_dynamic_path(self):
        # Execute the real configuration function using lightweight config
        # objects, without importing the Isaac Sim runtime.
        import robolab.assets.rpo_motor_catalog as catalog
        source = Path(catalog.__file__).parent / "robots/roboparty.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "configure_rpo_motors" for n in tree.body
        ), "Original RPO_CFG must not be auto-overwritten at import.")
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "configure_rpo_motors")
        namespace = {name: getattr(catalog, name) for name in (
            "DEFAULT_KNEE_MOTOR", "DEFAULT_ANKLE_MOTOR", "KNEE_MOTOR_CHOICES",
            "ANKLE_MOTOR_CHOICES", "STAGED_MOTOR_IDS", "get_motor_spec",
            "require_motor_actuator_ready", "load_torque_speed_envelope",
        )}
        namespace.update(ArticulationCfg=object, TorqueSpeedEnvelopeActuatorCfg=SimpleNamespace)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)

        def cfg():
            return SimpleNamespace(actuators={key: SimpleNamespace(
                joint_names_expr=[key], stiffness=1, damping=1,
                friction=0, dynamic_friction=0, viscous_friction=0,
                min_delay=0, max_delay=2,
            ) for key in ("knees", "feet")})

        original = cfg()
        result = namespace["configure_rpo_motors"](original)
        self.assertFalse(result["knee"]["dynamic_envelope"])
        self.assertFalse(result["ankle"]["dynamic_envelope"])
        self.assertEqual(original.actuators["knees"].effort_limit_sim, 120)
        self.assertAlmostEqual(original.actuators["feet"].velocity_limit_sim, 8)
        self.assertIsNone(original.actuators["knees"].effort_limit)
        self.assertEqual(original.actuators["feet"].armature, 0.01)

        dynamic = cfg()
        result = namespace["configure_rpo_motors"](
            dynamic, knee_motor="DM-J10010L-2EC", ankle_motor="DM-J4340P-2EC")
        self.assertTrue(result["knee"]["dynamic_envelope"])
        self.assertTrue(result["ankle"]["dynamic_envelope"])
        self.assertGreater(len(dynamic.actuators["feet"].torque_speed_lookup), 2)

        newly_selected = cfg()
        result = namespace["configure_rpo_motors"](
            newly_selected, knee_motor="DM-J6248P-2EC", ankle_motor="DM-J6216P-2EC")
        self.assertTrue(result["knee"]["dynamic_envelope"])
        self.assertTrue(result["ankle"]["dynamic_envelope"])
        self.assertAlmostEqual(newly_selected.actuators["knees"].armature, 0.124416)
        self.assertAlmostEqual(newly_selected.actuators["feet"].armature, 0.0125696)
        self.assertEqual(len(newly_selected.actuators["knees"].torque_speed_lookup), 464)
        self.assertEqual(len(newly_selected.actuators["feet"].torque_speed_lookup), 464)

    def test_existing_bo_motors_still_have_curves_and_inertia(self):
        for motor_id in (*KNEE_MOTOR_CANDIDATES, *ANKLE_MOTOR_CANDIDATES):
            with self.subTest(motor_id=motor_id):
                motor = get_motor_spec(motor_id)
                require_motor_ready(motor)
                self.assertGreater(len(load_torque_speed_envelope(motor)), 2)

    def test_new_models_cannot_silently_use_rectangles_or_legacy_armature(self):
        validated = set(KNEE_MOTOR_CANDIDATES + ANKLE_MOTOR_CANDIDATES)
        staged = set(KNEE_MOTOR_DESIGN_CANDIDATES + ANKLE_MOTOR_DESIGN_CANDIDATES) - validated
        self.assertEqual(len(staged), 8)
        for motor_id in staged:
            with self.subTest(motor_id=motor_id):
                motor = get_motor_spec(motor_id)
                self.assertIsNotNone(motor.torque_speed_curve_csv)
                self.assertIn("assumption", motor.armature_source)
                require_motor_actuator_ready(motor)
                self.assertIsNotNone(motor.urdf_model_id)
                require_motor_ready(motor)

    def test_actuator_can_be_verified_before_sw_fit(self):
        actuator_only = replace(get_motor_spec("RS06"), urdf_model_id=None)
        require_motor_actuator_ready(actuator_only)
        with self.assertRaisesRegex(ValueError, "SW-URDF GP"):
            require_motor_ready(actuator_only)

    def test_dm_versions_and_low_torque_design_choices_are_explicit(self):
        self.assertEqual(len(set(DM_STAGED_KNEE_IDS + DM_STAGED_ANKLE_IDS)), 8)
        old = get_motor_spec("DM-J4340P-2EC")
        new = get_motor_spec("DM-J4340P-2EC-V1.1-48V")
        self.assertEqual((old.peak_torque_nm, new.peak_torque_nm), (27.0, 40.0))
        self.assertIsNotNone(new.torque_speed_curve_csv)
        self.assertAlmostEqual(new.armature_kgm2, 0.03152)
        self.assertIn("DM-J4310P-2EC-48V", ANKLE_MOTOR_DESIGN_CANDIDATES)
        self.assertLess(get_motor_spec("DM-J4310P-2EC-48V").peak_torque_nm, 27.0)

    def test_out_of_voltage_range_motor_not_in_48v_catalog(self):
        self.assertNotIn("Unitree-A1", MOTOR_CATALOG)

    def test_factory_parameters_preserve_phase_and_inertia_definitions(self):
        motor = get_motor_spec("DM-J10010-2EC")
        source = motor.manufacturer_parameters
        self.assertAlmostEqual(source["flux_pm_wb"], 0.00825)
        self.assertAlmostEqual(source["line_to_line_resistance_ohm"], 0.125)
        self.assertEqual(source["line_to_line_inductance_uh"], 100)
        self.assertEqual(source["resistance_inductance_definition"], "line_to_line_as_labelled")
        self.assertEqual(source["inertia_axis_definition"], "rotational_inertia_axis_not_specified")
        self.assertAlmostEqual(motor.armature_kgm2, 0.055)
        self.assertEqual(motor.model_kind, "parameter_model")
        source["flux_pm_wb"] = 0
        self.assertAlmostEqual(motor.manufacturer_parameters["flux_pm_wb"], 0.00825)

    def test_factory_multiline_cells_are_assigned_to_the_correct_model(self):
        self.assertEqual(get_motor_spec("DM-J6248P-2EC").max_speed_rpm, 115)
        self.assertEqual(get_motor_spec("DM-J8006-2EC-V1.1").max_speed_rpm, 392.8)
        self.assertEqual(get_motor_spec("DM-J8009P-2EC-V1.1").max_speed_rpm, 335)

    def test_protocol_ranges_do_not_replace_physical_output_limits(self):
        motor = get_motor_spec("DM-J4340P-2EC-V1.1-48V")
        self.assertEqual(motor.peak_torque_nm, 40)
        self.assertEqual(motor.manufacturer_parameters["protocol_tmax_nm"], 28)
        self.assertEqual(motor.max_speed_rpm, 112)

    def test_disputed_models_are_deleted_not_merely_disabled(self):
        from robolab.assets.rpo_motor_catalog import ANKLE_MOTOR_CHOICES, KNEE_MOTOR_CHOICES
        for motor_id in ("DM-J6006-2EC", "DM-J6216L-2EC"):
            with self.subTest(motor_id=motor_id):
                self.assertNotIn(motor_id, MOTOR_CATALOG)
                self.assertNotIn(motor_id, KNEE_MOTOR_CHOICES)
                self.assertNotIn(motor_id, ANKLE_MOTOR_CHOICES)
                with self.assertRaisesRegex(ValueError, "Unknown motor"):
                    get_motor_spec(motor_id)

    def test_eight_selected_models_have_distinct_parameter_curves_and_sw_models(self):
        modeled = [m for m in MOTOR_CATALOG.values() if m.model_kind == "parameter_model"]
        self.assertEqual(len(modeled), 8)
        for motor in modeled:
            with self.subTest(motor_id=motor.motor_id):
                require_motor_actuator_ready(motor)
                curve = load_torque_speed_envelope(motor)
                self.assertEqual(len(curve), 464)
                self.assertEqual(curve[-1][1], 0)
                self.assertAlmostEqual(motor.armature_kgm2,
                                       motor.rotor_inertia_kgm2 * motor.gear_ratio**2)
                self.assertIn("SPMSM_Ld_equals_Lq_assumed", motor.parameter_model["assumptions"])
                self.assertIsNotNone(motor.urdf_model_id)
                require_motor_ready(motor)

    def test_bo_pools_are_segment_matched_gp_models_and_predict_physical_inertias(self):
        import csv
        import importlib
        import json
        import sys
        import robolab.assets.rpo_motor_catalog as catalog
        repo_root = Path(catalog.__file__).resolve().parents[2]
        motor_data = Path(catalog.__file__).parent / "motor_data"
        tools_dir = str(repo_root / "scripts" / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        generator = importlib.import_module("generate_urdf_sw")
        registry = json.loads((motor_data / "motor_urdf_gp_registry.json").read_text(encoding="utf-8"))
        self.assertEqual(set(generator.MODELS), set(registry))
        self.assertFalse((motor_data / "motor_urdf_models.json").exists())
        self.assertFalse((repo_root / "scripts/tools/fit_motor_urdf_csv.py").exists())
        for segment, ids, lengths in (
            ("thigh", KNEE_MOTOR_BO_CANDIDATES, (0.25, 0.325, 0.40)),
            ("calf", ANKLE_MOTOR_BO_CANDIDATES, (0.30, 0.375, 0.45)),
        ):
            for motor_id in ids:
                motor = get_motor_spec(motor_id)
                record = generator.MODELS[motor.urdf_model_id]
                self.assertEqual(record["segment"], segment)
                self.assertTrue(record["gp_model_file"].endswith(".json"))
                for length in lengths:
                    with self.subTest(motor=motor_id, length=length):
                        prediction = generator.predict_properties(motor.urdf_model_id, length)
                        self.assertGreater(prediction["mass_kg"], 0.0)
                        self.assertTrue(prediction["com_m"].shape == (3,))
                        self.assertTrue((__import__("numpy").linalg.eigvalsh(
                            prediction["inertia_kg_m2"]
                        ) > 0.0).all())

        # The non-L J10010 runtime export must reproduce the saved MATLAB GP
        # means across the entire 301-point dense prediction table.
        source_csv = (Path(catalog.__file__).parent / "motor_data/urdf_gp_sources"
                      / "DM-J10010-2EC/GP拟合/output/gp_dense_predictions_301.csv")
        with source_csv.open(newline="", encoding="utf-8-sig") as stream:
            reference_rows = list(csv.DictReader(stream))
        self.assertEqual(len(reference_rows), 301)
        maximum_error = 0.0
        for row in reference_rows:
            prediction = generator.predict_properties("DM10010_thigh", float(row["length_m"]))
            outputs = {
                "mass_kg": prediction["mass_kg"],
                "com_x_m": prediction["com_m"][0],
                "com_y_m": prediction["com_m"][1],
                "com_z_m": prediction["com_m"][2],
                "Lxx": prediction["inertia_kg_m2"][0, 0],
                "Lxy": -prediction["inertia_kg_m2"][0, 1],
                "Lxz": -prediction["inertia_kg_m2"][0, 2],
                "Lyy": prediction["inertia_kg_m2"][1, 1],
                "Lyz": -prediction["inertia_kg_m2"][1, 2],
                "Lzz": prediction["inertia_kg_m2"][2, 2],
            }
            maximum_error = max(maximum_error, *(abs(
                outputs[key] - float(row[key + "_mean"])
            ) for key in outputs))
        self.assertLess(maximum_error, 1e-9)

    def test_p_preference_and_incomplete_models_are_blocked_at_all_entries(self):
        from robolab.assets.rpo_motor_catalog import (
            ANKLE_MOTOR_CHOICES, KNEE_MOTOR_CHOICES, NOT_SELECTED_MOTOR_REASONS,
        )
        self.assertEqual(len(NOT_SELECTED_MOTOR_REASONS), 15)
        for motor_id in NOT_SELECTED_MOTOR_REASONS:
            with self.subTest(motor_id=motor_id):
                self.assertNotIn(motor_id, MOTOR_CATALOG)
                for allowed in (None, KNEE_MOTOR_CHOICES, ANKLE_MOTOR_CHOICES):
                    if allowed is not None:
                        self.assertNotIn(motor_id, allowed)
                    with self.assertRaisesRegex(ValueError, "not selected"):
                        get_motor_spec(motor_id, allowed)
        for motor_id in ("DM-J8009P-2EC-V1.1", "DM-J4340P-2EC-V1.1-48V", "DM-J4310P-2EC-48V"):
            self.assertIn(motor_id, ANKLE_MOTOR_CHOICES)

    def test_selected_data_and_archive_are_disjoint_and_preserve_factory_rows(self):
        import csv
        import robolab.assets.rpo_motor_catalog as catalog
        data = Path(catalog.__file__).parent / "motor_data"

        def records(path):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                return {row["motor_id"]: row for row in csv.DictReader(stream)}

        original = records(data / "DM_selection_2026_09_23.csv")
        selected = records(data / "selected/DM_selection_2026_09_23.csv")
        archived = records(data / "not_selected/DM_selection_2026_09_23.csv")
        self.assertEqual((len(selected), len(archived)), (10, 6))
        self.assertFalse(selected.keys() & archived.keys())
        self.assertEqual({**selected, **archived}, original)
        index = records(data / "selected/电机清单.csv")
        archive_index = records(data / "not_selected/电机清单.csv")
        expected = set(KNEE_MOTOR_DESIGN_CANDIDATES + ANKLE_MOTOR_DESIGN_CANDIDATES)
        self.assertEqual(set(index), expected)
        self.assertEqual(len(expected), 12)
        self.assertFalse(index.keys() & archive_index.keys())
        self.assertEqual(set(index), set(MOTOR_CATALOG) - {"legacy-knee", "legacy-ankle"})
        bo_index = records(data / "selected/bo_ready_candidates.csv")
        bo_ids = set(KNEE_MOTOR_BO_CANDIDATES + ANKLE_MOTOR_BO_CANDIDATES)
        self.assertEqual(set(bo_index), bo_ids)
        self.assertEqual(len(bo_index), 12)
        self.assertEqual({Path(row["torque_speed_curve_csv"]).parent.as_posix()
                          for row in bo_index.values()}, {"motor_data/torque_speed_curves"})
        self.assertEqual(len({Path(row["torque_speed_curve_csv"]).name
                              for row in bo_index.values()}), 12)
        self.assertTrue(all((Path(catalog.__file__).parent / row["torque_speed_curve_csv"]).is_file()
                            for row in bo_index.values()))
        self.assertEqual(bo_index["DM-J10010-2EC"]["urdf_model_id"], "DM10010_thigh")
        self.assertEqual(bo_index["DM-J10010L-2EC"]["urdf_model_id"], "DM10010L_thigh")

    def test_selected_model_registry_matches_runtime_files_and_hashes(self):
        import csv
        import hashlib
        import robolab.assets.rpo_motor_catalog as catalog
        assets = Path(catalog.__file__).parent
        with (assets / "motor_data/selected/DM_parameter_models_2026_10_09.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({r["motor_id"] for r in rows}, set(DM_STAGED_KNEE_IDS + DM_STAGED_ANKLE_IDS))
        self.assertEqual(len(rows), 8)
        for row in rows:
            path = assets / row["curve_csv"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["curve_sha256"])
        actual = {p.name for p in (assets / "motor_data/torque_speed_curves").glob("*.csv")}
        expected = {Path(r["curve_csv"]).name for r in rows} | {
            "DM_J10010L_gp_envelope.csv", "DM_J4340P_gp_envelope.csv",
            "RS04_gp_envelope.csv", "RS06_gp_envelope.csv",
        }
        self.assertEqual(actual, expected)

    def test_archived_models_are_recoverable_but_not_runtime_choices(self):
        import csv
        import hashlib
        import json
        import robolab.assets.rpo_motor_catalog as catalog
        assets = Path(catalog.__file__).parent
        with (assets / "motor_data/not_selected/DM_parameter_models_2026_10_09.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertNotIn(row["motor_id"], MOTOR_CATALOG)
            self.assertTrue(row["curve_csv"].startswith("motor_data/not_selected/"))
            self.assertEqual(hashlib.sha256((assets / row["curve_csv"]).read_bytes()).hexdigest(),
                             row["curve_sha256"])
            self.assertTrue(row["historical_curve_sha256"])
        snapshot = json.loads((assets / "motor_data/not_selected/catalog_reference.json").read_text(encoding="utf-8"))
        self.assertEqual({row["motor_id"] for row in snapshot}, set(catalog.NOT_SELECTED_MOTOR_REASONS))

    def test_6216p_is_ankle_only_and_bo_ready_after_8520p_exclusion(self):
        from robolab.assets.rpo_motor_catalog import KNEE_MOTOR_CHOICES, ANKLE_MOTOR_CHOICES
        self.assertNotIn("DM-J8520P-2EC", KNEE_MOTOR_CHOICES)
        self.assertNotIn("DM-J8520P-2EC", ANKLE_MOTOR_CHOICES)
        self.assertIn("DM-J6216P-2EC", ANKLE_MOTOR_CHOICES)
        self.assertNotIn("DM-J6216P-2EC", KNEE_MOTOR_CHOICES)
        for motor_id in ("DM-J6216P-2EC",):
            motor = get_motor_spec(motor_id)
            require_motor_actuator_ready(motor)
            self.assertEqual(motor.urdf_model_id, "DM6216P_calf")
            require_motor_ready(motor)

    def test_8520p_is_blocked_and_its_curve_is_archived(self):
        import robolab.assets.rpo_motor_catalog as catalog
        data = Path(catalog.__file__).parent / "motor_data"
        for allowed in (None, catalog.KNEE_MOTOR_CHOICES, catalog.ANKLE_MOTOR_CHOICES):
            with self.assertRaisesRegex(ValueError, "not selected"):
                get_motor_spec("DM-J8520P-2EC", allowed)
        self.assertFalse((data / "dm_parameter_models/DM-J8520P-2EC.csv").exists())
        self.assertTrue((data / "not_selected/dm_parameter_models/DM-J8520P-2EC.csv").is_file())
        self.assertEqual(len(KNEE_MOTOR_DESIGN_CANDIDATES), 5)

    def test_6216p_model_speed_does_not_relabel_unknown_factory_speed(self):
        motor = get_motor_spec("DM-J6216P-2EC")
        source = motor.manufacturer_parameters
        self.assertEqual(source["reported_no_load_speed_rpm"], 260.0)
        self.assertIsNone(source["no_load_output_speed_48v_rpm"])
        self.assertEqual(motor.parameter_model["speed_endpoint_source"], "user_assumed_48v_anchor")
        self.assertEqual(motor.parameter_model["manufacturer_no_load_48v_rpm"], "")
        self.assertEqual(motor.max_speed_rpm, 260.0)
        self.assertEqual(motor.peak_torque_nm, 45.0)
        self.assertAlmostEqual(motor.armature_kgm2, 0.0125696)
        self.assertAlmostEqual(load_torque_speed_envelope(motor)[-1][0], motor.max_speed_rad_s)
        self.assertIn("not a manufacturer-verified", motor.data_notes)
        self.assertIn("user-assumed 48 V anchor", motor.data_notes)
        self.assertEqual(float(motor.parameter_model["rated_phase_current_48v_a"]), 22.0)
        self.assertEqual(motor.parameter_model["rated_current_source"], "manual_V1.3_20261015_page6")
        self.assertEqual(source["rated_phase_current_48v_a"], 7.6)



if __name__ == "__main__":
    unittest.main()
