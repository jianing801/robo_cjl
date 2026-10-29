"""Preview configurable grid on CPU; --generate-assets freezes a NEW AMP bank."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def main():
    from amp_bank import KIND, select_grid, eligible_pools, context, load_configurable_bank
    from amp_support import write_new, source_identity, RECIPE
    from robolab.physical_snapshot import DEFAULT_ROOT, physical_model, generator
    from robolab.rough_geometry import file_sha, audit_candidate_urdf, neutral_foot_compensation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.template.json")))
    p.add_argument("--task-key", choices=["run"], default="run")
    p.add_argument("--template", help="Original baseline RPO URDF on THIS machine; never load old policies/banks")
    p.add_argument("--mesh-dir", help="Original RPO display meshes on THIS machine")
    p.add_argument("--contract-json", default=str(ROOT/"configs/rough_robot_contract.json"))
    p.add_argument("--output-dir")
    p.add_argument("--rough-bank", help="Reuse the exact existing Rough assets/context; write only a new AMP task manifest")
    p.add_argument("--generate-assets", action="store_true")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding="utf8"))
    config = config.get("grid", config)
    pools, excluded = eligible_pools(DEFAULT_ROOT)
    grid = select_grid(config, pools["knee"], pools["ankle"])
    print(json.dumps(dict(configuration=config, eligible_motor_pools=pools, excluded=excluded,
                         **{k: v for k, v in grid.items() if k != "designs"}), indent=2), flush=True)
    if not a.generate_assets:
        return
    if a.rough_bank:
        if not a.output_dir or a.template or a.mesh_dir:
            p.error('--rough-bank requires a NEW --output-dir; do not also supply template/mesh-dir')
        from amp_rough_alignment import adopt_rough_bank
        bank = adopt_rough_bank(a.rough_bank, config)
        out = Path(a.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=False)
        write_new(out/'bank.json', bank)
        load_configurable_bank(out/'bank.json')
        print('AMP_ROUGH_EXACT_MORPHOLOGY_VALIDATED', out/'bank.json', flush=True)
        return
    if not a.template or not a.mesh_dir or not a.output_dir:
        p.error("asset generation requires --template, --mesh-dir and a NEW --output-dir")
    out = Path(a.output_dir).resolve()
    # Validate all selected models/ranges before creating the output folder.
    pairs = {}
    ranges = dict(thigh=config["thigh_range_m"], calf=config["calf_range_m"])
    for knee in grid["knees"]:
        for ankle in grid["ankles"]:
            model = physical_model(knee_motor=knee, ankle_motor=ankle)
            gen = generator(model["snapshot_root"])
            for label, segment in (("knee", "thigh"), ("ankle", "calf")):
                lo, hi = gen.MODELS[model["selected"][label]["urdf_model_id"]]["prediction_range_m"]
                if not lo <= ranges[segment][0] < ranges[segment][1] <= hi:
                    raise ValueError(f"{segment} requested range exceeds {knee}/{ankle} calibrated GP bounds {[lo, hi]}")
            pairs[knee+"__"+ankle] = model
    template = Path(a.template).resolve()
    contract_path = Path(a.contract_json).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf8"))
    if set(contract) != {"joint_names", "default_joint_pos"} or len(contract["joint_names"]) != 23 or len(contract["default_joint_pos"]) != 23:
        raise ValueError("expected explicit 23-joint AMP order and unchanged neutral pose")
    mesh_dir = Path(a.mesh_dir).resolve()
    if not template.is_file() or not mesh_dir.is_dir():
        raise ValueError("template/mesh directory must exist on this machine")
    out.mkdir(parents=True, exist_ok=False)
    rows = grid["designs"]
    for index, row in enumerate(rows):
        pair = row["knee_motor"]+"__"+row["ankle_motor"]
        physical = pairs[pair]
        gen = generator(physical["snapshot_root"])
        urdf = Path(gen.generate_urdf(row["thigh_m"], row["calf_m"], out/row["id"], template,
            mesh_dir=mesh_dir, knee_motor=row["knee_motor"], ankle_motor=row["ankle_motor"]))
        report = urdf.with_suffix(".report.json")
        audit = audit_candidate_urdf(template, urdf)
        graph = dict(joint_names=contract["joint_names"])
        placement = neutral_foot_compensation(template, urdf, contract["joint_names"],
            contract["default_joint_pos"], max_height_shift_m=.31)
        row.update(pair_id=pair, urdf=str(urdf), urdf_sha256=file_sha(urdf), report=str(report),
            report_sha256=file_sha(report), graph=graph,
            context=context(row, physical, grid["knees"], grid["ankles"], ranges),
            root_height_compensation_m=placement["root_height_compensation_m"],
            total_mass_kg=audit["total_mass_kg"]["candidate"],
            visuals_are_original_meshes_not_parameterized_CAD=True)
        print(f"Generated {index+1}/{len(rows)}: {row['id']}", flush=True)
    training = dict(recipe=RECIPE, authorized_tasks=[a.task_key], source_hashes=source_identity(a.task_key),
                    retargeted_references=True)
    bank = dict(kind=KIND, schema_version=2, task_key=a.task_key, configuration=config, snapshot_root=str(DEFAULT_ROOT.resolve()),
        num_envs=grid["num_envs"], knees=grid["knees"], ankles=grid["ankles"],
        context_width=grid["context_width"], ranges_m=ranges, designs=rows, motor_pairs=pairs,
        original_template=str(template), original_template_sha256=file_sha(template),
        reference_contract_source=str(contract_path), reference_contract_sha256=file_sha(contract_path),
        canonical_semantic_contract=contract, training=training,
        no_implicit_holdout_or_update_budget=True)
    write_new(out/"bank.json", bank)
    load_configurable_bank(out/"bank.json")
    print("VALIDATED", out/"bank.json", file_sha(out/"bank.json"), flush=True)


if __name__ == "__main__":
    main()
