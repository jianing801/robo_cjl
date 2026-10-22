"""Preview configurable grid on CPU; --generate-assets freezes a NEW Rough bank."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def main():
    from robolab.configurable_morphology import KIND, select_grid, eligible_pools, context, load_configurable_bank
    from robolab.morph_experiments import load_bank, write_new
    from robolab.physical_snapshot import DEFAULT_ROOT, physical_model, generator
    from robolab.mxt_design_transfer import file_sha, audit_candidate_urdf, candidate_contract, neutral_foot_compensation
    from robolab.native_expert_contract import source_identity, validate_native_bank
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/spatial_rough_grid.json"))
    p.add_argument("--source-bank")
    p.add_argument("--output-dir")
    p.add_argument("--generate-assets", action="store_true")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding="utf8"))
    pools, excluded = eligible_pools(DEFAULT_ROOT)
    grid = select_grid(config, pools["knee"], pools["ankle"])
    print(json.dumps(dict(configuration=config, eligible_motor_pools=pools, excluded=excluded,
                         **{k: v for k, v in grid.items() if k != "designs"}), indent=2), flush=True)
    if not a.generate_assets:
        return
    if not a.source_bank or not a.output_dir:
        p.error("asset generation requires --source-bank and a NEW --output-dir")
    source = load_bank(a.source_bank)
    validate_native_bank(source)
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
    template = Path(source["original_template"])
    contract_path = Path(source["reference_contract_source"])
    contract = json.loads(contract_path.read_text(encoding="utf8"))
    mesh = next(ET.parse(source["designs"][0]["urdf"]).getroot().iter("mesh")).get("filename")
    mesh_dir = Path(mesh.removeprefix("file://")).parent
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
        graph = candidate_contract(contract, urdf)["mtg"]
        placement = neutral_foot_compensation(template, urdf, contract["joint_names"],
            contract["default_joint_pos"], max_height_shift_m=.31)
        row.update(pair_id=pair, urdf=str(urdf), urdf_sha256=file_sha(urdf), report=str(report),
            report_sha256=file_sha(report), graph=graph,
            context=context(row, physical, grid["knees"], grid["ankles"], ranges),
            root_height_compensation_m=placement["root_height_compensation_m"],
            total_mass_kg=audit["total_mass_kg"]["candidate"],
            visuals_are_original_meshes_not_parameterized_CAD=True)
        print(f"Generated {index+1}/{len(rows)}: {row['id']}", flush=True)
    training = deepcopy(source["training"])
    training.update(authorized_tasks=["stairs"], source_hashes=source_identity(),
                    retargeted_references=False)
    bank = dict(kind=KIND, schema_version=1, configuration=config, snapshot_root=str(DEFAULT_ROOT.resolve()),
        num_envs=grid["num_envs"], knees=grid["knees"], ankles=grid["ankles"],
        context_width=grid["context_width"], ranges_m=ranges, designs=rows, motor_pairs=pairs,
        original_template=str(template), original_template_sha256=file_sha(template),
        reference_contract_source=str(contract_path), reference_contract_sha256=file_sha(contract_path),
        canonical_semantic_contract=source["canonical_semantic_contract"], training=training,
        source_bank_sha256=file_sha(a.source_bank), no_implicit_holdout_or_update_budget=True)
    write_new(out/"bank.json", bank)
    load_configurable_bank(out/"bank.json")
    print("VALIDATED", out/"bank.json", file_sha(out/"bank.json"), flush=True)


if __name__ == "__main__":
    main()
