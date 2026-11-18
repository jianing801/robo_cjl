"""Configurable Cartesian morphology banks; no Isaac imports or live mutation."""
import itertools
import json
import math
import random
from pathlib import Path
from hybrid_support import digest, interpolate

KIND = "hybrid_rough_grid_bank_v1"


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def evenly_spaced(bounds, count):
    positive(count, "length_count")
    if (len(bounds) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) for v in bounds)
            or not 0 < bounds[0] < bounds[1]):
        raise ValueError("length range must be two finite positive increasing bounds")
    lo, hi = bounds
    if count == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def select_grid(config, knee_pool, ankle_pool):
    allowed = {"num_envs", "length_count", "thigh_range_m", "calf_range_m",
               "knee_motor_count", "ankle_motor_count", "motor_seed", "knee_pool", "ankle_pool", "gp_extrapolation"}
    if set(config) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(config)-allowed)}")
    if type(config.get('gp_extrapolation',False)) is not bool:raise ValueError('gp_extrapolation must be boolean')
    envs = positive(config["num_envs"], "num_envs")
    count = positive(config["length_count"], "length_count")
    thigh = evenly_spaced(config["thigh_range_m"], count)
    calf = evenly_spaced(config["calf_range_m"], count)
    seed = config["motor_seed"]
    if type(seed) is not int:
        raise ValueError("motor_seed must be an integer")
    selected = []
    for label, available in (("knee", knee_pool), ("ankle", ankle_pool)):
        pool = config.get(label + "_pool", available)
        if not isinstance(pool, (list, tuple)) or not pool or len(set(pool)) != len(pool):
            raise ValueError(f"{label} pool must contain unique model IDs")
        if not set(pool).issubset(available):
            raise ValueError(f"{label} pool includes unavailable/incomplete motors: {set(pool)-set(available)}")
        n = positive(config[label + "_motor_count"], label + "_motor_count")
        if n > len(pool):
            raise ValueError(f"{label} count {n} exceeds eligible pool size {len(pool)}")
        # Independent streams: changing knee count does not resample ankle models.
        rng = random.Random(digest({"seed": seed, "group": label}))
        selected.append(rng.sample(sorted(pool), n))
    knees, ankles = selected
    total = count * count * len(knees) * len(ankles)
    # Only the active stage subset must fit the live environment count.
    rows = []
    for t, c, k, a in itertools.product(thigh, calf, knees, ankles):
        row = dict(split="train", thigh_m=t, calf_m=c, knee_motor=k, ankle_motor=a)
        rows.append(dict(row, id="tr_" + digest(row)[:12]))
    return dict(num_envs=envs, thigh_values=thigh, calf_values=calf, knees=knees, ankles=ankles,
                design_count=total, context_width=18+len(knees)+len(ankles), designs=rows)


def eligible_pools(root):
    from robolab.physical_snapshot import catalog
    module = catalog(root)
    result, excluded = {}, {}
    for label in ("knee", "ankle"):
        result[label], excluded[label] = [], {}
        for motor in getattr(module, label.upper()+"_MOTOR_BO_CANDIDATES"):
            try:
                spec = module.get_motor_spec(motor)
                module.require_motor_ready(spec)
                if module.load_torque_speed_envelope(spec) is None:
                    raise ValueError("no registered motor curve")
                result[label].append(motor)
            except (ValueError, FileNotFoundError) as error:
                excluded[label][motor] = str(error)
    return result, excluded


def context(row, physical, knees, ankles, ranges):
    from robolab.physical_snapshot import catalog
    module = catalog(physical["snapshot_root"])
    values = [(row[field]-(ranges[seg][0]+ranges[seg][1])/2)/((ranges[seg][1]-ranges[seg][0])/2)
              for field, seg in (("thigh_m", "thigh"), ("calf_m", "calf"))]
    values += [float(row["knee_motor"] == k) for k in knees]
    values += [float(row["ankle_motor"] == k) for k in ankles]
    profiles = []
    for label in ("knee", "ankle"):
        if row[label+"_motor"] != physical[label+"_motor"]:
            raise ValueError("context/actual motor mismatch")
        spec = physical["selected"][label]
        values += [spec["peak_torque_nm"]/150, spec["max_speed_rad_s"]/25, spec["armature_kgm2"]/.1]
        curve = module.load_torque_speed_envelope(module.get_motor_spec(row[label+"_motor"]))
        profiles += [interpolate(curve, f*spec["max_speed_rad_s"])/spec["peak_torque_nm"]
                     for f in (0, .25, .5, .75, 1)]
    values += profiles
    if len(values) != 18+len(knees)+len(ankles) or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid configurable context")
    return values


def load_configurable_bank(path):
    from robolab.rough_geometry import file_sha
    from robolab.morphology_context import validate_physical_context, verified_raw_context
    from hybrid_support import source_identity
    from robolab.physical_snapshot import validate_physical_model
    bank = json.loads(Path(path).read_text(encoding="utf8"))
    if bank["kind"] != KIND or bank["schema_version"] not in (2, 3):
        raise ValueError("wrong configurable bank schema")
    physical_context = bank["schema_version"] == 3
    if not physical_context and "context_schema" in bank:
        raise ValueError("Legacy bank must not advertise a physical context schema")
    from robolab.gp_extrapolation import validate_authorization
    validate_authorization(bank['configuration'],bank['training'])
    pools, excluded = eligible_pools(bank["snapshot_root"])
    grid = select_grid(bank["configuration"], pools["knee"], pools["ankle"])
    expected_ranges = dict(thigh=bank["configuration"]["thigh_range_m"], calf=bank["configuration"]["calf_range_m"])
    if bank["ranges_m"] != expected_ranges:
        raise ValueError("range/context scaling differs from configuration")
    fields = ("num_envs", "knees", "ankles") if physical_context else ("num_envs", "knees", "ankles", "context_width")
    for field in fields:
        if bank[field] != grid[field]:
            raise ValueError(f"bank selection changed: {field}")
    task = bank.get("task_key", "stairs")
    if task not in ("stairs", "backpack"):
        raise ValueError("unknown independent expert")
    if bank["training"]["source_hashes"] != source_identity(task):
        raise ValueError("native source identity changed")
    if bank["training"]["authorized_tasks"] != [task]:
        raise ValueError("single-expert bank task identity mismatch")
    if bank["training"].get("retargeted_references"):
        raise ValueError("Rough must not use AMP references")
    if task=='backpack' and 'base_morphology_binding' in bank:
        binding=bank['base_morphology_binding']
        if file_sha(binding['bank'])!=binding['sha256']:raise ValueError('bound Rough bank changed')
        base=json.loads(Path(binding['bank']).read_text(encoding='utf8'))
        if base.get('task_key','stairs')!='stairs' or base['configuration']!=bank['configuration']:
            raise ValueError('Backpack/Rough grid mismatch')
        for key in ('schema_version', 'context_schema', 'context_fields', 'context_normalization'):
            if base.get(key) != bank.get(key):raise ValueError('Backpack/Rough context schema mismatch: '+key)
        if len(base['designs']) != len(bank['designs']):
            raise ValueError('Backpack/Rough design count mismatch')
        for actual,original in zip(bank['designs'],base['designs']):
            for key in ('id','thigh_m','calf_m','knee_motor','ankle_motor','context','pair_id','root_height_compensation_m'):
                if actual[key]!=original[key]:raise ValueError('Backpack/Rough structure mismatch: '+key)
            if actual['base_urdf_sha256']!=original['urdf_sha256'] or file_sha(actual['base_urdf'])!=actual['base_urdf_sha256']:
                raise ValueError('Backpack base URDF identity changed')
    if file_sha(bank["original_template"]) != bank["original_template_sha256"]:
        raise ValueError("template changed")
    if file_sha(bank["reference_contract_source"]) != bank["reference_contract_sha256"]:
        raise ValueError("semantic contract changed")
    if len(bank["designs"]) != len(grid["designs"]):
        raise ValueError("missing/extra Cartesian designs")
    for physical in bank["motor_pairs"].values():
        validate_physical_model(physical)
    gp_gen=None
    if bank['configuration'].get('gp_extrapolation'):
        from robolab.physical_snapshot import generator
        gp_gen=generator(bank['snapshot_root'])
    pair_suffixes={}
    samples = []
    for actual, expected in zip(bank["designs"], grid["designs"]):
        if any(actual[k] != v for k, v in expected.items()):
            raise ValueError("design differs from frozen Cartesian grid")
        if file_sha(actual["report"]) != actual["report_sha256"]:
            raise ValueError("report asset changed")
        if not physical_context and file_sha(actual["urdf"]) != actual["urdf_sha256"]:
            raise ValueError("urdf asset changed")
        pair = actual["knee_motor"]+"__"+actual["ankle_motor"]
        if gp_gen is not None:
            from robolab.gp_extrapolation import design_extrapolation
            if actual.get('gp_extrapolation')!=design_extrapolation(gp_gen,bank['motor_pairs'][pair],actual):
                raise ValueError('design GP extrapolation markers changed')
        if actual["pair_id"] != pair:
            raise ValueError("physical motor pair identity changed")
        if physical_context:
            samples.append(verified_raw_context(actual, bank["motor_pairs"][pair]))
            continue
        if pair not in pair_suffixes:
            pair_suffixes[pair]=context(actual,bank['motor_pairs'][pair],bank['knees'],bank['ankles'],bank['ranges_m'])[2:]
        lengths=[(actual[field]-(bank['ranges_m'][segment][0]+bank['ranges_m'][segment][1])/2)/
            ((bank['ranges_m'][segment][1]-bank['ranges_m'][segment][0])/2)
            for field,segment in (('thigh_m','thigh'),('calf_m','calf'))]
        if actual["context"] != lengths+pair_suffixes[pair]:
            raise ValueError("physical/context identity changed")
    if physical_context:
        validate_physical_context(bank, samples)
    return bank
