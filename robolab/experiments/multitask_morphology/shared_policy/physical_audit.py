"""CPU-readable startup physics audit; diagnostics are never actor inputs.

Call after native startup events and before attaching a backpack.  Mass checks
are exhaustive per body. Other recorded properties receive finite/physical
sanity checks, not an unsupported claim of full PhysX/config equivalence.
"""
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

import torch


MASS_ATOL = 2e-5
TOTAL_MASS_ATOL = 2e-4


def _tensor(value, label):
    result = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite {label}")
    return result


def _rows(value, n, label):
    result = _tensor(value, label)
    if result.ndim < 2 or result.shape[0] != n:
        raise ValueError(f"{label} must have one row per environment ({n}), got {tuple(result.shape)}")
    return result


def _json(value):
    if isinstance(value, torch.Tensor):
        return _tensor(value, "configuration").tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported/nonfinite physics configuration value: {type(value).__name__}")


def _selector(asset, names, field):
    patterns = getattr(asset, field + "_names", None)
    if patterns is not None:
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        selected = set()
        for pattern in patterns:
            matches = [i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
            if not matches:
                raise ValueError(f"Physics selector {pattern!r} matches no {field}")
            selected.update(matches)
        if not selected:
            raise ValueError(f"Empty {field} selector")
        return sorted(selected)
    ids = getattr(asset, field + "_ids", slice(None))
    if isinstance(ids, slice):
        return list(range(len(names)))[ids]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    ids = list(ids)
    if not ids or any(not isinstance(i, int) or not 0 <= i < len(names) for i in ids):
        raise ValueError(f"Invalid {field} IDs: {ids}")
    return sorted(set(ids))


def _inertia(value, n, bodies, label):
    result = _rows(value, n, label)
    if result.shape == (n, bodies, 9):
        matrix = result.reshape(n, bodies, 3, 3)
    elif result.shape == (n, bodies, 3, 3):
        matrix = result
    else:
        raise ValueError(f"Unexpected {label} shape: {tuple(result.shape)}")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=1e-8, rtol=1e-5):
        raise ValueError(f"Nonsymmetric {label}")
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.transpose(-1, -2)) / 2)
    if (eigenvalues <= 0).any():
        raise ValueError(f"Nonpositive {label} eigenvalue")
    return matrix, eigenvalues


def _mass_bounds(cfg, nominal, names, profile):
    low, high = nominal.clone(), nominal.clone()
    audit, claimed = {}, set()
    for name, event in vars(cfg.events).items():
        if event is None:
            continue
        params = getattr(event, "params", {})
        if "mass_distribution_params" not in params:
            continue
        if profile == "nominal":
            raise ValueError(f"Nominal profile still enables mass event {name}")
        if getattr(event, "mode", None) != "startup":
            raise ValueError(f"Mass audit supports startup events only: {name}")
        distribution = params.get("distribution", "uniform")
        if distribution not in ("uniform", "log_uniform"):
            raise ValueError(f"Cannot bound mass distribution {distribution!r}: {name}")
        bounds = params["mass_distribution_params"]
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid mass range: {name}")
        operation = params.get("operation")
        if operation not in ("add", "scale"):
            raise ValueError(f"Unsupported mass operation: {operation!r}")
        if distribution == "log_uniform" and bounds[0] <= 0:
            raise ValueError(f"Nonpositive log-uniform mass range: {name}")
        asset = params.get("asset_cfg")
        if asset is None or getattr(asset, "name", "robot") != "robot":
            raise ValueError(f"Mass event must explicitly target robot: {name}")
        ids = _selector(asset, names, "body")
        if claimed.intersection(ids):
            # Isaac versions may reset selected masses from their defaults on
            # each event. Do not guess composition semantics for overlapping DR.
            raise ValueError(f"Overlapping mass events require explicit semantics: {name}")
        claimed.update(ids)
        if operation == "add":
            low[:, ids] += bounds[0]
            high[:, ids] += bounds[1]
        else:
            if bounds[0] <= 0:
                raise ValueError(f"Nonpositive mass scale range: {name}")
            low[:, ids] *= bounds[0]
            high[:, ids] *= bounds[1]
        audit[name] = dict(mode="startup", distribution=distribution, operation=operation,
            range=list(bounds), body_indices=ids, body_names=[names[i] for i in ids])
    return low, high, audit


def _actuator_config(cfg, design_index, design_count):
    result = {}
    fields = ("joint_names_expr", "stiffness", "damping", "armature", "friction",
        "effort_limit", "effort_limit_sim", "velocity_limit", "velocity_limit_sim",
        "torque_speed_lookup")
    actuators = getattr(getattr(getattr(cfg, "scene", None), "robot", None), "actuators", {})
    for name, actuator in actuators.items():
        row = {field: _json(getattr(actuator, field)) for field in fields if hasattr(actuator, field)}
        spatial = getattr(actuator, "spatial_parameters", None)
        if spatial:
            if len(spatial) != design_count:
                raise ValueError(f"Actuator {name} spatial parameter routing mismatch")
            row["spatial_parameters"] = _json(spatial[design_index])
            # These shared config fields describe design zero; the per-design
            # spatial override is the executed nominal source for this group.
            for field in ("armature", "effort_limit", "effort_limit_sim", "velocity_limit", "velocity_limit_sim"):
                if field in spatial[design_index]:
                    row[field] = _json(spatial[design_index][field])
            if "curve" in spatial[design_index]:
                row["torque_speed_lookup"] = _json(spatial[design_index]["curve"])
        result[name] = row
    return result


def audit_physics(raw, cfg, designs, *, profile):
    """Check nominal identity and legal startup mass DR, returning JSON data.

    ``designs`` are the ordered selected bank rows; environment i belongs to
    designs[i % len(designs)]. Identity still requires the caller's USD design
    tag and geometry checks. Missing nominal mass/inertia fails explicitly.
    The report distinguishes actual buffers from configuration and unavailable
    optional getters. It never alters physics, observations, or random state.
    """
    if profile not in ("nominal", "robust"):
        raise ValueError(f"Unknown physics audit profile: {profile}")
    if not designs or len({d["id"] for d in designs}) != len(designs):
        raise ValueError("Physics audit requires nonempty unique designs")
    robot = raw.robot
    view, data = robot.root_physx_view, robot.data
    mass = _rows(view.get_masses(), raw.num_envs, "actual mass")
    if mass.ndim != 2:
        raise ValueError(f"Actual masses must have shape (environments, bodies), got {tuple(mass.shape)}")
    n, bodies = mass.shape
    names = list(robot.body_names)
    if len(names) != bodies or n < len(designs) or n % len(designs):
        raise ValueError("Physics environment/body/design routing dimensions differ")
    if not hasattr(data, "default_mass"):
        raise ValueError("Missing robot.data.default_mass: cannot verify nominal bank/URDF identity")
    nominal = _rows(data.default_mass, n, "nominal mass")
    if nominal.shape != mass.shape or (nominal <= 0).any() or (mass <= 0).any():
        raise ValueError("Actual and nominal body masses must have matching shape and be positive")
    if not hasattr(data, "default_inertia"):
        raise ValueError("Missing robot.data.default_inertia: cannot verify nominal inertia")
    inertia, eigenvalues = _inertia(view.get_inertias(), n, bodies, "actual inertia")
    nominal_inertia, _ = _inertia(data.default_inertia, n, bodies, "nominal inertia")
    if profile == "nominal" and not torch.allclose(inertia, nominal_inertia, rtol=1e-5, atol=1e-8):
        raise ValueError("Nominal profile actual inertia differs from default inertia")
    low, high, mass_events = _mass_bounds(cfg, nominal, names, profile)
    outside = (mass < low - MASS_ATOL) | (mass > high + MASS_ATOL)
    if outside.any():
        env_id, body_id = outside.nonzero()[0].tolist()
        raise ValueError(f"{profile} mass outside allowed range: env={env_id}, body={names[body_id]}, "
            f"actual={mass[env_id, body_id]:.8g}, range=[{low[env_id, body_id]:.8g}, {high[env_id, body_id]:.8g}]")

    properties, unavailable = {}, []
    def record(label, owner, attribute, *, getter=False, nonnegative=False):
        if not hasattr(owner, attribute):
            unavailable.append(label)
            return
        value = getattr(owner, attribute)
        value = value() if getter else value
        if value is None:
            unavailable.append(label)
            return
        value = _rows(value, n, label)
        if nonnegative and (value < 0).any():
            raise ValueError(f"Negative {label}")
        properties[label] = value

    record("physx_coms", view, "get_coms", getter=True)
    record("physx_material_properties", view, "get_material_properties", getter=True, nonnegative=True)
    for field in ("stiffness", "damping", "armature", "friction_coeff"):
        record("data_joint_" + field, data, "joint_" + field, nonnegative=True)
        record("default_joint_" + field, data, "default_joint_" + field, nonnegative=True)
    for field, getter in (("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings"),
            ("armature", "get_dof_armatures"), ("friction", "get_dof_friction_coefficients")):
        record("physx_joint_" + field, view, getter, getter=True, nonnegative=True)
    for name, actuator in getattr(robot, "actuators", {}).items():
        for field in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit"):
            record(f"actuator/{name}/{field}", actuator, field, nonnegative=True)

    design_audit = []
    for index, design in enumerate(designs):
        expected = float(design["total_mass_kg"])
        xml = ET.parse(design["urdf"]).getroot()
        urdf_values = [float(node.attrib["value"]) for node in xml.findall("link/inertial/mass")]
        if not urdf_values or any(not math.isfinite(x) or x < 0 for x in urdf_values):
            raise ValueError(f"Invalid URDF masses for {design['id']}")
        urdf_total = sum(urdf_values)
        if not math.isfinite(expected) or expected <= 0 or abs(urdf_total - expected) > TOTAL_MASS_ATOL:
            raise ValueError(f"Nominal bank/URDF total mass mismatch for {design['id']}")
        env_ids = list(range(index, n, len(designs)))
        for env_id in env_ids:
            if abs(float(nominal[env_id].sum()) - expected) > TOTAL_MASS_ATOL:
                raise ValueError(f"Nominal default_mass/bank total mass mismatch: env={env_id}, design={design['id']}")
            if not torch.allclose(nominal[env_id], nominal[index], rtol=0, atol=MASS_ATOL):
                raise ValueError(f"Nominal body masses differ between copies of {design['id']}")
        defaults = {}
        for field, values in properties.items():
            if field.startswith("default_joint_"):
                for env_id in env_ids:
                    if not torch.allclose(values[env_id], values[index], rtol=1e-6, atol=1e-8):
                        raise ValueError(f"Nominal actuator parameters differ between copies of {design['id']}")
                defaults[field] = values[index].tolist()
        actuator_nominal = dict(joint_names=list(getattr(robot, "joint_names", [])),
            defaults=defaults, configuration=_actuator_config(cfg, index, len(designs)))
        fingerprint = hashlib.sha256(json.dumps(actuator_nominal, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()
        design_audit.append(dict(design_id=design["id"], environment_ids=env_ids,
            bank_total_mass_kg=expected, urdf_total_mass_kg=urdf_total,
            nominal_body_masses_kg=nominal[index].tolist(),
            nominal_actuator_parameters=actuator_nominal, nominal_actuator_fingerprint=fingerprint))

    events = {}
    for name, event in vars(cfg.events).items():
        if getattr(event, "mode", None) not in ("startup", "interval"):
            continue
        params = dict(getattr(event, "params", {}))
        asset = params.pop("asset_cfg", None)
        events[name] = dict(mode=event.mode, parameters=_json(params))
        if getattr(event, "interval_range_s", None) is not None:
            events[name]["interval_range_s"] = _json(event.interval_range_s)
        if asset is not None:
            events[name]["asset_selection"] = {field: _json(getattr(asset, field))
                for field in ("name", "body_names", "joint_names") if hasattr(asset, field)}
    environments = []
    for i in range(n):
        environments.append(dict(environment_id=i, design_id=designs[i % len(designs)]["id"],
            actual_body_masses_kg=mass[i].tolist(), actual_total_mass_kg=float(mass[i].sum()),
            nominal_total_mass_kg=float(nominal[i].sum()),
            allowed_body_mass_min_kg=low[i].tolist(), allowed_body_mass_max_kg=high[i].tolist(),
            actual_inertias_kg_m2=inertia[i].tolist(), inertia_eigenvalues=eigenvalues[i].tolist(),
            properties={field: values[i].tolist() for field, values in properties.items()}))
    result = dict(version="shared_physics_audit_v1", profile=profile, timing="after_startup_before_payload",
        nominal_bank_urdf_mass_verified=True, actual_body_mass_ranges_verified=True,
        inertia_positive_definite_verified=True, actor_inputs_modified=False,
        body_names=names, joint_names=list(getattr(robot, "joint_names", [])),
        routing="environment_index_modulo_design_count; caller verifies USD tags and geometry",
        tolerances=dict(body_mass_kg=MASS_ATOL, total_mass_kg=TOTAL_MASS_ATOL),
        mass_events=mass_events, configured_events=events, designs=design_audit, environments=environments,
        unavailable_properties=unavailable,
        scope="COM/material/gain/armature are finite-value snapshots; their configured ranges are recorded, "
              "not fully verified against PhysX. Explicit actuator gains and PhysX drive gains are distinct.")
    json.dumps(result, allow_nan=False)
    return result
