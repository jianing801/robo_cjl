"""Attach a separate rigid backpack link to a generated URDF, without editing its source."""

from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET


DEFAULT_SIZE_M = (0.12, 0.15, 0.20)
DEFAULT_CENTER_M = (-0.17, 0.0, 0.11)


def sample_backpack_properties(
    count: int,
    *,
    seed: int = 2026,
    mass_range_kg: tuple[float, float] = (10.0, 10.0),
    inertia_size_scale_range: tuple[float, float] = (0.9, 1.1),
    mount_surface_offset_max_m: tuple[float, float] = (0.005, 0.005),
    com_offset_max_m: tuple[float, float, float] = (0.015, 0.010, 0.020),
) -> list[dict]:
    """Sample fixed-per-environment backpack mass properties.

    The visible/collision box and its mounting transform are shared by all
    environments. Per-environment mount tolerance is represented as an
    equivalent shift of the backpack CoM, while equivalent box dimensions are
    used only to calculate physically consistent principal inertias. This
    avoids importing a distinct full-robot URDF for every environment.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    mass_low, mass_high = mass_range_kg
    if not (
        math.isfinite(mass_low) and math.isfinite(mass_high)
        and 0 < mass_low <= mass_high
    ):
        raise ValueError("mass_range_kg must be finite, positive, and ordered")
    low, high = inertia_size_scale_range
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("inertia_size_scale_range must be finite, positive, and ordered")
    if len(mount_surface_offset_max_m) != 2 or not all(
        math.isfinite(value) and value >= 0 for value in mount_surface_offset_max_m
    ):
        raise ValueError("mount_surface_offset_max_m must contain two finite nonnegative values")
    if len(com_offset_max_m) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in com_offset_max_m
    ):
        raise ValueError("com_offset_max_m must contain three finite nonnegative values")
    if any(
        mount + com >= dimension / 2.0
        for mount, com, dimension in zip(
            (0.0, *mount_surface_offset_max_m), com_offset_max_m, DEFAULT_SIZE_M,
        )
    ):
        raise ValueError("Combined mount and CoM offsets must stay inside the fixed backpack box")
    rng = random.Random(seed)
    sx, sy, sz = DEFAULT_SIZE_M
    properties = []
    for index in range(count):
        mass = rng.uniform(mass_low, mass_high)
        inertia_scale = tuple(rng.uniform(low, high) for _ in range(3))
        inertia_size = (sx * inertia_scale[0], sy * inertia_scale[1], sz * inertia_scale[2])
        mount_offset = (
            0.0,
            rng.uniform(-mount_surface_offset_max_m[0], mount_surface_offset_max_m[0]),
            rng.uniform(-mount_surface_offset_max_m[1], mount_surface_offset_max_m[1]),
        )
        internal_com_offset = tuple(
            rng.uniform(-maximum, maximum) for maximum in com_offset_max_m
        )
        effective_com_offset = (
            internal_com_offset[0],
            mount_offset[1] + internal_com_offset[1],
            mount_offset[2] + internal_com_offset[2],
        )
        if any(abs(offset) >= dimension / 2.0 for offset, dimension in zip(effective_com_offset, DEFAULT_SIZE_M)):
            raise ValueError(
                f"Effective CoM offset {effective_com_offset} must stay inside fixed box {DEFAULT_SIZE_M}"
            )
        ixx = mass * (inertia_size[1] ** 2 + inertia_size[2] ** 2) / 12.0
        iyy = mass * (inertia_size[0] ** 2 + inertia_size[2] ** 2) / 12.0
        izz = mass * (inertia_size[0] ** 2 + inertia_size[1] ** 2) / 12.0
        com_position = tuple(center + offset for center, offset in zip(DEFAULT_CENTER_M, effective_com_offset))
        properties.append({
            "env_id": index,
            "mass_kg": mass,
            "size_scale": inertia_scale,
            "inertia_size_scale": inertia_scale,
            "size_m": DEFAULT_SIZE_M,
            "inertia_equivalent_size_m": inertia_size,
            "inertia_tensor_kg_m2": (ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
            "mount_offset_on_back_m": mount_offset,
            "mount_position_torso_m": DEFAULT_CENTER_M,
            "geometry_center_torso_m": DEFAULT_CENTER_M,
            "internal_com_offset_from_box_center_m": internal_com_offset,
            "com_offset_from_geometry_center_m": effective_com_offset,
            "com_position_torso_m": com_position,
            "center_m": DEFAULT_CENTER_M,
        })
    return properties


def apply_backpack_properties(robot, properties: list[dict]) -> dict:
    """Apply startup-sampled mass, CoM, and inertia to the shared robot asset.

    Isaac Lab's PhysX view addresses environment instances in a batch. The
    sampled properties are written once after scene creation and are not
    changed by episode resets.
    """
    import torch

    if "backpack_link" not in robot.data.body_names:
        raise RuntimeError("backpack_link was merged during URDF import; set merge_fixed_joints=False")
    body_id = robot.data.body_names.index("backpack_link")
    view = robot.root_physx_view
    masses = view.get_masses().clone()
    inertias = view.get_inertias().clone()
    coms = view.get_coms().clone()
    if masses.shape[0] != len(properties):
        raise RuntimeError(
            f"Expected one backpack mass sample per environment ({len(properties)}), got {masses.shape[0]}"
        )
    env_ids = torch.arange(len(properties), dtype=torch.long, device="cpu")
    masses[:, body_id] = torch.as_tensor(
        [item["mass_kg"] for item in properties], dtype=masses.dtype, device=masses.device,
    )
    inertias[:, body_id, :] = torch.as_tensor(
        [item["inertia_tensor_kg_m2"] for item in properties],
        dtype=inertias.dtype,
        device=inertias.device,
    )
    coms[:, body_id, :3] = torch.as_tensor(
        [item["com_offset_from_geometry_center_m"] for item in properties],
        dtype=coms.dtype,
        device=coms.device,
    )
    view.set_masses(masses, env_ids)
    view.set_coms(coms, env_ids)
    view.set_inertias(inertias, env_ids)

    actual_masses = view.get_masses()[:, body_id]
    requested_masses = torch.as_tensor(
        [item["mass_kg"] for item in properties], dtype=actual_masses.dtype, device=actual_masses.device,
    )
    if not torch.allclose(actual_masses, requested_masses, rtol=1e-5, atol=1e-5):
        raise RuntimeError("PhysX backpack mass write did not match the requested per-environment values")
    # The posture allowance follows the verified actual payload of each env.
    # Cache on the asset to avoid a CPU PhysX read in every reward calculation.
    robot.backpack_mass_kg = actual_masses.to(robot.device).clone()
    robot.backpack_com_x_m = torch.as_tensor(
        [item["com_position_torso_m"][0] for item in properties],
        dtype=actual_masses.dtype,
        device=robot.device,
    )
    return {
        "mass_min_kg": float(actual_masses.min().item()),
        "mass_max_kg": float(actual_masses.max().item()),
        "environment_count": len(properties),
        "body_id": body_id,
    }


def _numbers(raw: str, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected {count} finite numbers, got {raw!r}")
    return values


def write_backpack_urdf(
    source_path: str | Path,
    output_path: str | Path,
    mass_kg: float,
    *,
    size_m: tuple[float, float, float] = DEFAULT_SIZE_M,
    center_m: tuple[float, float, float] = DEFAULT_CENTER_M,
    geometry_center_torso_m: tuple[float, float, float] | None = None,
    com_offset_from_geometry_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_position_torso_m: tuple[float, float, float] | None = None,
) -> dict:
    """Write a separate URDF with a fixed, independently mass-randomizable pack link.

    The backpack link frame is at the box geometry center. The fixed joint
    carries the box-center position relative to the torso, while the inertial
    origin carries the internal CoM offset. PhysX can then randomize pack mass
    without merging the pack into the torso's mass properties.
    """
    source, output = Path(source_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("Backpack output must differ from the source URDF")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Backpack mass must be positive and finite")
    if len(size_m) != 3 or len(center_m) != 3:
        raise ValueError("Backpack size and center must each have three coordinates")
    if not all(math.isfinite(value) and value > 0 for value in size_m):
        raise ValueError("Backpack dimensions must be positive and finite")
    if len(com_offset_from_geometry_center_m) != 3 or not all(
        math.isfinite(value) for value in com_offset_from_geometry_center_m
    ):
        raise ValueError("Backpack CoM offset must contain three finite coordinates")
    if any(abs(offset) >= dimension / 2.0 for offset, dimension in zip(com_offset_from_geometry_center_m, size_m)):
        raise ValueError("Backpack CoM offset must stay strictly inside the box")
    geometry_center = geometry_center_torso_m if geometry_center_torso_m is not None else center_m
    if len(geometry_center) != 3 or not all(math.isfinite(value) for value in geometry_center):
        raise ValueError("Backpack geometry center must contain three finite coordinates")
    if mount_position_torso_m is not None and (
        len(mount_position_torso_m) != 3 or not all(math.isfinite(value) for value in mount_position_torso_m)
    ):
        raise ValueError("Backpack mount position must contain three finite coordinates")

    tree = ET.parse(source)
    root = tree.getroot()
    # The output may live in a different temporary directory. Preserve mesh
    # references by resolving relative paths against the original URDF.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename and not filename.startswith("package://") and not Path(filename).is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))
    links = [link for link in root.findall("link") if link.get("name") == "torso_link"]
    if len(links) != 1:
        raise ValueError("Expected exactly one torso_link")
    torso = links[0]
    if root.find("link[@name='backpack_link']") is not None or root.find("joint[@name='backpack_mount_joint']") is not None:
        raise ValueError("URDF already contains a backpack link")
    inertial = torso.find("inertial")
    if inertial is None or inertial.find("origin") is None or inertial.find("mass") is None or inertial.find("inertia") is None:
        raise ValueError("torso_link requires a complete inertial specification")
    origin, mass = inertial.find("origin"), inertial.find("mass")
    if any(abs(value) > 1e-9 for value in _numbers(origin.get("rpy", "0 0 0"), 3)):
        raise ValueError("Rotated torso inertia frame is not supported")
    old_mass = float(mass.get("value"))
    if not math.isfinite(old_mass) or old_mass <= 0:
        raise ValueError("Invalid original torso mass")
    sx, sy, sz = size_m
    link = ET.SubElement(root, "link", {"name": "backpack_link"})
    pack_inertial = ET.SubElement(link, "inertial")
    inertial_xyz = " ".join(f"{value:.12g}" for value in com_offset_from_geometry_center_m)
    ET.SubElement(pack_inertial, "origin", {"xyz": inertial_xyz, "rpy": "0 0 0"})
    ET.SubElement(pack_inertial, "mass", {"value": f"{mass_kg:.12g}"})
    ET.SubElement(pack_inertial, "inertia", {
        "ixx": f"{mass_kg * (sy * sy + sz * sz) / 12:.12g}",
        "ixy": "0", "ixz": "0",
        "iyy": f"{mass_kg * (sx * sx + sz * sz) / 12:.12g}",
        "iyz": "0",
        "izz": f"{mass_kg * (sx * sx + sy * sy) / 12:.12g}",
    })
    size = " ".join(f"{value:.12g}" for value in size_m)
    visual = ET.SubElement(link, "visual", {"name": "backpack_load"})
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(ET.SubElement(visual, "geometry"), "box", {"size": size})
    ET.SubElement(ET.SubElement(visual, "material", {"name": "backpack_blue"}), "color", {"rgba": "0.12 0.30 0.80 1"})
    collision = ET.SubElement(link, "collision", {"name": "backpack_load"})
    ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", {"size": size})
    mount = ET.SubElement(root, "joint", {"name": "backpack_mount_joint", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "torso_link"})
    ET.SubElement(mount, "child", {"link": "backpack_link"})
    ET.SubElement(mount, "origin", {
        "xyz": " ".join(f"{value:.12g}" for value in geometry_center), "rpy": "0 0 0",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "source_urdf": str(source), "output_urdf": str(output),
        "backpack_mass_kg": mass_kg, "backpack_mass_body": "backpack_link",
        "backpack_size_m": size_m,
        "backpack_center_torso_m": geometry_center,
        "backpack_geometry_center_torso_m": geometry_center,
        "backpack_com_offset_from_geometry_center_m": com_offset_from_geometry_center_m,
        "backpack_com_torso_m": tuple(
            center + offset for center, offset in zip(geometry_center, com_offset_from_geometry_center_m)
        ),
        "backpack_mount_position_torso_m": mount_position_torso_m,
        "torso_mass_kg": old_mass,
        "urdf_import_must_preserve_fixed_link": True,
    }
