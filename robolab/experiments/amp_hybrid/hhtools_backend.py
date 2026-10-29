"""Pinned HHTools CPU adapter, isolated from the Isaac Python environment.

Geometric R2R only. Does not claim contact dynamics or motor feasibility.
"""
from dataclasses import asdict, replace
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class Backend:
    def __init__(self, plan, output):
        from importlib.metadata import version
        from amp_references import write_new
        versions = {n:version(n) for n in ("hhtools", "newton", "warp-lang", "mujoco", "numpy", "yourdfpy")}
        for name, expected in (("hhtools", "0.1.0"), ("newton", "1.1.0"), ("warp-lang", "1.12.1")):
            if versions[name] != expected:
                raise ValueError(f"unvalidated HHTools dependency: {name} {versions[name]}")
        write_new(output / "runtime_versions.json", versions)
        # Must be set before importing Warp/Newton/HHTools.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HHTOOLS_IK_USE_CUDA_GRAPH"] = "0"
        for key, sub in (("WARP_CACHE_DIR", "warp"), ("HHTOOLS_ROBOT_DIR", "robots"),
                         ("HHTOOLS_CACHE_DIR", "cache"), ("XDG_CONFIG_HOME", "config")):
            os.environ[key] = str(output / "runtime" / sub)
        import warp as wp
        wp.config.enable_cuda = False
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_DIR"]
        wp.init()
        wp.set_device("cpu")
        from hhtools.retarget import robot_to_robot as r2r
        from scipy.spatial.transform import Rotation
        import yaml
        self.r2r, self.Rotation, self.yaml = r2r, Rotation, yaml
        self.plan, self.out = plan, output
        self.names = plan["joint_names"]
        (output / "models").mkdir()
        self.source = self.model("source", plan["source_urdf"])

    def model(self, name, path):
        from hhtools.robot.base import RobotPreset
        from hhtools.robot.loader import load_robot
        from hhtools.retarget.newton_basic.robot_model import build_newton_model
        bundle = self.out / "models" / name
        bundle.mkdir()
        tree = ET.parse(path)
        # Resolve only URI paths, never alter origins, geometry or inertias.
        for mesh in tree.findall(".//mesh"):
            text = mesh.get("filename")
            resolved = Path(text.removeprefix("file://"))
            if not resolved.is_absolute():
                resolved = (Path(path).parent / resolved).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"unresolved mesh, refusing guessed substitution: {text}")
            mesh.set("filename", resolved.as_posix())
        derived = bundle / "robot.urdf"
        tree.write(derived, encoding="utf8", xml_declaration=True)
        mapping = {"hips": "base_link", "chest": "torso_link"}
        for side in ("left", "right"):
            for canonical, link in (("shoulder", "arm_roll"), ("elbow", "elbow_pitch"),
                                    ("wrist", "elbow_yaw"), ("hip", "thigh_roll"),
                                    ("knee", "knee"), ("ankle", "ankle_roll")):
                mapping[side+"_"+canonical] = side+"_"+link+"_link"
        template = self.yaml.safe_load((Path(self.plan["hhtools_root"]) /
                    "configs/robots/_template/robot.yaml").read_text(encoding="utf8"))
        preset = RobotPreset(name=name, display_name=name, root_dir=bundle, urdf_path=derived,
                             dof_order=tuple(self.names), ik_map=mapping, weights=template["weights"],
                             feet={"left_contact_link": "left_ankle_roll_link",
                                   "right_contact_link": "right_ankle_roll_link", "ground_z": 0.},
                             meta={"retarget": {"num_initialization_frames": 10,
                                               "num_stabilization_frames": 5}})
        model = load_robot(preset, compile_mjcf=False)
        ctx = build_newton_model(model)
        labels = [str(n).split("/")[-1] for n in ctx.model.joint_label]
        starts = ctx.model.joint_q_start.numpy()
        coordinates = sorted((int(s), n) for s, n in zip(starts, labels) if n in self.names)
        if [s for s, _ in coordinates] != list(range(7, 30)):
            raise ValueError("unexpected Newton coordinate layout")
        native = [n for _, n in coordinates]
        by_name = {j.name: j for j in model.actuated_joints}
        if len(native) != 23 or set(native) != set(self.names):
            raise ValueError("not the same 23 actuated joints")
        # This pinned upstream pipeline labels raw Newton columns with preset
        # order. Correct internally, then explicitly permute back for AMP.
        model._preset = replace(preset, dof_order=tuple(native))
        model._actuated = tuple(by_name[n] for n in native)
        return model

    def fk_and_soles(self, model, q, key_bodies=None):
        from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q
        from amp_references import KEY_BODIES
        vertices = self.foot_vertices(model.preset.urdf_path)
        points, soles = [], []
        for row in q:
            tx = _collect_link_transforms_at_q(model, dict(zip(self.names, row[7:])))
            rot = self.Rotation.from_quat(row[3:7]).as_matrix()
            points.append([rot @ tx[n][:3, 3]+row[:3] for n in (key_bodies or KEY_BODIES)])
            soles.append([float(((v @ tx[n][:3, :3].T+tx[n][:3, 3]) @ rot.T+row[:3])[:, 2].min())
                          for n, v in vertices.items()])
        return np.asarray(points), np.asarray(soles)

    def foot_vertices(self, path):
        import trimesh
        result = {}
        root = ET.parse(path).getroot()
        for side in ("left", "right"):
            name = side+"_ankle_roll_link"
            link = next(l for l in root.findall("link") if l.get("name") == name)
            parts = []
            for col in link.findall("collision"):
                geom = col.find("geometry")
                mesh, box = geom.find("mesh"), geom.find("box")
                if mesh is not None:
                    p = Path(mesh.get("filename"))
                    m = trimesh.load(str(p), force="mesh", process=False)
                    pts = np.asarray(m.convex_hull.vertices)*np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                elif box is not None:
                    half = np.fromstring(box.get("size"), sep=" ")/2
                    pts = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
                else:
                    raise ValueError("unsupported foot collision primitive")
                origin = col.find("origin")
                if origin is not None:
                    rot = self.Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                    pts = pts @ rot.T+np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                parts.append(pts)
            if not parts:
                raise ValueError("missing actual foot collision geometry")
            result[name] = np.concatenate(parts)
        return result

    def limits(self, model):
        joints = {j.name: j for j in model.actuated_joints}
        return (np.asarray([joints[n].limit_lower if joints[n].limit_lower is not None else -np.inf for n in self.names]),
                np.asarray([joints[n].limit_upper if joints[n].limit_upper is not None else np.inf for n in self.names]))

    def solve(self, target, q, fps, clip):
        from hhtools.core.grounding import SOURCE_FLOOR_META_KEY
        from hhtools.retarget.newton_basic import NewtonBasicPipeline
        from hhtools.retarget.newton_basic.robot_model import IKMapping
        from hhtools.robot.retarget_profile import build_feet_stabilizer_config, build_pipeline_config_for_preset
        from amp_references import reorder_solver
        motion = self.r2r.source_trajectory_to_motion(self.source, q, tuple(self.names), framerate=fps, name=clip)
        motion = replace(motion, meta={**motion.meta, SOURCE_FLOOR_META_KEY: 0.})
        scaler, ref = self.r2r._build_scaler_config(self.source, target, target.zero_configuration())
        # Source data are already world-frame ROBOT FK, not human ankle
        # markers. Do not introduce a second sole-thickness/pelvis offset.
        # Root-anchored upstream scaling preserves the first root height even
        # when legs grow. Calibrate the height difference using the SAME rest
        # pose's actual collision soles, not the minimum frame in a clip.
        # Identity gives exactly zero. A constant geometry offset preserves
        # support/flight timing; it does not snap each frame onto the floor.
        rest = np.zeros((1, 30)); rest[0, 6] = 1.
        rest[0, 7:] = [self.source.zero_configuration()[n] for n in self.names]
        _, source_rest_z = self.fk_and_soles(self.source, rest)
        target_rest = rest.copy()
        target_rest[0, 7:] = [target.zero_configuration()[n] for n in self.names]
        _, target_rest_z = self.fk_and_soles(target, target_rest)
        geometric_height_offset = float(source_rest_z.min()-target_rest_z.min())
        scaler = replace(scaler, root_z_offset=geometric_height_offset, robot_pelvis_height=None)
        reference_key = "robot_"+self.source.preset.name
        cfg = replace(build_pipeline_config_for_preset(target.preset, reference_key, ik_iterations=24),
                      max_joint_velocity=0., smooth_joint_filter_weight=0., apply_feet_stabilizer=False,
                      post_ik_foot_clamps=False, clip_floor_snap=False)
        pipe = NewtonBasicPipeline(target, scaler_config=scaler, pipeline_config=cfg,
            feet_stabilizer_config=build_feet_stabilizer_config(target.preset, reference_key, model=target),
            human_height=float(ref.height_m), source_to_canonical={n:n for n in ref.joint_names}, configure_warp=False)
        pipe.ik_mapping = IKMapping(entries=tuple(e for e in pipe.ik_mapping.entries
                                                   if e.canonical_name in target.preset.ik_map))
        result = pipe.run(motion, progress_callback=lambda cur, total:
                          print(clip, cur, total, flush=True) if cur == total or cur % 200 == 0 else None)
        canonical = reorder_solver(result.joint_q, list(result.dof_names), self.names)
        return canonical, {"scaler": asdict(scaler), "pipeline": asdict(cfg),
                           "rest_collision_sole_height_offset_m": geometric_height_offset,
                           "solver_joint_names": list(result.dof_names), "metadata": result.meta}
