"""Offline native Z=0 infinite collision plane; source physics material retained."""
from pathlib import Path
from isaaclab.terrains import TerrainImporter
import isaaclab.sim as sim_utils


class OfflineNativePlane(TerrainImporter):
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        prim_path = self.cfg.prim_path + "/" + name
        if prim_path in self.terrain_prim_paths:
            raise ValueError("duplicate native plane")
        self.terrain_prim_paths.append(prim_path)
        asset = Path(__file__).resolve().parents[1] / "data/native_probe_ground.usda"
        cfg = sim_utils.GroundPlaneCfg(usd_path=str(asset), physics_material=self.cfg.physics_material,
                                      size=size, color=None)
        cfg.func(prim_path, cfg)
