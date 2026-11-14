"""Native mixed-asset spawn semantics, without GUI undo history for temporary templates."""
def spawn_without_template_undo(prim_path, cfg, translation=None, orientation=None,
                                clone_in_fabric=False, replicate_physics=False):
    import re
    import random
    import time
    import carb
    from pxr import Sdf, Usd
    import isaaclab.sim as sim
    stage = sim.get_current_stage()
    root_path, asset_name = prim_path.rsplit("/", 1)
    parents = (sim.find_matching_prim_paths(root_path)
               if root_path and re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None else [root_path])
    if not parents:
        raise RuntimeError(f"No source environments: {root_path}")
    template = sim.get_next_free_prim_path("/World/Template", stage=stage)
    sim.create_prim(template, "Scope", stage=stage)
    prototypes = []
    started = time.perf_counter()
    print(f'SPATIAL_SPAWN_BEGIN assets={len(cfg.assets_cfg)} environments={len(parents)}', flush=True)
    for index, asset in enumerate(cfg.assets_cfg):
        if cfg.semantic_tags is not None:
            if asset.semantic_tags is None:
                asset.semantic_tags = cfg.semantic_tags
            else:
                asset.semantic_tags += cfg.semantic_tags
        for name in ("mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"):
            value = getattr(cfg, name)
            if hasattr(asset, name) and value is not None:
                setattr(asset, name, value)
        path = f"{template}/Asset_{index:04d}"
        asset.func(path, asset, translation=translation, orientation=orientation,
                   clone_in_fabric=clone_in_fabric, replicate_physics=replicate_physics)
        prototypes.append(path)
        if (index+1) % 50 == 0:
            print(f'SPATIAL_PROTOTYPES_READY {index+1}', flush=True)
    paths = [f"{parent}/{asset_name}" for parent in parents]
    with Sdf.ChangeBlock():
        for index, path in enumerate(paths):
            spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), path)
            source = random.choice(prototypes) if cfg.random_choice else prototypes[index % len(prototypes)]
            Sdf.CopySpec(spec.layer, Sdf.Path(source), spec.layer, Sdf.Path(path))
            if (index+1) % 512 == 0:
                print(f'SPATIAL_COPY_QUEUED {index+1}', flush=True)
        # Disable only our temporary prototype scope in the SAME Sdf transaction.
        # Inactive prims are excluded from physics and normal stage traversal.
        # Retaining this inert specification avoids a second destructive namespace
        # edit over the entire heterogeneous scene. No on-disk asset is changed.
        template_spec = stage.GetRootLayer().GetPrimAtPath(template)
        if not template_spec:
            raise RuntimeError('Temporary template is not in the root layer')
        template_spec.active = False
    if stage.GetPrimAtPath(template).IsActive():
        raise RuntimeError("Temporary templates remained active")
    print(f'SPATIAL_SPAWN_COMPLETE seconds={time.perf_counter()-started:.2f}', flush=True)
    carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
    return stage.GetPrimAtPath(paths[0])
