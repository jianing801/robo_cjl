"""Narrow name remap for fixed-merged RPO base, not new randomization settings."""


def align_merged_base_events(cfg):
    if not getattr(cfg.scene.robot.spawn, 'merge_fixed_joints', False):
        return {}
    changed = {}
    allowed = {'add_base_mass': ('base_link', ['torso_link']),
               'randomize_rigid_body_com': (['torso_link', 'base_link'], ['torso_link'])}
    for name, (original, replacement) in allowed.items():
        event = getattr(cfg.events, name, None)
        if event is None:
            continue
        asset = event.params['asset_cfg']
        if asset.body_names == replacement:
            continue
        if asset.body_names != original:
            raise ValueError('Unknown AMP merged-base selector: '+name)
        asset.body_names = list(replacement)
        changed[name] = dict(before=original, after=list(replacement))
    return changed
