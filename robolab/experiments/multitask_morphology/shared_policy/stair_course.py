"""A straight evaluation flight sized for the measurement window, in SI units."""
import math

COURSE = 'bottom_start_long_stairs'
TRAVEL_MARGIN = 2.
WIDTH_M = 40.
APPROACH_M = 1.


def evaluation_flight(duration, speed, depth, margin=TRAVEL_MARGIN):
    if not all(math.isfinite(v) and v > 0 for v in (duration, speed, depth, margin)) or margin <= 1:
        raise ValueError('A timed stair flight needs positive dimensions and a travel margin above one')
    count = max(2, math.ceil(duration * speed * margin / depth))
    return dict(forward_steps=count, rear_steps=count, length_m=2 * (count * depth + APPROACH_M),
                origin_x_m=count * depth + APPROACH_M, approach_m=APPROACH_M)


def flight_mesh(_difficulty, cfg):
    import numpy as np
    import trimesh
    length, width = cfg.size
    height, depth = cfg.step_height_range[0], cfg.step_width
    back = cfg.rear_steps * depth
    front = back + 2 * APPROACH_M
    def slab(left, right, top):
        bottom = -.15
        transform = trimesh.transformations.translation_matrix(
            ((left + right) / 2, width / 2, (top + bottom) / 2))
        return trimesh.creation.box((right - left, width, top - bottom), transform=transform)

    meshes = [slab(back, front, 0.)]
    for i in range(cfg.rear_steps):
        meshes.append(slab(back - (i + 1) * depth, back - i * depth, (i + 1) * height))
    for i in range(math.ceil((length - front) / depth)):
        left, right = front + i * depth, min(front + (i + 1) * depth, length)
        if right - left > 1e-8:
            meshes.append(slab(left, right, (i + 1) * height))
    return meshes, np.array((back + APPROACH_M, width / 2, 0.))


def flight_tile(height, depth, proportion, rear_steps):
    from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg
    from isaaclab.utils import configclass

    @configclass
    class FlightCfg(MeshInvertedPyramidStairsTerrainCfg):
        function = flight_mesh
        rear_steps: int = 2

    return FlightCfg(proportion=proportion, step_height_range=(height, height), step_width=depth,
                     border_width=0., holes=False, rear_steps=rear_steps)


def outside_flight(x, y, origin_x, length):
    """Positions relative to the spawn origin. Keep a tread-width buffer for the feet."""
    clearance = .35
    return ((x <= clearance - origin_x) | (x >= length - origin_x - clearance)
            | (abs(y) >= WIDTH_M / 2 - clearance))
