"""A straight evaluation flight sized for the measurement window, in SI units."""
import math

COURSE = 'timed_centered_ascent'
TRAVEL_MARGIN = 2.
WIDTH_M = 40.


def evaluation_flight(duration, speed, depth, margin=TRAVEL_MARGIN):
    if not all(math.isfinite(v) and v > 0 for v in (duration, speed, depth, margin)) or margin <= 1:
        raise ValueError('A timed stair flight needs positive dimensions and a travel margin above one')
    count = max(2, math.ceil(duration * speed * margin / depth))
    return dict(forward_steps=count, rear_steps=count, length_m=(2 * count + 1) * depth,
                origin_x_m=(count + .5) * depth)


def flight_mesh(_difficulty, cfg):
    import numpy as np
    import trimesh
    length, width = cfg.size
    height, depth = cfg.step_height_range[0], cfg.step_width
    # A common tile size can give the last tread of a stratum a fractional width.
    meshes = []
    for index in range(math.ceil(length / depth)):
        left, right = index * depth, min((index + 1) * depth, length)
        if right - left <= 1e-8:
            break
        z = (index + 1) * height
        transform = trimesh.transformations.translation_matrix(
            ((left + right) / 2, width / 2, z / 2 - (cfg.rear_steps + 1) * height))
        meshes.append(trimesh.creation.box((right - left, width, z), transform=transform))
    return meshes, np.array(((cfg.rear_steps + .5) * depth, width / 2, 0.))


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
