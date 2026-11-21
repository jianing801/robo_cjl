"""A straight evaluation flight sized for the measurement window, in SI units."""
import math

COURSE = 'timed_continuous_ascent'
TRAVEL_MARGIN = 2.
REAR_TREADS = 8
WIDTH_M = 40.


def evaluation_flight(duration, speed, depth, margin=TRAVEL_MARGIN):
    if not all(math.isfinite(v) and v > 0 for v in (duration, speed, depth, margin)) or margin <= 1:
        raise ValueError('A timed stair flight needs positive dimensions and a travel margin above one')
    count = math.ceil(duration * speed * margin / depth)
    return dict(forward_steps=count, length_m=(REAR_TREADS + 1 + count) * depth,
                origin_x_m=(REAR_TREADS + .5) * depth)


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
        transform = trimesh.transformations.translation_matrix(((left + right) / 2, width / 2, z / 2))
        meshes.append(trimesh.creation.box((right - left, width, z), transform=transform))
    return meshes, np.array(((REAR_TREADS + .5) * depth, width / 2, (REAR_TREADS + 1) * height))


def flight_tile(height, depth, proportion):
    from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg
    from isaaclab.utils import configclass

    @configclass
    class FlightCfg(MeshInvertedPyramidStairsTerrainCfg):
        function = flight_mesh

    return FlightCfg(proportion=proportion, step_height_range=(height, height), step_width=depth,
                     border_width=0., holes=False)


def outside_flight(x, y, origin_x, length):
    """Positions relative to the spawn origin. Keep a tread-width buffer for the feet."""
    clearance = .35
    return ((x <= clearance - origin_x) | (x >= length - origin_x - clearance)
            | (abs(y) >= WIDTH_M / 2 - clearance))
