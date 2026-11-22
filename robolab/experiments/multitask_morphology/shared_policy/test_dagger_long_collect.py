import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


HERE = Path(__file__).resolve().parent


def fake_native_collector(collector, *, short_horizon=False, short_run=False):
    collector.mkdir()
    (collector/'scenario_sampling.py').write_text(
        'def verify_collection_layout(raw, designs, count):\n'
        '    assert raw.num_envs == count == 4096\n', encoding='utf8')
    (collector/'collection_grid.py').write_text('def select_collection_grid(keys): return keys\n', encoding='utf8')
    (collector/'randomization.py').write_text(
        'def configure_profile(cfg, task, **options): return cfg, {}\n', encoding='utf8')
    (collector/'contract.py').write_text(
        'import hashlib, json\n'
        'from pathlib import Path\n'
        'def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n'
        'def write_new(path, value): Path(path).write_text(json.dumps(value), encoding="utf8")\n'
        'def design_ids_for_keys(plan, keys): return {task: keys for task in ("backpack", "run")}\n', encoding='utf8')
    script = '''import argparse, json, os
from pathlib import Path
from types import SimpleNamespace as NS
from contract import sha, write_new
from randomization import configure_profile
from scenario_sampling import verify_collection_layout

def load_plan(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def main():
    parser = argparse.ArgumentParser()
    run = Path(__file__).name == 'rollout_amp.py'
    if not run:
        parser.add_argument('--task', required=True, choices=['backpack'])
    for key in ('plan', 'design-ids', 'output', 'student', 'profile'):
        parser.add_argument('--' + key, required=True)
    for key in ('steps', 'stride', 'num-envs', 'seed'):
        parser.add_argument('--' + key, required=True, type=int)
    parser.add_argument('--randomization-strength', type=float)
    args = parser.parse_args()
    task = 'run' if run else args.task
    load_plan(args.plan)
    cfg, _ = configure_profile(NS(episode_length_s=20.), task, profile=args.profile,
                               strength=args.randomization_strength, seed=args.seed)
    limit = round(cfg.episode_length_s / .02)
    raw = NS(num_envs=args.num_envs, step_dt=.02, cfg=cfg, max_episode_length=limit, episode_length=limit,
             common_step_counter=0)
    designs = [{'id': key} for key in args.design_ids.split(',')]
    verify_collection_layout(raw, designs, args.num_envs)
    output = Path(args.output)
    output.mkdir()
    raw.common_step_counter += args.steps
    (output/'data.pt').write_bytes(b'data')
    rows = [dict(environment=i, design=designs[i % len(designs)]['id']) for i in range(raw.num_envs)]
    result = 'REAL_AMP_ROLLOUT_COMPLETE' if run else 'REAL_ROLLOUT_COMPLETE'
    write_new(output/'summary.json', dict(result=result, task=task, steps=args.steps, seed=args.seed,
        design_ids=args.design_ids.split(','), rows=rows, student=sha(args.student), teacher_updates=0))
    os._exit(0)

if __name__ == '__main__':
    os._exit(77)
'''
    if short_horizon:
        script = script.replace('limit = round(cfg.episode_length_s / .02)', 'limit = 1000')
    if short_run:
        script = script.replace('raw.common_step_counter += args.steps', 'raw.common_step_counter += args.steps - 1')
    for filename in ('rollout.py', 'rollout_amp.py'):
        (collector/filename).write_text(script, encoding='utf8')


def collection_command(tmp_path, task, *, held_out=False, short_horizon=False, short_run=False):
    collector = tmp_path/'archived'
    fake_native_collector(collector, short_horizon=short_horizon, short_run=short_run)
    original = [f'design_{i}' for i in range(400)]
    ids = original[:-1] + ['held_out'] if held_out else original
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'selected_structure_keys': original}), encoding='utf8')
    student = tmp_path/'student.pt'
    student.write_bytes(b'r4 student')
    output = tmp_path/'output'
    command = [sys.executable, str(HERE/'dagger_long_collect.py'), '--collector', str(collector),
               '--task', task, '--', '--plan', str(plan), '--design-ids', ','.join(ids),
               '--output', str(output), '--student', str(student), '--steps', '1000', '--stride', '5',
               '--num-envs', '4096', '--seed', '50046', '--profile', 'robust', '--randomization-strength', '1.0']
    return command, output, collector


@pytest.mark.parametrize('task', ['backpack', 'run'])
def test_native_collection_receipt_survives_shutdown(tmp_path, task):
    command, output, collector = collection_command(tmp_path, task)
    sources = {path.name: path.read_bytes() for path in collector.glob('*.py')}
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output/'long_replay_collection.json').read_bytes())
    assert receipt['kind'] == 'dagger_long_collection_v1'
    assert (receipt['task'], receipt['steps'], receipt['stride'], receipt['seed']) == (task, 1000, 5, 50046)
    assert receipt['terrain'] == 'native'
    assert receipt['environment_count'] == 4096
    assert receipt['design_count'] == 400
    assert receipt['duration_s'] == 20.
    assert receipt['configured_episode_length_s'] >= 20.04
    assert receipt['native_episode_limit_steps'] >= 1002
    assert receipt['summary_sha256'] == hashlib.sha256((output/'summary.json').read_bytes()).hexdigest()
    assert {path.name: path.read_bytes() for path in collector.glob('*.py')} == sources


def test_long_replay_refuses_a_design_outside_the_training_grid(tmp_path):
    command, output, _ = collection_command(tmp_path, 'run', held_out=True)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'original training grid' in result.stderr
    assert not output.exists()


def test_long_replay_refuses_a_short_effective_simulator_horizon(tmp_path):
    command, output, _ = collection_command(tmp_path, 'backpack', short_horizon=True)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'native episode horizon differs' in result.stderr
    assert not output.exists()


def test_long_replay_refuses_a_summary_with_fewer_executed_control_steps(tmp_path):
    command, output, _ = collection_command(tmp_path, 'run', short_run=True)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'did not execute 1000 native control steps' in result.stderr
    assert not (output/'long_replay_collection.json').exists()


def test_stair_generator_extends_all_native_strata_from_the_bottom(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    import dagger_long_collect as worker
    from randomization import stairs_spec

    spec = stairs_spec('robust')
    count = len(spec['strata'])
    generator = NS(size=(40., 40.), border_width=2., num_rows=1, num_cols=count,
                   sub_terrains={f'stairs_{i}': NS() for i in range(count)})
    cfg = NS(scene=NS(terrain=NS(terrain_generator=generator)),
             scene_context=NS(terrain_generator=generator), commands=NS(ranges=NS(lin_vel_x=(.2, .6))),
             events=NS(reset_base=NS(params={'pose_range': {'x': (-.5, .5), 'y': (-.5, .5)}})))
    monkeypatch.setattr(worker.stair_course, 'flight_tile', lambda height, depth, proportion, rear_steps: NS(
        step_height_range=(height, height), step_width=depth, proportion=proportion, rear_steps=rear_steps,
        function=worker.stair_course.flight_mesh))
    audit = {'reset': {'base': {'pose_range': dict(cfg.events.reset_base.params['pose_range'])}}}
    evidence = worker.configure_long_stairs(cfg, spec, audit)
    assert evidence['stratum_count'] == count == 6
    assert cfg.scene_context.terrain_generator is generator
    assert cfg.events.reset_base.params['pose_range'] == {'x': (0., 0.), 'y': (-.5, .5)}
    assert audit['reset']['base']['pose_range'] == cfg.events.reset_base.params['pose_range']
    assert generator.border_width == 0.
    assert list(generator.sub_terrains) == [f'stairs_{i}' for i in range(6)]
    for tile, row in zip(generator.sub_terrains.values(), spec['strata'], strict=True):
        tile.size = generator.size
        meshes, origin = tile.function(.5, tile)
        assert meshes[0].bounds[1, 2] == pytest.approx(0.)
        assert meshes[0].bounds[1, 0] - origin[0] == pytest.approx(1.)
        ascending = meshes[tile.rear_steps+1:]
        assert ascending[-1].bounds[1, 0] - origin[0] >= 24.
        assert [mesh.bounds[1, 2] for mesh in ascending] == pytest.approx(
            [i * row['step_height_m'] for i in range(1, len(ascending)+1)])
        assert tile.step_width == row['step_width_m']


def test_verifies_actual_stair_spawn_after_native_routing(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(HERE))
    import dagger_long_collect as worker
    from scenario_sampling import route_stairs

    count = 4096
    stairs = dict(stratum_count=6, tile_length_m=26.4, origin_x_by_stratum_m=[13., 13.2]*3)
    width = worker.stair_course.WIDTH_M
    origins = torch.tensor([[(x - stairs['tile_length_m'] / 2, (i - 2.5) * width, 0.)
                             for i, x in enumerate(stairs['origin_x_by_stratum_m'])]])
    generator = NS(size=(stairs['tile_length_m'], width),
                   sub_terrains={i: NS(function=worker.stair_course.flight_mesh) for i in range(6)})
    terrain = NS(cfg=NS(terrain_type='generator', terrain_generator=generator), terrain_prim_paths=['/World/ground'],
                 terrain_origins=origins, terrain_types=torch.zeros(count, dtype=torch.long),
                 terrain_levels=torch.zeros(count, dtype=torch.long), env_origins=torch.zeros(count, 3))
    scene = NS(terrain=terrain, env_origins=torch.zeros(count, 3))
    validation = route_stairs(scene, 400, profile='robust', paired_validation=True)
    split = worker.stair_split(scene, validation, 400)
    assert split['training_copies_by_stratum'] == [400] * 6
    assert sum(split['validation_copies_by_stratum']) == 1696
    robot = NS(data=NS(root_pos_w=scene.env_origins + torch.tensor([0., 0., .8])))
    raw = NS(scene=scene, robot=robot)
    start = worker.verify_stair_start(raw, stairs)
    assert start == dict(initial_start_environments=count, first_riser_min_m=1., first_riser_max_m=1.)
    robot.data.root_pos_w[-1, 0] += 2.
    with pytest.raises(ValueError, match='start at the base'):
        worker.verify_stair_start(raw, stairs)
