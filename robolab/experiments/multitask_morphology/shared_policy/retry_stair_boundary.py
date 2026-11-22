"""Retry a single interrupted shared-policy stair batch on a longer flight."""
import argparse
import json
from math import isclose
from pathlib import Path
import os
import re
import runpy
import sys

from contract import load_plan, sha, write_new
from persistence import run_lock
from shared_bo_metrics import digest
from stair_course import COURSE, WIDTH_M, evaluation_flight, outside_flight

HERE = Path(__file__).resolve().parent
EXTRA_DISTANCE_M = 24.5


def extended_flight(duration, speed, depth):
    flight = evaluation_flight(duration, speed, depth)
    extra_treads = round(EXTRA_DISTANCE_M / depth)
    if not isclose(extra_treads * depth, EXTRA_DISTANCE_M):
        raise ValueError('Extra stair length must align with the original tread grid')
    flight['forward_steps'] += extra_treads
    flight['rear_steps'] += extra_treads
    flight['length_m'] += 2 * EXTRA_DISTANCE_M
    flight['origin_x_m'] += EXTRA_DISTANCE_M
    return flight


def check_longitudinal_exit(entries, course, strata):
    origins = course['origin_x_by_stratum_m']
    if not entries or any(abs(y) >= WIDTH_M / 2 - .35 or
                          not outside_flight(x, y, origins[strata[int(env)]], course['tile_length_m'])
                          for env, (_, x, y, _) in entries.items()):
        raise ValueError('A retry with extra length requires a longitudinal stair boundary exit')


def value(command, flag):
    return command[command.index(flag) + 1]


def read(path):
    return json.loads(path.read_bytes())


def native_worker(args):
    if value(args, '--mode') != 'evaluate' or value(args, '--task') != 'stairs' or '--bo-conditions' in args:
        raise ValueError('The extension is restricted to shared-policy stair evaluation')
    import randomization
    import scenario_sampling
    scenario_sampling.evaluation_flight = extended_flight
    configure_stairs = scenario_sampling.configure_stairs

    def configure_with_audit(*args, **kwargs):
        spec = configure_stairs(*args, **kwargs)
        spec['extra_distance_per_end_m'] = EXTRA_DISTANCE_M
        return spec

    scenario_sampling.configure_stairs = configure_with_audit
    rollout_metadata = randomization.rollout_metadata

    def metadata_with_extension(audit, requested):
        return dict(**rollout_metadata(audit, requested), stair_flight_extension=audit['stairs'])

    randomization.rollout_metadata = metadata_with_extension
    sys.argv = [str(HERE/'rollout.py'), *args]
    runpy.run_path(sys.argv[0], run_name='__main__')


def retry(root, log_dir):
    import pipeline
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment

    match = re.fullmatch(r'eval_selection_round_([1-3])_seed_(\d+)_0_student_\d{8}_\d{6}_\d{6}',
                         log_dir.name)
    if not match or log_dir.parent != root/'processes':
        raise ValueError('Choose the failed student stair process from this run')
    round_index, seed = map(int, match.groups())
    suite = f'selection/round_{round_index}/seed_{seed}'
    candidate = root/'evaluation'/suite/'0000_stairs_student'
    original = read(log_dir/'launch.json')['command']
    previous_output = Path(value(original, '--output')).resolve()
    if (Path(original[4]).resolve() != HERE/'rollout.py' or previous_output.parent != candidate
            or read(log_dir/'exit.json')['returncode'] != 1 or (previous_output/'summary.json').exists()):
        raise ValueError('The saved launch is not an uncommitted failed stair rollout')
    boundary = previous_output/'stair_boundary.json'
    previous = read(boundary)
    original_audit = read(previous_output/'randomization_audit.json')
    strata = {row['environment_id']: row['terrain_stratum'] for row in read(previous_output/'scenarios.json')}
    check_longitudinal_exit(previous['environments'], previous['course'], strata)
    if previous['course']['terrain'] != COURSE:
        raise ValueError('The failed trial used a different stair geometry')
    with run_lock(root):
        if any((root/name).exists() for name in ('USER_PAUSED', 'selection.json', 'completed.json')):
            raise RuntimeError('Run is paused or already finalized')
        manifest = read(root/'manifest.json')
        protocol, args = manifest['protocol'], manifest['arguments']
        if digest(protocol) != manifest['protocol_sha256']:
            raise ValueError('Selection manifest hash does not match')
        coordinator_hashes = {sha(HERE/'pipeline.py'), *pipeline.PRE_CONTINUATION_PIPELINE_SHA256}
        if (protocol['implementation_sha256'] not in coordinator_hashes
                or sha(HERE/'evaluation.py') != protocol['evaluation_implementation_sha256']
                or protocol['stair_evaluation'] != COURSE or seed not in protocol['selection_seeds']):
            raise ValueError('Selection implementation, course or seed differs')
        plan_path = Path(value(original, '--plan')).resolve()
        if sha(plan_path) != manifest['plan_sha256']:
            raise ValueError('Shared plan changed since the original evaluation')
        plan = load_plan(plan_path)
        fit = read(root/f'fit_round_{round_index}.json')
        if sha(fit['summary']) != fit['summary_sha256']:
            raise ValueError('Committed fit summary changed')
        actor = pipeline.last_checkpoint(read(Path(fit['summary'])), args['epochs'])
        ids = pipeline.design_ids_for_keys(plan, list(map(tuple, protocol['selection_keys'])))['stairs']
        profile, strength = protocol['selection']['profile'], protocol['selection']['strength']
        if pipeline.protocol_metadata(profile, strength) != protocol['selection']:
            raise ValueError('Selection rollout source changed')
        original_worker = pipeline.child

        def retry_child(command, folder, environment, timeout, control_root):
            output = Path(value(command, '--output'))
            comparable = command[:]
            comparable[comparable.index('--output') + 1] = str(previous_output)
            if comparable != original:
                raise ValueError('Recovered command differs from the failed batch')
            command = [*command[:4], str(Path(__file__).resolve()), '--native-worker', *command[5:]]
            original_worker(command, folder, environment, timeout, control_root)
            audit = read(output/'randomization_audit.json')
            applied = audit.pop('stairs')
            start = read(output/'stair_start.json')
            before = previous['course']
            summary = read(output/'summary.json')
            if (audit != {key: item for key, item in original_audit.items() if key != 'stairs'}
                    or summary['seed'] != seed or applied['terrain'] != COURSE
                    or applied['strata'] != previous['course']['strata']
                    or applied['duration_s'] != before['duration_s']
                    or applied['max_command_mps'] != before['max_command_mps']
                    or applied['extra_distance_per_end_m'] != EXTRA_DISTANCE_M
                    or not isclose(applied['tile_length_m'] - before['tile_length_m'], 2 * EXTRA_DISTANCE_M)
                    or any(not isclose(new - old, EXTRA_DISTANCE_M)
                           for new, old in zip(applied['origin_x_by_stratum_m'],
                                               before['origin_x_by_stratum_m'], strict=True))
                    or summary.get('stair_flight_extension') != applied
                    or start['first_riser_min_m'] != 1. or start['first_riser_max_m'] != 1.):
                raise ValueError('Recovered trial changed more than the stair length')
            write_new(output/'stair_recovery.json', dict(original_process=str(log_dir),
                original_boundary_sha256=sha(boundary), original_course=previous['course'],
                extended_course=applied, recovery_source_sha256=sha(__file__),
                summary_sha256=sha(output/'summary.json')))

        criteria = {key: protocol['evaluation_criteria'][key] for key in
                    ('duration_s', 'time_tolerance_s', 'min_command_progress_ratio', 'max_missing_tracking_steps')}
        choices = candidates(query_gpus(), os.environ, min_free_mib=args['gpu_min_free_gb']*1024,
                             max_utilization=args['gpu_max_utilization'], requested=args['device'])
        pipeline.child = retry_child
        try:
            with gpu_lease(choices, HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
                output, = pipeline.run_evaluation(plan, plan_path, root, [('stairs', ids)], actor['student'],
                    profile, strength, seed, suite, child_environment(gpu, os.environ),
                    args['timeout_seconds'], criteria)
        finally:
            pipeline.child = original_worker
        print('STAIR_BOUNDARY_RETRY_COMMITTED', output['summary'], flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--native-worker':
        native_worker(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--failed-process', type=Path, required=True)
    args = parser.parse_args(argv)
    retry(args.root.resolve(), args.failed_process.resolve())


if __name__ == '__main__':
    main()
