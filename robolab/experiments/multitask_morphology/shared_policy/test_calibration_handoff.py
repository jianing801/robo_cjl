import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_bo_calibration as handoff
from run_bo_calibration import initialize, own_distillation, wait_for_distillation


def test_handoff_accepts_only_the_expected_distillation_process(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'distill.py'
    command = ['python', '-u', str(script), '--output', str(tmp_path)]
    assert own_distillation(command, tmp_path)
    assert not own_distillation(command[:-1] + [str(tmp_path/'other')], tmp_path)
    assert not own_distillation(['python', 'something_else.py', *command[3:]], tmp_path)


def test_handoff_waits_for_gpu_release_and_refuses_incomplete_exit(tmp_path):
    actor = tmp_path / 'completed.json'
    checks = []
    process = SimpleNamespace(is_running=lambda: True, status=lambda: 'running')
    def advance(seconds):
        checks.append(seconds)
        if len(checks) == 1:
            actor.touch()
        else:
            process.is_running = lambda: False
    wait_for_distillation(tmp_path, process, advance)
    assert len(checks) == 2
    actor.unlink()
    with pytest.raises(RuntimeError, match='without verified completion'):
        wait_for_distillation(tmp_path, process, advance)


def test_handoff_obeys_manual_pause(tmp_path):
    (tmp_path / 'USER_PAUSED').touch()
    with pytest.raises(RuntimeError, match='USER_PAUSED'):
        wait_for_distillation(tmp_path, None)


def test_calibration_retries_without_reusing_manual_resume_permission(tmp_path):
    attempts = []
    def observe(command, work):
        attempts.append(command)
        if len(attempts) == 2:
            (work/'USER_PAUSED').touch()
        return 1
    with pytest.raises(subprocess.CalledProcessError):
        initialize(['bo.py'], tmp_path, resume_paused=True, observe=observe, sleep=lambda _: None)
    assert attempts == [['bo.py', '--resume-paused'], ['bo.py']]


def test_calibration_interrupt_during_first_report_scan_stops_controller(tmp_path, monkeypatch):
    first = tmp_path/'design_01'
    first.mkdir()
    (first/'manifest.json').touch()
    stopped = []

    class Process:
        def __enter__(self):
            (first/'bo_evaluation.json').touch()
            return self
        def __exit__(self, *_):
            return False
        def poll(self):
            return None
        def send_signal(self, _):
            stopped.append(self)
        def wait(self):
            return 130

    def interrupt(*_):
        raise KeyboardInterrupt

    import pipeline
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *_, **__: Process())
    monkeypatch.setattr(Path, 'rglob', interrupt)
    monkeypatch.setattr(pipeline, 'stop_child', stopped.append)
    with pytest.raises(KeyboardInterrupt):
        handoff.observe_calibration(['bo.py'], tmp_path)
    assert (tmp_path/'USER_PAUSED').is_file()
    assert len(stopped) == 1


def test_handoff_resumes_with_the_frozen_legacy_conditions(tmp_path, monkeypatch):
    import bo

    work = tmp_path/'calibration'
    work.mkdir()
    legacy = bo.LEGACY_FORMAL_CONDITIONS
    conditions_sha256 = bo.digest(bo.read_conditions(legacy))
    (work/'manifest.json').write_text(json.dumps(dict(conditions_sha256=conditions_sha256)))
    calls = []
    monkeypatch.setattr(handoff, 'wait_for_distillation', lambda *args: None)
    monkeypatch.setattr(handoff, 'shared_input', lambda *args, **kwargs: None)
    monkeypatch.setattr(handoff, 'initialize', lambda cmd, *args, **kwargs: calls.append(cmd))
    monkeypatch.setattr(handoff, 'summarize', lambda *args, conditions_path: calls.append(conditions_path))
    args = ['--root', str(tmp_path), '--output', str(work), '--resume-paused']
    handoff.main(args)
    command, summary_conditions = calls
    assert Path(command[command.index('--conditions')+1]) == legacy
    assert summary_conditions == legacy
    with pytest.raises(ValueError, match='frozen calibration'):
        handoff.main(args+['--conditions', str(bo.FORMAL_CONDITIONS)])
    assert len(calls) == 2


@pytest.mark.parametrize('legacy', [False, True])
def test_summary_checks_frozen_trials_and_recomputes_anchored_scales(tmp_path, monkeypatch, legacy):
    import bo
    plan, actor = tmp_path/'plan.json', tmp_path/'actor.pt'
    plan.touch()
    actor.write_bytes(b'actor')
    monkeypatch.setattr(handoff, 'shared_input', lambda *_, **__: (plan, {}, actor))
    monkeypatch.setattr(handoff, 'completed_actor', lambda *_, **__: actor)
    selected = bo.LEGACY_FORMAL_CONDITIONS if legacy else bo.FORMAL_CONDITIONS
    conditions = bo.read_conditions(selected)
    keys = bo.calibration_keys()
    reports, receipts = [], []
    for i, key in enumerate(keys, 1):
        folder = tmp_path/f'design_{i:02d}'
        folder.mkdir()
        manifest, trace = folder/'manifest.json', folder/'trace.npz'
        manifest.touch()
        trace.write_bytes(b'simulated trace fixture')
        native_trials, measurements = [], []
        for seed in conditions['seeds']:
            for task in bo.TASKS:
                native = dict(task=task, seed=seed, actor_sha256=bo.sha(actor),
                    conditions_sha256=bo.digest(conditions),
                    trace=str(trace), trace_sha256=bo.sha(trace),
                    trials=[dict(replica=0, status='success', mean_forward_mps=.5,
                                 metrics={metric: float(i) for metric in bo.METRICS[task]})])
                path = folder/f'{seed}_{task}.json'
                path.write_text(json.dumps(native))
                native_trials.append(native)
                measurements.append(dict(report=str(path), sha256=bo.sha(path)))
        report = dict(kind=bo.REPORT_VERSION, domain='simulation', candidate=dict(structure_key=list(key)),
            ppo_updates=200, student=str(actor), student_sha256=bo.sha(actor), conditions=conditions,
            conditions_sha256=bo.digest(conditions), evaluation_manifest_sha256=bo.sha(manifest),
            pairs=bo.paired_trials(native_trials, conditions), measurements=measurements)
        path = folder/'bo_evaluation.json'
        path.write_text(json.dumps(report))
        receipts.append(dict(report=path.relative_to(tmp_path).as_posix(), sha256=bo.sha(path)))
        reports.append(report)
    benchmark = dict(updates=200, initial_actor_sha256=bo.sha(actor), source_plan_sha256=bo.sha(plan),
                     reports=receipts)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(conditions_sha256=bo.digest(conditions))))
    scoring = tmp_path/'scoring.json'
    scoring.write_text(json.dumps(bo.panel_scoring(reports, keys, benchmark)))
    (tmp_path/'completed.json').write_text(json.dumps(dict(scoring_sha256=bo.sha(scoring),
                                                          manifest_sha256=bo.sha(manifest))))
    handoff.summarize(tmp_path, tmp_path, 200)
    result = json.loads((tmp_path/'calibration_summary.json').read_bytes())
    assert result['structures'] == 50
    assert result['trials'] == (450 if legacy else 150)
    assert result['paired_successes'] == (150 if legacy else 50)
    for task in result['normalization'].values():
        for metric in task.values():
            assert metric['multiplier'] * metric['p10'] + metric['offset'] == pytest.approx(0)
            assert metric['multiplier'] * metric['p90'] + metric['offset'] == pytest.approx(1)
    (tmp_path/'design_01/trace.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='raw trace changed'):
        handoff.summarize(tmp_path, tmp_path, 200)

def test_handoff_explicit_continuation_passes_same_actor_source_to_evaluation_and_summary(tmp_path, monkeypatch):
    extension = tmp_path/'shared_policy_round4'
    calls = []
    monkeypatch.setattr(handoff, 'wait_for_distillation',
                        lambda folder, process: calls.append(folder))
    monkeypatch.setattr(handoff, 'shared_input',
                        lambda *args, **kwargs: calls.append(kwargs['continuation']))
    monkeypatch.setattr(handoff, 'initialize',
                        lambda command, *args, **kwargs: calls.append(command))
    monkeypatch.setattr(handoff, 'summarize',
                        lambda *args, **kwargs: calls.append(kwargs['continuation']))
    handoff.main(['--root', str(tmp_path), '--output', str(tmp_path/'calibration'),
                  '--shared-continuation', str(extension)])
    wait_for, verify, command, summarize = calls
    assert wait_for == verify == summarize == extension.resolve()
    assert Path(command[command.index('--shared-continuation') + 1]) == extension.resolve()
