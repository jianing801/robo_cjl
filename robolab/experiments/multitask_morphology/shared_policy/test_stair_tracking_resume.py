"""A pending evaluation can change scoring without reusing incompatible evidence."""
import json

import pytest

import stair_resume
from shared_bo_metrics import digest


def manifest(protocol, *, selection_rule='strict tracking'):
    return dict(protocol=protocol, protocol_sha256=digest(protocol),
                selection_rule=selection_rule, arguments={'epochs': 10}, plan_sha256='original plan')


@pytest.fixture
def pending_run(tmp_path):
    history = tmp_path/'stair_evaluation_upgrade'
    history.mkdir()
    original = manifest(dict(collection='unchanged', selection_version='last_epoch_dagger_paired_200'))
    (history/'original_manifest.json').write_text(json.dumps(original), encoding='utf8')
    old_protocol = dict(original['protocol'], stair_evaluation='bottom_start_long_stairs',
                        implementation_sha256='2e146ef15d585613ed9c1e18e1913358c75a082cfc29078da329487b334eaa70',
                        evaluation_criteria={'tracking_rel_tol': .25}, evaluation_version='v3')
    old = manifest(old_protocol)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(old), encoding='utf8')
    (tmp_path/'USER_PAUSED').touch()
    current = manifest(dict(old_protocol, evaluation_velocity_frame='horizontal_heading',
                            implementation_sha256='new code', evaluation_version='v4',
                            evaluation_criteria={'min_command_progress_ratio': .5}),
                       selection_rule='report measured tracking')
    return tmp_path, path, history, old, current


def test_resume_archives_previous_evaluation_and_preserves_original(pending_run):
    root, path, history, old, current = pending_run
    original = (history/'original_manifest.json').read_bytes()
    previous = path.read_bytes()
    stair_resume.upgrade_tracking_manifest(root, old, current, explicit=True)
    assert json.loads(path.read_bytes()) == current
    assert (history/'original_manifest.json').read_bytes() == original
    assert [p.read_bytes() for p in history.glob('tracking_previous_*.json')] == [previous]
    assert (root/'USER_PAUSED').exists()


def test_direct_upgrade_from_original_adopts_new_evaluation_rule(pending_run):
    root, path, history, old, current = pending_run
    original_path = history/'original_manifest.json'
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    path.write_bytes(original_bytes)
    stair_resume.upgrade_manifest(root, original, current)
    assert json.loads(path.read_bytes()) == current
    assert original_path.read_bytes() == original_bytes


@pytest.mark.parametrize('blocked', ['implicit', 'finalized', 'training', 'source', 'wrong_frame'])
def test_resume_refuses_unapproved_or_unrelated_change(pending_run, blocked):
    root, path, history, old, current = pending_run
    if blocked == 'finalized':
        (root/'selection.json').touch()
    elif blocked == 'training':
        current = manifest(dict(current['protocol'], collection='different'))
    elif blocked == 'source':
        old = manifest(dict(old['protocol'], implementation_sha256='unexpected'))
        path.write_text(json.dumps(old), encoding='utf8')
    elif blocked == 'wrong_frame':
        current = manifest(dict(current['protocol'], evaluation_velocity_frame='body_axis'))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        stair_resume.upgrade_tracking_manifest(root, old, current, explicit=blocked != 'implicit')
    assert path.read_bytes() == original
    assert not list(history.glob('tracking_previous_*.json'))
