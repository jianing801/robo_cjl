"""Regression checks for exact checkpoint-cadence resume identity."""
from copy import deepcopy
from pathlib import Path

import pytest

from native_resume_compat import validate_native_resume_manifest
from finetune_settings import NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS


# Real LF receipt from the previously running e94fafa formal BO panel.
# The expected 17 SHA256 values deliberately do not come from the helper.
OLD_SOURCES = {
    'mixed_finetune.py': 'e1a47de4576b6fde0b2b28bd884b1095255f15c936decf9b6881c0daf78688f2',
    'finetune_common.py': '466423f4b356f53330daaca583c84302f5c9f208361d56405464c271e617d800',
    'mixed_scene.py': 'fc7e60c6a74bcbbbabe0b91388f651eb8d7b4f9e71397f7df1651284fd317f70',
    'mixed_batch.py': 'a974692776d125358cc7ca473aa9bd02b326b9a634b5be1f34b6eae06ece0904',
    'native_ppo.py': 'aa9d976f2128306a3c741d1666563bef8ccd05157c9fbca7cbc221dc572f3b48',
    'native_observations.py': '337ebb71724b1f615eee88a359d14993c2862617bcbf12e28c706884624dea1c',
    'rollout.py': 'ee89155427cc77dd27661041fe81ad8bc44a8ace452d059402fa4bee6c283c03',
    'rollout_amp.py': '122e2845a383fd637a1e385b1541d60f6d6f69f51447ccb602b70b3e0666f675',
    'student.py': 'fb485ddba283e30314d8b46fa27ae09bcba2f5ff4bbdcf111ffe934e7962022d',
    'training_reward.py': '415cb20c7e3fc6f7cbe6edacbc387cfa2012bc38e1f6b51c250cab222f9af398',
    'finetune_state.py': 'd7cf321f589f010b8dd99cd35b37b0573cc1cacbeffb21ac8350ce65ea945040',
    'finetune_settings.py': '932dcef2cf8189efcb12d27c41cdc1607862bb2f223552e5b6d919645e7f853b',
    '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py': '9e1836cdd9de070c2dbd82e95f09b9998f22745a6b03ca2212f13c0314f0c7ce',
    '../../../../rsl_rl/rsl_rl/algorithms/ppo.py': '9a817fa4e97de3baa48a6b1c9acc1da3873627eed28d494d11468851a7ba6a57',
    '../../../../rsl_rl/rsl_rl/storage/rollout_storage.py': 'aa3a33b4b8e912520fc6e3f157fd1350f730a3f507a856bdfab48249e0701202',
    '../../../../rsl_rl/rsl_rl/storage/circular_buffer.py': 'ac06641bc867d523cdf90b18148b2df244c036647564058342b3731365958723',
    '../../../../rsl_rl/rsl_rl/modules/amp.py': '988161dc5cdc4f5bba4fcb1bd348f9075f78e0b50ec87c8af963333b9e4c009f',
}
LATEST_SOURCES = dict(OLD_SOURCES, **{
    'mixed_finetune.py': 'ff360eeb24b769ff6d60ced4a26060367b5edeb15331b907432265f4c93b83b3',
    'finetune_settings.py': '546fca423f45b7fd4b4b6371f6550a472792d6e420f744a82bc19c0df74ca1b8',
    'native_ppo.py': '4646a7f7351daf761aa314d9a4580be7b7dc918b0eaf45bbce4b8b745e953f17',
    '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py': '5bcbf4f706fc854673f1f0ac9c2e3b1d0cfe74e4e765c7972a9c60a6dc806898',
})


def manifest(sources):
    return dict(kind='shared_candidate_finetuning_mixed', candidate={'structure_key': [0.25, 0.3, 'knee', 'ankle']},
                plan_sha256='1' * 64, initial_actor_sha256='2' * 64, updates=100,
                collection_protocol={'profile': 'robust'}, seed=20261104,
                ppo_algorithm=NATIVE_PPO_ALGORITHM, native_ppo_settings=deepcopy(NATIVE_PPO_SETTINGS),
                teacher_checkpoint_sha256={'stairs': '3' * 64, 'backpack': '4' * 64, 'run': '5' * 64},
                sources=dict(sources))


def new_sources():
    return dict(LATEST_SOURCES, **{'mixed_finetune.py': '6' * 64, 'finetune_state.py': '7' * 64,
                                   'native_resume_compat.py': '8' * 64})


@pytest.mark.parametrize('old_sources', [OLD_SOURCES, LATEST_SOURCES])
def test_accepts_exact_archived_native_source_set_without_mutating_receipt(old_sources):
    saved = manifest(old_sources)
    current = manifest(new_sources())
    before = deepcopy(saved)
    assert validate_native_resume_manifest(saved, current)
    assert saved == before
    assert validate_native_resume_manifest(current, deepcopy(current))


def test_verified_windows_line_endings_are_accepted():
    # Exact CRLF byte hashes of known Git blobs, mixed with the same revision
    # LF hashes; both line-ending checkouts have identical Python semantics.
    sources = dict(OLD_SOURCES, **{
        'mixed_finetune.py': '34ddf432ed39e6d4b61ac6868093eb66b9ca47cfa0aeb547ca5abde37451e061',
        'finetune_state.py': 'd84385896fa555f9d517ce7e0e792c947343baea1a3bba969cfd3623863fe7f0',
        'rollout.py': '61f300c3be3c4449d6bd797eec723dcf99cf17397a83c78ee400498afcf5ff83',
    })
    current_sources = dict(new_sources(), **{
        'rollout.py': '61f300c3be3c4449d6bd797eec723dcf99cf17397a83c78ee400498afcf5ff83',
        'native_ppo.py': '1e04070bd6041ba7486fb241eb24239c9e84aaa51c98a0ca9f2cf9b3389310a6',
    })
    assert validate_native_resume_manifest(manifest(sources), manifest(current_sources))


@pytest.mark.parametrize('change', ['unknown_old_source', 'unknown_new_source', 'unexpected_new_key',
                                    'mixed_old_revisions', 'old_helper_already_present', 'missing_old_key',
                                    'missing_new_helper', 'unknown_new_cadence_digest', 'learning_rate',
                                    'teacher_hash', 'seed', 'candidate', 'algorithm'])
def test_rejects_source_or_learning_protocol_changes(change):
    saved, current = manifest(OLD_SOURCES), manifest(new_sources())
    if change == 'unknown_old_source':
        saved['sources']['mixed_finetune.py'] = 'a' * 64
    elif change == 'unknown_new_source':
        current['sources']['native_ppo.py'] = 'a' * 64
    elif change == 'unexpected_new_key':
        current['sources']['unexpected.py'] = 'a' * 64
    elif change == 'mixed_old_revisions':
        saved['sources']['native_ppo.py'] = LATEST_SOURCES['native_ppo.py']
    elif change == 'old_helper_already_present':
        saved['sources']['native_resume_compat.py'] = '8' * 64
    elif change == 'missing_old_key':
        del saved['sources']['student.py']
    elif change == 'missing_new_helper':
        del current['sources']['native_resume_compat.py']
    elif change == 'unknown_new_cadence_digest':
        current['sources']['mixed_finetune.py'] = 'incorrect'
    elif change == 'learning_rate':
        saved['native_ppo_settings']['initial_learning_rate'] = 0.003
    elif change == 'teacher_hash':
        current['teacher_checkpoint_sha256']['run'] = 'a' * 64
    elif change == 'seed':
        current['seed'] = 1
    elif change == 'candidate':
        saved['candidate']['structure_key'][0] = 0.35
    elif change == 'algorithm':
        current['ppo_algorithm'] = 'not_v4'
    assert not validate_native_resume_manifest(saved, current)


def test_e94_has_no_experimental_profile_but_391_same_profile_is_eligible():
    saved, current = manifest(OLD_SOURCES), manifest(new_sources())
    for x in (saved, current):
        x['experimental_profile'] = 'fixed_amp_std'
        x['native_ppo_settings']['fixed_expert_std_tasks'] = ['run']
    assert not validate_native_resume_manifest(saved, current)
    saved['sources'] = dict(LATEST_SOURCES)
    assert validate_native_resume_manifest(saved, current)
    saved['native_ppo_settings']['fixed_expert_std_tasks'] = ['stairs']
    assert not validate_native_resume_manifest(saved, current)

def test_real_source_files_are_eligible_for_migrating_linux_receipt():
    import hashlib

    here = Path(__file__).resolve().parent
    current_sources = {key: hashlib.sha256((here / key).read_bytes()).hexdigest()
                       for key in (*LATEST_SOURCES, 'native_resume_compat.py')}
    assert validate_native_resume_manifest(manifest(OLD_SOURCES), manifest(current_sources))
