"""Narrow compatibility for checkpoint-cadence migration of native PPO V4.

Hashes below are SHA256 of the exact Git blobs (LF) and the same blobs
with CRLF line endings. These are not general permission to ignore source
changes: old sources must all belong to one explicitly audited revision and
current non-cadence sources must match 391e19c. The old manifest remains
immutable because its hash binds every committed checkpoint.
"""
import re

from finetune_settings import NATIVE_PPO_ALGORITHM, correction_settings, V4_STAIRS5120_EPOCH10_PROFILE


# e94fafa25a91c3ccf916082f58c69c500905bedd: original expert-prior V4.
_E94FAFA = {
    'mixed_finetune.py': (
        'e1a47de4576b6fde0b2b28bd884b1095255f15c936decf9b6881c0daf78688f2',
        '34ddf432ed39e6d4b61ac6868093eb66b9ca47cfa0aeb547ca5abde37451e061'),
    'finetune_common.py': (
        '466423f4b356f53330daaca583c84302f5c9f208361d56405464c271e617d800',
        '014ababc467edb1a2bac28de978ff4d4a2d1f835d049bdfb96269b129a4c5738'),
    'mixed_scene.py': (
        'fc7e60c6a74bcbbbabe0b91388f651eb8d7b4f9e71397f7df1651284fd317f70',
        'dfa200db7b35fc370c11af5bfe62457ec4d7db347ac6352311803864867242ba'),
    'mixed_batch.py': (
        'a974692776d125358cc7ca473aa9bd02b326b9a634b5be1f34b6eae06ece0904',
        'db6a13a3eef78f3741c99e93ffb8a2f81c1f4e5908e8f051cffb21076abd0b13'),
    'native_ppo.py': (
        'aa9d976f2128306a3c741d1666563bef8ccd05157c9fbca7cbc221dc572f3b48',
        '629842673a921f980246505c07838aea3a5e5246a6ed98396b47c6cc9fa406e7'),
    'native_observations.py': (
        '337ebb71724b1f615eee88a359d14993c2862617bcbf12e28c706884624dea1c',
        '5836deacbc14543ca72c3535a5aca6e362a458546b60e1c976b865139dfa7dd2'),
    'rollout.py': (
        'ee89155427cc77dd27661041fe81ad8bc44a8ace452d059402fa4bee6c283c03',
        '61f300c3be3c4449d6bd797eec723dcf99cf17397a83c78ee400498afcf5ff83'),
    'rollout_amp.py': (
        '122e2845a383fd637a1e385b1541d60f6d6f69f51447ccb602b70b3e0666f675',
        'ba48f47f056b9011017c27e6dba96bfd1c4ba3e1a5ac00acff6b9bb1e65ea240'),
    'student.py': (
        'fb485ddba283e30314d8b46fa27ae09bcba2f5ff4bbdcf111ffe934e7962022d',
        '53b76f65fb0dc0d089fc1971f5d2b5ef0283eee06f878bbe9ea5cbb05c1e45e8'),
    'training_reward.py': (
        '415cb20c7e3fc6f7cbe6edacbc387cfa2012bc38e1f6b51c250cab222f9af398',
        '9a2086d889b59922792d7ede0418f3058fee9239bedc6dbc839b9ee5020835bd'),
    'finetune_state.py': (
        'd7cf321f589f010b8dd99cd35b37b0573cc1cacbeffb21ac8350ce65ea945040',
        'd84385896fa555f9d517ce7e0e792c947343baea1a3bba969cfd3623863fe7f0'),
    'finetune_settings.py': (
        '932dcef2cf8189efcb12d27c41cdc1607862bb2f223552e5b6d919645e7f853b',
        'e5dec5fe07694e619e4a9313ee5799e767b75f0d87a039515766c6e1853e32b1'),
    '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py': (
        '9e1836cdd9de070c2dbd82e95f09b9998f22745a6b03ca2212f13c0314f0c7ce',
        '3ad21a131f4a86c19234f8e5ca9032db77c1c8016bbcc7b55571e1c1e125cf28'),
    '../../../../rsl_rl/rsl_rl/algorithms/ppo.py': (
        '9a817fa4e97de3baa48a6b1c9acc1da3873627eed28d494d11468851a7ba6a57',
        '00ed43a870c758806019ecd70067deb54a5eb76efb1744dc8a4cada580b409da'),
    '../../../../rsl_rl/rsl_rl/storage/rollout_storage.py': (
        'aa3a33b4b8e912520fc6e3f157fd1350f730a3f507a856bdfab48249e0701202',
        'f48d65fe8ccdde8688febd0de54c6af6733eafa3c9f17d702a2fe209c62dabda'),
    '../../../../rsl_rl/rsl_rl/storage/circular_buffer.py': (
        'ac06641bc867d523cdf90b18148b2df244c036647564058342b3731365958723',
        'fc55353418e8f23e496f3627290d1fa6afd86c659b255ee911fb6d4ccf1e6f34'),
    '../../../../rsl_rl/rsl_rl/modules/amp.py': (
        '988161dc5cdc4f5bba4fcb1bd348f9075f78e0b50ec87c8af963333b9e4c009f',
        '435dd2462f7a42fbe42835b8e236043bd8446fb518e5f5e0b27d383bfd8d1a09'),
}

# 391e19cd70430f2350612f64440a0b57def95de6: optional ablations only;
# their default native profile uses the original V4 PPO behavior.
_391E19C = dict(_E94FAFA, **{
    'mixed_finetune.py': (
        'ff360eeb24b769ff6d60ced4a26060367b5edeb15331b907432265f4c93b83b3',
        '5b6ee29ae3287e3ab5607527b14421cee2145478a2313738a146af6863bf7674'),
    'native_ppo.py': (
        '4646a7f7351daf761aa314d9a4580be7b7dc918b0eaf45bbce4b8b745e953f17',
        '1e04070bd6041ba7486fb241eb24239c9e84aaa51c98a0ca9f2cf9b3389310a6'),
    'finetune_settings.py': (
        '546fca423f45b7fd4b4b6371f6550a472792d6e420f744a82bc19c0df74ca1b8',
        'decc46dca1411bc9498a37de9f82dabc95d05c52581ea02ed01e60750a9f031f'),
    '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py': (
        '5bcbf4f706fc854673f1f0ac9c2e3b1d0cfe74e4e765c7972a9c60a6dc806898',
        'af2b26c6ccdcd92f0385fb3fec14c291812e390b565b893db5aedccfd898b2f7'),
})

_CADENCE_SOURCES = frozenset({'mixed_finetune.py', 'finetune_state.py'})
_HELPER = 'native_resume_compat.py'
# The new formal B defaults do not alter an explicitly selected legacy native
# profile. Pin the exact LF/CRLF settings files for that narrow migration.
_B_DEFAULT_SETTINGS_HASHES = (
    '2c7b0a28e5563e0cce93f6e166630af76d27c67aa60d86a9388a0e3b6fd4b19d',
    '9791e08cb73b7934f2e9505f514d99eeb49d1c953c65a972a48e60a87653f12f',
    'ed8314a036ad38bdb638a3e8944ec18f6aa276b0ed0b737beecc7acfb35c4c75',
    '7e967110c416ed17314109c17f9fda5708303b6be4ea94a9186183a3cb394d72',
    '9717f2ead989e4c8498be5e1a29e57d32ec8c7ea0c3ce627d17bf71ec170bfd4',
    '71f4517c19d9f416818ecf9a909bae0f0ace842e33d7f92f395a399823fcc4a9',
    '7811b45ff795c23c4429eab100f1116fa05150e4aea4895cc16638331775f4b1',
    '42618c479545deb8b91946d4138fbbaff967d8ecf45000c4446155aa586609e6',
    # Isolated V4 5120/512/512, ten-epoch profile; native and B settings unchanged.
    '244e4374ab1d6ea58a38d9d6fcd0b787000a34f64cd2ca2ec6e73c622766c6cb',
    'a6243fd2b1a3eb6bea79f4d105d6bb4c1ac353ed88f2ba27e61c8b377de05842',
)
_OPTIONAL_ACTOR_WEIGHT_HASHES = (
    'a2b3c9f8c202f2f4ec7604c65eec6bb0f5edc735b770b8080210fd2b64548acd',
    'd57a4073cf46a048734593efb8e31ca825d03be6bb2396d4ffce4d98981a59c1',
    '07eb1b0c78a27beeb77a97b5f41febc876c0e7af0746c7a3448ed969e72e486f',
    '3a36bd40c43a6fa463ecf167ccf3ca14429f2b1d2577b86fb36837b8dd0a0c6e',
)


# New multi-morphology routing is selected only when a mixed worker hosts
# more than one design. Existing singleton candidate behavior is unchanged.
_SPATIAL_PRETRAIN_ROLLOUT_HASHES = (
    'd5e87ee391452c6ff70bfb8566414ab01e3ec158ca18076b58cf04fed3ef588b',
    '84999aa23362a72889ccf37c3745ad83b16013611d1d548e4a0ead9003a4cb39',
)

def _native_manifest_fields(manifest):
    return {key: value for key, value in manifest.items() if key != 'sources'}


def _matches_known_sources(saved, known):
    return (isinstance(saved, dict) and saved.keys() == known.keys()
            and all(saved[key] in hashes for key, hashes in known.items()))


def _matches_current_sources(sources):
    # These two files intentionally change with cadence. The caller builds
    # the manifest by hashing actual files in the running checkout; all
    # other algorithm/collection files remain pinned to the audited commit.
    return (isinstance(sources, dict)
            and sources.keys() == _391E19C.keys() | {_HELPER}
            and all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                    for value in sources.values())
            and all(sources[name] in (expected + _B_DEFAULT_SETTINGS_HASHES if name == 'finetune_settings.py'
                                      else expected + _OPTIONAL_ACTOR_WEIGHT_HASHES if name == 'native_ppo.py'
                                      else expected + _SPATIAL_PRETRAIN_ROLLOUT_HASHES if name == 'rollout.py' else expected)
                    for name, expected in _391E19C.items()
                    if name not in _CADENCE_SOURCES))


def validate_native_resume_manifest(saved, current):
    """Accept exact identity or the one intentional V4 persistence migration.

    No non-source field can change, and no arbitrary source revision can be
    resumed. Do not edit either input manifest; the caller hashes the saved
    on-disk manifest and records current source hashes at attempt level.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if saved == current:
        return True
    if (_native_manifest_fields(saved) != _native_manifest_fields(current)
            or current.get('kind') != 'shared_candidate_finetuning_mixed'
            or current.get('ppo_algorithm') != NATIVE_PPO_ALGORITHM):
        return False
    profile = current.get('experimental_profile', 'native')
    # The new experiment has no historical migration path; require exact identity.
    if profile == V4_STAIRS5120_EPOCH10_PROFILE:
        return False
    # Native runs omit the experimental_profile field. Reject invented
    # combinations of old source identity and learning hyperparameters.
    if profile == 'native' and 'experimental_profile' in current:
        return False
    try:
        if current.get('native_ppo_settings') != correction_settings(profile):
            return False
    except ValueError:
        return False
    if not _matches_current_sources(current.get('sources')):
        return False
    old_sources = saved.get('sources')
    if _matches_known_sources(old_sources, _E94FAFA):
        return profile == 'native'
    return _matches_known_sources(old_sources, _391E19C)
