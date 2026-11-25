"""Shared native candidate budget, pause and committed Actor publication."""
import json
from pathlib import Path

import torch

from contract import sha, write_new
from finetune_settings import NATIVE_PPO_SETTINGS
from randomization import assert_protocol_unchanged
from student import load_student, save_student


STEPS_PER_TASK = NATIVE_PPO_SETTINGS['steps_per_env']


def check_pause(root):
    if (Path(root)/'USER_PAUSED').exists():
        raise KeyboardInterrupt('USER_PAUSED')


def publish_finetuned_actor(a, root, model, manifest_sha256, protocol):
    actor = root/'shared_actor_finetuned.pt'
    expected = dict(ppo_updates=a.updates, initial_actor_sha256=sha(a.student),
                    collection_protocol=protocol, manifest_sha256=manifest_sha256)
    if actor.exists():
        previous, metadata = load_student(actor)
        previous_state = previous.state_dict()
        if (any(metadata.get(key) != value for key, value in expected.items())
                or previous.schema != model.schema or previous.architecture != model.architecture
                or any(not torch.equal(value, previous_state[key]) for key, value in model.state_dict().items())):
            raise ValueError('Published finetuned actor differs from the committed learning state')
    else:
        save_student(actor, model, **expected)
    return actor


def publish_training_completion(a, root, model, candidate, manifest_sha256, protocol):
    check_pause(root)
    assert_protocol_unchanged(protocol)
    actor = publish_finetuned_actor(a, root, model, manifest_sha256, protocol)
    commit = root/f'checkpoints/update_{a.updates:06d}.json'
    record = dict(result='CANDIDATE_FINETUNED', candidate=candidate, ppo_updates=a.updates,
                  manifest_sha256=manifest_sha256, student=str(actor), student_sha256=sha(actor),
                  checkpoint_sha256=sha(commit))
    receipt = root/'training_completed.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf8')) != record:
            raise ValueError('Published training completion record changed')
    else:
        write_new(receipt, record)
    print('CANDIDATE_FINETUNED', receipt, flush=True)
