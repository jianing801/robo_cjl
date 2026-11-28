import json
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from singleton_resume_warmup import install_warmup


def runner_fixture():
    policy=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
    return SimpleNamespace(current_learning_iteration=5939,env=SimpleNamespace(num_envs=4096,episode_length_buf=torch.arange(4096)%1000,max_episode_length=1000),
        alg=SimpleNamespace(policy=policy,optimizer=optimizer,learning_rate=.001,storage=SimpleNamespace(step=0)),
        logger=SimpleNamespace(rewbuffer=deque([31.0]*100,maxlen=100),lenbuffer=deque([1000]*100,maxlen=100)))


def test_verified_warmup_publishes_audit_without_changing_update(tmp_path):
    runner=runner_fixture(); stage={'end_update':12000,'design_ids':['one']}; calls=[]
    def native(*args): calls.append('ledger')
    def warmup(runner, pause, *, independent):
        assert independent and pause==tmp_path/'USER_PAUSED'
        calls.append('warmup'); return {'window_episodes':100,'warmup_seconds':2.0}
    def publish(path, value): path.write_text(json.dumps(value))
    install_warmup(runner,tmp_path,tmp_path,stage,native,warmup,publish)
    audit=json.loads((tmp_path/'warmup_receipt.json').read_text())
    assert calls==['ledger','warmup']
    assert audit['policy_and_normalization_unchanged'] and audit['adam_unchanged'] and audit['update_count_unchanged']
    assert runner.current_learning_iteration==5939


def test_changed_policy_rejected(tmp_path):
    runner=runner_fixture(); stage={'end_update':12000,'design_ids':['one']}
    def warmup(runner, marker, *, independent):
        with torch.no_grad(): runner.alg.policy.weight.add_(1)
        return {'window_episodes':100}
    with pytest.raises(RuntimeError,match='Warm-up changed learning state'):
        install_warmup(runner,tmp_path,tmp_path,stage,lambda *_:None,warmup,lambda *_:None)
    assert not (tmp_path/'warmup_receipt.json').exists()
