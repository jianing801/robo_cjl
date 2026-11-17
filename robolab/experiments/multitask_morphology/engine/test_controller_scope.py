"""控制器修复兼容旧断点，但拒绝训练源码或身份变化。"""
import ast,hashlib
from pathlib import Path,PurePosixPath
import pytest
import hybrid_contract as contract


def identities():
    current={'bank':'same','hybrid_sources':{p.name:contract.sha(p) for p in contract.HERE.glob('*.py') if not p.name.startswith('test_')}}
    old=dict(current,hybrid_sources=dict(current['hybrid_sources']))
    old['hybrid_sources'].update({'train.py':'33e38a4c10a79c8f50fd62c975f2a54572c7c719bb31b9331fa83421ddfbd7c3','hybrid_contract.py':'4a86085ab8a5800ee838966e7126f1910ad3882a4c0f79e7a2fab42cc2006ccd'})
    return old,current


def test_old_checkpoint_identity_accepted_without_rebinding(tmp_path):
    old,current=identities()
    assert contract.controller_compatible(old,current,tmp_path)
    assert (tmp_path/'controller_scope_compatibility.json').exists()
    assert contract.controller_compatible(old,current,tmp_path)
    assert old!=current


@pytest.mark.parametrize('fault',['physics','reward','unknown_old'])
def test_other_identity_changes_rejected(tmp_path,fault):
    old,current=identities()
    if fault=='physics':current['bank']='changed'
    elif fault=='reward':current['hybrid_sources']['hybrid_runtime.py']='changed'
    else:old['hybrid_sources']['train.py']='unknown'
    assert not contract.controller_compatible(old,current,tmp_path)


def test_worker_guard_allows_other_bank_but_rejects_same_and_unknown():
    text=(contract.HERE/'train.py').read_text(encoding='utf8')
    tree=ast.parse(text)
    checks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and 'orphan/active same-bank' in ast.get_source_segment(text,n)]
    node=min(checks,key=lambda n:n.end_lineno-n.lineno)
    expr=compile(ast.Expression(node.test),'<guard>','eval')
    env=dict(Path=Path,bank_path=PurePosixPath('/rough/bank.json'),root=PurePosixPath('/rough/run'))
    for args,blocked in [(['python','persistent_worker.py','--bank','/backpack/bank.json','--root','/backpack/run'],False),(['python','persistent_worker.py','--bank','/rough/bank.json'],True),(['python','persistent_worker.py'],True)]:
        assert bool(eval(expr,dict(env,args=args)))==blocked
