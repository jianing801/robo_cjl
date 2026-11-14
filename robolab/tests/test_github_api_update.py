"""Offline Git fixtures: verified tree import and preservation of local commits."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/tools/update_github_api.py'
spec=importlib.util.spec_from_file_location('github_api_update',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def init(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    module.git(root,'config','user.name','Test')
    module.git(root,'config','user.email','test@example.com')
    module.git(root,'config','core.autocrlf','false')
    (root/'base.txt').write_text('base\n')
    module.git(root,'add','.');module.git(root,'commit','-qm','base')
    module.git(root,'branch','-M','feature/rough-hybrid')
    return root


def test_exact_archive_tree_and_mismatch_guard(tmp_path):
    root=init(tmp_path)
    (root/'nested').mkdir();(root/'nested/binary').write_bytes(bytes(range(256)))
    module.git(root,'add','.');module.git(root,'commit','-qm','nested binary')
    tree=module.git(root,'rev-parse','HEAD^{tree}').decode().strip()
    data=module.git(root,'archive','--format=tar.gz','--prefix=snapshot/','HEAD')
    assert module.import_archive(root,io.BytesIO(data),tree)==(tree,2)
    with pytest.raises(RuntimeError,match='does not match'):
        module.import_archive(root,io.BytesIO(data),'0'*40)
    assert module.git(root,'status','--porcelain')==b''


@pytest.mark.parametrize('conflict',[False,True])
def test_api_merge_preserves_local_history_and_conflicts(tmp_path,monkeypatch,conflict):
    root=init(tmp_path);base=module.git(root,'rev-parse','HEAD').decode().strip()
    module.git(root,'update-ref','refs/remotes/origin/feature/rough-hybrid',base)
    (root/('base.txt' if conflict else 'remote.txt')).write_text('upstream\n')
    module.git(root,'add','.');module.git(root,'commit','-qm','upstream')
    upstream=module.git(root,'rev-parse','HEAD').decode().strip()
    tree=module.git(root,'rev-parse','HEAD^{tree}').decode().strip()
    archive=module.git(root,'archive','--format=tar.gz','--prefix=snapshot/','HEAD')
    # Move only this disposable fixture's refs back to its saved base.
    module.git(root,'switch','-q','-c','fixture-local',base)
    module.git(root,'branch','-D','feature/rough-hybrid')
    module.git(root,'branch','-M','feature/rough-hybrid')
    (root/('base.txt' if conflict else 'local.txt')).write_text('local\n')
    module.git(root,'add','.');module.git(root,'commit','-qm','local')
    local=module.git(root,'rev-parse','HEAD').decode().strip()
    monkeypatch.setattr(module,'api',lambda p: {'object':{'sha':upstream}} if '/git/ref/' in p else {'tree':{'sha':tree}})
    monkeypatch.setattr(module,'request',lambda p:io.BytesIO(archive))
    monkeypatch.setattr(sys,'argv',['update','--repo',str(root),'--apply'])
    if conflict:
        with pytest.raises(SystemExit):module.main()
        assert module.git(root,'rev-parse','HEAD').decode().strip()==local
        assert 'UU base.txt' in module.git(root,'status','--short').decode()
    else:
        module.main()
        assert (root/'local.txt').read_text()=='local\n'
        assert (root/'remote.txt').read_text()=='upstream\n'
        assert subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',local,'HEAD']).returncode==0
    assert module.git(root,'rev-parse','refs/remotes/origin/feature/rough-hybrid').decode().strip()==base
    backups=module.git(root,'for-each-ref','--format=%(objectname)','refs/heads/backup/').decode().splitlines()
    assert local in backups
