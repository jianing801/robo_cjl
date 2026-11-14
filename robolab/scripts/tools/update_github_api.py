"""Online API snapshot merge when github.com Git transport is unavailable.

Preserves local commits using an explicitly labelled import branch and a normal
three-way merge. Does NOT pretend the import commit is the upstream commit or
rewrite origin tracking. A conflicted merge is left intact for human resolution.
Only the Python standard library and git are needed. Never runs training.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

API='https://api.github.com'


def git(repo,*args,data=None):
    result=subprocess.run(['git','-C',str(repo),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8','replace').strip())
    return result.stdout


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urllib.parse.urlparse(newurl).hostname not in ('api.github.com','codeload.github.com','github.com'):
            raise RuntimeError('Unexpected archive redirect host')
        result=super().redirect_request(req,fp,code,msg,headers,newurl)
        if result and urllib.parse.urlparse(newurl).hostname!='api.github.com':
            result.remove_header('Authorization')
        return result


def request(url):
    headers={'Accept':'application/vnd.github+json','User-Agent':'robo-cjl-api-updater'}
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:headers['Authorization']='Bearer '+token
    return urllib.request.build_opener(SafeRedirect()).open(urllib.request.Request(url,headers=headers),timeout=20)


def api(path):
    with request(API+path) as response:return json.load(response)


def import_archive(repo,stream,expected_tree):
    """Hash raw archive bytes into Git objects, without extracting to worktree."""
    directories=defaultdict(dict);seen=set();count=0
    with tarfile.open(fileobj=stream,mode='r|gz') as archive:
        for member in archive:
            parts=member.name.split('/')[1:]
            if not parts or member.isdir():continue
            if any(x in ('','..','.git') for x in parts):raise RuntimeError('Unsafe archive path')
            name='/'.join(parts)
            if name in seen:raise RuntimeError('Duplicate archive entry')
            seen.add(name)
            if member.issym():
                data=member.linkname.encode('utf8');mode='120000'
            elif member.isfile():
                data=archive.extractfile(member).read();mode='100755' if member.mode&0o111 else '100644'
            else:raise RuntimeError('Unsupported archive entry')
            blob=git(repo,'hash-object','-w','--stdin',data=data).decode().strip()
            parent='/'.join(parts[:-1]);directories[parent][parts[-1]]=(mode,'blob',blob)
            for index in range(len(parts)-1):directories['/'.join(parts[:index])]
            count+=1
            if count%200==0:print('API_FILES_VERIFIED',count,flush=True)
    if not count:raise RuntimeError('Empty archive')
    for folder in sorted(directories,key=lambda v:v.count('/')+(1 if v else 0),reverse=True):
        entries=directories[folder]
        payload=b''.join(f'{mode} {kind} {sha}\t'.encode()+name.encode('utf8')+b'\0'
            for name,(mode,kind,sha) in entries.items())
        tree=git(repo,'mktree','-z',data=payload).decode().strip()
        if folder:
            parent,_,name=folder.rpartition('/')
            directories[parent][name]=('040000','tree',tree)
        else:
            if tree!=expected_tree:raise RuntimeError('Archive tree does not match GitHub commit; merge refused')
            return tree,count


def uses_source_tree(command,cwd,repo):
    paths=[x.replace('\\','/') for x in command]
    matching=any(any(p in x for p in ('multitask_morphology/','rough_hybrid/','amp_hybrid/','rough_bo/'))
        and any(y in Path(x).name for y in ('train','worker','controller','prepare_references','prepare_assets','prepare_bank'))
        for x in paths)
    return matching and (cwd==repo or repo in cwd.parents or any(str(repo) in x for x in paths))


def check_idle(repo):
    proc=Path('/proc')
    if not proc.is_dir():return
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:
            if folder.stat().st_uid!=os.getuid():continue
            command=(folder/'cmdline').read_bytes().decode('utf8','replace').split('\0')
            cwd=(folder/'cwd').resolve()
            if uses_source_tree(command,cwd,repo):
                raise RuntimeError(f'Related training/reference/asset/controller PID {folder.name} is active; no code update performed')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--branch',default='feature/rough-hybrid')
    p.add_argument('--apply',action='store_true',help='Create backup/import branches and merge the verified snapshot')
    args=p.parse_args();repo=Path(args.repo).resolve()
    if git(repo,'status','--porcelain').strip():raise RuntimeError('Worktree is not clean; preserve local changes first')
    if git(repo,'branch','--show-current').decode().strip()!=args.branch:raise RuntimeError('Wrong local branch')
    git_dir=Path(git(repo,'rev-parse','--absolute-git-dir').decode().strip())
    if any((git_dir/x).exists() for x in ('rebase-apply','rebase-merge','MERGE_HEAD')):
        raise RuntimeError('Existing merge/rebase/am session; resolve it first')
    check_idle(repo)
    head=git(repo,'rev-parse','HEAD').decode().strip()
    tracking='refs/remotes/origin/'+args.branch
    base=git(repo,'merge-base',head,tracking).decode().strip()
    target=api('/repos/jianing801/robo_cjl/git/ref/heads/'+urllib.parse.quote(args.branch,safe='/'))['object']['sha']
    commit=api('/repos/jianing801/robo_cjl/git/commits/'+target)
    expected_tree=commit['tree']['sha']
    print('GITHUB_TARGET',target,flush=True)
    print('LOCAL_COMMITS_PRESERVED',head,'BASE',base,flush=True)
    if git(repo,'rev-parse','HEAD^{tree}').decode().strip()==expected_tree:
        print('ALREADY_IDENTICAL_TO_GITHUB');return
    if not args.apply:
        print('PREVIEW_ONLY; use --apply to import and merge');return
    # Archive transfer uses the API and its official codeload redirect, not Git.
    with tempfile.TemporaryFile() as stream:
        print('DOWNLOADING_GITHUB_SNAPSHOT',flush=True)
        with request('/'.join([API,'repos/jianing801/robo_cjl/tarball',target])) as response:
            total=0
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block);total+=len(block)
                if total%(10*1024*1024)==0:print('DOWNLOADED_MIB',total//1024//1024,flush=True)
                if total>1024**3:raise RuntimeError('Archive exceeds 1GiB transfer bound')
        stream.seek(0);tree,count=import_archive(repo,stream,expected_tree)
    # No source-file changes or branch changes have occurred before this point.
    print('EXACT_GITHUB_TREE_VERIFIED',tree,'FILES',count,flush=True)
    if git(repo,'rev-parse','HEAD').decode().strip()!=head or git(repo,'status','--porcelain').strip():
        raise RuntimeError('Local repository changed during download; merge refused')
    check_idle(repo)
    message=f'Import verified GitHub API snapshot {target}\n\nUpstream tree: {tree}\nPreserves local server commits via a three-way merge.\n'
    imported=git(repo,'-c','user.name=jianing801','-c','user.email=jianing801@gmail.com',
        'commit-tree',tree,'-p',base,data=message.encode()).decode().strip()
    suffix=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    backup='backup/server-before-api-update-'+suffix
    git(repo,'branch',backup,head)
    incoming='api-snapshot/'+target[:12]
    existing=subprocess.run(['git','-C',str(repo),'show-ref','--verify','--quiet','refs/heads/'+incoming]).returncode==0
    if existing:incoming+='-'+suffix
    git(repo,'branch',incoming,imported)
    print('BACKUP_BRANCH',backup,flush=True)
    result=subprocess.run(['git','-C',str(repo),'-c','user.name=jianing801','-c','user.email=jianing801@gmail.com',
        'merge','--no-ff','--no-edit',incoming])
    if result.returncode:
        print('MERGE_NEEDS_ATTENTION: local commits and backup retained; do not reset. Run git status.',flush=True)
        raise SystemExit(result.returncode)
    if subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',head,'HEAD']).returncode:
        raise RuntimeError('Local commit ancestry check failed')
    print('API_UPDATE_COMPLETE; local history preserved; origin tracking was not rewritten.',flush=True)
    print(git(repo,'log','-1','--oneline').decode().strip(),flush=True)
    print(git(repo,'status','--short').decode().strip() or 'WORKTREE_CLEAN',flush=True)


if __name__=='__main__':
    try:main()
    except (RuntimeError,urllib.error.URLError,OSError,tarfile.TarError) as error:
        raise SystemExit('UPDATE_STOPPED: '+str(error))
