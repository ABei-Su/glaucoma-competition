"""Foreground execution-host supervisor, one attempt and four-hour limit."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    from provenance import digest
    from utils import save_json
    review=json.loads((ROOT/'reports/v18_impl002_review.json').read_text('utf-8-sig'))
    if not review['approved']:raise ValueError('Independent approval required')
    for item in review['files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Reviewed file changed')
    manifest=ROOT/'output/v18/implementation_002/frozen_inputs.json'
    if digest(manifest)!=review['remediation_manifest_sha256']:raise ValueError('Freeze binding mismatch')
    frozen=json.loads(manifest.read_text('utf-8-sig'))
    for item in frozen['files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen file changed')
    if (ROOT.parent/'第十八版试交_v18').exists():raise ValueError('Existing delivery')
    with (ROOT/'output/v18/implementation_002/launch.reservation').open('x') as f:f.write(str(os.getpid()))
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',TEMP=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'),TORCH_HOME=str(ROOT/'pretrain_model'))
    with (ROOT/'reports/v18_impl002.log').open('w',encoding='utf-8') as log, (ROOT/'reports/v18_impl002.stderr').open('w',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.finalize_v18_impl002'],cwd=ROOT,env=env,stdout=log,stderr=err)
        save_json(ROOT/'reports/v18_impl002_launch.json',dict(pid=p.pid,supervisor_pid=os.getpid(),started_at=time.time(),deadline=time.time()+14400,launch='execution host foreground pipes'))
        try:
            code=p.wait(timeout=14400)
            result=dict(outcome='exited',exit_code=code,time=time.time())
        except subprocess.TimeoutExpired:
            p.kill();p.wait()
            code=1;result=dict(outcome='budget_exceeded',time=time.time())
        save_json(ROOT/'reports/v18_impl002_exit.json',result)
    return code


if __name__=='__main__':sys.exit(main())
