"""Finish the predeclared candidates once the running 224/cache jobs complete."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json
from cache_paths import cache_path


def status(phase,**extra):
    save_json(ROOT/'reports'/'optimization_status.json',dict(phase=phase,**extra))
    print(phase,extra,flush=True)


def run(command,logname):
    with (ROOT/'reports'/logname).open('a',encoding='utf-8') as file:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,check=True)


def main():
    try:
        status('waiting_for_224_and_384_cache')
        deadline=time.monotonic()+7200
        while True:
            state_path=ROOT/'runs/opt_r18_224_w2/status.json'
            state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            if state.get('state')=='failed':
                raise RuntimeError('224 candidate failed: '+state.get('error','unknown'))
            if state.get('state')=='complete' and (cache_path(384)/'READY.json').exists():
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Preparation/224 training timeout')
            time.sleep(10)
        for name,weight,seed in [('opt_r18_384_w2','2','20261023'),('opt_r18_384_w1','1','20261024')]:
            directory=ROOT/'runs'/name
            state_path=directory/'status.json'
            if state_path.exists() and json.loads(state_path.read_text())['state']=='complete':
                continue
            status('training',run=name)
            command=['train.py','--run','runs/'+name,'--cache',str(cache_path(384)),'--size','384','--batch','32',
                     '--epochs','8','--lr','0.0001','--pos-weight',weight,'--seed',seed,
                     '--init-checkpoint','runs/baseline_r18_224/best.pt']
            if (directory/'last.pt').exists():
                command+=['--resume']
            run(command,name+'.log')
        status('validation_comparison')
        run(['tools/compare_candidates.py'],'candidate_comparison.log')
        status('candidates_compared',note='No new test inference yet; validate final inference path and latency before selection')
    except BaseException as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__':
    main()
