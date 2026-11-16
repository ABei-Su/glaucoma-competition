"""Wait for audited 448 cache, train two declared continuations, compare."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def run(command,log):
    with (ROOT/'reports'/log).open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    state=ROOT/'reports/v4_status.json'
    try:
        save_json(state,dict(phase='waiting_for_448_cache'))
        deadline=time.monotonic()+7200
        while not (ROOT/'data/cache448/READY.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('448 cache wait exceeded two hours')
            time.sleep(10)
        capacity=json.loads((ROOT/'reports/v4_capacity.json').read_text())
        latency=json.loads((ROOT/'reports/v4_latency_probe.json').read_text())
        if latency['mean_ms']>100: raise ValueError('448 latency capacity gate failed')
        for name,size,epochs,lr,batch,seed in [
            ('v4_r50_448',448,6,'0.00003',capacity['selected_batch'],'20261027'),
            ('v4_r50_384',384,4,'0.00002',16,'20261028')]:
            directory=ROOT/'runs'/name
            if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text())['state']=='complete': continue
            save_json(state,dict(phase='training',run=name))
            command=['train.py','--architecture','resnet50','--run','runs/'+name,'--cache','data/cache'+str(size),
                     '--size',str(size),'--batch',str(batch),'--memory-format','contiguous',
                     '--validation-probability-precision','fp32','--epochs',str(epochs),'--lr',lr,'--pos-weight','1','--seed',seed,
                     '--init-checkpoint','runs/v3_r50_384_w1/best.pt']
            if (directory/'last.pt').exists(): command+=['--resume']
            run(command,name+'.log')
        save_json(state,dict(phase='comparison'))
        run(['tools/compare_v4.py'],'v4_comparison.log')
        save_json(state,dict(phase='candidates_compared'))
    except BaseException as error:
        save_json(state,dict(phase='failed',error=str(error)));raise


if __name__=='__main__': main()
