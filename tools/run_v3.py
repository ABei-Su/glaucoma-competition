"""Run the two declared supervised R50 stages, then compare fixed candidates."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def run(command,log):
    with (ROOT/'reports'/log).open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    state=ROOT/'reports/v3_status.json'
    try:
        probe=json.loads((ROOT/'reports/r50_capacity.json').read_text())
        latency=json.loads((ROOT/'reports/r50_latency_probe.json').read_text())
        if latency['mean_ms']>100: raise ValueError('Capacity gate failed')
        batch=probe['selected_batch']
        for name,epochs,lr,weight,seed,initial in [
            ('v3_r50_384_w2',8,'0.0002','2','20261025',None),
            ('v3_r50_384_w1',4,'0.00005','1','20261026','runs/v3_r50_384_w2/best.pt')]:
            directory=ROOT/'runs'/name
            if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text())['state']=='complete': continue
            save_json(state,dict(phase='training',run=name))
            command=['train.py','--architecture','resnet50','--run','runs/'+name,'--cache','data/cache384',
                     '--size','384','--batch',str(batch),'--memory-format',probe.get('selected_memory_format','contiguous'),
                     '--validation-probability-precision','fp32','--epochs',str(epochs),'--lr',lr,'--pos-weight',weight,'--seed',seed]
            if initial: command+=['--init-checkpoint',initial]
            if (directory/'last.pt').exists(): command+=['--resume']
            run(command,name+'.log')
        save_json(state,dict(phase='comparison'))
        run(['tools/compare_v3.py'],'v3_comparison.log')
        save_json(state,dict(phase='candidates_compared'))
    except BaseException as error:
        save_json(state,dict(phase='failed',error=str(error)))
        raise


if __name__=='__main__': main()
