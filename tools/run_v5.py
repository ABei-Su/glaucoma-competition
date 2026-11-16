"""Sequential, versioned internal validation then frozen-epoch full training."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json


def run(command,log):
    with (ROOT/'reports'/log).open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def train(name,architecture,seed,lr,batch,internal=False,checkpoint_epoch=0):
    directory=ROOT/'runs'/name
    state=directory/'status.json'
    if state.exists() and json.loads(state.read_text())['state']=='complete':return
    command=['train.py','--architecture',architecture,'--run','runs/'+name,'--cache','data/cache384',
             '--size','384','--epochs','8','--lr',str(lr),'--batch',str(batch),'--pos-weight','1',
             '--seed',str(seed),'--memory-format','contiguous','--validation-probability-precision','fp32',
             '--checkpoint-epoch',str(checkpoint_epoch)]
    if internal:command+=['--split-manifest','reports/v5_internal_split.json']
    if (directory/'last.pt').exists():command+=['--resume']
    run(command,name+'.log')


def main():
    status=ROOT/'reports/v5_status.json'
    try:
        capacity=json.loads((ROOT/'reports/v5_capacity.json').read_text())
        latency=json.loads((ROOT/'reports/v5_latency_probe.json').read_text())
        if latency['mean_ms']>100:raise ValueError('Capacity latency gate failed')
        for name,architecture,seed,lr in [
            ('v5_inner_r50','resnet50',20261030,.0002),
            ('v5_inner_convnext','convnext_tiny',20261031,.0001)]:
            save_json(status,dict(phase='internal_training',run=name))
            train(name,architecture,seed,lr,capacity['architectures'][architecture]['selected_batch'],internal=True)
        save_json(status,dict(phase='internal_comparison'))
        selected=ROOT/'output/v5/internal_selection.json'
        if not selected.exists():run(['tools/compare_v5_internal.py'],'v5_internal_comparison.log')
        selection_record=json.loads(selected.read_text())
        for evidence in selection_record['evidence']:
            if hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest()!=evidence['sha256']:
                raise ValueError('Internal selection evidence changed')
        if hashlib.sha256((ROOT/'reports/v5_internal_split.json').read_bytes()).hexdigest()!=selection_record['manifest_sha256']:
            raise ValueError('Selected internal split changed')
        frozen=selection_record['selected']
        save_json(status,dict(phase='full_training',run='v5_full',frozen_internal_choice=frozen))
        train('v5_full',frozen['architecture'],20261032,frozen['lr'],frozen['batch'],checkpoint_epoch=frozen['epoch'])
        output=ROOT/'output/v5'
        config=dict(schema_version=1,models=[dict(checkpoint='../../runs/v5_full/best.pt',tta=frozen['tta'])],
                    name='v5_internal_selected_'+frozen['architecture']+('_tta' if frozen['tta'] else ''),
                    batch=1,precision='fp16',probability_precision='fp32',selection_source='fixed internal train holdout')
        save_json(output/'candidate_00.json',config)
        save_json(output/'candidate_comparison.json',[dict(name=config['name'],models=[dict(checkpoint=str(ROOT/'runs/v5_full/best.pt'),tta=frozen['tta'])],
                   internal_validation_metrics={k:frozen[k] for k in ('auroc','average_precision','auprc_trapezoid')},
                   selection_source='single frozen candidate selected before full training; no official-valid ranking')])
        save_json(status,dict(phase='candidates_compared'))
    except BaseException as error:
        save_json(status,dict(phase='failed',error=str(error)));raise


if __name__=='__main__':main()
