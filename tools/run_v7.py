"""One new internal recipe, frozen deployment shape, then fixed-epoch full training."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json
from v7_integrity import requested_recipe,validate_completed_run


def run(command,log):
    with (ROOT/'reports'/log).open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def train(name,seed,batch,internal=False,epoch=0):
    directory=ROOT/'runs'/name;state=directory/'status.json'
    expected=requested_recipe('runs/'+name,seed,batch,internal,epoch)
    if state.exists() and json.loads(state.read_text())['state']=='complete':
        validate_completed_run(directory,expected,ROOT)
        return
    command=['train.py','--architecture','efficientnet_v2_s','--run','runs/'+name,'--cache','data/cache384',
             '--size','384','--epochs','8','--lr','0.0001','--batch',str(batch),'--pos-weight','1','--seed',str(seed),
             '--memory-format','contiguous','--validation-probability-precision','fp32','--checkpoint-epoch',str(epoch)]
    if internal:command+=['--split-manifest','reports/v5_internal_split.json']
    if (directory/'last.pt').exists():command+=['--resume']
    run(command,name+'.log')
    validate_completed_run(directory,expected,ROOT)


def main():
    state=ROOT/'reports/v7_status.json'
    try:
        speed=json.loads((ROOT/'reports/v7_latency_probe.json').read_text())
        if not any(speed['eligible'].values()):raise ValueError('No speed-qualified deployment shape')
        batch=json.loads((ROOT/'reports/v7_capacity.json').read_text())['selected_batch']
        save_json(state,dict(phase='internal_training',run='v7_inner_efficientnet'))
        train('v7_inner_efficientnet',20261033,batch,internal=True)
        save_json(state,dict(phase='internal_comparison'))
        target=ROOT/'output/v7/internal_selection.json'
        if not target.exists():run(['tools/compare_v7_internal.py'],'v7_internal_comparison.log')
        selection=json.loads(target.read_text())
        for item in selection['evidence']:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Internal selection evidence changed')
        if hashlib.sha256((ROOT/'reports/v5_internal_split.json').read_bytes()).hexdigest()!=selection['manifest_sha256']:
            raise ValueError('Internal split changed after selection')
        chosen=selection['selected']
        if chosen is None:
            save_json(state,dict(phase='not_promoted',reason='No internally selected speed-qualified recipe improved the fixed internal ConvNeXt reference'))
            return
        save_json(state,dict(phase='full_training',run='v7_full_efficientnet',frozen_internal_choice=chosen))
        frozen_batch=selection['frozen_recipe']['batch']
        train('v7_full_efficientnet',20261034,frozen_batch,epoch=chosen['efficientnet_epoch'])
        members=[dict(checkpoint='../../runs/v7_full_efficientnet/best.pt',tta=chosen['tta'])]
        if chosen['pair']:members.append(dict(checkpoint='../../runs/v5_full/best.pt',tta=chosen['tta']))
        output=ROOT/'output/v7';name='v7_internal_selected_'+chosen['name']
        config=dict(schema_version=1,name=name,models=members,batch=1,precision='fp16',probability_precision='fp32',
                    selection_source='frozen reused-train-holdout recipe, not official-valid ranking')
        save_json(output/'candidate_00.json',config)
        save_json(output/'candidate_comparison.json',[dict(name=name,models=[dict(checkpoint=str((output/m['checkpoint']).resolve()),tta=m['tta']) for m in members],
                   internal_validation_metrics={key:chosen[key] for key in ('auroc','average_precision','auprc_trapezoid')},
                   selection_source=config['selection_source'])])
        save_json(state,dict(phase='candidates_compared'))
    except BaseException as error:
        save_json(state,dict(phase='failed',error=str(error)));raise


if __name__=='__main__':main()
