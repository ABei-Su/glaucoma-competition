"""One color-robust candidate, conditional fixed-epoch full training."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json
from v7_integrity import validate_completed_run
from v8_integrity import recipe,frozen,verify_evidence


def run(command,log):
    with (ROOT/'reports'/log).open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def train(name,seed,internal=False,epoch=0):
    directory=ROOT/'runs'/name;state=directory/'status.json';expected=recipe(name,seed,internal,epoch)
    if state.exists() and json.loads(state.read_text())['state']=='complete':
        validate_completed_run(directory,expected,ROOT);return
    command=['train.py','--architecture','convnext_tiny','--run','runs/'+name,'--cache','data/cache384',
             '--size','384','--epochs','8','--batch','16','--workers','4','--lr','0.0001','--pos-weight','1',
             '--seed',str(seed),'--augmentation','color_robust','--memory-format','contiguous',
             '--validation-probability-precision','fp32','--checkpoint-epoch',str(epoch)]
    if internal:command+=['--split-manifest','reports/v5_internal_split.json']
    if (directory/'last.pt').exists():command+=['--resume']
    run(command,name+'.log');validate_completed_run(directory,expected,ROOT)


def main():
    state=ROOT/'reports/v8_status.json';output=ROOT/'output/v8'
    try:
        if not (ROOT/'reports/v8_latency_probe.json').exists():
            save_json(state,dict(phase='resource_probe'));run(['tools/probe_v8.py'],'v8_resource_probe.log')
        speed=json.loads((ROOT/'reports/v8_latency_probe.json').read_text());verify_evidence(speed['evidence'])
        if not any(speed['eligible'].values()):
            save_json(state,dict(phase='not_promoted',reason='Neither predeclared shape met 90ms initial margin'));return
        if not (output/'frozen_inputs.json').exists():run(['tools/prepare_v8.py'],'v8_preparation.log')
        frozen(ROOT);save_json(state,dict(phase='internal_training',run='v8_inner_color'))
        train('v8_inner_color',20261031,True)
        save_json(state,dict(phase='internal_comparison'))
        if not (output/'internal_selection.json').exists():run(['tools/compare_v8_internal.py'],'v8_internal_comparison.log')
        selection=json.loads((output/'internal_selection.json').read_text());verify_evidence(selection['evidence']);frozen(ROOT)
        chosen=selection['selected']
        if chosen is None:
            save_json(state,dict(phase='not_promoted',reason='Fixed color recipe did not improve both clean and worst stress AUROC by .0005'));return
        save_json(state,dict(phase='full_training',run='v8_full_color',frozen_internal_choice=chosen))
        train('v8_full_color',20261040,False,chosen['epoch']);frozen(ROOT)
        members=[dict(checkpoint='../../runs/v3_r50_384_w1/best.pt',tta=True),
                 dict(checkpoint='../../runs/v8_full_color/best.pt',tta=chosen['tta'])]
        name='v8_fixed_r50_color_convnext_'+chosen['mode']
        config=dict(schema_version=1,name=name,models=members,batch=1,precision='fp16',probability_precision='fp32',
                    selection_source='fixed color-robust internal gate; no official-valid ranking')
        save_json(output/'candidate_00.json',config)
        save_json(output/'candidate_comparison.json',[dict(name=name,models=[dict(checkpoint=str((output/m['checkpoint']).resolve()),tta=m['tta']) for m in members],
                  internal_convnext_clean_metrics=chosen['metrics']['clean'],internal_worst_auroc=chosen['worst_auroc'],
                  selection_source=config['selection_source'])])
        save_json(state,dict(phase='candidates_compared'))
    except BaseException as error:
        save_json(state,dict(phase='failed',error=str(error)));raise


if __name__=='__main__':main()
