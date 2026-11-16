"""Finite guarded internal Age experiment and conditional full retraining."""
import json
import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import digest,inference_fingerprint
from utils import save_json
from v8_integrity import control,verify_evidence
from v10_integrity import recipe,validate_completed_age,frozen

ALLOWED={'ready':{'verifying'},'verifying':{'internal_training','not_promoted','failed'},
 'internal_training':{'internal_comparison','failed'},'internal_comparison':{'full_training','not_promoted','failed'},
 'full_training':{'full_ready','failed'},'full_ready':set(),'not_promoted':set(),'failed':set()}


def transition(state,phase,details):
    if set(details)&{'phase','revision','history'} or phase not in ALLOWED[state['phase']]:
        raise ValueError('Illegal v10 coordinator transition')
    if phase=='internal_training' and details.get('resource_passed') is not True:
        raise ValueError('Resources must pass before supervised fitting')
    if phase=='full_training' and (details.get('gate',{}).get('passed') is not True or
            type(details.get('frozen_epoch')) is not int or not 1<=details['frozen_epoch']<=8):
        raise ValueError('Full training requires internal qualification and frozen epoch')
    if phase=='full_ready' and details.get('frozen_epoch')!=state['frozen_epoch']:
        raise ValueError('Full completion cannot change the frozen epoch')
    event=dict(phase=phase,revision=state['revision']+1,time=datetime.now().astimezone().isoformat(),details=copy.deepcopy(details))
    return dict(phase=phase,revision=event['revision'],history=copy.deepcopy(state['history'])+[event],**copy.deepcopy(details))


def run(command,log):
    with (ROOT/'reports'/log).open('x',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    output=ROOT/'output/v10';output.mkdir(exist_ok=True)
    status=ROOT/'reports/v10_status.json'
    if status.exists() or (output/'frozen_inputs.json').exists():raise FileExistsError('Preserve v10 operation; inspect before manual recovery')
    for name in ('v10_inner_age','v10_full_age'):
        if (ROOT/'runs'/name).exists():raise FileExistsError('Preserve existing Age run '+name)
    lock=output/'operation.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state=dict(phase='ready',revision=0,history=[])
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,details);save_json(status,state)
    try:
        advance('verifying')
        resource=json.loads((ROOT/'reports/v10_resource_probe.json').read_text(encoding='utf-8'))
        if not resource['passed']:
            advance('not_promoted',reason='Fixed resource gate failed');return
        verify_evidence(resource['evidence']);control(ROOT)
        previous=json.loads((ROOT/'output/v8/internal_selection.json').read_text(encoding='utf-8'));verify_evidence(previous['evidence'])
        sources=[ROOT/name for name in ('clinical.py','clinical_predict.py','train_age.py','v10_integrity.py',
            'tools/run_v10.py','tools/compare_v10_internal.py','tools/probe_v10.py',
            'tests/test_clinical.py','tests/test_clinical_predict.py','tests/test_age_training.py','tests/test_v10_integrity.py','tests/test_v10_protocol.py',
            'docs/OPTIMIZATION-v10.md','docs/ADR-009-age-input-contract.md','docs/REVIEW-v10-preflight.md',
            'reports/v10_verification.log','reports/v10_resource_probe.json','reports/v5_internal_split.json',
            'data/cache384/READY.json','data/raw/glaucoma_first/train.csv','data/raw/glaucoma_first/valid.csv',
            'pretrain_model/convnext_tiny-983f1562.pth','output/v8/internal_selection.json','output/v8/internal_predictions.npz',
            'model.py','data.py','train.py','predict.py','utils.py','internal_split.py','provenance.py','v7_integrity.py','v8_integrity.py',
            'requirements-local.lock.txt')]
        for name in ('config.json','best.pt','best_metrics.json','status.json'):
            sources.append(ROOT/'runs/v5_inner_convnext'/name)
        evidence=[dict(path=str(path),sha256=digest(path)) for path in sources]
        evidence+=inference_fingerprint(ROOT/'output/v6/selected_config.json')
        evidence+=resource['evidence']
        save_json(output/'frozen_inputs.json',dict(evidence=evidence,internal_recipe=recipe('runs/v10_inner_age'),
            full_seed=20261041,source='officialtrain only; Age stats fit actual training rows; no test or leaderboard fitting',
            declared_trial_csv_policy='One fixed full candidate may be saved after resource checks; validation regression disclosed'))
        frozen(ROOT);advance('internal_training',run='v10_inner_age',resource_passed=True)
        run(['train_age.py','--run','runs/v10_inner_age'],'v10_inner_age.log')
        validate_completed_age(ROOT/'runs/v10_inner_age',recipe('runs/v10_inner_age'),ROOT);frozen(ROOT)
        advance('internal_comparison')
        run(['tools/compare_v10_internal.py'],'v10_internal_comparison.log')
        selection=json.loads((output/'internal_selection.json').read_text(encoding='utf-8'));verify_evidence(selection['evidence'])
        if not selection['gate']['passed']:
            advance('not_promoted',gate=selection['gate'],reason='Actual-Age, AP or missing-Age internal gate failed');return
        epoch=selection['selected_epoch'];advance('full_training',run='v10_full_age',frozen_epoch=epoch,gate=selection['gate'])
        run(['train_age.py','--run','runs/v10_full_age','--full','--checkpoint-epoch',str(epoch),'--seed','20261041'],'v10_full_age.log')
        validate_completed_age(ROOT/'runs/v10_full_age',recipe('runs/v10_full_age',True,epoch),ROOT);frozen(ROOT)
        advance('full_ready',frozen_epoch=epoch,gate=selection['gate'])
        print('V10 FULL CHECKPOINT READY',flush=True)
    except BaseException as error:
        if state['phase'] not in ('failed','not_promoted','full_ready'):advance('failed',error=str(error))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
