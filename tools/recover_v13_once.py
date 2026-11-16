"""User-authorized single recovery of the interrupted fold2 control coordinator."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from tools.run_v13 import ROOT, OUT, frozen, completed, run_child, compare, require_promotion
from provenance import digest
from utils import save_json
from v13_protocol import transition, TERMINAL, remaining_timeout


def main():
    recovery=OUT/'recovery_001'
    recovery.mkdir(exist_ok=False)
    status=ROOT/'reports/v13_status.json'
    state=json.loads(status.read_text(encoding='utf-8'))
    deadline=state['deadline']
    remaining_timeout(deadline,time.time(),10800)
    if state['phase']!='fold_training' or state['pid']!=9544:raise ValueError('Unexpected recovery state')
    # Host-side launch and this check both require old owners to have exited.
    query=subprocess.run(['powershell','-NoProfile','-Command',
        "@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.ProcessId -in @(9544,4084) }).Count"],
        capture_output=True,text=True,check=True)
    if query.stdout.strip()!='0':raise RuntimeError('Old owners still live or query invalid')
    frozen()
    import torch
    run=ROOT/'runs/v13_fold2_control'
    saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    if saved['epoch']!=3 or saved['config']!=json.loads((run/'config.json').read_text(encoding='utf-8')):
        raise ValueError('Unexpected checkpoint/config')
    if saved['binding']!=json.loads((run/'binding.json').read_text(encoding='utf-8')):raise ValueError('Binding mismatch')
    for name,sha in saved['binding']['source_identity'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Resume source changed')
    if (run/'status.json').exists() or (ROOT/'runs/v13_fold2_mixup').exists():raise ValueError('Unexpected stage outputs')
    for fold in range(2):
        for treatment in ('control','mixup'):completed(ROOT/('runs/v13_fold'+str(fold)+'_'+treatment),fold,treatment)
    paths=[status,ROOT/'reports/v13_child.json',OUT/'operation.lock',run/'training.lock',run/'last.pt',run/'progress.json',run/'history.jsonl']
    evidence=[]
    for index,path in enumerate(paths):
        target=recovery/(str(index)+'_'+path.name);shutil.copy2(path,target)
        if digest(path)!=digest(target):raise ValueError('Recovery backup failed')
        evidence.append(dict(path=str(target),sha256=digest(target)))
    save_json(recovery/'record.json',dict(evidence=evidence,epoch=3,deadline=deadline,
        cause='Unknown abrupt process termination; no caught Python traceback; host did not reboot',
        authorization='User requested start and continue after interruption disclosure',
        policy='One resume, unchanged frozen source/recipe/deadline; retain four completed fits'))
    if (OUT/'operation.lock').read_text().strip()!='9544' or (run/'training.lock').read_text().strip()!='4084':
        raise ValueError('Unexpected stale lock ownership')
    # Explicit audited stale lock archival, never general automatic cleanup.
    (OUT/'operation.lock').rename(recovery/'original_operation.lock')
    (run/'training.lock').rename(recovery/'original_training.lock')
    with (OUT/'operation.lock').open('x') as stream:stream.write(str(os.getpid()))
    state=dict(state,pid=os.getpid(),revision=state['revision']+1,updated_at=time.time())
    state['history'].append(dict(event='authorized_recovery_001',phase='fold_training',time=time.time(),pid=os.getpid()))
    save_json(status,state)
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,**details);save_json(status,state)
    try:
        # Original per-fit deadline is retained, including the interruption interval.
        original_child=json.loads((recovery/'1_v13_child.json').read_text(encoding='utf-8'))
        fit_deadline=min(deadline,original_child['started_at']+10800)
        run_child(['train_mixup.py','--fold','2','--treatment','control','--run','runs/v13_fold2_control','--resume'],
                  'fold2_control_recovery001',fit_deadline)
        frozen();completed(run,2,'control')
        run_child(['train_mixup.py','--fold','2','--treatment','mixup','--run','runs/v13_fold2_mixup'],'fold2_mixup',deadline)
        frozen();completed(ROOT/'runs/v13_fold2_mixup',2,'mixup')
        advance('comparison');remaining_timeout(deadline,time.time(),10800);result=compare()
        remaining_timeout(deadline,time.time(),10800)
        if not result['passed']:advance('not_promoted',reason='Three-fold gate failed',comparison=result);return
        paths=[OUT/'comparison.json',OUT/'oof_predictions.npz']
        for fold in range(3):
            for treatment in ('control','mixup'):
                directory=ROOT/('runs/v13_fold'+str(fold)+'_'+treatment)
                paths += [directory/n for n in ('best.pt','config.json','binding.json','status.json','history.jsonl','validation_predictions.csv')]
        save_json(OUT/'promotion.json',dict(comparison_sha256=digest(OUT/'comparison.json'),
            evidence=[dict(path=str(p),sha256=digest(p)) for p in paths]))
        require_promotion(ROOT);full_deadline=time.time()+14400
        advance('full_training',full_deadline=full_deadline,comparison=result)
        run_child(['train_mixup.py','--full','--treatment','mixup','--run','runs/v13_full_mixup'],'full_mixup',full_deadline)
        frozen();completed(ROOT/'runs/v13_full_mixup',full=True);advance('finalization')
        run_child(['-m','tools.run_v13','--stage','finalize'],'finalization',full_deadline,14400)
        result=json.loads((OUT/'finalization/result.json').read_text(encoding='utf-8'));frozen()
        advance('awaiting_review' if result['passed'] else 'not_promoted',result=result)
    except (TimeoutError,subprocess.TimeoutExpired) as error:
        if state['phase'] not in TERMINAL:advance('budget_exceeded',error=repr(error))
        raise
    except BaseException as error:
        if state['phase'] not in TERMINAL:advance('failed',error=repr(error))
        raise
    finally:
        lock=OUT/'operation.lock'
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
