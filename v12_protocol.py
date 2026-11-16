"""Pure v12 decisions, immutable inference identities and completed-run guards."""
import copy
import json
import math
from datetime import datetime
from pathlib import Path
from provenance import digest,inference_fingerprint

ROOT=Path(__file__).resolve().parent
RUNTIME=['v12_protocol.py','tools/predict_v12.py','predict.py','model.py','data.py','utils.py','provenance.py']
STATES={
 'experiment':{'ready':{'preflight','failed'},'preflight':{'internal_training','not_promoted','failed'},
  'internal_training':{'internal_comparison','failed'},'internal_comparison':{'full_training','not_promoted','failed'},
  'full_training':{'full_ready','failed'},'full_ready':set(),'not_promoted':set(),'failed':set()},
 'finalization':{'ready':{'waiting_for_full','failed'},'waiting_for_full':{'timing','not_promoted','failed'},
  'timing':{'validating','not_promoted','failed'},'validating':{'predicting','failed'},
  'predicting':{'checking','failed'},'checking':{'complete','failed'},'complete':set(),'not_promoted':set(),'failed':set()}}


def recipe(internal,epoch=0):
    if type(internal) is not bool or type(epoch) is not int or (epoch!=0 if internal else not 1<=epoch<=8):
        raise ValueError('Invalid fixed internal/full checkpoint scope')
    return dict(data='data/raw/glaucoma_first',cache='data/cache384',architecture='convnext_tiny',
        memory_format='contiguous',validation_probability_precision='fp32',
        pretrain='pretrain_model/convnext_tiny-983f1562.pth',run='runs/v12_inner_focal' if internal else 'runs/v12_full_focal',
        epochs=8,size=384,batch=16,workers=4,seed=20261031 if internal else 20261032,resume=False,
        init_checkpoint=None,lr=.0001,pos_weight=1.,max_steps=0,
        split_manifest='reports/v5_internal_split.json' if internal else None,checkpoint_epoch=epoch,augmentation='mild',full=not internal,focal_gamma=2.,focal_alpha=-1.)


def training_command(internal,epoch=0):
    result=['train_focal.py']
    for key,value in recipe(internal,epoch).items():
        if value is None or value is False:continue
        if value is True:result.append('--'+key.replace('_','-'));continue
        result.extend(['--'+key.replace('_','-'),str(value)])
    return result


def resource_gate(peak,mean,delta,size):
    return (type(peak) is int and 0<peak<5_000_000_000 and math.isfinite(mean) and 0<mean<=100 and
            math.isfinite(delta) and 0<=delta<=1e-7 and type(size) is int and 0<size<=500_000_000)


def quality_gate(reference,candidate):
    values=[r.get(key,float('nan')) for r in (reference,candidate) for key in ('auroc','average_precision')]
    valid=all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values)
    auroc=candidate.get('auroc',float('nan'))-reference.get('auroc',float('nan'))
    ap=candidate.get('average_precision',float('nan'))-reference.get('average_precision',float('nan'))
    return dict(passed=bool(valid and auroc>0 and ap>=-1e-12),auroc_gain=auroc,average_precision_gain=ap)


def transition(state,phase,details):
    kind=state['kind']
    if set(details)&{'kind','phase','revision','history'} or phase not in STATES[kind][state['phase']]:
        raise ValueError('Illegal v12 state transition')
    if phase=='internal_training' and details.get('resource_passed') is not True:raise ValueError('Measured resources required')
    if phase=='full_training' and (details.get('gate',{}).get('passed') is not True or
            type(details.get('frozen_epoch')) is not int or not 1<=details['frozen_epoch']<=8):
        raise ValueError('Qualified internal gate and fixed epoch required')
    if phase=='full_ready' and details.get('frozen_epoch')!=state['frozen_epoch']:raise ValueError('Frozen epoch changed')
    if kind=='finalization':
        if phase=='timing' and details.get('full_ready') is not True:raise ValueError('Completed full model required')
        if phase=='predicting' and (details.get('resource_passed') is not True or details.get('trial_frozen') is not True):
            raise ValueError('Resource compliance and frozen trial required')
        if phase=='complete' and (details.get('rows')!=5855 or details.get('delivery_verified') is not True):
            raise ValueError('Complete verified delivery required')
    event=dict(phase=phase,time=datetime.now().astimezone().isoformat(),revision=state['revision']+1,details=copy.deepcopy(details))
    return dict(kind=kind,phase=phase,revision=event['revision'],history=copy.deepcopy(state['history'])+[event],**copy.deepcopy(details))


def verify_evidence(items):
    for item in items:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen v12 file changed: '+item['path'])


def frozen(root):
    result=json.loads((Path(root)/'output/v12/frozen_inputs.json').read_text(encoding='utf-8'))
    verify_evidence(result['evidence']);return result


def inference_identity(config):
    config=Path(config).resolve();spec=json.loads(config.read_text(encoding='utf-8'))
    if (spec.get('schema_version')!=1 or len(spec.get('models',[]))!=2 or spec.get('batch')!=1 or
            spec.get('precision')!='fp16' or spec.get('probability_precision')!='fp32' or spec.get('threshold')!=.5):
        raise ValueError('Unsupported fixed v12 pair contract')
    for item in spec['models']:
        if item.get('tta') is not True or item.get('fuse_conv_bn',False) is not False:raise ValueError('Fixed TTA pair required')
        if digest(config.parent/item['checkpoint'])!=item.get('sha256'):raise ValueError('Pinned member hash changed')
    return inference_fingerprint(config)+[dict(path=str(ROOT/name),sha256=digest(ROOT/name)) for name in RUNTIME]


def validate_completed(root,internal,epoch=0):
    from v7_integrity import validate_completed_run
    root=Path(root);expected=recipe(internal,epoch);run=root/expected['run']
    checked=validate_completed_run(run,expected,root)
    history=[json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['epoch'] for r in history]!=list(range(1,9)):raise ValueError('Full eight-epoch history required')
    from focal_loss import loss_spec
    if checked['checkpoint'].get('loss_spec')!=loss_spec():raise ValueError('Completed focal loss contract mismatch')
    if checked['checkpoint'].get('public_pretrain_sha256')!=digest(root/expected['pretrain']):
        raise ValueError('Completed focal public-source mismatch')
    selected=checked['checkpoint']['metrics']
    if history[selected['epoch']-1]!=selected:raise ValueError('Selected checkpoint/history binding mismatch')
    if internal:
        binding=json.loads((run/'split_binding.json').read_text(encoding='utf-8'))
        if (binding['training_rows']!=43686 or binding['validation_rows']!=7710 or
                binding['data_fingerprint']!=checked['checkpoint']['data_fingerprint'] or
                binding['manifest_sha256']!=checked['checkpoint']['split_manifest_sha256']):
            raise ValueError('Internal row/split binding mismatch')
    return checked
