"""Fixed v13 paired-fold trainer; independent from all frozen legacy trainers."""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FundusDataset, audit_inputs, load_rows
from model import build_model
from provenance import digest
from utils import Lifecycle, metrics, save_json
from v13_core import Mixup, soft_bce, validate_folds
from v13_protocol import require_promotion, training_guard

ROOT = Path(__file__).resolve().parent
PUBLIC_SHA = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--fold', type=int, choices=range(3), default=0)
    parser.add_argument('--treatment', choices=('control', 'mixup'), default='control')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.full and args.treatment != 'mixup': parser.error('Full mode requires promoted Mixup')
    args.seed = 20261104 if args.full else 20261101+args.fold
    args.epochs, args.size, args.batch, args.checkpoint_epoch = 8, 384, 16, 8
    args.workers, args.lr, args.augmentation = 4, .0001, 'mild'
    args.architecture = 'convnext_tiny'
    return args


def validation_probability(model, images):
    return (model(images).flatten().float().sigmoid()
            + model(images.flip(-1)).flatten().float().sigmoid())/2


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % 2**32)
    np.random.seed(torch.initial_seed() % 2**32)


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_identity():
    return {name: digest(ROOT/name) for name in
            ('train_mixup.py', 'v13_core.py', 'v13_protocol.py', 'data.py', 'model.py', 'utils.py')}


def commit_predictions(path, names, labels, probabilities):
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer);writer.writerow(['Image_name','True_Label','Probability'])
    writer.writerows(zip(names,labels,probabilities));content=buffer.getvalue().encode('utf-8')
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Preserve different interrupted epoch predictions')
        return
    temporary=path.with_suffix('.tmp');temporary.write_bytes(content);os.replace(temporary,path)


def resume_result(saved):
    result=saved.get('result')
    if not isinstance(result,dict) or result.get('epoch')!=saved['epoch']:
        raise ValueError('Resume checkpoint lacks committed epoch result')
    return result


def main(argv=None):
    args = parse_args(argv)
    run = (ROOT/args.run).resolve()
    if run.parent != ROOT/'runs': raise ValueError('Run must be directly inside project runs')
    state=json.loads((ROOT/'reports/v13_status.json').read_text(encoding='utf-8'))
    training_guard(state,(ROOT/'output/v13/operation.lock').read_text(),os.getppid(),run.name,
                   args.fold,args.treatment,args.full)
    if args.full: require_promotion(ROOT)
    if run.exists() and not args.resume: raise FileExistsError('Preserve existing run')
    if args.resume and (not (run/'last.pt').exists() or (run/'status.json').exists()):
        raise ValueError('Resume requires nonterminal complete-epoch checkpoint')
    run.mkdir(parents=True, exist_ok=args.resume)
    lock = run/'training.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    lifecycle = Lifecycle()
    try:
        save_json(run/'progress.json', dict(phase='data_audit', pid=os.getpid()))
        raw, cache = ROOT/'data/raw/glaucoma_first', ROOT/'data/cache384'
        public = ROOT/'pretrain_model/convnext_tiny-983f1562.pth'
        if digest(public) != PUBLIC_SHA: raise ValueError('Public weights changed')
        sources = source_identity()
        audit = audit_inputs(raw, cache, 384)
        manifest_path = ROOT/'output/v13_planning/three_fold_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        subsets = validate_folds(load_rows(raw, 'train'), (raw/'train.csv').read_bytes(),
                                 Path(audit['manifest_path']).read_bytes(), manifest, args.fold)
        fingerprint = hashlib.sha256((audit['manifest_sha256']+digest(manifest_path)).encode()).hexdigest()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required for declared campaign')
        device = torch.device('cuda')
        generator = torch.Generator().manual_seed(args.seed)
        mixer = Mixup(args.seed)
        train = FundusDataset(raw, 'train', 384, cache, True, augmentation='mild')
        valid = FundusDataset(raw, 'valid' if args.full else 'train', 384, cache)
        if not args.full: train.rows, valid.rows = subsets
        trainloader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4,
                                 pin_memory=True, worker_init_fn=seed_worker, generator=generator)
        # Separate evaluation generator prevents evaluation from consuming model RNG.
        valid_generator = torch.Generator().manual_seed(args.seed+100)
        validloader = DataLoader(valid, batch_size=16, shuffle=False, num_workers=4,
                                 pin_memory=True, worker_init_fn=seed_worker, generator=valid_generator)
        model = build_model(public, architecture='convnext_tiny').to(device)
        initial_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial_hash.update(name.encode()); initial_hash.update(value.cpu().numpy().tobytes())
        initial_hash = initial_hash.hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 8, eta_min=.00001)
        scaler = torch.amp.GradScaler('cuda')
        contract = dict(name='soft_bce_fp32', mixup_alpha=.2 if args.treatment == 'mixup' else None,
                        probability=1. if args.treatment == 'mixup' else 0., reduction='mean')
        binding = dict(data_fingerprint=fingerprint, source_identity=sources,
                       public_pretrain_sha256=PUBLIC_SHA, loss_spec=contract,
                       initial_model_sha256=initial_hash, scope='full_official_train' if args.full else 'three_fold_train',
                       train_rows=len(train), holdout_rows=len(valid), fold=None if args.full else args.fold,
                       split_manifest_sha256=digest(manifest_path))
        start_epoch = 0
        result = None
        if args.resume:
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            if saved['binding'] != binding: raise ValueError('Resume binding changed')
            expected_config = dict(vars(args), resume=False)
            if saved['config'] != expected_config: raise ValueError('Resume configuration changed')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
            random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
            torch.set_rng_state(saved['rng_torch']); torch.cuda.set_rng_state_all(saved['rng_cuda'])
            generator.set_state(saved['rng_loader']); valid_generator.set_state(saved['rng_valid'])
            mixer.load_state_dict(saved['rng_mixup']); start_epoch = saved['epoch']
            result=resume_result(saved)
            if start_epoch==8:
                for name,sha in saved['final_artifacts'].items():
                    if digest(run/name)!=sha:raise ValueError('Interrupted final epoch artifacts changed')
        else:
            save_json(run/'config.json', dict(vars(args), resume=False))
            save_json(run/'binding.json', binding)
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0))
        if args.resume:
            if json.loads((run/'environment.json').read_text(encoding='utf-8'))!=environment:
                raise ValueError('Resume environment changed')
        else:save_json(run/'environment.json',environment)
        lifecycle.transition('training')
        for epoch in range(start_epoch, 8):
            started = time.monotonic(); model.train(); total, count = 0., 0
            for step, (images, labels, names) in enumerate(trainloader):
                if epoch == 0 and step < 2:
                    save_json(run/('batch_identity_'+str(step)+'.json'), list(names))
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if args.treatment == 'mixup': images, labels = mixer(images, labels)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'): logits = model(images).flatten()
                loss = soft_bce(logits, labels)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer); scaler.update()
                total += loss.item()*len(labels); count += len(labels)
                if step % 100 == 0:
                    progress = dict(phase='training', epoch=epoch+1, step=step, steps=len(trainloader),
                                    loss=total/count, elapsed_seconds=time.monotonic()-started, pid=os.getpid())
                    save_json(run/'progress.json', progress); print(json.dumps(progress), flush=True)
            lifecycle.transition('validating')
            save_json(run/'progress.json', dict(phase='validating', epoch=epoch+1, pid=os.getpid()))
            model.eval(); y, p, names = [], [], []
            with torch.inference_mode():
                for images, labels, batch_names in validloader:
                    with torch.autocast('cuda'): probabilities = validation_probability(model, images.cuda(non_blocking=True))
                    y.extend(labels.tolist()); p.extend(probabilities.cpu().tolist()); names.extend(batch_names)
            result = dict(metrics(y, p), epoch=epoch+1, train_loss=total/count, seconds=time.monotonic()-started)
            scheduler.step(); lifecycle.transition('saving')
            if epoch == 7:
                atomic_save(dict(model=model.state_dict(), schema_version=1, architecture='convnext_tiny',
                                 size=384, memory_format='contiguous', validation_probability_precision='fp32',
                                 threshold=.5, metrics=result, checkpoint_epoch=8, binding=binding), run/'best.pt')
                commit_predictions(run/'validation_predictions.csv',names,y,p)
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             scaler=scaler.state_dict(), epoch=epoch+1, result=result, config=dict(vars(args), resume=False), binding=binding,
                             rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                             rng_cuda=torch.cuda.get_rng_state_all(), rng_loader=generator.get_state(),
                             rng_valid=valid_generator.get_state(), rng_mixup=mixer.state_dict(),
                             final_artifacts={n:digest(run/n) for n in ('best.pt','validation_predictions.csv')} if epoch==7 else {}), run/'last.pt')
            with (run/'history.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True)
            lifecycle.transition('complete' if epoch == 7 else 'training')
        if result is None or result['epoch']!=8:raise ValueError('No complete fixed epoch result')
        if sources != source_identity(): raise ValueError('Training source changed')
        save_json(run/'status.json', dict(state='complete', epoch=8, metrics=result, binding=binding))
    except BaseException as error:
        save_json(run/'status.json', dict(state='failed', phase=lifecycle.state, error=repr(error)))
        raise
    finally:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()): lock.unlink()


if __name__ == '__main__': main()
