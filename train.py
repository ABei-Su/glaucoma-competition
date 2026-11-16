"""Reproducible supervised baseline. Official validation never enters training."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FundusDataset, audit_inputs
from model import build_model,probabilities_from_logits
from utils import Lifecycle, is_better, metrics, save_json
from internal_split import partition_rows


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/glaucoma_first')
    parser.add_argument('--cache', default='data/cache224')
    parser.add_argument('--architecture',choices=['resnet18','resnet50','convnext_tiny','efficientnet_v2_s'],default='resnet18')
    parser.add_argument('--memory-format',choices=['contiguous','channels_last'],default='contiguous')
    parser.add_argument('--validation-probability-precision',choices=['fp16','fp32'],default=None)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--run', default='runs/baseline_r18_224')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init-checkpoint', default=None, help='Initialize a new supervised run from a previous trusted model')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pos-weight', type=float, default=None)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke test only; use a separate run')
    parser.add_argument('--split-manifest',default=None,help='Partition official TRAIN into training and internal validation')
    parser.add_argument('--checkpoint-epoch',type=int,default=0,help='Save this preselected epoch instead of choosing with validation; zero keeps legacy selection')
    parser.add_argument('--augmentation',choices=['mild','color_robust'],default='mild')
    args = parser.parse_args(argv)
    if args.pretrain is None:
        args.pretrain='pretrain_model/'+{'resnet18':'resnet18-f37072fd.pth','resnet50':'resnet50-11ad3fa6.pth',
                                        'convnext_tiny':'convnext_tiny-983f1562.pth',
                                        'efficientnet_v2_s':'efficientnet_v2_s-dd5fe13b.pth'}[args.architecture]
    if args.validation_probability_precision is None:
        args.validation_probability_precision='fp16' if args.architecture=='resnet18' else 'fp32'
    if not math.isfinite(args.lr) or args.lr<=0 or args.epochs<1 or args.batch<1 or args.size<32 or args.workers<0:
        parser.error('Invalid learning rate, epochs, batch, size or workers')
    if args.pos_weight is not None and (not math.isfinite(args.pos_weight) or args.pos_weight<=0):
        parser.error('Positive class weight must be finite and positive')
    if args.split_manifest and args.init_checkpoint:
        parser.error('Internal holdout requires fresh public pretraining; exposed supervised initialization forbidden')
    if not 0<=args.checkpoint_epoch<=args.epochs:
        parser.error('Checkpoint epoch must be zero or within the training schedule')
    return args


def main():
    args = parse_args()
    audit_path = Path(args.cache)/'READY.json'
    if not audit_path.is_file():
        raise RuntimeError('Complete tools/prepare_data.py and leakage audit before training')
    audit = audit_inputs(args.data,args.cache,args.size)
    data_fingerprint=audit['manifest_sha256']
    split_rows=None
    split_fingerprint=None
    if args.split_manifest:
        from data import load_rows
        split_rows=partition_rows(load_rows(args.data,'train'),(Path(args.data)/'train.csv').read_bytes(),args.split_manifest)
        split_fingerprint=split_rows[2]
        data_fingerprint=hashlib.sha256((data_fingerprint+'|'+split_fingerprint).encode()).hexdigest()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'training.lock'
    # Exclusive lock is intentionally not auto-cleared after an unclean kill.
    with lock.open('x') as file:
        file.write(str(os.getpid()))
    lifecycle = Lifecycle()
    try:
        if (run / 'last.pt').exists() and not args.resume:
            raise ValueError('Existing run: use --resume or a new run directory')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        memory_format=torch.channels_last if args.memory_format=='channels_last' else torch.contiguous_format
        generator = torch.Generator().manual_seed(args.seed)
        train = FundusDataset(args.data, 'train', args.size, args.cache, True,augmentation=args.augmentation)
        valid = FundusDataset(args.data, 'train' if split_rows else 'valid', args.size, args.cache)
        if split_rows:
            train.rows,valid.rows=split_rows[:2]
        trainloader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                 pin_memory=device.type=='cuda', worker_init_fn=seed_worker, generator=generator)
        validloader = DataLoader(valid, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                 pin_memory=device.type=='cuda', worker_init_fn=seed_worker)
        model = build_model(args.pretrain if not args.resume and not args.init_checkpoint else None,
                            architecture=args.architecture).to(device,memory_format=memory_format)
        if args.init_checkpoint and not args.resume:
            initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
            if initial.get('architecture','resnet18')!=args.architecture:
                raise ValueError('Incompatible initialization architecture')
            model.load_state_dict(initial['model'])
            save_json(run/'initialization.json',{'checkpoint':args.init_checkpoint,
                      'sha256':hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
                      'kind':'supervised continuation on official train only'})
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=min(1e-5,args.lr/10))
        scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
        positives = sum(int(r['True_Label']) for r in train.rows)
        # Mild imbalance weighting; ranking remains validation-selected, no threshold fitting.
        pos_weight = args.pos_weight if args.pos_weight is not None else math.sqrt((len(train)-positives)/positives)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        start_epoch, best = 0, None
        if args.resume:
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            legacy_defaults={'lr':3e-4,'pos_weight':None,'init_checkpoint':None,'architecture':'resnet18',
                             'memory_format':'contiguous','validation_probability_precision':'fp16',
                             'split_manifest':None,'checkpoint_epoch':0,'augmentation':'mild'}
            for key in ['architecture','memory_format','validation_probability_precision','size','epochs','seed','batch','max_steps','workers','lr','pos_weight','init_checkpoint','split_manifest','checkpoint_epoch','augmentation']:
                if saved['config'].get(key,legacy_defaults.get(key)) != vars(args)[key]:
                    raise ValueError('Resume configuration mismatch: '+key)
            if saved.get('data_fingerprint') != data_fingerprint:
                raise ValueError('Resume data fingerprint mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            start_epoch, best = saved['epoch']+1, saved['best']
            random.setstate(saved['rng_python'])
            np.random.set_state(saved['rng_numpy'])
            torch.set_rng_state(saved['rng_torch'])
            if device.type=='cuda':
                torch.cuda.set_rng_state_all(saved['rng_cuda'])
            generator.set_state(saved['rng_loader'])
        if not args.resume:
            save_json(run/'config.json', vars(args))
        else:
            with (run/'resume_events.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps({'start_epoch':start_epoch,'config':vars(args),'time':time.time()})+'\n')
        environment = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                         'device': torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu',
                                         'positive_weight':pos_weight}
        if args.resume:
            if json.loads((run/'environment.json').read_text(encoding='utf-8')) != environment:
                raise ValueError('Resume environment mismatch')
        else:
            save_json(run/'environment.json', environment)
        if split_rows:
            save_json(run/'split_binding.json',dict(source='official train only',manifest_sha256=split_fingerprint,
                      training_rows=len(train),validation_rows=len(valid),data_fingerprint=data_fingerprint))
        lifecycle.transition('training')
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            model.train()
            running, count = 0., 0
            for step, (images, labels, _) in enumerate(trainloader):
                images, labels = images.to(device, non_blocking=True,memory_format=memory_format), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                    loss = loss_fn(model(images).flatten(), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()*len(labels)
                count += len(labels)
                if step % 100 == 0:
                    progress = {'epoch':epoch+1,'step':step,'steps':len(trainloader),'loss':running/count,
                                'elapsed_seconds':round(time.monotonic()-started)}
                    print(json.dumps(progress), flush=True)
                    save_json(run/'progress.json', progress)
                if args.max_steps and step+1 >= args.max_steps:
                    break
            lifecycle.transition('validating')
            model.eval()
            y, p, names = [], [], []
            with torch.inference_mode():
                for images, labels, batch_names in validloader:
                    with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                        logits=model(images.to(device,non_blocking=True,memory_format=memory_format)).flatten()
                        probabilities=(probabilities_from_logits(logits) if args.validation_probability_precision=='fp32'
                                       else logits.sigmoid().float()).cpu().tolist()
                    y.extend(labels.tolist()); p.extend(probabilities); names.extend(batch_names)
            result = dict(metrics(y,p), epoch=epoch+1, train_loss=running/count, seconds=time.monotonic()-started)
            scheduler.step()
            lifecycle.transition('saving')
            if ((args.checkpoint_epoch==0 and is_better(result,best)) or epoch+1==args.checkpoint_epoch):
                best = result
                atomic_save({'model':model.state_dict(), 'size':args.size, 'architecture':args.architecture,
                             'memory_format':args.memory_format,'validation_probability_precision':args.validation_probability_precision,
                             'metrics':best, 'threshold':.5, 'schema_version':1,
                             'data_fingerprint':data_fingerprint,
                             'validation_partition':'internal_train_holdout' if split_rows else 'official_valid',
                             'split_manifest_sha256':split_fingerprint,'checkpoint_epoch':args.checkpoint_epoch,
                             'precision':'fp16' if device.type=='cuda' else 'fp32','augmentation':args.augmentation}, run/'best.pt')
                with (run/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as file:
                    writer=csv.writer(file); writer.writerow(['Image_name','True_Label','Probability'])
                    writer.writerows(zip(names,y,p))
            atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                         'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'config':vars(args),
                         'data_fingerprint':data_fingerprint,'schema_version':1,
                         'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
                         'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                         'rng_loader':generator.get_state()},run/'last.pt')
            with (run/'history.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(result)+'\n')
            save_json(run/'best_metrics.json',best)
            print(json.dumps(result),flush=True)
            lifecycle.transition('complete' if epoch+1==args.epochs else 'training')
        save_json(run/'status.json',{'state':'complete','best':best})
    except BaseException as error:
        save_json(run/'status.json',{'state':'failed','phase':lifecycle.state,'error':str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
