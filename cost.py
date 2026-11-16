"""Organizer timing protocol; actual official 100+5 images must be supplied."""
import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import torch
from predict import Predictor
from utils import save_json


def reference_index(rows):
    reference={r['Image_name'].casefold():float(r['Probability']) for r in rows}
    if len(reference)!=len(rows):
        raise ValueError('Duplicate reference filenames after case normalization')
    return reference


def image_files(directory,expected):
    paths=sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if len(paths)!=expected or len({p.name for p in paths})!=expected:
        raise ValueError('Expected '+str(expected)+' unique images')
    return paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('submission')
    parser.add_argument('pictures_100')
    parser.add_argument('pictures_5')
    parser.add_argument('--checkpoint',default='model/best.pt')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--precision',default='fp16')
    parser.add_argument('--report',default='cost_report.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    with open(args.submission,encoding='utf-8-sig',newline='') as file:
        rows=list(csv.DictReader(file))
    reference=reference_index(rows)
    pictures=image_files(args.pictures_100,100)
    warmup=image_files(args.pictures_5,5)
    if {p.resolve() for p in pictures} & {p.resolve() for p in warmup}:
        raise ValueError('Warmup and measured images overlap')
    predictor=Predictor(args.checkpoint,args.device,args.precision)
    for path in warmup:
        predictor.predict_one(path)
    durations={p.name:[] for p in pictures}
    mismatches=set()
    with tempfile.TemporaryDirectory(dir=Path(args.report).resolve().parent) as directory:
        for repeat in range(3):
            for path in pictures:
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                start=time.perf_counter()
                probability=predictor.predict_one(path)
                output=Path(directory)/(path.name+'.csv')
                with output.open('w',encoding='utf-8',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow(['Image_name','Probability','Pred_Label'])
                    writer.writerow([path.name,format(probability,'.10g'),int(probability>=.5)])
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                durations[path.name].append((time.perf_counter()-start)*1000)
                if path.name.casefold() not in reference or abs(reference[path.name.casefold()]-probability)>1e-7:
                    mismatches.add(path.name)
    mean=sum(sum(times)/3 for times in durations.values())/100
    report={'mean_ms':mean,'limit_ms':100,'within_limit':mean<=100,'mismatches':len(mismatches),
            'batch':1,'rounds':3,'precision':args.precision,
            'gpu':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
            'timings_ms':durations,'note':'Official compliance only established on organizer hardware and samples'}
    save_json(args.report,report)
    print('[CONCLUSION]',json.dumps({k:v for k,v in report.items() if k!='timings_ms'},ensure_ascii=False))
    if mismatches or mean>100:
        raise SystemExit(1)


if __name__=='__main__':
    main()
